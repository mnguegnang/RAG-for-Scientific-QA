import requests
import logging
import json
import sys
import torch
from typing import List, Dict, Any

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default HuggingFace model used when backend="transformers"
# Llama-3.1-8B-Instruct fits in ~16GB VRAM (BF16); well within one A100-80GB.
# Reference: Meta AI (2024) https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


class LocalLLMGenerator:
    """
    Unified LLM generator supporting two backends:

    - ``"ollama"``        : Calls a local Ollama daemon via HTTP (default on CPU / laptop).
                            Requires ``ollama serve`` to be running.
    - ``"transformers"`` : Loads the model directly into GPU memory using HuggingFace
                            ``transformers.pipeline`` + ``accelerate`` device_map="auto".
                            No daemon needed; works in SLURM batch jobs. (default on GPU)
    - ``"auto"``         : Selects ``"transformers"`` when a CUDA GPU is available,
                            otherwise falls back to ``"ollama"``.

    Args:
        backend (str):      One of ``"auto"``, ``"ollama"``, ``"transformers"``.
        ollama_model (str): Ollama model tag, e.g. ``"llama3"``.
        hf_model (str):     HuggingFace model ID, e.g. ``"meta-llama/Llama-3.1-8B-Instruct"``.
        api_url (str):      Ollama REST endpoint (only used when backend=="ollama").
    """

    def __init__(
        self,
        backend: str = "auto",
        ollama_model: str = "llama3",
        hf_model: str = DEFAULT_HF_MODEL,
        api_url: str = "http://localhost:11434/api/generate",
    ):
        # ── Resolve backend ──────────────────────────────────────────────────
        if backend == "auto":
            backend = "transformers" if torch.cuda.is_available() else "ollama"
            logging.info("[LLMGenerator] backend=auto resolved to: %s", backend)

        self.backend = backend
        self.api_url = api_url

        if self.backend == "ollama":
            self.model_name = ollama_model
            logging.info("[LLMGenerator] Using Ollama backend  | model: %s | endpoint: %s",
                         self.model_name, self.api_url)

        elif self.backend == "transformers":
            self.model_name = hf_model
            self._load_hf_pipeline()

        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'auto', 'ollama', or 'transformers'.")

    # ── HuggingFace pipeline loader ───────────────────────────────────────────
    def _load_hf_pipeline(self):
        """Loads the HuggingFace text-generation pipeline onto available GPU(s)."""
        try:
            from transformers import pipeline, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "The 'transformers' package is required for the transformers backend. "
                "Install it with: pip install transformers accelerate"
            ) from e

        logging.info("[LLMGenerator] Loading HF model '%s' with device_map=auto (BF16)...",
                     self.model_name)
        logging.info("[LLMGenerator] Available GPUs: %d", torch.cuda.device_count())

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # device_map="auto" uses HuggingFace Accelerate to automatically distribute
        # the model across all available GPUs. torch_dtype=bfloat16 halves VRAM usage
        # with no accuracy loss on A100s (native BF16 support).
        # Reference: HuggingFace Accelerate docs — https://huggingface.co/docs/accelerate
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        logging.info("[LLMGenerator] HF pipeline ready on device_map=auto")

    def _build_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Constructs the Chain-of-Thought and Citation prompt.

        HybridRetriever.search() returns dicts with keys: 'text', 'doc_id', 'score'.
        We use 'doc_id' directly as the source label so citations like [Doc 1] can
        be traced back to real paper IDs rather than 'Unknown Source'.
        """
        # 1. Format the context
        context_str = ""
        for i, doc in enumerate(retrieved_docs, 1):
            # Use doc_id from HybridRetriever output; fall back gracefully if absent
            doc_id = doc.get('doc_id', doc.get('id', f'doc_{i}'))
            score  = doc.get('rerank_score', doc.get('score', None))
            score_str = f"  [relevance: {score:.4f}]" if score is not None else ""

            context_str += f"[Doc {i}]\nSource: {doc_id}{score_str}\nContent: {doc.get('text', '')}\n\n"

        # 2. Construct the strict instructional prompt
        prompt = f"""You are a precise scientific AI research assistant. Answer the user's query based ONLY on the provided context.

<Context>
{context_str}
</Context>

<Instructions>
1. Comprehension: Read the context carefully. If the context does not contain the answer, reply exactly with: "The retrieved documents do not contain enough information to answer this." Do not guess.
2. Chain of Thought: Provide a brief <Reasoning> section where you outline your logic based on the documents.
3. Citations: Provide a <Final Answer> section. Every factual claim MUST end with the citation tag of the document it originated from, e.g., [Doc 1] or [Doc 2].
</Instructions>

<User Query>
{query}

<Output Format>
<Reasoning>
(your step-by-step thinking)
</Reasoning>
<Final Answer>
(your synthesized, cited answer)
</Final Answer>
"""
        return prompt

    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generates an answer using the configured backend (Ollama or HuggingFace transformers).
        Routes automatically based on self.backend set at init time.
        """
        if not retrieved_docs:
            return "No documents were retrieved. Cannot generate an answer."

        full_prompt = self._build_prompt(query, retrieved_docs)

        if self.backend == "ollama":
            return self._generate_ollama(full_prompt)
        else:
            return self._generate_transformers(full_prompt)

    # ── Ollama backend ────────────────────────────────────────────────────────
    def _generate_ollama(self, full_prompt: str) -> str:
        """Calls the local Ollama HTTP API and streams tokens to stdout."""
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": 0.1  # Low temperature reduces hallucination
            }
        }
        try:
            logging.info("[Ollama] Sending prompt to %s. Awaiting stream...", self.api_url)
            response = requests.post(self.api_url, json=payload, stream=True, timeout=300)
            response.raise_for_status()

            print("\n" + "="*40 + " LLM OUTPUT (Ollama) " + "="*40 + "\n")

            full_response = ""
            char_count = 0
            for line in response.iter_lines():
                if line:
                    body = json.loads(line.decode("utf-8"))
                    token = body.get("response", "")
                    full_response += token
                    # Text wrapping at 80 chars
                    if token == "\n":
                        char_count = 0
                        sys.stdout.write(token)
                    else:
                        char_count += len(token)
                        if char_count >= 80 and token.startswith(" "):
                            sys.stdout.write("\n" + token.lstrip())
                            char_count = len(token.lstrip())
                        else:
                            sys.stdout.write(token)
                    sys.stdout.flush()
                    if body.get("done"):
                        break

            print("\n\n" + "="*92 + "\n")
            return full_response

        except json.JSONDecodeError as e:
            logging.error("[Ollama] Failed to parse stream: %s", e)
            return "System Error: JSON Parsing Failure."
        except requests.exceptions.RequestException as e:
            logging.error("[Ollama] Connection failed: %s", e)
            return f"System Error: Could not connect to Ollama at {self.api_url}. Is 'ollama serve' running?"

    # ── HuggingFace transformers backend ──────────────────────────────────────
    def _generate_transformers(self, full_prompt: str) -> str:
        """Generates answer using HuggingFace pipeline directly on GPU."""
        logging.info("[HF] Generating answer with %s on device_map=auto...", self.model_name)

        print("\n" + "="*40 + " LLM OUTPUT (HuggingFace) " + "="*40 + "\n")

        # pipeline returns a list of dicts: [{'generated_text': '...'}]
        # max_new_tokens=1024 allows detailed answers; temperature=0.1 for factual determinism.
        outputs = self.pipe(
            full_prompt,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            return_full_text=False,  # Return only newly generated tokens, not the prompt
        )
        answer = outputs[0]["generated_text"]
        print(answer)
        print("\n" + "="*92 + "\n")
        return answer