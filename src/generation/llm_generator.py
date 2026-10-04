import os
import time
import yaml
import requests
import logging
import json
import sys
import torch
from pathlib import Path
from typing import List, Dict, Any

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Load prompt templates from configs/prompts.yaml ──────────────────────
_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml"
try:
    with open(_PROMPTS_PATH) as _f:
        _GEN_CFG = yaml.safe_load(_f).get("generation", {})
except (FileNotFoundError, Exception):
    _GEN_CFG = {}

def format_context_blocks(retrieved_docs: List[Dict[str, Any]]) -> List[str]:
    """One "[Doc N] <text>" block per doc; shared by the static template and DSPy."""
    return [f"[Doc {i}] {doc.get('text', '')}" for i, doc in enumerate(retrieved_docs, 1)]


def format_context_str(retrieved_docs: List[Dict[str, Any]]) -> str:
    return "\n\n".join(format_context_blocks(retrieved_docs))


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

    # vLLM backend: frequency penalty for the single retry after an output
    # hits max_tokens (a repetition loop). See _generate_vllm.
    LOOP_RETRY_FREQUENCY_PENALTY = 0.5

    def __init__(
        self,
        backend: str = "auto",
        ollama_model: str = "llama3",
        hf_model: str = DEFAULT_HF_MODEL,
        api_url: str = "http://localhost:11434/api/generate",
    ):
        # ── Resolve backend ──────────────────────────────────────────────────
        if backend == "auto":
            # GENERATOR_BACKEND env var lets the SLURM script request the
            # vllm backend (pointing to an already-running vLLM server)
            # without changing Python code.  Falls back to original heuristic.
            env_backend = os.environ.get("GENERATOR_BACKEND", "").strip().lower()
            if env_backend in {"vllm", "ollama", "transformers"}:
                backend = env_backend
            elif torch.cuda.is_available():
                backend = "transformers"
            else:
                backend = "ollama"
            logging.info("[LLMGenerator] backend=auto resolved to: %s", backend)

        self.backend = backend
        self.api_url = api_url

        # ── Generation statistics (GenAI §2 — cost/latency monitoring) ────────
        self._stats = {
            "total_calls": 0,
            "total_latency_s": 0.0,
            "total_prompt_tokens_approx": 0,
            "total_completion_tokens_approx": 0,
            "backend": None,  # filled below
        }

        if self.backend == "ollama":
            self.model_name = ollama_model
            self._ollama_max_retries = 5
            self._ollama_base_wait = 2.0   # seconds; doubles each retry
            self._verify_ollama_reachable()
            logging.info("[LLMGenerator] Using Ollama backend  | model: %s | endpoint: %s",
                         self.model_name, self.api_url)

        elif self.backend == "transformers":
            self.model_name = hf_model
            self._load_hf_pipeline()

        elif self.backend == "vllm":
            # Route generation to a pre-running vLLM server that exposes an
            # OpenAI-compatible API.  No local weights are loaded in this
            # process, freeing GPU memory for SPECTER2 and the reranker.
            self.model_name = hf_model
            self.vllm_url = os.environ.get("VLLM_API_URL", "http://localhost:8000/v1")
            logging.info(
                "[LLMGenerator] Using vLLM backend | url: %s | model: %s",
                self.vllm_url, self.model_name,
            )

        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                "Choose 'auto', 'ollama', 'transformers', or 'vllm'."
            )

        self._stats["backend"] = self.backend

        # ── Optional DSPy-compiled prompt (Finding #5, key_metrics_improvements.md)
        # Opt-in only: unset (default) leaves generate_answer's behavior byte-for-
        # byte unchanged (static configs/prompts.yaml template). Set
        # DSPY_COMPILED_PROMPT_PATH to a compiled-program JSON (produced by
        # src/evaluation/compile_dspy_prompt.py) to route generation through it
        # instead — mirrors the existing GENERATOR_BACKEND env-var opt-in pattern.
        self._dspy_module = None
        dspy_path = os.environ.get("DSPY_COMPILED_PROMPT_PATH", "").strip()
        if dspy_path:
            if self.backend != "vllm":
                logging.warning(
                    "[LLMGenerator] DSPY_COMPILED_PROMPT_PATH is set but backend=%s "
                    "— DSPy generation is only wired to the vllm backend so it can "
                    "reuse the already-running Llama vLLM server instead of loading "
                    "a second copy of the model. Ignoring; using the static template.",
                    self.backend,
                )
            elif not os.path.exists(dspy_path):
                logging.warning(
                    "[LLMGenerator] DSPY_COMPILED_PROMPT_PATH=%s does not exist — "
                    "ignoring, using the static template.", dspy_path,
                )
            else:
                # dspy_module first: it patches typing/litellm for Python 3.10
                # before dspy/litellm are imported (see that module's header).
                from src.generation.dspy_module import ScientificRAGModule
                import dspy
                dspy.settings.configure(
                    lm=dspy.LM(f"openai/{self.model_name}", api_base=self.vllm_url,
                              api_key="EMPTY")
                )
                self._dspy_module = ScientificRAGModule()
                self._dspy_module.load(dspy_path)
                logging.info(
                    "[LLMGenerator] Loaded DSPy-compiled prompt from %s — generation "
                    "will route through it instead of configs/prompts.yaml.", dspy_path,
                )

    # ── Ollama health check ───────────────────────────────────────────────────
    def _verify_ollama_reachable(self):
        """Fail-fast check that Ollama is reachable before processing begins.

        Waits up to ~60 s with exponential backoff so Ollama has time to
        cold-start (model loading can take 10-30 s on CPU).  If Ollama is
        still unreachable after all attempts, raises ConnectionError with
        an actionable message instead of silently poisoning 54/150 rows
        with "System Error" strings.
        """
        # Ollama ≥0.1.29 exposes GET /api/tags (list loaded models).
        # Older versions respond to any GET on the root with 200.
        health_url = self.api_url.replace("/api/generate", "/api/tags")
        max_attempts = 8
        wait = 2.0
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    logging.info(
                        "[Ollama] Health check passed on attempt %d/%d",
                        attempt, max_attempts,
                    )
                    return
                logging.warning(
                    "[Ollama] Health check returned HTTP %d (attempt %d/%d)",
                    resp.status_code, attempt, max_attempts,
                )
            except requests.exceptions.RequestException as exc:
                logging.warning(
                    "[Ollama] Health check failed (attempt %d/%d): %s",
                    attempt, max_attempts, exc,
                )
            if attempt < max_attempts:
                logging.info("[Ollama] Retrying in %.0f s...", wait)
                time.sleep(wait)
                wait = min(wait * 2, 30)
        raise ConnectionError(
            f"Ollama is not reachable at {health_url} after {max_attempts} "
            "attempts (~60 s).  Start the server with 'ollama serve' before "
            "running the pipeline, or set GENERATOR_BACKEND=vllm for GPU mode."
        )

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

        Each block is "[Doc N] <chunk text>"; the chunk text already starts with
        its "Title: … Section: …" header. Blocks follow the order of
        retrieved_docs (paper order by default), and [Doc N] maps to
        retrieved_docs[N-1] — the same indexing ALCE uses on `contexts`.
        No paper ID or reranker score is shown: after paper-scoped retrieval
        the ID is identical on every block, and raw scores invite the model to
        anchor on rank (key_metrics_improvements.md, 2026-10-03, P6).
        """
        context_str = format_context_str(retrieved_docs)

        # Build prompt from configs/prompts.yaml; inline fallback if unavailable.
        template = _GEN_CFG.get("rag_prompt_template", None)
        if template:
            return template.format(context_str=context_str, query=query)
        return (
            "You are a precise scientific research assistant. Answer the question using "
            "ONLY the excerpts below. All excerpts come from the same research paper and "
            "are listed in the order they appear in the paper.\n\n"
            f"<Context>\n{context_str}\n</Context>\n\n"
            "<Instructions>\n"
            "1. Find every excerpt that bears on the question and combine them when needed.\n"
            "2. Write a brief <Reasoning> section naming the excerpts you used.\n"
            "3. Write a <Final Answer> of one or two sentences that states the specific fact "
            "first, copying names, numbers and units exactly.\n"
            "4. End every sentence with its citation in EXACTLY the format [Doc N], citing "
            "only the excerpts that directly state the fact (usually one, at most two).\n"
            "5. Only if no excerpt mentions what the question asks about, reply exactly with: "
            "\"The retrieved documents do not contain enough information to answer this.\"\n"
            f"</Instructions>\n\n<User Query>\n{query}\n\n"
            "<Output Format>\n<Reasoning>\n(which excerpts answer the question)\n"
            "</Reasoning>\n<Final Answer>\n(one or two cited sentences)\n</Final Answer>\n"
        )

    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generates an answer using the configured backend (Ollama or HuggingFace transformers).
        Routes automatically based on self.backend set at init time.

        Tracks per-call latency and approximate token counts (GenAI §2).
        """
        if not retrieved_docs:
            return "No documents were retrieved. Cannot generate an answer."

        if self._dspy_module is not None:
            return self._generate_via_dspy(query, retrieved_docs)

        full_prompt = self._build_prompt(query, retrieved_docs)

        # ── Latency + token tracking ─────────────────────────────────────────
        # Approximate prompt tokens via whitespace split (±10% vs BPE for
        # Llama 3 tokenizer).  Exact counting would require loading the
        # tokenizer at init, which is unnecessary for observability.
        prompt_tokens_approx = len(full_prompt.split())

        t0 = time.time()
        if self.backend == "ollama":
            answer = self._generate_ollama(full_prompt)
        elif self.backend == "vllm":
            answer = self._generate_vllm(full_prompt)
        else:
            answer = self._generate_transformers(full_prompt)
        elapsed = time.time() - t0

        completion_tokens_approx = len(answer.split())

        # Update cumulative stats
        self._stats["total_calls"] += 1
        self._stats["total_latency_s"] += elapsed
        self._stats["total_prompt_tokens_approx"] += prompt_tokens_approx
        self._stats["total_completion_tokens_approx"] += completion_tokens_approx

        logging.info(
            "[LLMGenerator] call=%d | backend=%s | latency=%.1fs | "
            "prompt_tok≈%d | completion_tok≈%d | cumulative_latency=%.1fs",
            self._stats["total_calls"], self.backend, elapsed,
            prompt_tokens_approx, completion_tokens_approx,
            self._stats["total_latency_s"],
        )
        return answer

    def _generate_via_dspy(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generation path for the opt-in DSPy-compiled prompt (Finding #5).
        Renders the module's typed output back into the exact tag shape
        extract_final_answer expects (dspy_module.render_tagged_output), and
        tracks latency/token stats the same way generate_answer does so
        log_generation_summary behaves identically either way.
        """
        from src.generation.dspy_module import (
            DEFAULT_PAPER_FOCUS_HINT,
            format_context_blocks,
            render_tagged_output,
        )

        context_blocks = format_context_blocks(retrieved_docs)
        prompt_tokens_approx = sum(len(c.split()) for c in context_blocks) + len(query.split())

        t0 = time.time()
        pred = self._dspy_module(
            context=context_blocks,
            paper_focus_hint=DEFAULT_PAPER_FOCUS_HINT,
            question=query,
        )
        answer = render_tagged_output(pred.reasoning, pred.cited_answer)
        elapsed = time.time() - t0

        completion_tokens_approx = len(answer.split())
        self._stats["total_calls"] += 1
        self._stats["total_latency_s"] += elapsed
        self._stats["total_prompt_tokens_approx"] += prompt_tokens_approx
        self._stats["total_completion_tokens_approx"] += completion_tokens_approx

        logging.info(
            "[LLMGenerator] call=%d | backend=dspy | latency=%.1fs | "
            "prompt_tok≈%d | completion_tok≈%d | cumulative_latency=%.1fs",
            self._stats["total_calls"], elapsed,
            prompt_tokens_approx, completion_tokens_approx,
            self._stats["total_latency_s"],
        )
        return answer

    def log_generation_summary(self) -> Dict[str, Any]:
        """Log and return cumulative generation statistics.

        Call this after the evaluation run completes to get a summary of
        total latency, token usage, and per-call averages.  Useful for
        cost estimation and performance budgeting (GenAI §2).
        """
        s = self._stats
        n = s["total_calls"] or 1  # avoid division by zero
        summary = {
            "backend": s["backend"],
            "total_calls": s["total_calls"],
            "total_latency_s": round(s["total_latency_s"], 2),
            "avg_latency_s": round(s["total_latency_s"] / n, 2),
            "total_prompt_tokens_approx": s["total_prompt_tokens_approx"],
            "total_completion_tokens_approx": s["total_completion_tokens_approx"],
            "total_tokens_approx": (
                s["total_prompt_tokens_approx"] + s["total_completion_tokens_approx"]
            ),
        }
        logging.info(
            "[LLMGenerator] === Generation Summary ===\n"
            "  Backend:              %s\n"
            "  Total calls:          %d\n"
            "  Total latency:        %.1f s (avg %.1f s/call)\n"
            "  Prompt tokens (≈):    %d\n"
            "  Completion tokens (≈):%d\n"
            "  Total tokens (≈):     %d",
            summary["backend"], summary["total_calls"],
            summary["total_latency_s"], summary["avg_latency_s"],
            summary["total_prompt_tokens_approx"],
            summary["total_completion_tokens_approx"],
            summary["total_tokens_approx"],
        )
        return summary

    # ── Ollama backend ────────────────────────────────────────────────────────
    def _generate_ollama(self, full_prompt: str) -> str:
        """Calls the local Ollama HTTP API with retry + exponential backoff.

        Ollama on CPU is prone to transient failures:
          - Cold model loading (10-30 s for 8B params on CPU)
          - GC pauses under memory pressure
          - Connection refused if 'ollama serve' was restarted mid-run

        Without retries the original code immediately returned a permanent
        "System Error" string, which poisoned 54/150 rows in the evaluation
        dataset.  Retrying with backoff recovers from all transient modes.
        """
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": 0.1  # Low temperature reduces hallucination
            }
        }

        last_exc = None
        wait = self._ollama_base_wait

        for attempt in range(1, self._ollama_max_retries + 1):
            try:
                logging.info(
                    "[Ollama] Attempt %d/%d — sending prompt to %s",
                    attempt, self._ollama_max_retries, self.api_url,
                )
                response = requests.post(
                    self.api_url, json=payload, stream=True, timeout=300,
                )
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

            except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
                last_exc = exc
                logging.warning(
                    "[Ollama] Attempt %d/%d failed: %s",
                    attempt, self._ollama_max_retries, exc,
                )
                if attempt < self._ollama_max_retries:
                    logging.info("[Ollama] Retrying in %.0f s...", wait)
                    time.sleep(wait)
                    wait = min(wait * 2, 60)

        # All retries exhausted
        logging.error(
            "[Ollama] All %d attempts failed. Last error: %s",
            self._ollama_max_retries, last_exc,
        )
        return (
            f"System Error: Could not connect to Ollama at {self.api_url} "
            f"after {self._ollama_max_retries} retries. Is 'ollama serve' running?"
        )

    # ── vLLM backend (OpenAI-compatible server) ──────────────────────────────────
    def _generate_vllm(self, full_prompt: str) -> str:
        """Calls a vLLM server's OpenAI-compatible chat-completions endpoint."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for the vllm backend. "
                "Install it with: pip install openai"
            ) from exc

        logging.info(
            "[vLLM] Sending prompt to %s | model=%s",
            self.vllm_url, self.model_name,
        )
        client = OpenAI(base_url=self.vllm_url, api_key="EMPTY")

        def _complete(**extra):
            return client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=1024,
                temperature=0.1,
                **extra,
            ).choices[0]

        choice = _complete()
        answer = choice.message.content or ""
        if choice.finish_reason == "length":
            # A short cited answer never needs 1024 tokens: hitting the cap
            # means a repetition loop (seen when the answer is absent from the
            # excerpts), and the truncated output has no <Final Answer>, so the
            # whole loop would be scored as the answer. Retry once with a
            # frequency penalty, which (unlike repetition_penalty) only counts
            # generated tokens, so copying numbers from the context is not
            # discouraged. Normal rows never reach this branch.
            logging.warning("[vLLM] Output hit max_tokens (likely a repetition loop); "
                            "retrying once with frequency_penalty=%.1f.", self.LOOP_RETRY_FREQUENCY_PENALTY)
            retry = _complete(frequency_penalty=self.LOOP_RETRY_FREQUENCY_PENALTY)
            answer = retry.message.content or answer
        print("\n" + "="*40 + " LLM OUTPUT (vLLM) " + "="*40 + "\n")
        print(answer)
        print("\n" + "="*92 + "\n")
        return answer

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

    # ── HyDE: Hypothetical Document Embedding ──────────────────────────────────
    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generates a brief hypothetical passage for HyDE dense retrieval.

        Asks the LLM to write a concise 2-3 sentence factual passage that
        would directly answer *query*, as if extracted from a scientific paper.
        This passage is then encoded by SPECTER2 in place of the raw query,
        closing the query-document semantic gap.

        The prompt deliberately carries no paper ID: the model cannot map an
        arXiv ID to the paper's content, so the note only added noise
        (key_metrics_improvements.md, 2026-10-03, P5).

        Reference:
            Gao et al. (2022). Precise Zero-Shot Dense Retrieval without
            Relevance Labels (HyDE). arXiv:2212.10496. ACL 2023.
            — Encoding a hypothetical passage improves nDCG@10 by 5-15 points
              on academic benchmarks vs. encoding the raw question.
        """
        hyde_prompt = (
            "You are a scientific assistant. Write a concise 2-3 sentence "
            "factual passage that would directly answer the following question, "
            "as if it were extracted verbatim from a scientific paper. "
            "State facts directly without hedging.\n\n"
            f"Question: {query}\n\n"
            "Hypothetical passage:"
        )
        if self.backend == "ollama":
            payload = {
                "model": self.model_name,
                "prompt": hyde_prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200},
            }
            try:
                resp = requests.post(self.api_url, json=payload, timeout=60)
                resp.raise_for_status()
                return resp.json().get("response", "").strip() or query
            except Exception as exc:
                logging.warning("[HyDE/Ollama] generation failed: %s", exc)
                return query  # fallback: use original query
        elif self.backend == "vllm":
            try:
                from openai import OpenAI
                client = OpenAI(base_url=self.vllm_url, api_key="EMPTY")
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": hyde_prompt}],
                    max_tokens=200,
                    temperature=0.3,
                )
                text = (resp.choices[0].message.content or "").strip()
                return text if text else query
            except Exception as exc:
                logging.warning("[HyDE/vLLM] generation failed: %s", exc)
                return query
        else:
            outputs = self.pipe(
                hyde_prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.3,
                return_full_text=False,
            )
            text = outputs[0]["generated_text"].strip()
            return text if text else query
