import os
import ast
import re
import logging
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple
from datasets import Dataset

# Import NLTK for robust sentence separation
import nltk

# RAGAS specific imports
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from openai import OpenAI

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# Globally force HuggingFace to trust custom architectures (Fixes the Nomic bug)
os.environ["HF_TRUST_REMOTE_CODE"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ALCE_RAGASevaluator:
    """
    Implements the ALCE (Gao et al., 2023) metrics for Citation Precision and Recall 
    using a local LLM-as-a-Judge for Natural Language Inference (NLI) Entailment.
    And perform RAGAS Evaluation.
    """
    def __init__(self, llm):
        self.llm = llm

        logging.info("Verifying NLTK tokenizer models...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True) 
        except Exception as e:
            logging.warning(f"Failed to verify NLTK models: {e}")

    def _check_entailment(self, claim: str, cited_text: str) -> bool:
        """Prompts the local LLM to perform an NLI check."""
        if not cited_text.strip():
            return False
            
        prompt = f"""
        Task: Natural Language Inference.
        Determine if the following CLAIM is fully supported by the provided CITED DOCUMENT.
        
        CITED DOCUMENT: "{cited_text}"
        CLAIM: "{claim}"
        
        Instructions: 
        If the document contains evidence that proves the claim, output exactly the word "TRUE".
        If the document does not contain the evidence, or contradicts the claim, output exactly the word "FALSE".
        Output nothing else.
        """
        try:
            # FIX: Handle different input/output formats between Ollama and HuggingFace
            if isinstance(self.llm, ChatOllama):
                response = self.llm.invoke([HumanMessage(content=prompt)])
                answer_text = response.content
            else:
                # HuggingFace pipeline takes string and returns string
                answer_text = self.llm.invoke(prompt)
                
            return "TRUE" in answer_text.upper()
        except Exception as e:
            logging.error(f"Local LLM Entailment check failed: {e}")
            return False

    def calculate_metrics(self, answer: str, contexts: list) -> Tuple[float, float]:
        sentences = nltk.sent_tokenize(answer)
        
        if not sentences:
            return 0.0, 0.0

        supported_sentences = 0
        sentences_with_citations = 0

        for sentence in sentences:
            citations = re.findall(r'$$Doc (\d+)$$', sentence)
            
            if citations:
                sentences_with_citations += 1
                
                cited_texts = []
                for doc_id_str in citations:
                    doc_idx = int(doc_id_str) - 1
                    if 0 <= doc_idx < len(contexts):
                        cited_texts.append(contexts[doc_idx])
                
                combined_cited_text = " ".join(cited_texts)
                
                if self._check_entailment(sentence, combined_cited_text):
                    supported_sentences += 1

        precision = supported_sentences / sentences_with_citations if sentences_with_citations > 0 else 0.0
        recall = supported_sentences / len(sentences)
        
        return precision, recall

def get_hardware_aware_models():
    """Detects GPU. Connects to local vLLM server if A100 is present."""
    
    #MATCH THIS PORT TO YOUR vLLM LOGS (8000)
    VLLM_PORT = 8000 
    
    if torch.cuda.is_available():
        logging.info(f"GPU Detected! Connecting to local vLLM server on port {VLLM_PORT}...")
        is_gpu = True
        
        # 1. Create an OpenAI client pointing to your local vLLM server
        local_client = OpenAI(
            base_url=f"http://localhost:{VLLM_PORT}/v1",
            api_key="EMPTY"  # vLLM does not require an API key
        )
        
        # 2. Use modern Ragas llm_factory
        ragas_llm = llm_factory(
            model="meta-llama/Llama-3.1-8B-Instruct", 
            client=local_client
        )
        
        # 3. Provide the LangChain equivalent for your ALCE NLI checks
        local_judge_llm = ChatOpenAI(
            model="meta-llama/Llama-3.1-8B-Instruct",
            base_url=f"http://localhost:{VLLM_PORT}/v1",
            api_key="EMPTY",
            temperature=0.0
        )
        
        # 4. FIX: Use LangChain's HuggingFace wrapper which correctly passes trust_remote_code
        #lc_embeddings = HuggingFaceEmbeddings(
        #    model_name="nomic-ai/nomic-embed-text-v1.5",
        #    model_kwargs={'device': 'cuda', 'trust_remote_code': True},
        #    encode_kwargs={'normalize_embeddings': True}
        #)
        #ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

        # The os.environ["HF_TRUST_REMOTE_CODE"] = "1" at the top makes this safe!
        lc_embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={"device": "cuda", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

    else:
        logging.info("No GPU Detected. Connecting to local Ollama server...")
        is_gpu = False
        
        ollama_client = OpenAI(
            base_url="http://localhost:11434/v1", 
            api_key="ollama"
        )
        
        ragas_llm = llm_factory(model="llama3", client=ollama_client)
        local_judge_llm = ChatOllama(model="llama3", temperature=0.0)
        
        # CPU Fallback also uses LangChain wrapper
        lc_embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)
        
    return local_judge_llm, ragas_llm, ragas_embeddings, is_gpu

def run_evaluation(input_csv: str = None, output_csv: str = None):
    
    _project_root = Path(__file__).resolve().parents[2]
    if input_csv is None:
        input_csv = str(_project_root / "data" / "evaluation_dataset.csv")
    if output_csv is None:
        output_csv = str(_project_root / "data" / "evaluation_report.csv")

    logging.info(f"Loading generated dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    df['contexts'] = df['contexts'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # 1. Catch the is_gpu flag
    #local_judge_llm, local_embeddings, is_gpu = get_hardware_aware_models()
    local_judge_llm, ragas_llm, ragas_embeddings, is_gpu = get_hardware_aware_models()
    
    #ragas_llm = LangchainLLMWrapper(local_judge_llm)
    #ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)
    
    # --- ALCE EVALUATION ---
    logging.info("Calculating ALCE Citation Precision & Recall (Local NLI checks)...")
    alce_evaluator = ALCE_RAGASevaluator(llm=local_judge_llm)
    
    precisions, recalls = [], []
    total_rows = len(df)
    
    for index, row in df.iterrows():
        logging.info(f"Running ALCE Entailment for row {index + 1}/{total_rows}...")
        p, r = alce_evaluator.calculate_metrics(row['answer'], row['contexts'])
        precisions.append(p)
        recalls.append(r)
        
    df['alce_citation_precision'] = precisions
    df['alce_citation_recall'] = recalls
    
    # --- RAGAS EVALUATION ---
    # RAGAS receives the full top-10 reranked contexts, identical to what ALCE uses.
    # Giving RAGAS all 10 docs ensures ContextRecall can find every relevant chunk
    # and Faithfulness/ContextPrecision have the complete evidence set to grade against.
    eval_dataset = Dataset.from_pandas(df)
    metrics = [ContextPrecision(llm=ragas_llm), 
               ContextRecall(llm=ragas_llm), 
               Faithfulness(llm=ragas_llm), 
               AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)]

    # 2. DYNAMIC HARDWARE CONFIGURATION
    if is_gpu:
        eval_batch_size = 16
        max_workers = 16
        timeout = 600      # 10 minutes max (GPUs are fast)
    else:
        eval_batch_size = 1
        max_workers = 1    # CRITICAL: Prevent Ollama queue timeouts
        timeout = 2400     # 40 minutes max (CPUs are slow)

<<<<<<< HEAD
    # --- RunConfig tuned for a single local CPU Ollama instance ---
    #
    # max_workers=1  Ollama serialises all requests (one llama3 process, no true
    #                parallelism). Sending >1 concurrent job means later jobs wait
    #                in Ollama's queue while the timeout clock is already ticking —
    #                structurally guaranteed TimeoutErrors. Set to 1 to match reality.
    #
    # timeout=2400   40 min per call. top_k=10 roughly doubles the prompt length
    #                compared to top_k=5. llama3 on CPU needs 10–30 min to grade
    #                one row with 10 full contexts; 2400 s covers the worst case.
    #
    # max_retries=2  Local failures are slow inference, not transient network blips.
    #                The default 10 retries × 2400 s = up to 6.7 h wasted on one
    #                row before recording NaN. 2 retries catches rare Ollama hiccups.
    #
    # max_wait=30    Short back-off is fine; no rate-limited remote API to respect.
    run_config = RunConfig(
        timeout=2400,    # 40 min per LLM call — covers top_k=10 contexts on CPU
        max_retries=2,   # low: failures are slow inference, not network blips
        max_wait=30,     # short back-off; no rate-limit to respect
        max_workers=1,   # CRITICAL: matches Ollama's true concurrency (serial)
=======
    run_config = RunConfig(
        timeout=timeout,    
        max_retries=2,   
        max_wait=30,     
        max_workers=max_workers,   
>>>>>>> cc6e01ad33bfbf2fa9000592545c986b7eeb4561
    )

    logging.info(f"Starting Local RAGAS Evaluation (GPU Mode: {is_gpu})...")
    ragas_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        #llm=ragas_llm,
        #embeddings=ragas_embeddings,
        run_config=run_config,
        raise_exceptions=False,
        #batch_size=eval_batch_size  # Automatically scales based on hardware
    )
    
    ragas_df = ragas_result.to_pandas().reset_index(drop=True)
    base_df = df[['question', 'ground_truth', 'answer', 'alce_citation_precision', 'alce_citation_recall']].reset_index(drop=True)
    final_df = pd.concat([base_df, ragas_df.drop(columns=[c for c in ['question', 'answer', 'ground_truth', 'contexts'] if c in ragas_df.columns], errors='ignore')], axis=1)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    
    def _fmt(series: pd.Series) -> str:
<<<<<<< HEAD
        """Mean of valid (non-NaN) values; reports how many jobs timed out."""
        valid = series.dropna()
        if len(valid) == 0:
            return f"N/A — all {len(series)} rows failed"
        n_failed = len(series) - len(valid)
        suffix = f"  [{n_failed} NaN skipped]" if n_failed > 0 else ""
        return f"{valid.mean():.4f}{suffix}"

    logging.info("\n========== EVALUATION REPORT ==========")
=======
        valid = series.dropna()
        if len(valid) == 0:
            return f"N/A — all {len(series)} rows failed (timeout/NaN)"
        n_failed = len(series) - len(valid)
        suffix = f"  [{n_failed} NaN/timeout skipped]" if n_failed > 0 else ""
        return f"{valid.mean():.4f}{suffix}"

    logging.info("\n========== LOCAL EVALUATION REPORT ==========")
>>>>>>> cc6e01ad33bfbf2fa9000592545c986b7eeb4561
    logging.info(f"Context Precision:       {{_fmt(final_df.get('context_precision', pd.Series(dtype=float)))}}")
    logging.info(f"Context Recall:          {{_fmt(final_df.get('context_recall', pd.Series(dtype=float)))}}")
    logging.info(f"Faithfulness:            {{_fmt(final_df.get('faithfulness', pd.Series(dtype=float)))}}")
    logging.info(f"Answer Relevancy:        {{_fmt(final_df.get('answer_relevancy', pd.Series(dtype=float)))}}")
    logging.info(f"ALCE Citation Precision: {{_fmt(final_df['alce_citation_precision'])}}")
    logging.info(f"ALCE Citation Recall:    {{_fmt(final_df['alce_citation_recall'])}}")
    logging.info("=======================================")
    logging.info("======================================================")
=======
    logging.info("=======================================")
>>>>>>> cc6e01ad33bfbf2fa9000592545c986b7eeb4561

if __name__ == "__main__":
    run_evaluation()