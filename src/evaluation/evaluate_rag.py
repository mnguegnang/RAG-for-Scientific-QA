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
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from ragas.llms import llm_factory
from openai import OpenAI, AsyncOpenAI
import httpx

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

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
            response = self.llm.invoke([HumanMessage(content=prompt)])
            # All LangChain chat models (ChatOpenAI, ChatOllama, etc.) return
            # a BaseMessage; extract the text content before calling .upper()
            if hasattr(response, "content"):
                answer_text = response.content
            else:
                answer_text = str(response)
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
            citations = re.findall(r'\[Doc (\d+)\]', sentence)
            
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
        
        # 1. Async client for RAGAS (llm_factory uses async internally).
        # Using a sync OpenAI client inside RAGAS's asyncio eval loop
        # exhausts the connection pool and raises "Connection error" on all
        # 804 judge calls.  AsyncOpenAI is the correct client type here.
        # Reference: RAGAS docs — https://docs.ragas.io/en/latest/
        # Explicit httpx connection limits prevent pool exhaustion when
        # RAGAS fires many concurrent sub-calls (statement extraction,
        # NLI checks, question generation).  Without limits the default
        # pool of 100 connections fills up → "Connection error" on all
        # remaining requests.
        # keepalive_expiry must stay strictly below the vLLM/Uvicorn
        # server-side keep-alive idle timeout.  Uvicorn's hard default is 5 s;
        # setting keepalive_expiry=4.0 ensures httpx evicts idle connections
        # 1 s before the server closes the socket, preventing stale-connection
        # RemoteProtocolError.  (The --timeout-keep-alive CLI flag was removed
        # in the installed vLLM version, so the default cannot be overridden.)
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=4.0,  # 1 s margin below Uvicorn default 5 s
            ),
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=60.0),
        )
        async_client = AsyncOpenAI(
            base_url=f"http://localhost:{VLLM_PORT}/v1",
            api_key="EMPTY",
            http_client=_http_client,
        )

        # 2. RAGAS structured-output judge (uses instructor + JSON mode)
        # max_tokens=2048: ContextPrecision needs ~50 tokens × 7 docs = ~350 min;
        # Faithfulness extracts then verifies all answer claims (~700-1500 tokens).
        # 512 was too small → truncated JSON → InstructorRetryException × 804.
        ragas_llm = llm_factory(
            model="meta-llama/Llama-3.1-8B-Instruct",
            client=async_client,
            max_tokens=8192, #2048, increased to 8192 to handle larger outputs and avoid truncation that affects Answer Relevancy metric.
            temperature=0.0,
        )
        
        # 3. Provide the LangChain equivalent for your ALCE NLI checks
        local_judge_llm = ChatOpenAI(
            model="meta-llama/Llama-3.1-8B-Instruct",
            base_url=f"http://localhost:{VLLM_PORT}/v1",
            api_key="EMPTY",
            temperature=0.0,
            max_tokens=8,     # ALCE only needs TRUE/FALSE — cap to prevent runaway generation
        )
        
        # 4. LangChain HuggingFaceEmbeddings wrapped with LangchainEmbeddingsWrapper.
        # Root cause of AttributeError: RagasHuggingFaceEmbeddings (huggingface_provider)
        # extends BaseRagasEmbedding (embed_text interface only), NOT BaseRagasEmbeddings
        # (embed_query interface).  AnswerRelevancy internally calls embed_query, which is
        # absent on the old class.  LangchainEmbeddingsWrapper extends BaseRagasEmbeddings
        # and delegates embed_query to the wrapped LangChain object, which implements it.
        # langchain_huggingface docs confirm: prompts, default_prompt_name, trust_remote_code
        # are valid model_kwargs (forwarded to SentenceTransformer constructor, sbert.net docs).
        ragas_embeddings = LangchainEmbeddingsWrapper(
            LCHuggingFaceEmbeddings(
                model_name="nomic-ai/nomic-embed-text-v1.5",
                model_kwargs={
                    "device": "cuda",
                    "trust_remote_code": True,
                    "prompts": {"query": "search_query: ", "document": "search_document: "},
                    "default_prompt_name": "query",
                },
                encode_kwargs={"normalize_embeddings": True},
            )
        )

    else:
        logging.info("No GPU Detected. Connecting to local Ollama server...")
        is_gpu = False
        
        ollama_client = OpenAI(
            base_url="http://localhost:11434/v1", 
            api_key="ollama"
        )
        
        ragas_llm = llm_factory(model="llama3", client=ollama_client, max_tokens=512, temperature=0.0)
        local_judge_llm = ChatOllama(model="llama3", temperature=0.0, num_predict=8)
        
        # CPU Fallback — same wrapper pattern as GPU path
        ragas_embeddings = LangchainEmbeddingsWrapper(
            LCHuggingFaceEmbeddings(
                model_name="nomic-ai/nomic-embed-text-v1.5",
                model_kwargs={
                    "device": "cpu",
                    "trust_remote_code": True,
                    "prompts": {"query": "search_query: ", "document": "search_document: "},
                    "default_prompt_name": "query",
                },
                encode_kwargs={"normalize_embeddings": True},
            )
        )
        
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
    eval_dataset = Dataset.from_pandas(df)
    metrics = [ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy()]

    # 2. DYNAMIC HARDWARE CONFIGURATION
    if is_gpu:
        eval_batch_size = 16
        max_workers = 20 #vLLM can handle 20 concurrent    # ≤2 concurrent rows: limits simultaneous httpx connections to vLLM
        timeout = 600.0    # 10 minutes max (GPUs are fast); float required by httpx
    else:
        eval_batch_size = 1
        max_workers = 1    # CRITICAL: Prevent Ollama queue timeouts
        timeout = 2400.0   # 40 minutes max (CPUs are slow); float required by httpx

    # FIX: RunConfig(timeout=...) sets the RAGAS-level wait but does NOT propagate
    # to the httpx socket layer used by LangChain / the OpenAI SDK.  httpx has its
    # own default read timeout (~300 s) that drops the socket before RAGAS fires
    # when vLLM is generating long batches.  We must inject the hardware-derived
    # timeout directly into a custom httpx.AsyncClient passed to ChatOpenAI.
    # Sources:
    #   OpenAI Python SDK docs — custom httpx.Client for timeout override.
    #   LangChain ChatOpenAI API — http_client / http_async_client kwargs.
    if is_gpu:
        _ragas_http_sync = httpx.Client(
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=4.0,  # 1 s margin below Uvicorn default 5 s
            ),
            timeout=httpx.Timeout(connect=30.0, read=timeout, write=30.0, pool=60.0),
        )
        _ragas_http_async = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=4.0,  # 1 s margin below Uvicorn default 5 s
            ),
            timeout=httpx.Timeout(connect=30.0, read=timeout, write=30.0, pool=60.0),
        )
        ragas_llm = ChatOpenAI(
            model="meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            temperature=0.0,
            max_tokens=2048,
            max_retries=6,
            http_client=_ragas_http_sync,
            http_async_client=_ragas_http_async,
        )

    run_config = RunConfig(
        timeout=timeout,
        max_retries=6,     # More retries: vLLM may queue briefly under load
        max_wait=120,      # Longer backoff ceiling (was 60 s)
        max_workers=max_workers,
    )

    logging.info(f"Starting Local RAGAS Evaluation (GPU Mode: {is_gpu})...")
    ragas_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
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
        valid = series.dropna()
        if len(valid) == 0:
            return f"N/A — all {len(series)} rows failed"
        n_failed = len(series) - len(valid)
        suffix = f"  [{n_failed} NaN skipped]" if n_failed > 0 else ""
        return f"{valid.mean():.4f}{suffix}"

    logging.info("\n========== EVALUATION REPORT ==========")
    logging.info(f"Context Precision:       {_fmt(final_df.get('context_precision', pd.Series(dtype=float)))}")
    logging.info(f"Context Recall:          {_fmt(final_df.get('context_recall', pd.Series(dtype=float)))}")
    logging.info(f"Faithfulness:            {_fmt(final_df.get('faithfulness', pd.Series(dtype=float)))}")
    logging.info(f"Answer Relevancy:        {_fmt(final_df.get('answer_relevancy', pd.Series(dtype=float)))}")
    logging.info(f"ALCE Citation Precision: {_fmt(final_df['alce_citation_precision'])}")
    logging.info(f"ALCE Citation Recall:    {_fmt(final_df['alce_citation_recall'])}")
    logging.info("=======================================")

if __name__ == "__main__":
    run_evaluation()