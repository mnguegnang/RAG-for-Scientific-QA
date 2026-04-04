import os
import ast
import re
import logging
import pandas as pd
from typing import Tuple
from datasets import Dataset

# Import NLTK for robust sentence separation (boundary) e.g., identifies that Dr. is not sentence
import nltk


# RAGAS specific imports
from ragas import evaluate
from ragas.run_config import RunConfig

# Import from ragas.metrics (classic API) — compatible with evaluate()
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Modern LangChain Core & Local Ollama Imports
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ALCE_RAGASevaluator:
    """
    Implements the ALCE (Gao et al., 2023) metrics for Citation Precision and Recall 
    using a local LLM-as-a-Judge for Natural Language Inference (NLI) Entailment.
    And peform RAGAS Evaluation
    """
    def __init__(self, llm: ChatOllama):
        self.llm = llm

        # Safely download the NLTK Punkt tokenizer models on initialization.
        # quiet=True prevents spamming the terminal if it's already downloaded.
        logging.info("Verifying NLTK tokenizer models...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True) # Required for NLTK 3.8+ compatibility
        except Exception as e:
            logging.warning(f"Failed to verify NLTK models. Tokenization may fail if not cached: {e}")

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
            # Robust parsing for local models that might still add a period (e.g., "TRUE.")
            return "TRUE" in response.content.upper()
        except Exception as e:
            logging.error(f"Local LLM Entailment check failed: {e}")
            return False

    def calculate_metrics(self, answer: str, contexts: list) -> Tuple[float, float]:
        """Calculates Citation Precision and Citation Recall for a single row."""
        # Split answer into sentences based on punctuation followed by space
        sentences = nltk.sent_tokenize(answer) #[s.strip() for s in re.split(r'(?<=[.!?])\s+', str(answer)) if s.strip()]
        
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
                
                # NLI check using the local model
                if self._check_entailment(sentence, combined_cited_text):
                    supported_sentences += 1

        precision = supported_sentences / sentences_with_citations if sentences_with_citations > 0 else 0.0
        recall = supported_sentences / len(sentences)
        
        return precision, recall

def run_evaluation(input_csv: str = "data/evaluation_dataset.csv", output_csv: str = "data/evaluation_report.csv"):
    """
    Orchestrates RAGAS and the ALCE Entailment metrics using entirely local infrastructure (CPU).
    """
    logging.info(f"Loading generated dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Safely convert contexts from string representation to actual Python lists
    df['contexts'] = df['contexts'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Initialize the Local LLM Judge Models. Using Llama 3 for text grading
    local_judge_llm = ChatOllama(model="llama3", temperature=0.0)
    # Using nomic-embed-text for fast, local RAGAS Answer Relevancy math
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Wrap for RAGAS compatibility — evaluate() expects BaseRagasLLM / BaseRagasEmbeddings
    ragas_llm = LangchainLLMWrapper(local_judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)
    
    # --- CUSTOM METRICS: ALCE EVALUATION ---
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
    metrics = [ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy()]

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
    )

    logging.info("Starting Local RAGAS Evaluation. (NOTE: This will take time with a local model)...")
    ragas_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
        raise_exceptions=False,
        batch_size=1
    )
    
    # Merge results — use concat by index since RAGAS 0.4.x drops original columns from result df
    ragas_df = ragas_result.to_pandas().reset_index(drop=True)
    base_df = df[['question', 'ground_truth', 'answer', 'alce_citation_precision', 'alce_citation_recall']].reset_index(drop=True)
    final_df = pd.concat([base_df, ragas_df.drop(columns=[c for c in ['question', 'answer', 'ground_truth', 'contexts'] if c in ragas_df.columns], errors='ignore')], axis=1)
    
    # Save Report
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    
    def _fmt(series: pd.Series) -> str:
        """Mean of valid (non-NaN) values; reports how many jobs timed out."""
        valid = series.dropna()
        if len(valid) == 0:
            return f"N/A — all {len(series)} rows failed (timeout/NaN)"
        n_failed = len(series) - len(valid)
        suffix = f"  [{n_failed} NaN/timeout skipped]" if n_failed > 0 else ""
        return f"{valid.mean():.4f}{suffix}"

    logging.info("\n========== LOCAL EVALUATION REPORT ==========")
    logging.info(f"Context Precision:       {_fmt(final_df.get('context_precision', pd.Series(dtype=float)))}")
    logging.info(f"Context Recall:          {_fmt(final_df.get('context_recall', pd.Series(dtype=float)))}")
    logging.info(f"Faithfulness:            {_fmt(final_df.get('faithfulness', pd.Series(dtype=float)))}")
    logging.info(f"Answer Relevancy:        {_fmt(final_df.get('answer_relevancy', pd.Series(dtype=float)))}")
    logging.info(f"ALCE Citation Precision: {_fmt(final_df['alce_citation_precision'])}")
    logging.info(f"ALCE Citation Recall:    {_fmt(final_df['alce_citation_recall'])}")
    logging.info("======================================================")

if __name__ == "__main__":
    run_evaluation()