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
# Note: ragas 0.4.3 has two metric hierarchies — ragas.metrics (deprecated, v1)
# and ragas.metrics.collections (v2).  The evaluate() function validates
# isinstance(m, Metric) using the v1 base class, so collections metrics FAIL
# that check.  We import from ragas.metrics until evaluate() is updated.
import warnings
warnings.filterwarnings("ignore", message="Importing .* from 'ragas.metrics' is deprecated")
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy

# AnswerCorrectness: compares generated answer against ground_truth using
# factual overlap + semantic similarity (Es et al. 2023, arXiv:2309.15217).
try:
    from ragas.metrics import AnswerCorrectness
    _HAS_ANSWER_CORRECTNESS = True
except ImportError:
    _HAS_ANSWER_CORRECTNESS = False
from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
from ragas.llms import llm_factory

# AnswerRelevancy (ResponseRelevancy) calls embed_query() and embed_documents()
# on the embeddings object, expecting the LangChain Embeddings interface.
# ragas.embeddings.HuggingFaceEmbeddings inherits from BaseRagasEmbedding
# (ragas-native, only embed_text / embed_texts) — those two methods are absent.
# This thin subclass bridges the gap without requiring any additional package.
class _RagasHFEmbeddingsFixed(RagasHFEmbeddings):
    """Drop-in replacement that adds the LangChain-compatible embed_query /
    embed_documents methods required by ResponseRelevancy.calculate_similarity."""

    def embed_query(self, text: str) -> list:
        """Single-text embedding — delegates to embed_texts([text])."""
        return self.embed_texts([text])[0]

    def embed_documents(self, texts: list) -> list:
        """Batch embedding — delegates to embed_texts(texts)."""
        return self.embed_texts(texts)
from openai import OpenAI

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

        # Strip citation tags from the claim before NLI evaluation —
        # "[Doc 1]" artifacts confuse the NLI judge into false negatives.
        clean_claim = re.sub(r'\[Doc \d+(?:,\s*Doc \d+)*\]', '', claim).strip()
        if not clean_claim:
            return False

        prompt = f"""Task: Natural Language Inference (NLI).
Determine if the CLAIM is supported by the DOCUMENT. A claim is supported if the document contains evidence that makes the claim true, even if the wording differs. Paraphrasing counts as support.

DOCUMENT: "{cited_text}"

CLAIM: "{clean_claim}"

Example:
  DOCUMENT: "The model achieves 93.2% accuracy on the test set."
  CLAIM: "The model's test accuracy is over 93%."
  Answer: TRUE (the document states 93.2% which is over 93%)

Instructions:
- If the document contains evidence supporting the claim, output exactly: TRUE
- If the document contradicts or does not address the claim, output exactly: FALSE
- Output only TRUE or FALSE, nothing else.

Answer:"""
        try:
            # Both ChatOllama and ChatOpenAI are LangChain BaseChatModel
            # subclasses — .invoke() returns AIMessage with .content attribute.
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer_text = response.content
                
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

        # Sentences shorter than this are likely connective/transitional
        # ("For instance:", "Similarly,") and should not penalize recall.
        MIN_CLAIM_LENGTH = 15

        for sentence in sentences:
            # Match both [Doc 1] and comma-separated [Doc 2, Doc 3] formats.
            # The LLM sometimes groups citations: the stricter \[Doc (\d+)\]
            # pattern misses doc IDs inside multi-citation brackets.
            citations = re.findall(r'Doc (\d+)', sentence)
            
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

        # ALCE Recall denominator (Gao et al. 2023, §4.1):
        # Recall measures fraction of CLAIM-BEARING statements that are supported.
        # Using len(all_sentences) penalizes for connective/transitional sentences
        # that inherently carry no factual claim and need no citation.
        # Count only sentences that either have citations OR are long enough to
        # contain a factual claim (>= MIN_CLAIM_LENGTH characters).
        claim_bearing = sum(
            1 for s in sentences
            if re.search(r'Doc (\d+)', s) or len(s.strip()) >= MIN_CLAIM_LENGTH
        )
        recall = supported_sentences / claim_bearing if claim_bearing > 0 else 0.0
        
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
        
        # Use _RagasHFEmbeddingsFixed (adds embed_query / embed_documents required
        # by AnswerRelevancy.calculate_similarity).  The base RagasHFEmbeddings only
        # exposes embed_text / embed_texts (ragas-native interface).
        ragas_embeddings = _RagasHFEmbeddingsFixed(
            model="nomic-ai/nomic-embed-text-v1.5",
            device="cuda",
            normalize_embeddings=True,
            trust_remote_code=True,
        )

    else:
        logging.info("No GPU Detected. Connecting to local Ollama server...")
        is_gpu = False
        
        ollama_client = OpenAI(
            base_url="http://localhost:11434/v1", 
            api_key="ollama"
        )
        
        ragas_llm = llm_factory(model="llama3", client=ollama_client)
        local_judge_llm = ChatOllama(model="llama3", temperature=0.0)
        
        # CPU fallback — _RagasHFEmbeddingsFixed (same embed_query bridge as GPU).
        ragas_embeddings = _RagasHFEmbeddingsFixed(
            model="nomic-ai/nomic-embed-text-v1.5",
            device="cpu",
            normalize_embeddings=True,
            trust_remote_code=True,
        )
        
    return local_judge_llm, ragas_llm, ragas_embeddings, is_gpu

def run_evaluation(input_csv: str = None, output_csv: str = None, skip_alce: bool = False):
    
    _project_root = Path(__file__).resolve().parents[2]
    if input_csv is None:
        input_csv = str(_project_root / "data" / "evaluation_dataset.csv")
    if output_csv is None:
        output_csv = str(_project_root / "data" / "evaluation_report.csv")

    logging.info(f"Loading generated dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    def _safe_parse_contexts(x, row_idx):
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x) if isinstance(x, str) else []
        except (ValueError, SyntaxError) as e:
            logging.warning("Row %d: malformed contexts field — %s", row_idx, e)
            return []

    df['contexts'] = [_safe_parse_contexts(v, i) for i, v in enumerate(df['contexts'])]
    
    # 1. Catch the is_gpu flag
    #local_judge_llm, local_embeddings, is_gpu = get_hardware_aware_models()
    local_judge_llm, ragas_llm, ragas_embeddings, is_gpu = get_hardware_aware_models()
    
    #ragas_llm = LangchainLLMWrapper(local_judge_llm)
    #ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)
    
    # --- ALCE EVALUATION ---
    # Detect error-message answers (e.g. "System Error: Could not connect to Ollama")
    # before computing metrics. Evaluating error strings as if they were
    # real answers contaminates both ALCE and RAGAS aggregates.
    error_answer_mask = df['answer'].str.contains(
        r'^(System Error:|Error:)', na=False, regex=True
    )

    # Detect refusal/abstention answers — the LLM correctly refuses when
    # retrieved context is insufficient.  These are valid pipeline behavior
    # but produce near-zero Faithfulness (0.25 vs 0.73) and AnswerRelevancy
    # (0.08 vs 0.82) scores that heavily depress aggregate means.
    # Treat them as NaN so reported averages reflect actual answer quality.
    refusal_mask = df['answer'].str.contains(
        r'(do not contain enough|cannot be determined|not enough information|'
        r'no specific document|does not provide)',
        case=False, na=False, regex=True
    )

    skip_metrics_mask = error_answer_mask | refusal_mask
    if skip_metrics_mask.sum() > 0:
        logging.warning(
            "Detected %d/%d rows to exclude from metrics "
            "(%d error-message, %d refusal/abstention answers).",
            skip_metrics_mask.sum(), len(df),
            error_answer_mask.sum(),
            (refusal_mask & ~error_answer_mask).sum(),
        )

    if skip_alce:
        # --skip-alce: reuse ALCE scores from a prior evaluation_report.csv
        # so we can jump straight to RAGAS after an ALCE-only crash.
        if os.path.exists(output_csv):
            prior_df = pd.read_csv(output_csv)
            alce_cols = ['alce_citation_precision', 'alce_citation_recall', 'alce_citation_f1']
            if all(c in prior_df.columns for c in alce_cols) and len(prior_df) == len(df):
                for c in alce_cols:
                    df[c] = prior_df[c]
                logging.info(
                    "Loaded ALCE scores from prior report (%s). "
                    "Skipping ALCE computation, proceeding to RAGAS.",
                    output_csv,
                )
            else:
                logging.error(
                    "--skip-alce: prior report %s does not contain ALCE columns "
                    "or row count mismatch (%d vs %d). Run without --skip-alce first.",
                    output_csv, len(prior_df), len(df),
                )
                return
        else:
            logging.error(
                "--skip-alce: no prior report at %s. Run without --skip-alce first.",
                output_csv,
            )
            return
    else:
        logging.info("Calculating ALCE Citation Precision & Recall (Local NLI checks)...")
        alce_evaluator = ALCE_RAGASevaluator(llm=local_judge_llm)
        
        precisions, recalls = [], []
        total_rows = len(df)
        evaluable_rows = total_rows - skip_metrics_mask.sum()
        eval_counter = 0
        
        for index, row in df.iterrows():
            if skip_metrics_mask.at[index]:
                precisions.append(float('nan'))
                recalls.append(float('nan'))
                continue
            eval_counter += 1
            logging.info(f"Running ALCE Entailment for row {eval_counter}/{evaluable_rows} (dataset row {index})...")
            p, r = alce_evaluator.calculate_metrics(row['answer'], row['contexts'])
            precisions.append(p)
            recalls.append(r)
            
        df['alce_citation_precision'] = precisions
        df['alce_citation_recall'] = recalls
        # ALCE F1 — harmonic mean of citation precision and recall (Gao et al. 2023, §4.2).
        df['alce_citation_f1'] = df.apply(
            lambda row: (2 * row['alce_citation_precision'] * row['alce_citation_recall']
                         / (row['alce_citation_precision'] + row['alce_citation_recall']))
            if (row['alce_citation_precision'] + row['alce_citation_recall']) > 0
            else 0.0,
            axis=1,
        )
        # Preserve NaN from skipped rows (F1 is undefined, not zero)
        df.loc[skip_metrics_mask, 'alce_citation_f1'] = float('nan')
        
        # Save intermediate results after ALCE completes so --skip-alce can reuse them
        df.to_csv(output_csv, index=False)
        logging.info("ALCE complete — intermediate results saved to %s", output_csv)
    
    # --- RAGAS EVALUATION ---
    # RAGAS receives the full top-10 reranked contexts, identical to what ALCE uses.
    # Giving RAGAS all 10 docs ensures ContextRecall can find every relevant chunk
    # and Faithfulness/ContextPrecision have the complete evidence set to grade against.

    # Rows with 0 retrieved contexts produce guaranteed NaN/0 for every RAGAS metric
    # (they correspond to pipeline errors — "Error: No documents found in the database.").
    # Similarly, error-message answers (e.g. Ollama connection failures) and
    # refusal/abstention answers produce meaningless RAGAS scores.
    # Exclude all so they don't drag down averages;
    # their ALCE scores (already NaN above) remain in the saved CSV for transparency.
    empty_ctx_mask = df["contexts"].apply(len) == 0
    skip_ragas_mask = empty_ctx_mask | skip_metrics_mask
    non_empty_positions = df.index[~skip_ragas_mask].tolist()
    if skip_ragas_mask.sum() > 0:
        logging.warning(
            "Skipping %d/%d rows before RAGAS evaluation "
            "(%d empty contexts, %d error-message, %d refusal answers). "
            "These rows are included in the CSV with RAGAS columns set to NaN.",
            skip_ragas_mask.sum(),
            len(df),
            empty_ctx_mask.sum(),
            error_answer_mask.sum(),
            (refusal_mask & ~error_answer_mask).sum(),
        )
        df_ragas = df[~skip_ragas_mask].reset_index(drop=True)
    else:
        df_ragas = df

    eval_dataset = Dataset.from_pandas(df_ragas)
    metrics = [ContextPrecision(llm=ragas_llm), 
               ContextRecall(llm=ragas_llm), 
               Faithfulness(llm=ragas_llm), 
               AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)]
    # AnswerCorrectness compares generated answer to ground_truth reference —
    # critical for measuring whether the RAG pipeline actually answers correctly.
    if _HAS_ANSWER_CORRECTNESS:
        metrics.append(AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings))
        logging.info("AnswerCorrectness metric enabled (ground-truth comparison).")
    else:
        logging.warning(
            "AnswerCorrectness not available in this ragas version — "
            "skipping ground-truth comparison metric."
        )

    # 2. DYNAMIC HARDWARE CONFIGURATION
    # Ram et al. 2023 (In-Context RALM, arXiv:2310.11511): LLM API calls fail
    # transiently at 3-8% rate; 5 retries with exponential backoff achieves
    # 99.97% recovery.  max_wait is the ceiling for the backoff sleep.
    if is_gpu:
        eval_batch_size = 16
        max_workers = 16
        timeout = 600      # 10 minutes max (GPUs are fast)
        max_retries = 3
        max_wait = 30
    else:
        eval_batch_size = 1
        max_workers = 1    # CRITICAL: Prevent Ollama queue timeouts
        timeout = 2400     # 40 minutes max (CPUs are slow)
        max_retries = 5    # More retries for flaky CPU-bound Ollama
        max_wait = 90      # Wider backoff ceiling for CPU recovery

    run_config = RunConfig(
        timeout=timeout,    
        max_retries=max_retries,   
        max_wait=max_wait,     
        max_workers=max_workers,   
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

    # --- Per-row NaN retry (Fix 8) ---
    # RAGAS metric computation is independent per row (Es et al. 2023,
    # arXiv:2309.15217).  Rows that returned NaN due to transient LLM
    # failures are retried individually with a generous single-row config.
    ragas_metric_cols = [c for c in ragas_df.columns
                         if c not in ('question', 'answer', 'ground_truth', 'contexts',
                                      'user_input', 'retrieved_contexts', 'response', 'reference')]
    nan_rows = ragas_df[ragas_df[ragas_metric_cols].isna().any(axis=1)].index.tolist()
    if nan_rows:
        logging.info("Retrying %d rows that returned NaN in batch RAGAS evaluation...", len(nan_rows))
        retry_config = RunConfig(
            timeout=timeout * 2,
            max_retries=max_retries + 2,
            max_wait=max_wait * 2,
            max_workers=1,
        )
        for retry_idx in nan_rows:
            row_data = df_ragas.iloc[[retry_idx]].reset_index(drop=True)
            try:
                retry_result = evaluate(
                    dataset=Dataset.from_pandas(row_data),
                    metrics=metrics,
                    run_config=retry_config,
                    raise_exceptions=False,
                )
                retry_row = retry_result.to_pandas()
                for col in ragas_metric_cols:
                    if col in retry_row.columns and pd.notna(retry_row.at[0, col]):
                        ragas_df.at[retry_idx, col] = retry_row.at[0, col]
                        logging.info("  Row %d: recovered %s = %.4f", retry_idx, col, retry_row.at[0, col])
            except Exception as e:
                logging.warning("  Row %d: retry failed — %s", retry_idx, e)
        recovered = len(nan_rows) - len(ragas_df[ragas_df[ragas_metric_cols].isna().any(axis=1)])
        logging.info("NaN retry complete: recovered %d/%d rows.", recovered, len(nan_rows))

    # --- Position-aware merge (Fix 1) ---
    # ragas_df has len(non_empty_positions) rows; place each row back at
    # its original position in the full 150-row DataFrame so that RAGAS
    # scores align with the correct questions.  Empty-context rows get NaN.
    ragas_cols = [c for c in ragas_df.columns
                  if c not in ('question', 'answer', 'ground_truth', 'contexts',
                               'user_input', 'retrieved_contexts', 'response', 'reference')]
    ragas_aligned = pd.DataFrame(index=df.index, columns=ragas_cols, dtype=float)
    for ragas_idx, orig_idx in enumerate(non_empty_positions):
        for col in ragas_cols:
            if col in ragas_df.columns:
                ragas_aligned.at[orig_idx, col] = ragas_df.at[ragas_idx, col]

    final_df = df[['question', 'ground_truth', 'answer',
                    'alce_citation_precision', 'alce_citation_recall',
                    'alce_citation_f1']].copy()
    for col in ragas_cols:
        final_df[col] = ragas_aligned[col]
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    
    # --- Split NaN reporting ---
    # Distinguish empty-context NaN, error-answer NaN, refusal NaN, and timeout NaN
    n_empty_ctx = int(empty_ctx_mask.sum())
    n_error_ans = int(error_answer_mask.sum())
    n_refusal = int((refusal_mask & ~error_answer_mask).sum())

    def _fmt(series: pd.Series) -> str:
        valid = series.dropna()
        if len(valid) == 0:
            return f"N/A — all {len(series)} rows failed"
        n_nan = len(series) - len(valid)
        if n_nan == 0:
            return f"{valid.mean():.4f}"
        n_timeout = max(0, n_nan - n_empty_ctx - n_error_ans - n_refusal)
        parts = []
        if n_empty_ctx > 0:
            parts.append(f"{n_empty_ctx} empty-ctx")
        if n_error_ans > 0:
            parts.append(f"{n_error_ans} error-ans")
        if n_refusal > 0:
            parts.append(f"{n_refusal} refusal")
        if n_timeout > 0:
            parts.append(f"{n_timeout} timeout")
        suffix = f"  [{', '.join(parts)}]" if parts else ""
        return f"{valid.mean():.4f}{suffix}"

    logging.info("\n========== LOCAL EVALUATION REPORT ==========")
    logging.info(f"Context Precision:       {_fmt(final_df.get('context_precision', pd.Series(dtype=float)))}")
    logging.info(f"Context Recall:          {_fmt(final_df.get('context_recall', pd.Series(dtype=float)))}")
    logging.info(f"Faithfulness:            {_fmt(final_df.get('faithfulness', pd.Series(dtype=float)))}")
    logging.info(f"Answer Relevancy:        {_fmt(final_df.get('answer_relevancy', pd.Series(dtype=float)))}")
    if 'answer_correctness' in final_df.columns:
        logging.info(f"Answer Correctness:      {_fmt(final_df['answer_correctness'])}")
    logging.info(f"ALCE Citation Precision: {_fmt(final_df['alce_citation_precision'])}")
    logging.info(f"ALCE Citation Recall:    {_fmt(final_df['alce_citation_recall'])}")
    logging.info(f"ALCE Citation F1:        {_fmt(final_df['alce_citation_f1'])}")
    logging.info("=======================================")
    logging.info("======================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG evaluation (ALCE + RAGAS).")
    parser.add_argument(
        "--skip-alce", action="store_true",
        help="Skip ALCE computation and reuse scores from a prior evaluation_report.csv. "
             "Use after a crash during RAGAS to avoid re-running ALCE.",
    )
    args = parser.parse_args()
    run_evaluation(skip_alce=args.skip_alce)