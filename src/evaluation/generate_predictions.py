import os
import re
import json
import logging
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict

# Importing the orchestrator that runs the full RAG pipeline (Retriever + Reranker + Generator)
from src.run_rag import ScientificRAGPipeline 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_final_answer(full_response: str) -> str:
    """
    Extracts only the <Final Answer> section from the LLM output.

    The generator prompt asks the model to produce:
        <Reasoning> ... </Reasoning>
        <Final Answer> ... </Final Answer>

    ALCE and RAGAS should only evaluate the cited answer text, not the reasoning
    chain. Reasoning sentences have no [Doc N] citations and artificially lower
    Citation Recall. This function strips the reasoning block so the evaluation
    columns contain only the attributable answer.

    Three known failure modes handled here:

    1. Missing closing tag — llama3 frequently omits </Final Answer>.
       The pattern uses (?:</Final\s+Answer>|(?=<Final\s+Answer>)|$) so the
       block is correctly terminated by: a closing tag, the start of a new
       block (lookahead), or end-of-string — whichever comes first.

    2. Multiple <Final Answer> blocks — the LLM sometimes emits a refusal
       ("The retrieved documents do not contain enough information...") as a
       first block, then produces the real cited answer in a second block.
       re.finditer collects all matches; taking the LAST one ensures we always
       return the substantive answer, not the refusal.

    3. Line-wrapped opening tag — llama3 occasionally wraps the tag across a
       newline: "<Final\\nAnswer>" instead of "<Final Answer>". A literal-space
       pattern would silently miss this, triggering the full-response fallback
       and sending the <Reasoning> block to ALCE/RAGAS. Using \\s+ in both the
       opening and closing patterns absorbs any whitespace (space, newline, \\r).

    If no tags are found at all the full response is returned unchanged so
    no evaluation data is silently lost.
    """
    matches = list(re.finditer(
        r'<Final\s+Answer>\s*(.*?)\s*(?:</Final\s+Answer>|(?=<Final\s+Answer>)|$)',
        full_response,
        re.DOTALL | re.IGNORECASE,
    ))
    if matches:
        # Take the LAST block: the LLM puts the substantive cited answer last
        extracted = matches[-1].group(1).strip()
        if extracted:
            if len(matches) > 1:
                logging.warning(
                    f"Found {len(matches)} <Final Answer> blocks; "
                    "using the last one (earlier blocks are likely refusals)."
                )
            return extracted
    # Fallback: model omitted tags entirely — return full response
    logging.warning("<Final Answer> tag not found in LLM output; using full response for evaluation.")
    return full_response


def fetch_qasper_sample(num_samples: int = 10) -> List[Dict]:
    """
    Fetches a subset of the QASPER validation dataset, filtering for 
    questions that have a definitive free-form text answer.
    """
    logging.info("Loading QASPER validation dataset...")
    dataset = load_dataset("allenai/qasper", split="train") # not split="validation" because indexing is done on the train set
    
    qa_pairs = []
    for row in dataset:
        questions = row['qas']['question']
        answers = row['qas']['answers']
        
        for q_idx, q in enumerate(questions):
            ans_list = answers[q_idx]['answer']
            for ans in ans_list:
                if ans['free_form_answer']:
                    qa_pairs.append({
                        "question":     q,
                        "ground_truth": ans['free_form_answer'],
                        "paper_id":     row["id"],  # QASPER paper ID — used to scope retrieval
                    })
                    break # Stop if we found a valid answer for this question
        
        if len(qa_pairs) >= num_samples:
            break
            
    return qa_pairs

def generate_evaluation_dataset(output_path: str = None):
    """
    Passes QASPER questions through the RAG Pipeline and formats 
    the output exactly as RAGAS expects.
    """
    _project_root = Path(__file__).resolve().parents[2]
    if output_path is None:
        output_path = str(_project_root / "data" / "evaluation_dataset.csv")

    # 1. Initialize the Orchestrator (loads FAISS, Cross-Encoder, and the
    #    configured LLM backend).
    #
    #    When GENERATOR_BACKEND=vllm is set in the environment (e.g. from
    #    run_evaluation.sh), the pipeline routes generation requests to the
    #    already-running vLLM server instead of loading the weights here.
    #    This frees GPU memory on this process for SPECTER2 and the reranker.
    generator_backend = os.environ.get("GENERATOR_BACKEND", "auto")
    logging.info(
        "Initializing Scientific RAG Pipeline (generator_backend=%s)...",
        generator_backend,
    )
    rag_pipeline = ScientificRAGPipeline(
        dense_index_path="data/indices/dense.index",
        dense_meta_path="data/indices/dense.index.meta",
        sparse_index_path="data/indices/sparse.pkl",
        generator_backend=generator_backend,
    )
    
    # 2. Get the evaluation questions use 150 for sample testing
    qa_pairs = fetch_qasper_sample(150)#num_samples=150
    
    results = []
    logging.info(f"Generating RAG answers for {len(qa_pairs)} questions...")
    
    for i, qa in enumerate(qa_pairs):
        logging.info(f"Processing question {i+1}/{len(qa_pairs)}")
        
        
        # The pipeline's ask() returns {"answer": <full LLM output>, "retrieved_docs": [...]}
        pipeline_output = rag_pipeline.ask(qa["question"], filter_paper_id=qa.get("paper_id"))
        full_answer = pipeline_output["answer"]
        retrieved_docs = pipeline_output["retrieved_docs"]

        # Fix 5 — Fallback for empty-context rows (Lewis et al., 2020, NeurIPS
        # Section 4.3): RAG with zero passages degenerates to unconditioned
        # generation.  If paper-scoped retrieval failed, retry without the
        # filter so every question gets at least some context for RAGAS/ALCE.
        if not retrieved_docs and qa.get("paper_id"):
            logging.warning(
                "Question %d/%d: 0 docs with paper_id='%s'; "
                "retrying without paper filter (fallback).",
                i + 1, len(qa_pairs), qa["paper_id"],
            )
            pipeline_output = rag_pipeline.ask(qa["question"], filter_paper_id=None)
            full_answer = pipeline_output["answer"]
            retrieved_docs = pipeline_output["retrieved_docs"]

        # Strip <Reasoning> block: ALCE and RAGAS evaluate only the cited <Final Answer>.
        # Reasoning sentences have no [Doc N] tags and artificially lower Citation Recall.
        answer = extract_final_answer(full_answer)

        # RAGAS requires 'contexts' to be a list of strings.
        # The order here matches the [Doc N] citation numbering used by the generator.
        context_strings = [doc["text"] for doc in retrieved_docs]

        # best_rerank_score: the highest logit from the cross-encoder over
        # the reranked docs.  Saved here so calibrate_crag.py can derive
        # a data-driven CRAG threshold without requiring RAGAS scores first.
        best_score = max(
            (d.get("rerank_score", float("-inf")) for d in retrieved_docs),
            default=float("-inf"),
        )
        results.append({
            "question":          qa["question"],
            "ground_truth":      qa["ground_truth"],
            "contexts":          context_strings,
            "answer":            answer,          # Final Answer only — used by RAGAS + ALCE
            "full_answer":       full_answer,     # Full LLM output — kept for debugging
            "best_rerank_score": best_score,      # Used by calibrate_crag.py
            "crag_triggered":    pipeline_output.get("crag_triggered", False),
        })
        
    # 3. Save to disk
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved evaluation dataset to {output_path}")

    # Log generation statistics (latency, token usage)
    rag_pipeline.generator.log_generation_summary()


def regenerate_error_rows(csv_path: str = None):
    """Re-process only rows whose 'answer' contains a System Error string.

    This avoids re-running the full 150-question pipeline when Ollama was
    temporarily unreachable during the original generation pass.

    The function:
      1. Loads the existing evaluation_dataset.csv
      2. Identifies rows matching "System Error:" or "Error:"
      3. Re-initializes the RAG pipeline (with retry-enabled Ollama)
      4. Re-generates only those rows, preserving all successful rows
      5. Overwrites the CSV in-place

    Typical usage (local CPU):
        python -m src.evaluation.generate_predictions --fix-errors
    """
    _project_root = Path(__file__).resolve().parents[2]
    if csv_path is None:
        csv_path = str(_project_root / "data" / "evaluation_dataset.csv")

    if not os.path.exists(csv_path):
        logging.error("Cannot fix errors: %s does not exist. Run full generation first.", csv_path)
        return

    df = pd.read_csv(csv_path)

    # Detect error rows
    error_mask = df['answer'].str.contains(
        r'^(System Error:|Error:)', na=False, regex=True
    )
    n_errors = error_mask.sum()

    if n_errors == 0:
        logging.info("No error rows found in %s — nothing to regenerate.", csv_path)
        return

    logging.info(
        "Found %d/%d error rows in %s. Re-generating...",
        n_errors, len(df), csv_path,
    )

    # Initialize pipeline (Ollama health check will run at init)
    import ast
    generator_backend = os.environ.get("GENERATOR_BACKEND", "auto")
    rag_pipeline = ScientificRAGPipeline(
        dense_index_path="data/indices/dense.index",
        dense_meta_path="data/indices/dense.index.meta",
        sparse_index_path="data/indices/sparse.pkl",
        generator_backend=generator_backend,
    )

    fixed = 0
    for error_num, idx in enumerate(df.index[error_mask], 1):
        row = df.loc[idx]
        question = row['question']
        logging.info(
            "Re-generating error %d/%d (dataset row %d): %s",
            error_num, n_errors, idx, question[:80],
        )

        pipeline_output = rag_pipeline.ask(question)
        full_answer = pipeline_output["answer"]
        retrieved_docs = pipeline_output["retrieved_docs"]

        # Fallback: retry without paper filter if zero docs
        if not retrieved_docs:
            pipeline_output = rag_pipeline.ask(question, filter_paper_id=None)
            full_answer = pipeline_output["answer"]
            retrieved_docs = pipeline_output["retrieved_docs"]

        answer = extract_final_answer(full_answer)
        context_strings = [doc["text"] for doc in retrieved_docs]
        best_score = max(
            (d.get("rerank_score", float("-inf")) for d in retrieved_docs),
            default=float("-inf"),
        )

        # Check if this attempt also failed
        if answer.startswith("System Error:") or answer.startswith("Error:"):
            logging.warning("Row %d still failed: %s", idx, answer[:100])
            continue

        # Update the row in-place
        df.at[idx, 'answer'] = answer
        df.at[idx, 'full_answer'] = full_answer
        df.at[idx, 'contexts'] = str(context_strings)
        df.at[idx, 'best_rerank_score'] = best_score
        df.at[idx, 'crag_triggered'] = pipeline_output.get("crag_triggered", False)
        fixed += 1

    df.to_csv(csv_path, index=False)
    remaining = n_errors - fixed
    logging.info(
        "Re-generation complete: %d/%d errors fixed, %d still failing. Saved to %s",
        fixed, n_errors, remaining, csv_path,
    )

    # Log generation statistics (latency, token usage)
    rag_pipeline.generator.log_generation_summary()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate or fix RAG evaluation predictions."
    )
    parser.add_argument(
        "--fix-errors", action="store_true",
        help="Re-generate only rows with 'System Error' answers "
             "(requires Ollama to be running for CPU mode).",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to evaluation_dataset.csv (default: data/evaluation_dataset.csv).",
    )
    args = parser.parse_args()

    if args.fix_errors:
        regenerate_error_rows(csv_path=args.csv)
    else:
        generate_evaluation_dataset(output_path=args.csv)
