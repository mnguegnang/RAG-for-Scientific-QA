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
    import os
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
    
    # 2. Get the evaluation questions use 250 for sample testing 
    qa_pairs = fetch_qasper_sample(200)#num_samples=250
    
    results = []
    logging.info(f"Generating RAG answers for {len(qa_pairs)} questions...")
    
    for i, qa in enumerate(qa_pairs):
        logging.info(f"Processing question {i+1}/{len(qa_pairs)}")
        
        
        # The pipeline's ask() returns {"answer": <full LLM output>, "retrieved_docs": [...]}
        pipeline_output = rag_pipeline.ask(qa["question"], filter_paper_id=qa.get("paper_id"))
        full_answer = pipeline_output["answer"]
        retrieved_docs = pipeline_output["retrieved_docs"]

        # Strip <Reasoning> block: ALCE and RAGAS evaluate only the cited <Final Answer>.
        # Reasoning sentences have no [Doc N] tags and artificially lower Citation Recall.
        answer = extract_final_answer(full_answer)

        # RAGAS requires 'contexts' to be a list of strings.
        # The order here matches the [Doc N] citation numbering used by the generator.
        context_strings = [doc["text"] for doc in retrieved_docs]

        results.append({
            "question":     qa["question"],
            "ground_truth": qa["ground_truth"],
            "contexts":     context_strings,
            "answer":       answer,          # Final Answer only — used by RAGAS + ALCE
            "full_answer":  full_answer,     # Full LLM output — kept for debugging
        })
        
    # 3. Save to disk
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved evaluation dataset to {output_path}")

if __name__ == "__main__":
    generate_evaluation_dataset()
