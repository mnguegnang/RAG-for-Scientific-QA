import os
import json
import logging
import pandas as pd
from datasets import load_dataset
from typing import List, Dict

# Importing the orchestrator that runs the full RAG pipeline (Retriever + Reranker + Generator)
from src.run_rag import ScientificRAGPipeline 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_qasper_sample(num_samples: int = 10) -> List[Dict]:
    """
    Fetches a subset of the QASPER validation dataset, filtering for 
    questions that have a definitive free-form text answer.
    """
    logging.info("Loading QASPER validation dataset...")
    dataset = load_dataset("allenai/qasper", split="validation")
    
    qa_pairs = []
    for row in dataset:
        questions = row['qas']['question']
        answers = row['qas']['answers']
        
        for q_idx, q in enumerate(questions):
            ans_list = answers[q_idx]['answer']
            for ans in ans_list:
                if ans['free_form_answer']:
                    qa_pairs.append({
                        "question": q,
                        "ground_truth": ans['free_form_answer']
                    })
                    break # Stop if we found a valid answer for this question
        
        if len(qa_pairs) >= num_samples:
            break
            
    return qa_pairs

def generate_evaluation_dataset(output_path: str = "data/evaluation_dataset.csv"):
    """
    Passes QASPER questions through the RAG Pipeline and formats 
    the output exactly as RAGAS expects.
    """
    # 1. Initialize the Orchestrator (loads FAISS, Cross-Encoder, Ollama)
    logging.info("Initializing Scientific RAG Pipeline...")
    rag_pipeline = ScientificRAGPipeline(
        dense_index_path="data/indices/dense.index",
        dense_meta_path="data/indices/dense.index.meta",
        sparse_index_path="data/indices/sparse.pkl"
        
    )
    
    # 2. Get the evaluation questions currently limited to 20 for quick testing but will be increase in production
    qa_pairs = fetch_qasper_sample()#num_samples=20
    
    results = []
    logging.info(f"Generating RAG answers for {len(qa_pairs)} questions...")
    
    for i, qa in enumerate(qa_pairs):
        logging.info(f"Processing question {i+1}/{len(qa_pairs)}")
        
        
        #The pipeline run 'ask()' method and returns a dict with keys "answer" and "retrieved_docs".
        pipeline_output = rag_pipeline.ask(qa["question"])
        answer = pipeline_output["answer"]
        retrieved_docs = pipeline_output["retrieved_docs"] 
        
        # RAGAS requires 'contexts' to be a list of strings
        context_strings = [doc["text"] for doc in retrieved_docs]
        
        results.append({
            "question": qa["question"],
            "ground_truth": qa["ground_truth"],
            "contexts": context_strings,
            "answer": answer
        })
        
    # 3. Save to disk
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved evaluation dataset to {output_path}")

if __name__ == "__main__":
    generate_evaluation_dataset()