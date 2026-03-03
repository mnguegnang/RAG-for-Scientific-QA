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
from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy
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
    eval_dataset = Dataset.from_pandas(df)
    metrics = [ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy()]
    
    logging.info("Starting Local RAGAS Evaluation. (NOTE: This is CPU-bound and will take time)...")
    ragas_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False 
    )
    
    # Merge results
    ragas_df = ragas_result.to_pandas()
    final_df = pd.merge(
        df[['question', 'ground_truth', 'answer', 'alce_citation_precision', 'alce_citation_recall']], 
        ragas_df, 
        on=['question', 'answer']
    )
    
    # Save Report
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    
    logging.info("\n========== LOCAL EVALUATION REPORT ==========")
    logging.info(f"Context Precision:  {final_df.get('context_precision', pd.Series([0])).mean():.4f}")
    logging.info(f"Context Recall:     {final_df.get('context_recall', pd.Series([0])).mean():.4f}")
    logging.info(f"Faithfulness:       {final_df.get('faithfulness', pd.Series([0])).mean():.4f}")
    logging.info(f"Answer Relevancy:   {final_df.get('answer_relevancy', pd.Series([0])).mean():.4f}")
    logging.info(f"ALCE Citation Precision:      {final_df['alce_citation_precision'].mean():.4f}")
    logging.info(f"ALCE Citation Recall:         {final_df['alce_citation_recall'].mean():.4f}")
    logging.info("======================================================")

if __name__ == "__main__":
    run_evaluation()