import argparse
import logging
import sys
import os

# Import Retriever Components
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

# Import Generator Component
from src.generation.llm_generator import LocalLLMGenerator

# Configure logging to track the pipeline's progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScientificRAGPipeline:
    def __init__(self, 
                 dense_index_path: str = "data/indices/dense.index",
                 dense_meta_path: str = "data/indices/dense.index.meta",
                 sparse_index_path: str = "data/indices/sparse.pkl"):
        """
        Initializes the entire end-to-end RAG system with the exact artifact paths.
        """
        logging.info("Initializing the Hybrid Retriever...")
        
        # Load the physical index files from the hard drive into RAM
        self.retriever = HybridRetriever(
            dense_index_path=dense_index_path, 
            dense_meta_path=dense_meta_path,
            sparse_index_path=sparse_index_path,
            embedding_model_name="BAAI/bge-small-en-v1.5" # Ensure query embedding matches document embedding
        )
        
        logging.info("Initializing the Cross-Encoder Reranker...")
        self.reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        logging.info("Initializing the Local LLM Generator (Ollama - llama3)...")
        # Ensure Ollama is running (`ollama serve`)
        self.generator = LocalLLMGenerator(model_name="llama3")
        
        logging.info("System Ready.")

    def ask(self, query: str) -> dict:
        """
        Executes the full RAG pipeline for a given query.
        """
        logging.info(f"Processing Query: '{query}'")
        
        # 1. RETRIEVE (Recall)
        logging.info("Stage 1: Fetching top 50 candidates using Hybrid Search (Dense and Sparse)...")
        broad_results = self.retriever.search(query, k=50)
        
        if not broad_results:
            return {"answer": "Error: No documents found in the database.", "retrieved_docs": []}

        # 2. RERANK (Precision)
        logging.info("Stage 2: Reranking candidates using Cross-Encoder...")
        top_10_docs = self.reranker.rerank(query, broad_results, top_k=10) # Keep top 10 for more context to the generator
        
        # 3. THE HANDOFF & GENERATION
        logging.info("Stage 3: Passing top 10 documents to LLM to use as context for generation...")
        
        # The generator will stream the answer directly to the terminal, 
        # so we just let it execute.
        final_answer = self.generator.generate_answer(query, top_10_docs)
        
        return {"answer": final_answer,
                "retrieved_docs": top_10_docs}

def main():
    # Setup Argument Parser for Command Line Execution
    parser = argparse.ArgumentParser(description="Query the Scientific NLP RAG System.")
    parser.add_argument("--query", type=str, required=True, help="The scientific question to ask.")
    
    # Exact requested file paths set as Command Line Interface (CLI) defaults
    parser.add_argument("--dense-index", type=str, default="data/indices/dense.index")
    parser.add_argument("--dense-meta", type=str, default="data/indices/dense.index.meta")
    parser.add_argument("--sparse-index", type=str, default="data/indices/sparse.pkl")
    
    args = parser.parse_args()

    try:
        # Initialize passing the correct CLI or Default paths
        rag_system = ScientificRAGPipeline(
            dense_index_path=args.dense_index,
            dense_meta_path=args.dense_meta,
            sparse_index_path=args.sparse_index
        )
        
        print("\n" + "*"*60)
        print(f"QUESTION: {args.query}")
        print("*"*60 + "\n")
        
        # Execute the pipeline. The streaming UI will handle printing the response.
        _ = rag_system.ask(args.query)
        
    except Exception as e:
        logging.error(f"Pipeline crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()