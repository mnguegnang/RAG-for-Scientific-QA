from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
import os

# CONFIG
DENSE_INDEX = 'data/indices/dense.index'
DENSE_META = 'data/indices/dense.index.meta'
SPARSE_INDEX = 'data/indices/sparse.pkl'

def main():
    # 1. Initialize Components
    print("--- Initializing Retrieval Pipeline ---")
    retriever = HybridRetriever(DENSE_INDEX, DENSE_META, SPARSE_INDEX)
    reranker = Reranker()
    
    # 2. Simulate User Query
    query = "What are the limitations of the Transformer architecture?"
    print(f"\nQuery: {query}")
    
    # 3. Stage 1: Hybrid Retrieval (High Recall)
    # Get top 50 candidates to cast a wide net
    print("\n--- Stage 1: Hybrid Retrieval (Top 50) ---")
    initial_results = retriever.search(query, k=50)
    
    # Print top 3 just to see what the initial search found
    print("Top 3 before Re-ranking:")
    for res in initial_results[:3]:
        print(f"- [{res['score']:.4f}] {res['text'][:100]}...")
        
    # 4. Stage 2: Re-ranking (High Precision)
    # Filter down to the absolute best 5
    print("\n--- Stage 2: Re-ranking (Top 5) ---")
    final_results = reranker.rerank(query, initial_results, top_k=5)
    
    # 5. Final Output
    print("\n=== FINAL ANSWER CONTEXT ===")
    for i, res in enumerate(final_results):
        print(f"\nRank {i+1} (Score: {res['rerank_score']:.4f}):")
        print(f"Source: {res['doc_id']}")
        print(f"Text: {res['text']}")
        
if __name__ == "__main__":
    main()