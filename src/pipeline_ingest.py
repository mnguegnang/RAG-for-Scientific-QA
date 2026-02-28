import os
from data.make_dataset import load_and_inspect_qasper
from retrieval.chunking import QasperChunker
from retrieval.vector_store import DenseIndexer
from retrieval.sparse_store import SparseIndexer

def run_ingestion_pipeline():
    print("=== Data Ingestion & Indexing ===")    
    # 1. Load Data
    raw_data = load_and_inspect_qasper()
    
    # Fo Development phase: Limit to 50 papers to keep processing time under 2 minutes
    # In production, we will use the whole dataset i.e., len(raw_data) = 1585 papers
    subset_data = [raw_data[i] for i in range(50)] 
    print(f"Processing subset of {len(subset_data)} papers.")
    
    # 2. Chunking
    print("\n--- Chunking ---")
    chunker = QasperChunker()
    all_chunks = []
    
    for paper in subset_data:
        paper_chunks = chunker.process_paper(paper)
        all_chunks.extend(paper_chunks)
        
    print(f"Result: Generated {len(all_chunks)} chunks.")

    # 3. Dense Indexing (Vectors)
    print("\n--- Dense Indexing (FAISS) ---")
    dense_idx = DenseIndexer()
    dense_idx.build_index(all_chunks)
    dense_idx.save()

    # 4. Sparse Indexing (BM25)
    print("\n--- Sparse Indexing (BM25) ---")
    sparse_idx = SparseIndexer()
    sparse_idx.build_index(all_chunks)
    sparse_idx.save()

    print("\n=== Ingestion Complete ===")
    print("Files created in data/indices/:")
    print(os.listdir("data/indices/"))

if __name__ == "__main__":
    run_ingestion_pipeline()