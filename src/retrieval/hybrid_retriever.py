import faiss
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"

class HybridRetriever:
    def __init__(self, 
                 dense_index_path: str, 
                 dense_meta_path: str, 
                 sparse_index_path: str,
                 embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        
        # 1. Load Embedding Model (for query encoding)
        print("Loading Embedding Model...")
        self.encoder = SentenceTransformer(embedding_model_name, device=device)
        
        # 2. Load Dense Index (FAISS)
        print(f"Loading FAISS Index from {dense_index_path}...")
        self.dense_index = faiss.read_index(dense_index_path)
        
        # 3. Load Metadata (To map FAISS IDs back to text)
        with open(dense_meta_path, 'rb') as f:
            self.dense_meta = pickle.load(f)
            
        # 4. Load Sparse Index (BM25)
        print(f"Loading BM25 Index from {sparse_index_path}...")
        with open(sparse_index_path, 'rb') as f:
            self.bm25_package = pickle.load(f)
            # The pickle contains the object and the corpus, we unpack it
            self.bm25 = self.bm25_package['model']
            # Extract Metadata it format. It was saved as: {'model': bm25, 'metadata': [...]}
            # We need to separate the text and IDs for easy lookup during search
            metadata_list = self.bm25_package['metadata']

            self.bm25_corpus = [chunk['text'] for chunk in metadata_list] # The actual chunks
            self.bm25_ids = [chunk['paper_id'] for chunk in metadata_list]
            #self.bm25_corpus = self.bm25_package['metadata'] # The actual chunks
            #self.bm25_ids = self.bm25_package['doc_ids'] # The IDs
            
    def _search_dense(self, query: str, k: int):
        """Standard Vector Search"""
        # Encode query
        q_vec = self.encoder.encode([query])
        # FAISS expects float32 normalized vectors for IP search (if model output is normalized)
        # Note: BGE output is usually normalized, but good practice to ensure.
        faiss.normalize_L2(q_vec)
        
        # Search
        scores, indices = self.dense_index.search(q_vec, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # FAISS returns -1 if not enough neighbors
                results.append({
                    "doc_id": self.dense_meta[idx]['paper_id'], # Assuming metadata structure
                    "text": self.dense_meta[idx]['text'],
                    "rank": i + 1  # 1-based rank
                })
        return results

    def _search_sparse(self, query: str, k: int):
        """Standard BM25 Search"""
        # We need to tokenize the query exactly how we tokenized the documents for sparce store indexing
        tokenized_query = query.lower().split() 
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top K indices using numpy argpartition (faster than full sort)
        top_n = np.argpartition(scores, -k)[-k:]
        # Sort these top K by score descending
        best_indices = top_n[np.argsort(scores[top_n])][::-1]
        
        results = []
        for rank, idx in enumerate(best_indices):
            results.append({
                "doc_id": self.bm25_ids[idx],
                "text": self.bm25_corpus[idx], # Assuming corpus is list of text
                "rank": rank + 1 # 1-based rank
            })
        return results

    def search(self, query: str, k: int = 10, rrf_k: int = 60):
        """
        Performs Hybrid Search using RRF.
        """
        # 1. Get Independent Results
        dense_res = self._search_dense(query, k)
        sparse_res = self._search_sparse(query, k)
        
        # 2. Apply RRF
        # Map unique text/ID to accumulated score
        # We use text as key to de-duplicate, assuming unique text per chunk
        score_map = defaultdict(float)
        content_map = {} # Keep track of content so we can return it
        
        # Process Dense
        for item in dense_res:
            score_map[item['text']] += 1 / (rrf_k + item['rank'])
            content_map[item['text']] = item
            
        # Process Sparse
        for item in sparse_res:
            score_map[item['text']] += 1 / (rrf_k + item['rank'])
            content_map[item['text']] = item # Overwrite is fine, content is same
            
        # 3. Sort and Format
        sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for text, score in sorted_items[:k]: # Return top K from the fused list
            meta = content_map[text]
            final_results.append({
                "text": text,
                "doc_id": meta['doc_id'],
                "score": score
            })
            
        return final_results