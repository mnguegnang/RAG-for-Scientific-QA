import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class DenseIndexer:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", index_path="data/indices/dense.index"):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.dimension = 384 # Specific to BGE-Small
        self.index = None
        self.metadata_store = [] # FAISS only stores numbers; we need to store the text separately

    def build_index(self, chunks: List[Dict]):
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        texts = [c['text'] for c in chunks]
        
        # 1. Encode
        # normalize_embeddings=True prepares vectors for Inner Product search
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        
        # 2. Initialize FAISS Index
        # IndexFlatIP = Exact search using Inner Product (fastest for small datasets)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 3. Add vectors (must be float32 for FAISS)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # 4. Store metadata
        self.metadata_store = chunks
        print(f"Dense Index built with {self.index.ntotal} vectors.")

    def save(self):
        # Save the FAISS index (the vectors)
        faiss.write_index(self.index, self.index_path)
        
        # Save the metadata (the text and IDs) using pickle
        with open(self.index_path + ".meta", "wb") as f:
            pickle.dump(self.metadata_store, f)
        print(f"Saved dense index to {self.index_path}")