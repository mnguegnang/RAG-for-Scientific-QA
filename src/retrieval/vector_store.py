import faiss
import numpy as np
import pickle
import torch
from typing import List, Dict
from src.retrieval.encoders import Specter2Encoder

device = "cuda" if torch.cuda.is_available() else "cpu"

class DenseIndexer:
    def __init__(self, index_path="data/indices/dense.index"):
        # SPECTER2: scientifically pre-trained encoder (Singh et al., 2022, SciRepEval)
        # Dimension 768 vs. 384 for bge-small — richer representation for scientific text
        
        # Optimization: Use mixed precision (FP16) on GPU to accelerate the forward 
        # pass and save VRAM when embedding thousands of QASPER chunks.
        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        
        self.model = Specter2Encoder(
            device=device, 
            model_kwargs=model_kwargs  # Pass to the underlying transformer
        )
        self.index_path = index_path
        self.dimension = Specter2Encoder.DIMENSION  # 768
        self.index = None
        self.metadata_store = []  # FAISS only stores numbers; we need to store the text separately

    def build_index(self, chunks: List[Dict]):
        print(f"Generating embeddings for {len(chunks)} chunks on {device}...")
        
        texts = [c['text'] for c in chunks]
        
        # 1. Encode
        # normalize_embeddings=True mathematically converts Inner Product to Cosine Similarity
        # batch_size=32 (or 64/128) maximizes GPU parallelism.
        embeddings = self.model.encode(
            texts, 
            batch_size=64, 
            normalize_embeddings=True, 
            show_progress_bar=True
        )
        
        # 2. Initialize FAISS Index
        # IndexFlatIP = Exact search using Inner Product (Cosine Similarity because vectors are normalized)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 3. Add vectors to FAISS
        # CRITICAL: Even if embeddings were generated in FP16 on the GPU, 
        # FAISS CPU indexes mathematically require float32. 
        float32_embeddings = np.array(embeddings).astype('float32')
        self.index.add(float32_embeddings)
        
        # 4. Store metadata
        self.metadata_store = chunks
        print(f"Dense Index built with {self.index.ntotal} vectors.")

    def save(self):
        # Save the FAISS index (the raw float32 vectors)
        faiss.write_index(self.index, self.index_path)
        
        # Save the metadata (the text and QASPER IDs) using pickle
        with open(self.index_path + ".meta", "wb") as f:
            pickle.dump(self.metadata_store, f)
        print(f"Saved dense index to {self.index_path}")
        print(f"Saved metadata to {self.index_path}.meta")