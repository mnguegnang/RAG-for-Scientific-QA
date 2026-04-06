import faiss
import numpy as np
import pickle
import torch
from pathlib import Path
from typing import List, Dict
from src.retrieval.encoders import Specter2Encoder

# Default index path is resolved relative to this file's location so the
# path is correct regardless of the working directory the script is run from.
_DEFAULT_DENSE_INDEX = str(Path(__file__).resolve().parents[2] / "data" / "indices" / "dense.index")

device = "cuda" if torch.cuda.is_available() else "cpu"

class DenseIndexer:
    def __init__(self, index_path=_DEFAULT_DENSE_INDEX):
        # SPECTER2: scientifically pre-trained encoder (Singh et al., 2022, SciRepEval)
        # Dimension 768 vs. 384 for bge-small — richer representation for scientific text
<<<<<<< HEAD
        
        # Optimization: Use mixed precision (FP16) on GPU to accelerate the forward 
        # pass and save VRAM when embedding thousands of QASPER chunks.
        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        
        self.model = Specter2Encoder(
            device=device, 
            model_kwargs=model_kwargs  # Pass to the underlying transformer
        )
=======
        
        # Optimization: Use mixed precision (FP16) on GPU to accelerate the forward 
        # pass and save VRAM when embedding thousands of QASPER chunks.
        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        
        self.model = Specter2Encoder(
            device=device, 
            model_kwargs=model_kwargs  # Pass to the underlying transformer
        )
>>>>>>> cc6e01ad33bfbf2fa9000592545c986b7eeb4561
        self.index_path = index_path
        self.dimension = Specter2Encoder.DIMENSION  # 768
        self.index = None
        self.metadata_store = []  # FAISS only stores numbers; we need to store the text separately

    def build_index(self, chunks: List[Dict]):
        n_gpu = torch.cuda.device_count() if device == "cuda" else 1
        # Scale batch_size proportionally: DataParallel splits each batch
        # evenly across all GPUs, so a larger total batch keeps every
        # device saturated.  64 per-GPU x n_gpu → 192 with 3 GPUs.
        batch_size = 64 * max(1, n_gpu)
        print(
            f"Generating embeddings for {len(chunks)} chunks "
            f"on {device} ({n_gpu} GPU(s))  |  batch_size={batch_size}..."
        )

        texts = [c['text'] for c in chunks]

        # 1. Encode
        # normalize_embeddings=True converts Inner Product to Cosine Similarity.
        # batch_size scales with n_gpu to maximise DataParallel throughput.
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
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
        import os
        os.makedirs(os.path.dirname(os.path.abspath(self.index_path)), exist_ok=True)
        # Save the FAISS index (the raw float32 vectors)
        faiss.write_index(self.index, self.index_path)
        
        # Save the metadata (the text and QASPER IDs) using pickle
        with open(self.index_path + ".meta", "wb") as f:
            pickle.dump(self.metadata_store, f)
        print(f"Saved dense index to {self.index_path}")
        print(f"Saved metadata to {self.index_path}.meta")