import faiss
import hashlib
import os
import pickle
import logging
import numpy as np
import torch
from collections import defaultdict
from src.retrieval.encoders import Specter2Encoder
from src.retrieval.sparse_store import tokenize_for_bm25

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Project root for path-traversal checks
_PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _load_pickle_verified(path: str) -> object:
    """Load a pickle file with path-traversal protection and SHA-256 integrity check.

    Security rationale (OWASP A8 — Software and Data Integrity Failures):
      pickle.load() can execute arbitrary code.  This wrapper adds two layers:
        1. Path-traversal guard: rejects paths that resolve outside the project root.
        2. SHA-256 sidecar verification: if ``<path>.sha256`` exists, the pickle
           bytes are hashed and compared before deserialisation.

    If no .sha256 sidecar is found the file is loaded with a logged warning
    (backwards compatible with indices generated before this check was added).
    """
    resolved = os.path.realpath(path)
    if not resolved.startswith(_PROJECT_ROOT + os.sep) and resolved != _PROJECT_ROOT:
        raise ValueError(
            f"Path traversal blocked: '{path}' resolves to '{resolved}' "
            f"which is outside the project root '{_PROJECT_ROOT}'."
        )

    with open(resolved, 'rb') as fh:
        raw = fh.read()

    sha256_sidecar = resolved + ".sha256"
    if os.path.exists(sha256_sidecar):
        actual_hash = hashlib.sha256(raw).hexdigest()
        with open(sha256_sidecar) as fh:
            expected_hash = fh.read().strip()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Integrity check FAILED for {path}: "
                f"expected sha256={expected_hash}, got {actual_hash}. "
                "The file may have been tampered with or corrupted. "
                "Re-run ingestion to regenerate indices."
            )
        logger.info("SHA-256 integrity check passed for %s", path)
    else:
        logger.warning(
            "No .sha256 sidecar for %s — skipping integrity check. "
            "Re-run ingestion to generate integrity files.", path,
        )

    return pickle.loads(raw)


class HybridRetriever:
    def __init__(self, 
                 dense_index_path: str, 
                 dense_meta_path: str, 
                 sparse_index_path: str):
        
        # 1. Load SPECTER2 Encoder (for query encoding)
        # Must match the encoder used at ingestion time (DenseIndexer)
        logger.info("Loading SPECTER2 Encoder...")
        self.encoder = Specter2Encoder(device=device)
        
        # 2. Load Dense Index (FAISS)
        logger.info("Loading FAISS Index from %s...", dense_index_path)
        self.dense_index = faiss.read_index(dense_index_path)
        
        # 3. Load Metadata (To map FAISS IDs back to text) — verified pickle load
        self.dense_meta = _load_pickle_verified(dense_meta_path)
            
        # 4. Load Sparse Index (BM25) — verified pickle load
        logger.info("Loading BM25 Index from %s...", sparse_index_path)
        self.bm25_package = _load_pickle_verified(sparse_index_path)
        # The pickle contains the object and the corpus, we unpack it
        self.bm25 = self.bm25_package['model']
        # Extract Metadata it format. It was saved as: {'model': bm25, 'metadata': [...]}
        # We need to separate the text and IDs for easy lookup during search
        metadata_list = self.bm25_package['metadata']

        self.bm25_corpus = [chunk['text'] for chunk in metadata_list] # The actual chunks
        self.bm25_ids = [chunk['paper_id'] for chunk in metadata_list]
        #self.bm25_corpus = self.bm25_package['metadata'] # The actual chunks
        #self.bm25_ids = self.bm25_package['doc_ids'] # The IDs
            
    def _search_dense(self, query: str, k: int, dense_query: str = None):
        """Standard Vector Search.

        Parameters
        ----------
        dense_query : str, optional
            When provided (e.g. a HyDE hypothetical passage), this text is
            encoded for dense search instead of *query*. The original *query*
            is still used for BM25 sparse search. Defaults to None.
        """
        # Encode query (or a HyDE hypothetical passage when provided)
        q_vec = self.encoder.encode([dense_query if dense_query else query])
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
        # tokenize_for_bm25: identical pipeline to index-time (LaTeX strip +
        # lowercase + NLTK word_tokenize + stop-word removal + Porter stemming).
        # Robertson & Zaragoza (2009), BM25 and Beyond.
        tokenized_query = tokenize_for_bm25(query)
        
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

    def search(self, query: str, k: int = 10, rrf_k: int = 60,
               dense_query: str = None, filter_paper_id: str = None):
        """
        Performs Hybrid Search using RRF.

        Parameters
        ----------
        dense_query : str, optional
            Override the text encoded for the dense (SPECTER2) leg only.
            Pass a HyDE hypothetical passage here for short queries.
            BM25 sparse search always uses the original *query*.
        """
        # 1. Get Independent Results
        # Fetch more candidates when paper-scoped filtering is active so that
        # after the paper_id post-filter we still return up to k results.
        # Dasigi et al. (2021 NAACL) — QASPER questions are anchored to one paper;
        # cross-paper retrieval always produces ALCE=0 because cited evidence
        # never entails the ground truth from a different paper.
        broad_k = k * 4 if filter_paper_id else k
        dense_res = self._search_dense(query, broad_k, dense_query=dense_query)
        sparse_res = self._search_sparse(query, broad_k)

        if filter_paper_id:
            dense_res  = [r for r in dense_res  if r.get("doc_id") == filter_paper_id][:k]
            sparse_res = [r for r in sparse_res if r.get("doc_id") == filter_paper_id][:k]
        
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