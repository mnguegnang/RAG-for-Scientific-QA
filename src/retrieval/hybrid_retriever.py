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

        # 5. paper_id -> row indices, for PRE-filtered retrieval.
        # Filtering after a global top-k starves paper-scoped queries: the
        # global ranking is dominated by the other ~887 papers, so a paper
        # holding dozens of relevant chunks can survive with one or none.
        # Restricting the candidate set *before* ranking makes top-k mean
        # "top-k within this paper", which is what QASPER asks for
        # (Dasigi et al., NAACL 2021).
        self._dense_rows_by_paper = defaultdict(list)
        for row, chunk in enumerate(self.dense_meta):
            self._dense_rows_by_paper[chunk['paper_id']].append(row)
        self._sparse_rows_by_paper = defaultdict(list)
        for row, paper_id in enumerate(self.bm25_ids):
            self._sparse_rows_by_paper[paper_id].append(row)
        #self.bm25_corpus = self.bm25_package['metadata'] # The actual chunks
        #self.bm25_ids = self.bm25_package['doc_ids'] # The IDs
            
    def _search_dense(self, query: str, k: int, dense_query: str = None,
                      filter_paper_id: str = None):
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
        
        # Search. With a paper filter, hand FAISS an IDSelector so the top-k is
        # computed *within* the paper's rows rather than filtered out of a
        # global top-k. IndexFlatIP supports this through SearchParameters.
        if filter_paper_id is not None:
            rows = self._dense_rows_by_paper.get(filter_paper_id, [])
            if not rows:
                return []
            selector = faiss.IDSelectorBatch(np.asarray(rows, dtype='int64'))
            params = faiss.SearchParameters(sel=selector)
            scores, indices = self.dense_index.search(q_vec, min(k, len(rows)),
                                                      params=params)
        else:
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

    def _search_sparse(self, query: str, k: int, filter_paper_id: str = None):
        """BM25 search, optionally restricted to one paper's chunks."""
        # tokenize_for_bm25: identical pipeline to index-time (LaTeX strip +
        # lowercase + NLTK word_tokenize + stop-word removal + Porter stemming).
        # Robertson & Zaragoza (2009), BM25 and Beyond.
        tokenized_query = tokenize_for_bm25(query)
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)

        # Restrict the candidate pool to the paper before ranking (see the
        # note on self._dense_rows_by_paper). Ranking globally and filtering
        # afterwards discards in-paper chunks that never made the global top-k.
        if filter_paper_id is not None:
            candidate_rows = np.asarray(
                self._sparse_rows_by_paper.get(filter_paper_id, []), dtype='int64')
            if candidate_rows.size == 0:
                return []
            candidate_scores = scores[candidate_rows]
        else:
            candidate_rows = np.arange(len(scores), dtype='int64')
            candidate_scores = scores

        # Top-k within the candidate pool. argpartition needs k < n, so clamp.
        k_eff = min(k, candidate_rows.size)
        if k_eff < candidate_rows.size:
            top_n = np.argpartition(candidate_scores, -k_eff)[-k_eff:]
        else:
            top_n = np.arange(candidate_rows.size)
        top_n = top_n[np.argsort(candidate_scores[top_n])][::-1]
        best_indices = candidate_rows[top_n]
        
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
        # 1. Get Independent Results.
        # Both legs pre-filter on paper_id, so each returns its own top-k drawn
        # from that paper alone. No over-fetch factor and no post-filter: those
        # only ever recovered whatever happened to survive a global ranking.
        dense_res = self._search_dense(query, k, dense_query=dense_query,
                                       filter_paper_id=filter_paper_id)
        sparse_res = self._search_sparse(query, k,
                                         filter_paper_id=filter_paper_id)

        if filter_paper_id and not dense_res and not sparse_res:
            logger.warning(
                "No chunks indexed for paper '%s'; returning no results. "
                "Answering from other papers would be worse than abstaining.",
                filter_paper_id,
            )
            return []

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