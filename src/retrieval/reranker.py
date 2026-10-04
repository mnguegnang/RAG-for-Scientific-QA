import logging
import sys
import types
import torch

# ── Compatibility shim for ragatouille 0.0.9.x + langchain >= 1.0 ───────────
# ragatouille 0.0.9.post2 imports BaseDocumentCompressor from the pre-1.0 path
# `langchain.retrievers.document_compressors.base`, which was removed in
# langchain 1.0.  The class now lives in `langchain_core.documents.compressor`.
# Injecting stub modules at the old path before the ragatouille import resolves
# the ModuleNotFoundError without modifying any installed package files.
try:
    import langchain.retrievers  # noqa: F401 — already available, nothing to do
except ModuleNotFoundError:
    from langchain_core.documents.compressor import BaseDocumentCompressor
    _stub_ret = types.ModuleType("langchain.retrievers")
    _stub_dc = types.ModuleType("langchain.retrievers.document_compressors")
    _stub_base = types.ModuleType("langchain.retrievers.document_compressors.base")
    _stub_base.BaseDocumentCompressor = BaseDocumentCompressor
    _stub_dc.base = _stub_base
    _stub_ret.document_compressors = _stub_dc
    sys.modules.setdefault("langchain.retrievers", _stub_ret)
    sys.modules.setdefault("langchain.retrievers.document_compressors", _stub_dc)
    sys.modules.setdefault("langchain.retrievers.document_compressors.base", _stub_base)
# ─────────────────────────────────────────────────────────────────────────────

from ragatouille import RAGPretrainedModel

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


class ColBERTv2Reranker:
    """
    ColBERT v2 Late Interaction Reranker.

    Reference: Santhanam et al. (2022), "ColBERTv2: Effective and Efficient
    Retrieval via Lightweight Late Interaction" — NAACL 2022.
    https://arxiv.org/abs/2112.01488

    Key advantages over the previous cross-encoder (BAAI/bge-reranker-v2-m3):
      - Documents are encoded independently (no query-document cross-attention),
        so encoding is a single batched forward pass over all candidates.
      - Scoring uses a cheap MaxSim operation over pre-computed token embeddings:
            score(q, d) = Σ_i max_j cos(q_i, d_j)
      - Equivalent or better MRR@10 vs. full cross-encoders (Section 5.1, Table 1).
      - 100–1000× faster when document representations are pre-indexed (Section 5.3).

    Uses the RAGatouille library for ColBERT v2 model loading and MaxSim scoring.
    """

    def __init__(self, model_name: str = 'colbert-ir/colbertv2.0'):
        """
        Loads the ColBERT v2 checkpoint via RAGatouille.

        The underlying model is BERT-base with a 128-dim linear projection,
        fine-tuned with the ColBERT late-interaction objective on MS MARCO.
        Total parameters: ~110M (vs. ~568M for XLM-RoBERTa-large cross-encoder).
        """
        logger.info("Loading ColBERT v2 reranker: %s on device: %s", model_name, device)
        self.model = RAGPretrainedModel.from_pretrained(model_name)
        # Optional (query, text) -> score memo. Off by default; the offline
        # retrieval harness turns it on so sweeping many configurations over
        # the same questions scores each chunk once.
        self._cache = None
        logger.info("ColBERT v2 reranker ready.")

    def enable_cache(self) -> None:
        self._cache = {}

    def score(self, query: str, texts: list) -> list:
        """
        ColBERT v2 MaxSim score for each text, in input order.

        Also used by CRAG knowledge refinement to score sentence-level strips
        with the same evaluator that scored the documents (Yan et al. 2024,
        CRAG §4.4), instead of lexical overlap.
        """
        if not texts:
            return []
        # A throwaway dict when caching is off keeps one code path.
        cache = self._cache if self._cache is not None else {}
        todo = [t for t in dict.fromkeys(texts) if (query, t) not in cache]
        if todo:
            # ragatouille .rerank() returns [{'content', 'score', 'rank'}]
            reranked = self.model.rerank(query=query, documents=todo, k=len(todo))
            cache.update({(query, r['content']): float(r['score']) for r in reranked})
        return [cache.get((query, t), float('-inf')) for t in texts]

    def rerank(self, query: str, documents: list, top_k: int = 10) -> list:
        """
        Re-ranks documents using ColBERT v2 late-interaction scoring (MaxSim).

        Unlike the previous cross-encoder which required O(n) full forward passes
        (one per query-doc pair), ColBERT v2 encodes query and documents separately,
        then scores via cheap token-level MaxSim.

        params:
          query: The user question.
          documents: List of dicts. Must contain a 'text' key.
                     (These come from the Hybrid Retriever)
          top_k: Number of results to return after re-ranking. None returns
                 every candidate (used when the ranking is fused afterwards).

        returns:
          List of top_k documents sorted by ColBERT MaxSim score (descending).
        """
        if not documents:
            return []

        # Score ALL candidates (not just top_k) so downstream CRAG analysis and
        # rank fusion see the full score distribution.
        scores = self.score(query, [doc['text'] for doc in documents])
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = score

        sorted_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return sorted_docs if top_k is None else sorted_docs[:top_k]