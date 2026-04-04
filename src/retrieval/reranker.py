import logging
import torch
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
        logger.info("ColBERT v2 reranker ready.")

    def rerank(self, query: str, documents: list, top_k: int = 7) -> list:
        """
        Re-ranks documents using ColBERT v2 late-interaction scoring (MaxSim).

        Unlike the previous cross-encoder which required O(n) full forward passes
        (one per query-doc pair), ColBERT v2 encodes query and documents separately,
        then scores via cheap token-level MaxSim.

        params:
          query: The user question.
          documents: List of dicts. Must contain a 'text' key.
                     (These come from the Hybrid Retriever)
          top_k: Number of results to return after re-ranking.

        returns:
          List of top_k documents sorted by ColBERT MaxSim score (descending).
        """
        if not documents:
            return []

        # 1. Extract text for ColBERT scoring
        texts = [doc['text'] for doc in documents]

        # 2. Score ALL documents via ColBERT v2 MaxSim
        # ragatouille .rerank() returns: [{'content': str, 'score': float, 'rank': int}]
        # We score all candidates (not just top_k) so downstream CRAG analysis
        # can see the full score distribution for multi-signal confidence.
        reranked = self.model.rerank(
            query=query,
            documents=texts,
            k=len(texts)
        )

        # 3. Build content → score lookup
        score_map = {r['content']: r['score'] for r in reranked}

        # 4. Attach MaxSim scores to original document dicts
        for doc in documents:
            doc['rerank_score'] = float(score_map.get(doc['text'], float('-inf')))

        # 5. Sort by MaxSim score descending
        sorted_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        return sorted_docs[:top_k]