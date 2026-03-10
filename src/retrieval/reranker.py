import logging
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

class Reranker:
    def __init__(self, model_name: str = 'BAAI/bge-reranker-base'):
        """
        Initializes the Cross-Encoder reranker.

        Default: BAAI/bge-reranker-base
          - Trained on diverse (query, passage) pairs including academic text
          - Outperforms ms-marco-MiniLM on BEIR academic subsets (SCIDOCS, SciFact)
          - Zhang et al. (2023) https://huggingface.co/BAAI/bge-reranker-base
          - Outputs raw logits; sigmoid(logit) > 0.5 ≈ relevant
        """
        logger.info("Loading Reranker model: %s on device: %s", model_name, device)
        self.model = CrossEncoder(model_name, device=device)
        
    def rerank(self, query: str, documents: list, top_k: int = 10):
        """
        Re-ranks a list of documents based on relevance to the query.
        
        params:
        query: The user question.
        documents: List of dictionaries. Must contain a 'text' key.
                   (These come from the Hybrid Retriever)
        top_k: Number of results to return after re-ranking.
        """
        if not documents:
            return []
            
        # 1. Prepare pairs for the model
        # The model needs a list of [Query, Text] pairs
        pairs = [[query, doc['text']] for doc in documents]
        
        # 2. Predict scores
        # scores will be a numpy array of floats (logits)
        scores = self.model.predict(pairs)
        
        # 3. Attach scores to documents
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])
            
        # 4. Sort by new score
        # High logit = High relevance
        sorted_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        # 5. Return Top K
        return sorted_docs[:top_k]