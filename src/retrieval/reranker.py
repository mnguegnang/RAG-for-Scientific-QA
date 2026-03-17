import logging
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

class Reranker:
<<<<<<< HEAD
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3'):
        """
        Initializes the Cross-Encoder reranker.

        Based on BGE-M3 (Chen et al., 2024) & C-Pack (Xiao et al., 2024):
          - Backbone: XLM-RoBERTa-large (~568M parameters).
          - Performance: Yields 2-4 point nDCG@10 improvements on academic datasets 
            over 'base' models due to deeper cross-attention capacities.
          - Context: Supports up to 8192 tokens natively.
        """
        logger.info("Loading Reranker model: %s on device: %s", model_name, device)
        
        # Optimization: Use mixed precision (FP16) on GPU to handle the larger 
        # 568M parameter model efficiently, preventing Out-Of-Memory errors.
        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        
        # We explicitly set max_length=1024. 
        # Since our dense retrieval chunks are capped at 500 tokens (Specter2 limit),
        # Query + Chunk will always fall comfortably under 1024. This bounds VRAM usage 
        # while taking full advantage of the model's capacity without truncation.
        self.model = CrossEncoder(
            model_name, 
            device=device,
            max_length=1024,
            model_kwargs=model_kwargs
        )
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
=======
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3'):
        """
        Initializes the Cross-Encoder reranker.

        Based on BGE-M3 (Chen et al., 2024) & C-Pack (Xiao et al., 2024):
          - Backbone: XLM-RoBERTa-large (~568M parameters).
          - Performance: Yields 2-4 point nDCG@10 improvements on academic datasets 
            over 'base' models due to deeper cross-attention capacities.
          - Context: Supports up to 8192 tokens natively.
        """
        logger.info("Loading Reranker model: %s on device: %s", model_name, device)
        
        # Optimization: Use mixed precision (FP16) on GPU to handle the larger 
        # 568M parameter model efficiently, preventing Out-Of-Memory errors.
        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        
        # We explicitly set max_length=1024. 
        # Since our dense retrieval chunks are capped at 500 tokens (Specter2 limit),
        # Query + Chunk will always fall comfortably under 1024. This bounds VRAM usage 
        # while taking full advantage of the model's capacity without truncation.
        self.model = CrossEncoder(
            model_name, 
            device=device,
            max_length=1024,
            model_kwargs=model_kwargs
        )
>>>>>>> cc6e01ad33bfbf2fa9000592545c986b7eeb4561
        
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
        # CrossEncoders process [CLS] Query [SEP] Passage [SEP] simultaneously
        pairs = [[query, doc['text']] for doc in documents]
        
        # 2. Predict scores
        # Returns a numpy array of raw logits. High logit = high relevance.
        scores = self.model.predict(pairs)
        
        # 3. Attach scores to documents
        for i, doc in enumerate(documents):
            # Convert to float to ensure JSON serializability later
            doc['rerank_score'] = float(scores[i])
            
        # 4. Sort by new score
        # Descending order: highest logit first
        sorted_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        # 5. Return Top K
        return sorted_docs[:top_k]