from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the Cross-Encoder model.
        """
        print(f"Loading Reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, documents: list, top_k: int = 5):
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