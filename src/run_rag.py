import argparse
import logging
import sys
import os
import traceback

# Import Retriever Components
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

# Import Generator Component
from src.generation.llm_generator import LocalLLMGenerator

# Configure logging to track the pipeline's progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScientificRAGPipeline:
    def __init__(self, 
                 dense_index_path: str = "data/indices/dense.index",
                 dense_meta_path: str = "data/indices/dense.index.meta",
                 sparse_index_path: str = "data/indices/sparse.pkl",
                 generator_backend: str = "auto",
                 ollama_model: str = "llama3",
                 hf_model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initializes the entire end-to-end RAG system with the exact artifact paths.
        """
        logging.info("Initializing the Hybrid Retriever (SPECTER2 encoder)...")

        # SPECTER2: scientifically pre-trained, 768-dim (Singh et al., 2022)
        # Query encoder must match the encoder used during ingestion (DenseIndexer)
        self.retriever = HybridRetriever(
            dense_index_path=dense_index_path,
            dense_meta_path=dense_meta_path,
            sparse_index_path=sparse_index_path,
        )

        logging.info("Initializing the Cross-Encoder Reranker (bge-reranker-v2-m3)...")
        # We place BAAI/bge-reranker-base with 'BAAI/bge-reranker-v2-m3' is ideal here since it outperforms ms-marco-MiniLM on academic BEIR subsets
        self.reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3")
        
        logging.info("Initializing the LLM Generator (backend=%s)...", generator_backend)
        # backend="auto": uses transformers on GPU (supercomputer), ollama on CPU (laptop)
        self.generator = LocalLLMGenerator(
            backend=generator_backend,
            ollama_model=ollama_model,
            hf_model=hf_model,
        )
        
        logging.info("System Ready.")

    # CRAG relevance threshold (Yan et al., 2024 — Corrective RAG)
    # bge-reranker-base outputs raw logits; sigmoid(0.0) = 0.5.
    # A best-document logit below this value means no retrieved document
    # is likely relevant — generation is suppressed to avoid hallucination.
    # Tune empirically via: python -m src.evaluation.calibrate_crag
    CRAG_THRESHOLD: float = 0.517#-2#0.0

    # HyDE word-count threshold (Gao et al., 2022 — arXiv:2212.10496).
    # Queries shorter than this are considered "vague" and benefit from
    # generating a hypothetical answer before encoding for dense search.
    # Examples of short queries: "What are the results?" (4 words),
    #                             "How big is the Japanese data?" (7 words).
    HYDE_QUERY_WORD_THRESHOLD: int = 10

    def _generate_hyde_query(self, query: str) -> str:
        """
        Generates a hypothetical passage (HyDE) for dense retrieval.

        Short queries produce weak embedding signals because SPECTER2 was
        pre-trained on passage-level text, not question-style strings.
        Encoding a hypothetical answer passage instead closes this gap.

        Reference:
            Gao et al. (2022). Precise Zero-Shot Dense Retrieval without
            Relevance Labels (HyDE). arXiv:2212.10496. ACL 2023.
        """
        return self.generator.generate_hypothetical_answer(query)

    def ask(self, query: str, filter_paper_id: str = None) -> dict:
        """
        Executes the full RAG pipeline for a given query.

        Stages
        ------
        1. Hybrid retrieval  — top-50 candidates (dense + sparse via RRF)
        2. Reranking         — bge-reranker-base narrows to top-7
        3. CRAG gate         — suppresses generation when context is irrelevant
        4. Generation        — Llama3 with CoT + citation prompt
        """
        logging.info(f"Processing Query: '{query}'")

        # 1. RETRIEVE (Recall)
        # HyDE (Gao et al., 2022): for short/vague queries, generate a
        # hypothetical answer and encode *that* for dense search.
        # BM25 always uses the original query for exact keyword matching.
        dense_query = None
        query_word_count = len(query.split())
        if query_word_count < self.HYDE_QUERY_WORD_THRESHOLD:
            logging.info(
                "Stage 1a: Short query (%d words < threshold %d) — running HyDE...",
                query_word_count, self.HYDE_QUERY_WORD_THRESHOLD,
            )
            try:
                dense_query = self._generate_hyde_query(query)
                logging.info(
                    "HyDE passage generated (%d chars): %.80s...",
                    len(dense_query), dense_query,
                )
            except Exception as hyde_err:
                logging.warning(
                    "HyDE generation failed (%s); falling back to raw query.",
                    hyde_err,
                )
                dense_query = None
        logging.info("Stage 1: Fetching top 50 candidates via Hybrid Search (Dense + Sparse)...")
        broad_results = self.retriever.search(query, k=50, dense_query=dense_query, filter_paper_id=filter_paper_id)

        if not broad_results:
            return {"answer": "Error: No documents found in the database.", "retrieved_docs": [],
                    "crag_triggered": False}

        # 2. RERANK (Precision)
        # top_k=7: Liu et al. (2023) 'Lost in the Middle' shows LLM accuracy
        # peaks with 3–5 high-quality passages we increase to 7 for more context.
        logging.info("Stage 2: Reranking with bge-reranker-v2-m3, keeping top 7...")
        top_7_docs = self.reranker.rerank(query, broad_results, top_k=7)

        # 3. CRAG RELEVANCE GATE (Yan et al., 2024)
        # Check the highest rerank score. If even the best document is below
        # threshold, the retrieved context is too noisy for reliable generation.
        best_score = max(d.get("rerank_score", 0.0) for d in top_7_docs)
        logging.info(f"CRAG check — best rerank logit: {best_score:.4f} (threshold: {self.CRAG_THRESHOLD})")
        # umcomment the following later after calibrating the CRAG threshold empirically on some sample queries. 
        #For now, we want to see the full pipeline in action without the CRAG gate suppressing generation.
        if best_score < self.CRAG_THRESHOLD:
            logging.warning("CRAG gate triggered: retrieved context below relevance threshold.")
            return {
                "answer": (
                    "The retrieved documents do not contain enough information "
                    "to answer this question reliably."
                ),
                "retrieved_docs": top_7_docs,
                "crag_triggered": True,
            }

        # 4. GENERATION
        logging.info("Stage 3: Passing top 7 documents to LLM for generation...")
        final_answer = self.generator.generate_answer(query, top_7_docs)

        return {
            "answer": final_answer,
            "retrieved_docs": top_7_docs,
            "crag_triggered": False,
        }

def main():
    # Setup Argument Parser for Command Line Execution
    parser = argparse.ArgumentParser(description="Query the Scientific NLP RAG System.")
    parser.add_argument("--query", type=str, required=True, help="The scientific question to ask.")
    
    # Exact requested file paths set as Command Line Interface (CLI) defaults
    parser.add_argument("--dense-index", type=str, default="data/indices/dense.index")
    parser.add_argument("--dense-meta", type=str, default="data/indices/dense.index.meta")
    parser.add_argument("--sparse-index", type=str, default="data/indices/sparse.pkl")
    parser.add_argument("--crag-threshold", type=float, default=None,
        help="Override the CRAG relevance gate threshold (bge-reranker raw logit). "
             "Default: 0.0 (sigmoid=0.5, i.e., uncertain relevance)")
    parser.add_argument("--backend", type=str, default="auto",
        choices=["auto", "ollama", "transformers"],
        help="LLM backend. 'auto' picks transformers on GPU, ollama on CPU (default: auto)")
    parser.add_argument("--ollama-model", type=str, default="llama3",
        help="Ollama model tag (used when backend=ollama, default: llama3)")
    parser.add_argument("--hf-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID (used when backend=transformers)")

    args = parser.parse_args()

    try:
        # Initialize passing the correct CLI or Default paths
        rag_system = ScientificRAGPipeline(
            dense_index_path=args.dense_index,
            dense_meta_path=args.dense_meta,
            sparse_index_path=args.sparse_index,
            generator_backend=args.backend,
            ollama_model=args.ollama_model,
            hf_model=args.hf_model,
        )
        if args.crag_threshold is not None:
            rag_system.CRAG_THRESHOLD = args.crag_threshold
            logging.info(f"CRAG threshold overridden to {args.crag_threshold}")
        
        print("\n" + "*"*60)
        print(f"QUESTION: {args.query}")
        print("*"*60 + "\n")
        
        result = rag_system.ask(args.query)

        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result["answer"])

        if result.get("crag_triggered"):
            print("\n[CRAG] Generation suppressed: retrieved context below relevance threshold.")

        print("\n" + "-"*60)
        print(f"Retrieved {len(result['retrieved_docs'])} documents used as context.")
        print("-"*60 + "\n")
        
    except Exception as e:
        logging.error(f"Pipeline crashed: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()