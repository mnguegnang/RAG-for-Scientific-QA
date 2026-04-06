import argparse
import logging
import sys
import os
import traceback

# Import Retriever Components
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import ColBERTv2Reranker
from src.retrieval.crag_evaluator import CRAGEvaluator

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
                 hf_model: str = "meta-llama/Llama-3.1-8B-Instruct",
                 crag_correct_threshold: float = 20.0,
                 crag_ambiguous_threshold: float = 12.0,
                 crag_consistency_ratio: float = 0.3):
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

        logging.info("Initializing ColBERT v2 Late-Interaction Reranker...")
        # Santhanam et al. (2022) — ColBERTv2 achieves equivalent or better MRR@10
        # than full cross-encoders while being 100-1000x faster via MaxSim scoring.
        # Replaces the previous BAAI/bge-reranker-v2-m3 cross-encoder.
        self.reranker = ColBERTv2Reranker(model_name="colbert-ir/colbertv2.0")

        logging.info("Initializing CRAG Retrieval Evaluator (Yan et al., 2024)...")
        # Full Corrective RAG framework: {Correct, Incorrect, Ambiguous} classification
        # with multi-signal confidence and knowledge refinement.
        # Replaces the single-threshold CRAG gate.
        self.crag_evaluator = CRAGEvaluator(
            correct_threshold=crag_correct_threshold,
            ambiguous_threshold=crag_ambiguous_threshold,
            consistency_ratio=crag_consistency_ratio,
        )

        logging.info("Initializing the LLM Generator (backend=%s)...", generator_backend)
        # backend="auto": uses transformers on GPU (supercomputer), ollama on CPU (laptop)
        self.generator = LocalLLMGenerator(
            backend=generator_backend, 
            ollama_model=ollama_model, 
            hf_model=hf_model 
            )
                
        logging.info("System Ready.")

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
        1. Hybrid retrieval  — top-100 candidates (dense + sparse via RRF)
        2. Reranking         — ColBERT v2 late-interaction narrows to top-7
        3. CRAG evaluation   — three-way {Correct, Incorrect, Ambiguous} gate
                               with knowledge refinement (Yan et al., 2024)
        4. Generation        — Llama3 with CoT + citation prompt
        """
        logging.info(f"Processing Query: '{query}'")

        # ── Stage 1: RETRIEVE (Recall) ──────────────────────────────────────
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
        # Fetch 100 candidates — ColBERT v2 is efficient enough to rerank a
        # larger pool, improving recall before CRAG precision filtering.
        logging.info("Stage 1: Fetching top 100 candidates via Hybrid Search (Dense + Sparse)...")
        broad_results = self.retriever.search(query, k=100, dense_query=dense_query, filter_paper_id=filter_paper_id)

        if not broad_results and filter_paper_id:
            # The paper-scoped filter returned 0 hits — the paper's chunks
            # may be indexed under a different doc_id key.  Fall back to
            # unfiltered search so the query still gets an answer rather
            # than a hard error that produces 0-context rows in the dataset.
            logging.warning(
                "Stage 1: filter_paper_id='%s' returned 0 results; "
                "retrying without paper filter (fallback).",
                filter_paper_id,
            )
            broad_results = self.retriever.search(
                query, k=100, dense_query=dense_query, filter_paper_id=None
            )

        if not broad_results:
            return {
                "answer": "Error: No documents found in the database.",
                "retrieved_docs": [],
                "crag_triggered": False,
                "crag_action": None,
                "crag_details": {},
            }

        # ── Stage 2: RERANK (Precision) ─────────────────────────────────────
        # ColBERT v2 (Santhanam et al., 2022) late-interaction scoring via MaxSim.
        # top_k=7: Liu et al. (2023) 'Lost in the Middle' shows LLM accuracy
        # peaks with 3–5 high-quality passages; we keep 7 for broader context.
        logging.info("Stage 2: Reranking with ColBERT v2 (MaxSim), keeping top 7...")
        top_7_docs = self.reranker.rerank(query, broad_results, top_k=7)

        # ── Stage 3: CRAG EVALUATION (Yan et al., 2024) ────────────────────
        # Full Corrective RAG framework:
        #   Section 3.1 — Per-document classification: {Correct, Incorrect, Ambiguous}
        #   Multi-signal confidence with self-consistency ratio
        #   Section 3.2 — Knowledge refinement: strip decomposition + filtering
        logging.info("Stage 3: CRAG retrieval evaluation (three-way classification)...")
        crag_action, refined_docs, crag_details = self.crag_evaluator.evaluate_and_refine(
            query, top_7_docs
        )

        if crag_action == 'Incorrect':
            logging.warning(
                "CRAG action=Incorrect: all documents below relevance threshold. "
                "Generation suppressed to avoid hallucination."
            )
            return {
                "answer": (
                    "The retrieved documents do not contain enough information "
                    "to answer this question reliably."
                ),
                "retrieved_docs": top_7_docs,
                "crag_triggered": True,
                "crag_action": crag_action,
                "crag_details": crag_details,
            }

        if crag_action == 'Ambiguous':
            logging.info(
                "CRAG action=Ambiguous: using refined knowledge from %d documents "
                "(original top-7 had mixed relevance).",
                len(refined_docs),
            )

        # ── Stage 4: GENERATION ─────────────────────────────────────────────
        logging.info(
            "Stage 4: Passing %d refined documents to LLM for generation...",
            len(refined_docs),
        )
        final_answer = self.generator.generate_answer(query, refined_docs)

        return {
            "answer": final_answer,
            "retrieved_docs": refined_docs,
            "crag_triggered": crag_action == 'Ambiguous',
            "crag_action": crag_action,
            "crag_details": crag_details,
        }
    
def main():
    # Setup Argument Parser for Command Line Execution
    parser = argparse.ArgumentParser(description="Query the Scientific NLP RAG System.")
    parser.add_argument("--query", type=str, required=True, help="The scientific question to ask.")
    
    # Exact requested file paths set as Command Line Interface (CLI) defaults
    parser.add_argument("--dense-index", type=str, default="data/indices/dense.index")
    parser.add_argument("--dense-meta", type=str, default="data/indices/dense.index.meta")
    parser.add_argument("--sparse-index", type=str, default="data/indices/sparse.pkl")
    parser.add_argument("--crag-correct", type=float, default=20.0,
        help="ColBERT MaxSim threshold for CRAG 'Correct' label (default: 20.0)")
    parser.add_argument("--crag-ambiguous", type=float, default=12.0,
        help="ColBERT MaxSim threshold for CRAG 'Ambiguous' label (default: 12.0)")
    parser.add_argument("--crag-consistency", type=float, default=0.3,
        help="Self-consistency ratio — min fraction of docs labeled Correct (default: 0.3)")
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
            crag_correct_threshold=args.crag_correct,
            crag_ambiguous_threshold=args.crag_ambiguous,
            crag_consistency_ratio=args.crag_consistency,
        )
        
        print("\n" + "*"*60)
        print(f"QUESTION: {args.query}")
        print("*"*60 + "\n")
        
        result = rag_system.ask(args.query)

        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result["answer"])

        if result.get("crag_triggered"):
            print(f"\n[CRAG] Action: {result.get('crag_action', 'N/A')}")
            if result['crag_action'] == 'Incorrect':
                print("[CRAG] Generation suppressed: all documents below relevance threshold.")
            elif result['crag_action'] == 'Ambiguous':
                print("[CRAG] Partial confidence: used refined knowledge from ambiguous documents.")
            details = result.get('crag_details', {})
            if details:
                print(f"[CRAG] Correct: {details.get('n_correct', 0)}, "
                      f"Ambiguous: {details.get('n_ambiguous', 0)}, "
                      f"Incorrect: {details.get('n_incorrect', 0)} | "
                      f"Consistency: {details.get('correct_ratio', 0):.2f}")

        print("\n" + "-"*60)
        print(f"Retrieved {len(result['retrieved_docs'])} documents used as context.")
        print("-"*60 + "\n")
        
    except Exception as e:
        logging.error(f"Pipeline crashed: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()