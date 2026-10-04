import argparse
import logging
import sys
import traceback

# Import Retriever Components
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import ColBERTv2Reranker
from src.retrieval.crag_evaluator import CRAGEvaluator
from src.retrieval.context_selection import (
    apply_token_budget, chunk_sort_key, ranks_from_scores, rrf_fuse,
)

# Configure logging to track the pipeline's progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ScientificRAGPipeline:
    # Stage-1 candidate pool. With paper-scoped retrieval, 94% of QASPER papers
    # have <= 100 chunks, so this is effectively "the whole paper" and the
    # reranker (optionally fused with the stage-1 ranks) decides what is kept.
    CANDIDATE_K: int = 100

    # HyDE word-count threshold (Gao et al., 2022 — arXiv:2212.10496).
    # Queries shorter than this are considered "vague" and benefit from
    # generating a hypothetical answer before encoding for dense search.
    HYDE_QUERY_WORD_THRESHOLD: int = 10

    def __init__(self,
                 dense_index_path: str = "data/indices/dense.index",
                 dense_meta_path: str = "data/indices/dense.index.meta",
                 sparse_index_path: str = "data/indices/sparse.pkl",
                 generator_backend: str = "auto",
                 ollama_model: str = "llama3",
                 hf_model: str = "meta-llama/Llama-3.1-8B-Instruct",
                 # See the --crag-correct/--crag-ambiguous CLI help below for the
                 # 2026-09-02 recalibration rationale (calibrate_crag.py). This is
                 # the default generate_predictions.py actually uses (it does not
                 # pass CRAG thresholds explicitly), so it must match the CLI default.
                 crag_correct_threshold: float = 14.4403,
                 crag_ambiguous_threshold: float = 8.0,
                 crag_consistency_ratio: float = None,
                 crag_mode: str = "signal",
                 context_k: int = 20,
                 context_order: str = "paper",
                 final_ranking: str = "colbert",
                 max_context_tokens: int = 6000,
                 load_generator: bool = True):
        """
        Initializes the end-to-end RAG system.

        Context-selection settings (key_metrics_improvements.md, 2026-10-03, P3-P5):
          crag_mode          "signal" (default) | "refine" | "legacy" — see CRAGEvaluator.
          context_k          chunks passed to the generator after ranking (default 20).
          context_order      "paper" (default; order-preserving RAG, Yu et al. 2024)
                             or "rank" (relevance-descending, the old behaviour).
          final_ranking      "colbert" (default) or "rrf" (ColBERT + BM25 + SPECTER2
                             ranks fused with RRF, Cormack et al. 2009).
          max_context_tokens approximate budget for the context block; the
                             lowest-ranked chunks are dropped beyond it.
          load_generator     False builds stages 1-3 only (no LLM; HyDE disabled),
                             for the offline retrieval harness.
        """
        if context_order not in ("paper", "rank"):
            raise ValueError("context_order must be 'paper' or 'rank'")
        if final_ranking not in ("colbert", "rrf"):
            raise ValueError("final_ranking must be 'colbert' or 'rrf'")
        if crag_consistency_ratio is None:
            # 0.3 was the legacy self-consistency gate; CRAG §4.3 has none.
            crag_consistency_ratio = 0.3 if crag_mode == "legacy" else 0.0

        self.context_k = context_k
        self.context_order = context_order
        self.final_ranking = final_ranking
        self.max_context_tokens = max_context_tokens

        logging.info("Initializing the Hybrid Retriever (SPECTER2 encoder)...")
        # SPECTER2: scientifically pre-trained, 768-dim (Singh et al., 2022)
        # Query encoder must match the encoder used during ingestion (DenseIndexer)
        self.retriever = HybridRetriever(
            dense_index_path=dense_index_path,
            dense_meta_path=dense_meta_path,
            sparse_index_path=sparse_index_path,
        )

        logging.info("Initializing ColBERT v2 Late-Interaction Reranker...")
        # Santhanam et al. (2022) — ColBERTv2 late-interaction MaxSim scoring.
        self.reranker = ColBERTv2Reranker(model_name="colbert-ir/colbertv2.0")

        logging.info("Initializing CRAG Retrieval Evaluator (Yan et al., 2024, mode=%s)...",
                     crag_mode)
        self.crag_evaluator = CRAGEvaluator(
            correct_threshold=crag_correct_threshold,
            ambiguous_threshold=crag_ambiguous_threshold,
            consistency_ratio=crag_consistency_ratio,
            mode=crag_mode,
            # CRAG §4.4: strips are scored by the same evaluator as documents.
            strip_scorer=self.reranker.score,
        )

        self.generator = None
        if load_generator:
            # Imported lazily so the retrieval-only harness does not need the
            # generation dependencies.
            from src.generation.llm_generator import LocalLLMGenerator
            logging.info("Initializing the LLM Generator (backend=%s)...", generator_backend)
            self.generator = LocalLLMGenerator(
                backend=generator_backend,
                ollama_model=ollama_model,
                hf_model=hf_model,
            )

        logging.info("System Ready.")

    def config(self) -> dict:
        """Context-selection settings, persisted with every evaluation run."""
        return {
            "context_k": self.context_k,
            "context_order": self.context_order,
            "final_ranking": self.final_ranking,
            "max_context_tokens": self.max_context_tokens,
            "crag_mode": self.crag_evaluator.mode,
            "crag_correct_threshold": self.crag_evaluator.correct_threshold,
            "crag_ambiguous_threshold": self.crag_evaluator.ambiguous_threshold,
            "crag_consistency_ratio": self.crag_evaluator.consistency_ratio,
        }

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

    def _hyde_can_matter(self, filter_paper_id: str) -> bool:
        """
        HyDE only changes the dense leg. With ColBERT-only final ranking and a
        paper that fits in the candidate pool, stage 1 returns every chunk of
        the paper whatever the dense query is, so the LLM call is wasted.
        With final_ranking="rrf" the dense *rank* is fused in, so it matters.
        """
        if self.final_ranking == "rrf" or not filter_paper_id:
            return True
        return self.retriever.paper_chunk_count(filter_paper_id) > self.CANDIDATE_K

    def _final_ranking(self, ranked: list) -> list:
        """Order the reranked candidates by the configured final ranking."""
        if self.final_ranking == "colbert":
            return ranked
        colbert = ranks_from_scores({d['text']: d['rerank_score'] for d in ranked})
        dense = {d['text']: d['dense_rank'] for d in ranked if d.get('dense_rank')}
        sparse = {d['text']: d['sparse_rank'] for d in ranked if d.get('sparse_rank')}
        fused = rrf_fuse({"colbert": colbert, "dense": dense, "sparse": sparse})
        for d in ranked:
            d['fused_score'] = fused.get(d['text'], 0.0)
        return sorted(ranked, key=lambda d: d['fused_score'], reverse=True)

    def retrieve_context(self, query: str, filter_paper_id: str = None,
                         use_hyde: bool = True) -> dict:
        """
        Stages 1-3: retrieve, rank, CRAG-evaluate, and arrange the context.

        Returns a dict with "docs" (in the order the generator sees them;
        each doc carries "rerank_rank", its 1-based position in the final
        ranking), "crag_action", "crag_details", "crag_triggered", "hyde_used".
        """
        # ── Stage 1: RETRIEVE (Recall) ──────────────────────────────────────
        # HyDE (Gao et al., 2022): for short/vague queries, generate a
        # hypothetical answer and encode *that* for dense search.
        # BM25 always uses the original query for exact keyword matching.
        dense_query = None
        if (use_hyde and self.generator is not None
                and len(query.split()) < self.HYDE_QUERY_WORD_THRESHOLD
                and self._hyde_can_matter(filter_paper_id)):
            logging.info("Stage 1a: Short query — running HyDE...")
            try:
                dense_query = self._generate_hyde_query(query)
                logging.info("HyDE passage generated (%d chars): %.80s...",
                             len(dense_query), dense_query)
            except Exception as hyde_err:
                logging.warning("HyDE generation failed (%s); falling back to raw query.",
                                hyde_err)
                dense_query = None

        logging.info("Stage 1: Fetching top %d candidates via Hybrid Search (Dense + Sparse)...",
                     self.CANDIDATE_K)
        broad_results = self.retriever.search(query, k=self.CANDIDATE_K,
                                              dense_query=dense_query,
                                              filter_paper_id=filter_paper_id)

        # No unfiltered retry. On a paper-anchored benchmark, answering from a
        # different paper is strictly worse than abstaining: the cited evidence
        # cannot entail the ground truth, so such rows scored context_recall
        # 0.038 / answer_correctness 0.125 against 0.373 / 0.468 for
        # paper-scoped rows. Retrieval now pre-filters, so 0 results means the
        # paper genuinely has no indexed chunks.
        if not broad_results:
            return {"docs": [], "crag_action": None, "crag_details": {},
                    "crag_triggered": False, "hyde_used": dense_query is not None}

        # ── Stage 2: RANK ───────────────────────────────────────────────────
        # ColBERT v2 scores every candidate; the final ranking is ColBERT alone
        # or ColBERT fused with the stage-1 ranks (final_ranking="rrf").
        logging.info("Stage 2: Ranking (%s), keeping top %d...",
                     self.final_ranking, self.context_k)
        ranked = self._final_ranking(self.reranker.rerank(query, broad_results, top_k=None))
        top_docs = ranked[:self.context_k]
        for rank, doc in enumerate(top_docs, 1):
            doc['rerank_rank'] = rank

        # ── Stage 3: CRAG EVALUATION (Yan et al., 2024) ────────────────────
        logging.info("Stage 3: CRAG retrieval evaluation (mode=%s)...", self.crag_evaluator.mode)
        crag_action, refined_docs, crag_details = self.crag_evaluator.evaluate_and_refine(
            query, top_docs
        )
        if crag_action == 'Incorrect' and self.crag_evaluator.mode == 'legacy':
            # Legacy graceful fallback: top-5 docs by rerank score instead of
            # refusing (CRAG §3.3 triggers web search, not refusal; we have none).
            refined_docs = sorted(top_docs, key=lambda x: x.get('rerank_score', 0.0),
                                  reverse=True)[:5]
            logging.warning("CRAG action=Incorrect: falling back to top-%d docs by rerank score.",
                            len(refined_docs))

        # ── Arrange the context ─────────────────────────────────────────────
        # Budget first (drops the lowest-ranked docs), then presentation order.
        # Order-preserving RAG (Yu et al. 2024, OP-RAG, §3 Eq. 2): present the
        # kept chunks in their order in the paper, not by score.
        docs = apply_token_budget(sorted(refined_docs, key=lambda d: d['rerank_rank']),
                                  self.max_context_tokens)
        if self.context_order == "paper":
            docs = sorted(docs, key=chunk_sort_key)

        return {
            "docs": docs,
            "crag_action": crag_action,
            "crag_details": crag_details,
            "crag_triggered": crag_action in ('Ambiguous', 'Incorrect'),
            "hyde_used": dense_query is not None,
        }

    def ask(self, query: str, filter_paper_id: str = None) -> dict:
        """
        Executes the full RAG pipeline for a given query.

        Stages
        ------
        1. Hybrid retrieval  — candidates (dense + sparse via RRF), paper-scoped
        2. Ranking           — ColBERT v2 (optionally fused with stage-1 ranks)
        3. CRAG evaluation   — {Correct, Incorrect, Ambiguous} (Yan et al., 2024)
        4. Generation        — Llama 3.1 with citation prompt, context in paper order
        """
        if self.generator is None:
            raise RuntimeError("Pipeline was built with load_generator=False.")
        logging.info(f"Processing Query: '{query}'")
        ctx = self.retrieve_context(query, filter_paper_id=filter_paper_id)

        if not ctx["docs"]:
            return {
                "answer": "Error: No documents found in the database.",
                "retrieved_docs": [],
                "crag_triggered": False,
                "crag_action": None,
                "crag_details": {},
            }

        # ── Stage 4: GENERATION ─────────────────────────────────────────────
        logging.info("Stage 4: Passing %d documents to LLM for generation...", len(ctx["docs"]))
        final_answer = self.generator.generate_answer(query, ctx["docs"])

        return {
            "answer": final_answer,
            "retrieved_docs": ctx["docs"],
            "crag_triggered": ctx["crag_triggered"],
            "crag_action": ctx["crag_action"],
            "crag_details": ctx["crag_details"],
        }


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Context-selection / CRAG flags shared by every entry point that builds the pipeline."""
    # correct_threshold recalibrated 2026-09-02 via
    # `python -m src.evaluation.calibrate_crag` against the post-paper-scoping-fix
    # evaluation_report.csv (123 labelled rows: Incorrect=73, Ambiguous=4, Correct=46):
    # F1-maximising boundary for {Incorrect,Ambiguous} vs Correct = 14.4403 (F1=0.5542).
    # ambiguous_threshold intentionally left at its prior value: the same run's
    # boundary search for Incorrect vs {Ambiguous,Correct} also landed at 14.4403
    # (i.e. collapsed onto correct_threshold), because the Ambiguous class had only
    # 4 examples — not enough signal to place a distinct boundary.
    parser.add_argument("--crag-correct", type=float, default=14.44,
        help="ColBERT MaxSim threshold for CRAG 'Correct' label (default: 14.44)")
    parser.add_argument("--crag-ambiguous", type=float, default=8.0,
        help="ColBERT MaxSim threshold for CRAG 'Ambiguous' label (default: 8.0)")
    parser.add_argument("--crag-consistency", type=float, default=None,
        help="Min fraction of docs labeled Correct for action=Correct "
             "(default: 0.0, or 0.3 in legacy mode)")
    parser.add_argument("--crag-mode", choices=CRAGEvaluator.MODES, default="signal",
        help="signal: label only (default); refine: + ColBERT strip refinement; "
             "legacy: pre-2026-10 filtering behaviour")
    parser.add_argument("--context-k", type=int, default=20,
        help="Chunks passed to the generator (default: 20)")
    parser.add_argument("--context-order", choices=["paper", "rank"], default="paper",
        help="Present chunks in paper order (OP-RAG, default) or by rank")
    parser.add_argument("--final-ranking", choices=["colbert", "rrf"], default="colbert",
        help="ColBERT alone (default) or RRF of ColBERT + BM25 + SPECTER2 ranks")
    parser.add_argument("--max-context-tokens", type=int, default=6000,
        help="Approximate token budget for the context block (default: 6000)")


def pipeline_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        crag_correct_threshold=args.crag_correct,
        crag_ambiguous_threshold=args.crag_ambiguous,
        crag_consistency_ratio=args.crag_consistency,
        crag_mode=args.crag_mode,
        context_k=args.context_k,
        context_order=args.context_order,
        final_ranking=args.final_ranking,
        max_context_tokens=args.max_context_tokens,
    )


def main():
    # Setup Argument Parser for Command Line Execution
    parser = argparse.ArgumentParser(description="Query the Scientific NLP RAG System.")
    parser.add_argument("--query", type=str, required=True, help="The scientific question to ask.")
    parser.add_argument("--paper-id", type=str, default=None,
        help="Restrict retrieval to one QASPER paper (arXiv id).")

    # Exact requested file paths set as Command Line Interface (CLI) defaults
    parser.add_argument("--dense-index", type=str, default="data/indices/dense.index")
    parser.add_argument("--dense-meta", type=str, default="data/indices/dense.index.meta")
    parser.add_argument("--sparse-index", type=str, default="data/indices/sparse.pkl")
    add_pipeline_args(parser)
    parser.add_argument("--backend", type=str, default="auto",
        choices=["auto", "ollama", "transformers", "vllm"],
        help="LLM backend. 'auto' picks transformers on GPU, ollama on CPU (default: auto)")
    parser.add_argument("--ollama-model", type=str, default="llama3",
        help="Ollama model tag (used when backend=ollama, default: llama3)")
    parser.add_argument("--hf-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID (used when backend=transformers)")

    args = parser.parse_args()

    try:
        rag_system = ScientificRAGPipeline(
            dense_index_path=args.dense_index,
            dense_meta_path=args.dense_meta,
            sparse_index_path=args.sparse_index,
            generator_backend=args.backend,
            ollama_model=args.ollama_model,
            hf_model=args.hf_model,
            **pipeline_kwargs(args),
        )

        print("\n" + "*"*60)
        print(f"QUESTION: {args.query}")
        print("*"*60 + "\n")

        result = rag_system.ask(args.query, filter_paper_id=args.paper_id)

        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result["answer"])

        if result.get("crag_triggered"):
            print(f"\n[CRAG] Action: {result.get('crag_action', 'N/A')}")
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
