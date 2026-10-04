"""
Offline retrieval harness: Evidence Recall against QASPER's gold evidence.

Runs stages 1-3 of the real pipeline (ScientificRAGPipeline.retrieve_context,
no LLM, no judge) over the evaluation questions for a grid of
context-selection configurations, and reports how much of the annotators'
gold evidence ends up in the generator's context. Evidence selection is the
quantity QASPER itself scores (Evidence-F1, Dasigi et al. 2021 §4.1), and its
oracle experiments show most of the answer headroom comes from it (§5.2).

Use it to decide context_k, final_ranking and crag_mode, and to measure the
table/figure indexing, before spending GPU hours on generation + judging
(key_metrics_improvements.md, 2026-10-03, P5).

Metrics (means over questions that have the relevant evidence type):
    text_recall   — fraction of gold text paragraphs present in the context
    text_hit      — at least one gold text paragraph present
    float_hit     — at least one gold table/figure ("FLOAT SELECTED") present
    all_recall    — fraction of all gold evidence items present (text + float)
    n_ctx / ctx_tokens — context size actually passed to the generator

Usage:
    python -m src.evaluation.evaluate_retrieval                     # full grid
    python -m src.evaluation.evaluate_retrieval --split tune        # disjoint tuning questions
    python -m src.evaluation.evaluate_retrieval --configs baseline,default
    python -m src.evaluation.evaluate_retrieval --indices-dir data/indices_text_only --tag text_only

HyDE is not run (no LLM); with final_ranking="colbert" it cannot change the
result for papers within the candidate pool anyway.
"""
import argparse
import copy
import itertools
import logging
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.evaluation.generate_predictions import fetch_qasper_sample
from src.retrieval.context_selection import approx_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FLOAT_PREFIX = "FLOAT SELECTED:"

# The pre-2026-10 pipeline: ColBERT top-10 by rank, legacy CRAG filtering.
BASELINE = dict(context_k=10, final_ranking="colbert", crag_mode="legacy",
                crag_consistency_ratio=0.3, max_context_tokens=0)
# The new defaults (run_rag.py).
DEFAULT = dict(context_k=20, final_ranking="colbert", crag_mode="signal",
               crag_consistency_ratio=0.0, max_context_tokens=6000)


def grid() -> Dict[str, dict]:
    configs = {"baseline": BASELINE, "default": DEFAULT}
    for k, ranking, mode in itertools.product((10, 20, 30), ("colbert", "rrf"),
                                              ("legacy", "signal", "refine")):
        configs[f"k{k}_{ranking}_{mode}"] = dict(
            context_k=k, final_ranking=ranking, crag_mode=mode,
            crag_consistency_ratio=0.3 if mode == "legacy" else 0.0,
            max_context_tokens=0 if mode == "legacy" else 6000)
    return configs


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _body(context: str) -> str:
    """Chunk text without the chunker's 'Title: … Section: …' header line."""
    if context.startswith("Title:") and "\n" in context:
        return context.split("\n", 1)[1]
    return context


def evidence_found(evidence: str, contexts: List[str], probe: int = 120) -> bool:
    """
    Is a gold evidence item present in the contexts?

    Text paragraph: its first *probe* normalised characters occur in a context,
    or a context body starts inside the paragraph (later pieces of a paragraph
    the chunker split). Table/figure: the caption occurs in a context.
    """
    norm_contexts = [_norm(c) for c in contexts]
    if evidence.startswith(_FLOAT_PREFIX):
        caption = _norm(evidence[len(_FLOAT_PREFIX):])[:probe]
        return bool(caption) and any(caption in c for c in norm_contexts)
    ev = _norm(evidence)
    if not ev:
        return False
    head = ev[:probe]
    for context, norm_context in zip(contexts, norm_contexts):
        body = _norm(_body(context))
        if head in norm_context or (len(body) >= 60 and body[:probe] in ev):
            return True
    return False


def score_question(evidence: List[str], contexts: List[str]) -> dict:
    items = [e for e in evidence if e and e.strip()]
    text = [e for e in items if not e.startswith(_FLOAT_PREFIX)]
    floats = [e for e in items if e.startswith(_FLOAT_PREFIX)]
    text_found = [evidence_found(e, contexts) for e in text]
    float_found = [evidence_found(e, contexts) for e in floats]
    found = text_found + float_found
    return {
        "text_recall": sum(text_found) / len(text) if text else None,
        "text_hit": float(any(text_found)) if text else None,
        "float_hit": float(any(float_found)) if floats else None,
        "all_recall": sum(found) / len(found) if found else None,
        "n_ctx": len(contexts),
        "ctx_tokens": sum(approx_tokens(c) for c in contexts),
    }


def configure(pipeline, cfg: dict) -> None:
    pipeline.context_k = cfg["context_k"]
    pipeline.final_ranking = cfg["final_ranking"]
    pipeline.max_context_tokens = cfg["max_context_tokens"]
    pipeline.crag_evaluator.mode = cfg["crag_mode"]
    pipeline.crag_evaluator.consistency_ratio = cfg["crag_consistency_ratio"]


def main():
    parser = argparse.ArgumentParser(description="Evidence recall of the retrieval stages.")
    parser.add_argument("--split", choices=["eval", "tune"], default="eval",
                        help="eval: the 150 evaluation questions; tune: 150 disjoint ones.")
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--configs", default="all",
                        help="Comma-separated config names (see grid()), or 'all'.")
    parser.add_argument("--indices-dir", default=str(_PROJECT_ROOT / "data" / "indices"))
    parser.add_argument("--tag", default="", help="Label stored with the results (e.g. the index).")
    parser.add_argument("--out", default=str(_PROJECT_ROOT / "data" / "retrieval_eval.csv"),
                        help="Summary CSV; rows are appended so runs can be compared.")
    parser.add_argument("--per-question-out", default=None,
                        help="Optional CSV with one row per (config, question).")
    parser.add_argument("--crag-correct", type=float, default=14.44)
    parser.add_argument("--crag-ambiguous", type=float, default=8.0)
    args = parser.parse_args()

    configs = grid()
    if args.configs != "all":
        names = [n.strip() for n in args.configs.split(",") if n.strip()]
        unknown = [n for n in names if n not in configs]
        if unknown:
            parser.error(f"Unknown configs {unknown}. Known: {sorted(configs)}")
        configs = {n: configs[n] for n in names}

    qa = fetch_qasper_sample(args.num_samples, seed=args.seed)
    if args.split == "tune":
        exclude = {(q["paper_id"], q["question"]) for q in qa}
        qa = fetch_qasper_sample(args.num_samples, seed=args.seed + 1, exclude_questions=exclude)
    logging.info("Scoring %d %s questions over %d configs.", len(qa), args.split, len(configs))

    from src.run_rag import ScientificRAGPipeline
    indices = Path(args.indices_dir)
    pipeline = ScientificRAGPipeline(
        dense_index_path=str(indices / "dense.index"),
        dense_meta_path=str(indices / "dense.index.meta"),
        sparse_index_path=str(indices / "sparse.pkl"),
        crag_correct_threshold=args.crag_correct,
        crag_ambiguous_threshold=args.crag_ambiguous,
        load_generator=False,
    )
    # Every config sees the same candidates and ColBERT scores: cache both.
    pipeline.reranker.enable_cache()
    search = pipeline.retriever.search
    search_cache = {}

    def cached_search(query, k=10, rrf_k=60, dense_query=None, filter_paper_id=None):
        key = (query, k, filter_paper_id)
        if key not in search_cache:
            search_cache[key] = search(query, k=k, rrf_k=rrf_k, dense_query=dense_query,
                                       filter_paper_id=filter_paper_id)
        return copy.deepcopy(search_cache[key])
    pipeline.retriever.search = cached_search

    rows, per_question = [], []
    for name, cfg in configs.items():
        configure(pipeline, cfg)
        scores = []
        for i, item in enumerate(qa):
            ctx = pipeline.retrieve_context(item["question"], filter_paper_id=item["paper_id"],
                                            use_hyde=False)
            contexts = [d["text"] for d in ctx["docs"]]
            s = score_question(item.get("evidence") or [], contexts)
            s["crag_action"] = ctx["crag_action"]
            scores.append(s)
            per_question.append({"config": name, "q_idx": i, "question": item["question"],
                                 "paper_id": item["paper_id"], **s})
        df = pd.DataFrame(scores)
        row = {"tag": args.tag, "split": args.split, "config": name, **cfg,
               "n_questions": len(df)}
        for col in ("all_recall", "text_recall", "text_hit", "float_hit", "n_ctx", "ctx_tokens"):
            row[col] = round(pd.to_numeric(df[col], errors="coerce").mean(), 4)
        row["crag_triggered"] = round(df["crag_action"].isin(["Ambiguous", "Incorrect"]).mean(), 4)
        rows.append(row)
        logging.info("%-22s all_recall=%.3f text_recall=%.3f text_hit=%.3f float_hit=%.3f "
                     "n_ctx=%.1f tokens=%.0f", name, row["all_recall"], row["text_recall"],
                     row["text_hit"], row["float_hit"], row["n_ctx"], row["ctx_tokens"])

    summary = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, mode="a", header=not out.exists(), index=False)
    if args.per_question_out:
        pd.DataFrame(per_question).to_csv(args.per_question_out, index=False)

    cols = ["config", "all_recall", "text_recall", "text_hit", "float_hit", "n_ctx",
            "ctx_tokens", "crag_triggered"]
    print("\n" + summary[cols].sort_values("all_recall", ascending=False).to_string(index=False))
    print(f"\nAppended {len(summary)} rows to {out}" + (f" (tag={args.tag})" if args.tag else ""))


if __name__ == "__main__":
    main()
