"""
Retrieval paper-scoping validator (key_metrics_improvements.md, Finding #2).

The dominant pre-fix failure mode was cross-paper contamination: a query's
top-10 retrieved chunks coming from a different paper than its QASPER ground
truth, even at good rerank scores (Finding #2's evidence: dataset rows 3, 11,
30, 36, each 10/10 chunks from the wrong paper). The working-tree fix
restricts the FAISS/BM25 candidate pool to `paper_id` *before* ranking
(`HybridRetriever.search(..., filter_paper_id=...)`), rather than filtering
after a global top-k.

This script is a reusable regression check for that fix, not a one-off:
    1. Cross-paper check — every row in evaluation_dataset.csv should have
       all of its retrieved contexts drawn from a single paper title. A row
       whose contexts span >1 distinct title is contamination.
    2. ID-namespace check — Finding #2's own fallback recommendation: verify
       the `paper_id` values stored in the dense index metadata are actually
       QASPER train-split ids (same namespace `generate_predictions.py`
       filters on), not silently mismatched/reformatted.

Exit code is 0 (pass) / 1 (fail) so this can be re-run as a CI-style guard
after any future retrieval change.

Usage:
    python -m src.evaluation.validate_retrieval_scoping
"""
import argparse
import ast
import logging
import re
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd

from src.retrieval.hybrid_retriever import _load_pickle_verified

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TITLE_RE = re.compile(r"^Title:\s*(.*?)\.\s*Section:")


def _safe_parse_contexts(x) -> List[str]:
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except (ValueError, SyntaxError):
        return []


def _titles_in_row(contexts: List[str]) -> Set[str]:
    titles = set()
    for c in contexts:
        m = _TITLE_RE.match(c)
        if m:
            titles.add(m.group(1))
    return titles


def check_cross_paper_contamination(dataset_csv: str) -> bool:
    """Returns True (pass) iff no row's contexts span >1 distinct paper title."""
    df = pd.read_csv(dataset_csv)
    df["contexts"] = df["contexts"].apply(_safe_parse_contexts)
    df["n_titles"] = df["contexts"].apply(lambda cs: len(_titles_in_row(cs)))

    contaminated = df[df["n_titles"] > 1]
    n_empty = int((df["contexts"].apply(len) == 0).sum())

    logging.info(
        "Cross-paper check: %d/%d rows span >1 distinct paper title "
        "(%d rows had empty contexts, excluded from this check).",
        len(contaminated), len(df), n_empty,
    )
    if not contaminated.empty:
        for idx, row in contaminated.head(10).iterrows():
            logging.warning(
                "  Row %d: %d distinct titles — %.80s",
                idx, row["n_titles"], row["question"],
            )
    return contaminated.empty


def check_paper_id_namespace(dense_meta_path: str, sample_size: int = 888) -> bool:
    """
    Returns True (pass) iff the paper_id values stored in the dense index
    metadata are (mostly) valid QASPER train-split ids — i.e. the same
    namespace generate_predictions.py's fetch_qasper_sample filters on.
    """
    from datasets import load_dataset

    dense_meta = _load_pickle_verified(dense_meta_path)
    index_paper_ids = {chunk["paper_id"] for chunk in dense_meta}

    logging.info("Loading QASPER train split ids for namespace comparison...")
    qasper_ids = set(
        load_dataset("allenai/qasper", split="train").select_columns(["id"])["id"]
    )

    overlap = index_paper_ids & qasper_ids
    missing = index_paper_ids - qasper_ids
    coverage = len(overlap) / len(index_paper_ids) if index_paper_ids else 0.0

    logging.info(
        "Paper-ID namespace check: %d/%d index paper_ids found in the QASPER "
        "train id set (%.1f%% coverage). %d indexed papers not in QASPER train.",
        len(overlap), len(index_paper_ids), coverage * 100, len(missing),
    )
    if missing:
        logging.warning("  Example unmatched paper_id(s): %s", sorted(missing)[:5])

    # Full coverage is the pass criterion — any mismatch means filter_paper_id
    # (built from QASPER ids) can silently miss an indexed paper.
    return coverage == 1.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the paper-scoped retrieval fix (Finding #2)."
    )
    parser.add_argument(
        "--dataset-csv", type=str,
        default=str(_PROJECT_ROOT / "data" / "evaluation_dataset.csv"),
    )
    parser.add_argument(
        "--dense-meta", type=str,
        default=str(_PROJECT_ROOT / "data" / "indices" / "dense.index.meta"),
    )
    parser.add_argument(
        "--skip-namespace-check", action="store_true",
        help="Skip the QASPER-id namespace check (downloads the QASPER train "
             "split metadata; the cross-paper check alone needs no network).",
    )
    args = parser.parse_args()

    logging.info("========== RETRIEVAL PAPER-SCOPING VALIDATION ==========")
    ok_contamination = check_cross_paper_contamination(args.dataset_csv)

    ok_namespace = True
    if not args.skip_namespace_check:
        ok_namespace = check_paper_id_namespace(args.dense_meta)
    else:
        logging.info("Namespace check skipped (--skip-namespace-check).")

    passed = ok_contamination and ok_namespace
    logging.info("==========================================================")
    logging.info("RESULT: %s", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
