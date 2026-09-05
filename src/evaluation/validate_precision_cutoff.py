"""
Context-precision cutoff validation (key_metrics_improvements.md, Finding #6).

`run_prometheus_metrics` (evaluate_rag.py) scores context_precision on only
the top `max_precision_contexts=3` reranked chunks (of the 10 that reach
generation), to bound Prometheus API calls — justified by "Lost in the
Middle" (Liu et al. 2023). Finding #6 asks whether that cutoff still holds
after the Finding #2 paper-scoping fix changed what the top-10 actually look
like: "No change needed unless the post-fix numbers show top-3 precision
diverging materially from a full top-10 spot check."

This script gathers that evidence directly: for a fixed-seed sample of rows,
it scores context_precision on ALL 10 retrieved contexts (not just the top
3) via the same PrometheusJudge used in evaluate_rag.py, and reports
rank 1-3 vs rank 4-10 vs full top-10 means. It does NOT change
max_precision_contexts itself — that's a judgment call on the evidence,
made explicitly, not automated here.

Usage:
    python -m src.evaluation.validate_precision_cutoff
    python -m src.evaluation.validate_precision_cutoff --n-rows 20 --seed 20260902
"""
import argparse
import ast
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.evaluation.evaluate_rag import get_prometheus_judge

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 20260902


def _safe_parse_contexts(x) -> List[str]:
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except (ValueError, SyntaxError):
        return []


def run_validation(
    dataset_csv: Optional[str] = None,
    n_rows: int = 18,
    seed: int = DEFAULT_SEED,
    current_cutoff: int = 3,
) -> pd.DataFrame:
    dataset_csv = dataset_csv or str(_PROJECT_ROOT / "data" / "evaluation_dataset.csv")

    logging.info("Loading dataset from %s ...", dataset_csv)
    df = pd.read_csv(dataset_csv)
    df["contexts"] = df["contexts"].apply(_safe_parse_contexts)

    eligible = df[df["contexts"].apply(len) >= 10].index.tolist()
    if len(eligible) < n_rows:
        logging.warning(
            "Only %d/%d rows have >=10 contexts; sampling all of them.",
            len(eligible), n_rows,
        )
        n_rows = len(eligible)

    import random
    sampled = sorted(random.Random(seed).sample(eligible, n_rows))
    logging.info("Sampled %d rows (seed=%d): %s", len(sampled), seed, sampled)

    logging.info("Initializing Prometheus 2 judge ...")
    judge, _is_gpu = get_prometheus_judge()

    records = []
    for pos, row_idx in enumerate(sampled, start=1):
        question = str(df.at[row_idx, "question"])
        contexts = df.at[row_idx, "contexts"][:10]
        logging.info("[%d/%d] row %d: scoring 10 contexts individually...", pos, len(sampled), row_idx)
        for rank, ctx in enumerate(contexts, start=1):
            score = judge.score_context_precision(question, ctx)
            records.append({"row_id": row_idx, "rank": rank, "context_precision": score})

    result_df = pd.DataFrame.from_records(records)

    top_k = result_df[result_df["rank"] <= current_cutoff]
    rest = result_df[result_df["rank"] > current_cutoff]

    top_mean = top_k["context_precision"].mean()
    rest_mean = rest["context_precision"].mean() if not rest.empty else float("nan")
    full_mean = result_df["context_precision"].mean()
    gap = top_mean - rest_mean

    logging.info("========== CONTEXT-PRECISION CUTOFF VALIDATION ==========")
    logging.info(
        "rank 1-%d mean   : %.4f  (n=%d)", current_cutoff, top_mean, len(top_k)
    )
    logging.info(
        "rank %d-10 mean  : %.4f  (n=%d)", current_cutoff + 1, rest_mean, len(rest)
    )
    logging.info("full top-10 mean : %.4f  (n=%d)", full_mean, len(result_df))
    logging.info("gap (top - rest) : %+.4f", gap)

    if pd.isna(gap):
        recommendation = "Not enough data to compare (rest bucket empty)."
    elif gap < 0.10:
        recommendation = (
            f"Gap is small (<0.10) — rank {current_cutoff + 1}-10 chunks score "
            f"nearly as well as the top-{current_cutoff}. max_precision_contexts="
            f"{current_cutoff} may be leaving usable precision signal out; "
            "consider raising it."
        )
    else:
        recommendation = (
            f"Gap is substantial (>=0.10) — rank {current_cutoff + 1}-10 chunks "
            f"score meaningfully lower than the top-{current_cutoff}. Current "
            f"max_precision_contexts={current_cutoff} cutoff looks justified; "
            "no change recommended."
        )
    logging.info("Recommendation: %s", recommendation)
    logging.info("===========================================================")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spot-check whether max_precision_contexts=3 (evaluate_rag.py) "
                     "is leaving relevant context-precision signal out of scope."
    )
    parser.add_argument("--dataset-csv", type=str, default=None)
    parser.add_argument("--n-rows", type=int, default=18)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--current-cutoff", type=int, default=3)
    args = parser.parse_args()
    run_validation(
        dataset_csv=args.dataset_csv,
        n_rows=args.n_rows,
        seed=args.seed,
        current_cutoff=args.current_cutoff,
    )
