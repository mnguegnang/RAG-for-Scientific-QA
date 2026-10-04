"""
Compare two evaluation reports (evaluate_rag.py output) metric by metric.

Means over "all scored rows" are misleading when the two runs refuse
different questions (refusals are excluded from every metric), so this
reports, per metric:
  - each run's mean over its own scored rows, and how many rows that is;
  - the paired difference (B - A) on the questions scored in BOTH runs,
    with a 95% bootstrap confidence interval. An interval that excludes 0 is
    a difference the 150-question sample can actually resolve.

Rows are matched on the question text, so both reports must come from the
same question sample (same --num-samples / seed).

Usage:
    python -m src.evaluation.compare_reports data/runs/baseline_report.csv data/evaluation_report.csv
    python -m src.evaluation.compare_reports A.csv B.csv --labels old new --out data/runs/compare.csv
"""
import argparse

import numpy as np
import pandas as pd

METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy",
           "answer_correctness", "alce_citation_precision", "alce_citation_recall",
           "alce_citation_f1"]


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(a: pd.DataFrame, b: pd.DataFrame, labels=("A", "B")) -> pd.DataFrame:
    a = a.drop_duplicates("question").set_index("question")
    b = b.drop_duplicates("question").set_index("question")
    rows = []
    for metric in METRICS:
        if metric not in a.columns or metric not in b.columns:
            continue
        sa, sb = a[metric].dropna(), b[metric].dropna()
        common = sa.index.intersection(sb.index)
        diffs = (sb[common] - sa[common]).to_numpy(dtype=float)
        lo, hi = bootstrap_ci(diffs) if len(diffs) > 1 else (float("nan"), float("nan"))
        rows.append({
            "metric": metric,
            f"{labels[0]}_mean": round(sa.mean(), 4), f"{labels[0]}_n": len(sa),
            f"{labels[1]}_mean": round(sb.mean(), 4), f"{labels[1]}_n": len(sb),
            "paired_n": len(common),
            "paired_delta": round(float(diffs.mean()), 4) if len(diffs) else float("nan"),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "resolved": bool(len(diffs) > 1 and (lo > 0 or hi < 0)),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Paired comparison of two evaluation reports.")
    parser.add_argument("report_a")
    parser.add_argument("report_b")
    parser.add_argument("--labels", nargs=2, default=["A", "B"])
    parser.add_argument("--out", default=None, help="Optional CSV for the comparison table.")
    args = parser.parse_args()

    a, b = pd.read_csv(args.report_a), pd.read_csv(args.report_b)
    table = compare(a, b, tuple(args.labels))
    print(f"A = {args.report_a}\nB = {args.report_b}\n")
    print(table.to_string(index=False))
    for label, df in zip(args.labels, (a, b)):
        scored = df["answer_correctness"].notna().sum() if "answer_correctness" in df else len(df)
        print(f"{label}: {scored}/{len(df)} questions scored (the rest are refusals, errors "
              f"or empty contexts)")
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
