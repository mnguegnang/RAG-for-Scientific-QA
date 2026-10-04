"""
Re-score the sentence-level support metrics of an existing evaluation report.

Recomputes faithfulness, ALCE citation precision/recall/F1 and (optionally)
context recall for every scorable row of a dataset/report pair, with a chosen
grounding checker, and writes a new report. Prometheus's rubric metrics
(context precision, answer relevancy, answer correctness) are copied
unchanged, so switching checkers does not require re-running generation or the
rubric judge.

Use it to put several runs on the same evaluator after the checker changes:
    python -m src.evaluation.rescore_grounding \
        --dataset data/runs/new_k20_dataset.csv --report data/runs/new_k20_report.csv \
        --out data/runs/new_k20_report_rescored.csv --checker prometheus

Short ground truths: GroundingJudge drops ground-truth sentences under 10
characters, so answers such as "97.32%" or "SVM" got no context recall at all.
With --short-gt-as-claim (default on) a ground truth with no sentence of 10+
characters is checked as one claim. Rows are skipped exactly as
evaluate_rag.py skips them (errors, refusals, empty contexts).
"""
import argparse
import ast
import logging

import nltk
import pandas as pd

from src.evaluation.evaluate_rag import (
    ALCEEvaluator, GROUNDING_CHECKERS, build_grounding_judges, get_prometheus_judge,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_REFUSAL_RE = (r"(do not contain enough|cannot be determined|not enough information|"
               r"no specific document|does not provide)")


def recall_items(judge, contexts, ground_truth, short_gt_as_claim=True):
    items = judge._score_context_recall_items("", contexts, ground_truth)
    if items or not short_gt_as_claim or not contexts:
        return items
    claim = str(ground_truth).strip()
    if not claim or claim.lower() in ("nan", "none"):
        return []
    return [(claim, judge._supported(claim, contexts, "recall")[0])]


def main():
    parser = argparse.ArgumentParser(description="Re-score support metrics of a report.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checker", choices=GROUNDING_CHECKERS, default="hybrid")
    parser.add_argument("--metrics", default="faithfulness,alce,context_recall",
                        help="Comma-separated subset of faithfulness, alce, context_recall.")
    parser.add_argument("--no-short-gt-as-claim", dest="short_gt", action="store_false")
    args = parser.parse_args()
    metrics = set(m.strip() for m in args.metrics.split(","))

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    data = pd.read_csv(args.dataset)
    report = pd.read_csv(args.report)
    assert len(data) == len(report) and (data["question"].values == report["question"].values).all(), \
        "dataset and report rows must align"
    data["contexts"] = data["contexts"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    prometheus, _ = get_prometheus_judge()
    judges = build_grounding_judges(args.checker, prometheus)
    alce = ALCEEvaluator(judge=judges["alce"])

    skip = (data["answer"].str.contains(r"^(System Error:|Error:)", na=False, regex=True)
            | data["answer"].str.contains(_REFUSAL_RE, case=False, na=False, regex=True))
    out = report.copy()
    for i, row in data.iterrows():
        contexts, answer = row["contexts"], str(row["answer"])
        if skip.at[i] or not contexts:
            continue
        logging.info("Row %d/%d", i + 1, len(data))
        if "faithfulness" in metrics:
            out.at[i, "faithfulness"] = judges["faithfulness"].score_faithfulness(
                row["question"], answer, contexts)
        if "alce" in metrics:
            p, r = alce.calculate_metrics(answer, contexts)
            out.at[i, "alce_citation_precision"], out.at[i, "alce_citation_recall"] = p, r
            out.at[i, "alce_citation_f1"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if "context_recall" in metrics:
            items = recall_items(judges["recall"], contexts, str(row["ground_truth"]), args.short_gt)
            out.at[i, "context_recall"] = sum(v for _, v in items) / len(items) if items else float("nan")
    out["grounding_checker"] = ",".join(f"{m}={j.name}" for m, j in judges.items())
    out.to_csv(args.out, index=False)
    cols = ["context_recall", "faithfulness", "alce_citation_precision",
            "alce_citation_recall", "alce_citation_f1"]
    logging.info("Means after re-scoring (%s): %s", args.checker,
                 out[cols].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
