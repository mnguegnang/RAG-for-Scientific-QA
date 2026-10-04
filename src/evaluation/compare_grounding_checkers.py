"""
Grounding-checker validation: MiniCheck vs Prometheus 2 against reference labels.

Re-decides every labelled atomic unit from the judge-validation set
(data/judge_validation_key.csv, built by validate_judge.py) with MiniCheck,
and reports Cohen's kappa of each checker against the reference labels
(human, or the independent second judge from second_judge_label.py), per
metric. Both checkers see exactly the same claim and context the labeller saw
(`unit_text`, `context_shown`), so the comparison is like-for-like.

This is the adoption gate for key_metrics_improvements.md (2026-10-03) P1:
use MiniCheck for context recall / faithfulness / ALCE only if its kappa is
higher than Prometheus'.

Usage:
    python -m src.evaluation.compare_grounding_checkers
    python -m src.evaluation.compare_grounding_checkers \\
        --labels data/judge_validation_blind.csv      # human labels instead
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.evaluation.compute_agreement import _parse_human_label
from src.evaluation.grounding import MiniCheckJudge, _CITATION_TAG_RE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _agreement(name: str, labels: pd.Series, decisions: pd.Series) -> dict:
    labels, decisions = labels.astype(bool), decisions.astype(bool)
    kappa = (cohen_kappa_score(labels, decisions)
             if labels.nunique() > 1 or decisions.nunique() > 1 else float("nan"))
    return {
        "checker": name,
        "agreement": round(float((labels == decisions).mean()), 3),
        "kappa": round(float(kappa), 3),
        "false_neg": int((labels & ~decisions).sum()),   # labeller True, checker False
        "false_pos": int((~labels & decisions).sum()),
        "checker_true_rate": round(float(decisions.mean()), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare grounding checkers against labels.")
    parser.add_argument("--key", default=str(_PROJECT_ROOT / "data" / "judge_validation_key.csv"))
    parser.add_argument("--labels",
                        default=str(_PROJECT_ROOT / "data" / "judge_validation_blind_sonnet.csv"),
                        help="Blind CSV with a filled `human_label` column.")
    parser.add_argument("--out", default=str(_PROJECT_ROOT / "data" / "grounding_checker_comparison.csv"))
    args = parser.parse_args()

    key = pd.read_csv(args.key)
    labels = pd.read_csv(args.labels)[["unit_id", "human_label"]]
    df = key.merge(labels, on="unit_id")
    df["label"] = df["human_label"].apply(_parse_human_label)
    df = df[df["label"].notna()].copy()
    df["prometheus"] = df["prometheus_decision"].apply(_parse_human_label).astype(bool)
    logging.info("%d labelled units (%s).", len(df), dict(df["metric"].value_counts()))

    checker = MiniCheckJudge()
    decisions, probs = [], []
    for _, unit in df.iterrows():
        claim = _CITATION_TAG_RE.sub("", str(unit["unit_text"])).strip()
        context = str(unit["context_shown"])
        if not claim or not context.strip():
            decisions.append(False)
            probs.append(float("nan"))
            continue
        prob = checker.support_probability(context, claim)
        decisions.append(prob > checker.THRESHOLD)
        probs.append(prob)
    df["minicheck"] = decisions
    df["minicheck_p_support"] = probs

    rows = []
    for metric, group in list(df.groupby("metric")) + [("ALL", df)]:
        for name in ("prometheus", "minicheck"):
            rows.append({"metric": metric, "n": len(group),
                         "label_true_rate": round(float(group["label"].astype(bool).mean()), 3),
                         **_agreement(name, group["label"], group[name])})
    report = pd.DataFrame(rows)
    report.to_csv(args.out, index=False)
    df.drop(columns=["prometheus_prompt"], errors="ignore").to_csv(
        Path(args.out).with_name("grounding_checker_units.csv"), index=False)
    print("\n" + report.to_string(index=False))
    print(f"\nSaved {args.out} (and per-unit decisions next to it).")


if __name__ == "__main__":
    main()
