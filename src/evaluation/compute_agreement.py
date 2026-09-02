"""
Judge validation — agreement between human labels and Prometheus 2's atomic
decisions (key_metrics_improvements.md, Finding #4).

Run after labeling: fill `human_label` (True/False) in
data/judge_validation_blind.csv, then run this script. It joins the labeled
blind file back to data/judge_validation_key.csv on `unit_id`, and reports,
per metric (context_recall / faithfulness / alce_entailment) and overall:
    - n labeled
    - raw agreement rate
    - Cohen's kappa (chance-corrected agreement; the metric Finding #4 calls
      for — atomic labels here are binary True/False, not the old aggregate
      [0,1] float scores, so kappa is the right statistic, not Pearson r)
    - a 2x2 confusion breakdown (human label x Prometheus decision), which is
      what actually shows *which direction* the judge is wrong in — e.g. the
      false-negative pattern already observed live during sampling (units
      where Prometheus's own generated text said "does support the claim"
      but the True/False parser found no literal True/False/yes/no token and
      silently defaulted to False; see check_nli_entailment /
      _score_context_recall_items / _score_faithfulness_items,
      evaluate_rag.py).

Usage:
    python -m src.evaluation.compute_agreement
    python -m src.evaluation.compute_agreement --blind-csv data/judge_validation_blind.csv
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import cohen_kappa_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_TRUE_STRINGS = {"true", "1", "yes", "y", "t"}
_FALSE_STRINGS = {"false", "0", "no", "n", "f"}


def _parse_human_label(value) -> Optional[bool]:
    """Blank/NaN -> None (not yet labeled). Accepts True/False/yes/no/1/0."""
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s == "":
        return None
    if s in _TRUE_STRINGS:
        return True
    if s in _FALSE_STRINGS:
        return False
    logging.warning("Unparseable human_label value %r — treating as unlabeled.", value)
    return None


def _report_bucket(name: str, human: pd.Series, prom: pd.Series) -> dict:
    n = len(human)
    if n == 0:
        logging.info("%-16s n=0 (nothing labeled yet)", name)
        return {"metric": name, "n": 0}

    agreement = (human == prom).mean()
    if human.nunique() < 2 or prom.nunique() < 2:
        kappa = float("nan")
        logging.info(
            "%-16s n=%-3d agreement=%.3f  kappa=N/A (one side is constant — "
            "not enough label variety yet)",
            name, n, agreement,
        )
    else:
        kappa = cohen_kappa_score(human, prom)
        logging.info(
            "%-16s n=%-3d agreement=%.3f  kappa=%.3f",
            name, n, agreement, kappa,
        )

    tp = int(((human == True) & (prom == True)).sum())
    tn = int(((human == False) & (prom == False)).sum())
    # Prometheus True, human False: judge over-credits (false positive).
    fp = int(((human == False) & (prom == True)).sum())
    # Prometheus False, human True: judge under-credits (false negative) —
    # this is the failure mode Finding #4's row-0 example and the parse-
    # fallback warnings observed during sampling both point at.
    fn = int(((human == True) & (prom == False)).sum())
    logging.info(
        "%-16s   confusion (human x prometheus): TP=%d TN=%d  "
        "FP(judge says True, human False)=%d  FN(judge says False, human True)=%d",
        "", tp, tn, fp, fn,
    )

    return {
        "metric": name, "n": n, "agreement": agreement, "cohen_kappa": kappa,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def compute_agreement(
    blind_csv: Optional[str] = None,
    key_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    blind_csv = blind_csv or str(_PROJECT_ROOT / "data" / "judge_validation_blind.csv")
    key_csv = key_csv or str(_PROJECT_ROOT / "data" / "judge_validation_key.csv")

    blind_df = pd.read_csv(blind_csv)
    key_df = pd.read_csv(key_csv)

    if "human_label" not in blind_df.columns:
        raise RuntimeError(f"{blind_csv} has no `human_label` column.")

    merged = key_df.merge(
        blind_df[["unit_id", "human_label"]], on="unit_id", how="inner", validate="one_to_one"
    )
    if len(merged) != len(key_df):
        logging.warning(
            "%d/%d key rows had no matching unit_id in the blind file.",
            len(key_df) - len(merged), len(key_df),
        )

    merged["human_label_parsed"] = merged["human_label"].apply(_parse_human_label)
    merged["prometheus_decision"] = merged["prometheus_decision"].astype(bool)

    labeled = merged[merged["human_label_parsed"].notna()].copy()
    n_unlabeled = len(merged) - len(labeled)
    if n_unlabeled:
        logging.info(
            "%d/%d units are not yet labeled (blank human_label) — excluded "
            "from agreement stats.",
            n_unlabeled, len(merged),
        )
    if labeled.empty:
        raise RuntimeError(
            "No labeled units found. Fill `human_label` (True/False) in "
            f"{blind_csv} first."
        )

    logging.info("========== JUDGE VALIDATION — HUMAN vs PROMETHEUS 2 ==========")
    rows = []
    for metric_name, group in labeled.groupby("metric"):
        rows.append(_report_bucket(
            metric_name, group["human_label_parsed"], group["prometheus_decision"]
        ))
    rows.append(_report_bucket(
        "OVERALL", labeled["human_label_parsed"], labeled["prometheus_decision"]
    ))
    logging.info("================================================================")

    summary_df = pd.DataFrame(rows)

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        labeled.drop(columns=["human_label"]).rename(
            columns={"human_label_parsed": "human_label"}
        ).to_csv(out_path, index=False)
        logging.info("Wrote merged labeled units to %s", out_path)

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute human-vs-Prometheus-2 agreement on labeled "
                     "atomic judge validation units."
    )
    parser.add_argument("--blind-csv", type=str, default=None)
    parser.add_argument("--key-csv", type=str, default=None)
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Optional path to write the merged (human_label + "
             "prometheus_decision) rows, e.g. data/judge_validation_merged.csv",
    )
    args = parser.parse_args()
    compute_agreement(
        blind_csv=args.blind_csv,
        key_csv=args.key_csv,
        output_csv=args.output_csv,
    )
