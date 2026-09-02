"""
Judge validation harness — human-vs-Prometheus agreement at the atomic-unit level.

Implements key_metrics_improvements.md, Finding #4 ("The Prometheus 2 judge has
not been validated against human labels"): Kim et al. (2024, §4) report
Prometheus 2 achieving r=0.897 with GPT-4 on FeedbackBench's *original* rubrics —
that correlation has not been re-established for this project's QASPER-specific
custom rubrics (evaluate_rag.py:100-228), a materially different distribution.

Why atomic units, not aggregate scores:
    score_context_recall / score_faithfulness / check_nli_entailment each loop
    over atomic binary True/False judgments internally and collapse them into
    one float, discarding the intermediate decisions. Comparing two aggregate
    fractions (Prometheus says 0.6, human says 0.5) says almost nothing about
    *where* they diverge. This script instead calls the itemized helpers added
    to PrometheusJudge (_score_context_recall_items, _score_faithfulness_items)
    and the (bool, prompt_used) form of check_nli_entailment to re-derive the
    same (sentence, passage-set) -> True/False calls a human can independently
    label.

Pipeline:
    1. Load data/evaluation_dataset.csv (question/ground_truth/contexts/answer)
       and data/evaluation_report.csv (context_recall/faithfulness), merged
       *positionally* by row index — there is no join key (confirmed by
       reading run_evaluation's load path: both files are written from the
       same row-ordered DataFrame). The final evaluation_report.csv also drops
       the `contexts` column (see run_evaluation's final_df construction), so
       `contexts`/`ground_truth`/`answer` are always read from the dataset
       file, never the report.
    2. Stratify rows into {0, (0,0.5], (0.5,1]} buckets and draw a fixed-seed
       sample of ~n_per_bucket rows per bucket (~25-30 rows total), mirroring
       generate_predictions.py's `random.Random(seed).sample(...)` pattern.
       A row's bucket is the *worse* of its context_recall and faithfulness
       bucket (see `_combined_bucket`) so the sample covers the score range
       for both metrics without a combinatorial 3x3 stratification blowing
       past the ~25-30 row target.
    3. Re-run the itemized judge helpers over the sampled rows to expand them
       into ~100-150 atomic (unit, Prometheus-decision) pairs across three
       metrics: context_recall, faithfulness, and alce_entailment (the joint
       per-sentence NLI check ALCE citation recall is built on).
    4. Export two CSVs, kept separate to avoid anchoring bias during labeling
       (Huyen, *AI Engineering* — seeing the model's decision while labeling
       biases agreement upward):
         - judge_validation_blind.csv — unit text + context, no decision.
         - judge_validation_key.csv   — same rows + Prometheus's decision.
       Both carry a `unit_id` column; join on it after labeling.

Usage:
    python -m src.evaluation.validate_judge
    python -m src.evaluation.validate_judge --n-per-bucket 10 --seed 20260902
"""
import argparse
import ast
import logging
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

import nltk
import pandas as pd

from src.evaluation.evaluate_rag import PrometheusJudge, get_prometheus_judge

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_REPORT_COLS = ["context_recall", "faithfulness"]

# Same fixed-seed convention as generate_predictions.py::fetch_qasper_sample.
DEFAULT_SEED = 20260902
BUCKET_ORDER = ["0", "mid", "high"]  # worst -> best; index doubles as rank


def _safe_parse_contexts(x) -> List[str]:
    """Mirrors evaluate_rag.py::run_evaluation's `_safe_parse_contexts`."""
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except (ValueError, SyntaxError):
        return []


def _bucket(value: float) -> Optional[str]:
    """Map a [0,1] score to one of {0, (0,0.5], (0.5,1]}; None for NaN."""
    if pd.isna(value):
        return None
    if value <= 0:
        return "0"
    if value <= 0.5:
        return "mid"
    return "high"


def _combined_bucket(context_recall: float, faithfulness: float) -> Optional[str]:
    """
    A row's stratification bucket is the *worse* of its two metric buckets.

    Rationale: context_recall's zero-rate (68.1%, per key_metrics_improvements.md)
    dwarfs faithfulness's (17.2%), so pure context_recall stratification would
    already dominate the sample. Taking the worse-of-two still lets a row with
    good context_recall but zero faithfulness pull the sample toward that
    metric's failure mode, while keeping the single 3-bucket scheme needed to
    hit ~25-30 total rows at ~8-10/bucket (a 3x3 joint stratification would
    roughly double the target sample size for the same rows-per-cell density).
    Rows where both metrics are NaN (e.g. skipped by run_evaluation's
    empty-context/error/refusal masks) are excluded — there's nothing to
    validate on a row the judge never scored.
    """
    ranked = [b for b in (_bucket(context_recall), _bucket(faithfulness)) if b is not None]
    if not ranked:
        return None
    return min(ranked, key=BUCKET_ORDER.index)


def stratified_sample(
    report_df: pd.DataFrame, seed: int, n_per_bucket: int
) -> List[int]:
    """
    Draw a fixed-seed stratified sample of row indices from report_df.

    Returns sorted row indices (positions into both evaluation_dataset.csv and
    evaluation_report.csv, which are row-aligned).
    """
    buckets = [
        _combined_bucket(cr, fa)
        for cr, fa in zip(report_df["context_recall"], report_df["faithfulness"])
    ]
    sampled: List[int] = []
    for rank, bucket_name in enumerate(BUCKET_ORDER):
        pool = [i for i, b in enumerate(buckets) if b == bucket_name]
        k = min(n_per_bucket, len(pool))
        if k < n_per_bucket:
            logging.warning(
                "Bucket %r: only %d eligible rows (wanted %d).",
                bucket_name, len(pool), n_per_bucket,
            )
        # Distinct seed per bucket so the three draws aren't correlated by
        # sharing one Random stream, while each remains independently
        # reproducible.
        picked = random.Random(seed + rank).sample(pool, k) if pool else []
        sampled.extend(picked)
        logging.info("Bucket %r: sampled %d/%d rows.", bucket_name, k, len(pool))

    sampled = sorted(set(sampled))
    logging.info("Total sampled rows: %d", len(sampled))
    return sampled


def _iter_alce_citation_sentences(
    answer: str, contexts: List[str]
) -> List[Tuple[str, str]]:
    """
    Yield (sentence, joint_cited_passage) for each answer sentence carrying at
    least one valid [Doc N] citation — the exact unit ALCE citation recall's
    numerator is built on (see ALCEEvaluator.calculate_metrics's first
    self._entails(sentence, joint, memo) call, evaluate_rag.py).

    Only the joint-citation check is reproduced here (not the per-citation
    precision drill-down) — that joint check is the canonical "does this
    claim's cited evidence support it" unit Finding #4 asks to validate under
    the label "ALCE-entailment".
    """
    try:
        sentences = nltk.sent_tokenize(answer)
    except Exception as exc:
        logging.warning(
            "[ALCE] sent_tokenize failed (%s) — skipping alce_entailment units "
            "for this row. Run nltk.download('punkt'/'punkt_tab') first.", exc
        )
        return []

    items = []
    for sentence in sentences:
        seen, cited_idx = set(), []
        for doc_id_str in re.findall(r"Doc (\d+)", sentence):
            idx = int(doc_id_str) - 1
            if 0 <= idx < len(contexts) and idx not in seen:
                seen.add(idx)
                cited_idx.append(idx)
        if not cited_idx:
            continue
        joint = " ".join(contexts[i] for i in cited_idx)
        items.append((sentence, joint))
    return items


def expand_to_atomic_units(
    judge: PrometheusJudge,
    dataset_df: pd.DataFrame,
    report_df: pd.DataFrame,
    row_indices: List[int],
) -> List[dict]:
    """
    Re-run the itemized judge helpers over the sampled rows and flatten the
    result into one dict per atomic unit, ready to write to CSV.
    """
    records = []
    total = len(row_indices)
    for pos, row_idx in enumerate(row_indices, start=1):
        question = str(dataset_df.at[row_idx, "question"])
        ground_truth = str(dataset_df.at[row_idx, "ground_truth"])
        answer = str(dataset_df.at[row_idx, "answer"])
        contexts = _safe_parse_contexts(dataset_df.at[row_idx, "contexts"])
        report_cr = report_df.at[row_idx, "context_recall"]
        report_faith = report_df.at[row_idx, "faithfulness"]

        logging.info(
            "[%d/%d] row %d: expanding atomic units (cr=%.2f, faith=%.2f)...",
            pos, total, row_idx,
            0.0 if pd.isna(report_cr) else report_cr,
            0.0 if pd.isna(report_faith) else report_faith,
        )

        # ── context_recall ──────────────────────────────────────────────────
        cr_items = judge._score_context_recall_items(question, contexts, ground_truth)
        ctx_block = "\n---\n".join(
            f"[Passage {i + 1}] {c}" for i, c in enumerate(contexts[:10])
        )
        for k, (gt_sentence, decision) in enumerate(cr_items):
            records.append({
                "unit_id": f"{row_idx}-context_recall-{k}",
                "row_id": row_idx,
                "metric": "context_recall",
                "question": question,
                "unit_text": gt_sentence,
                "context_shown": ctx_block,
                "prometheus_decision": decision,
                "prometheus_prompt": "",
                "report_context_recall": report_cr,
                "report_faithfulness": report_faith,
            })

        # ── faithfulness ─────────────────────────────────────────────────────
        faith_items = judge._score_faithfulness_items(question, answer, contexts)
        faith_ctx_block = "\n---\n".join(
            f"[Passage {i + 1}] {c}" for i, c in enumerate(contexts[:5])
        )
        for k, (claim_sentence, decision) in enumerate(faith_items):
            records.append({
                "unit_id": f"{row_idx}-faithfulness-{k}",
                "row_id": row_idx,
                "metric": "faithfulness",
                "question": question,
                "unit_text": claim_sentence,
                "context_shown": faith_ctx_block,
                "prometheus_decision": decision,
                "prometheus_prompt": "",
                "report_context_recall": report_cr,
                "report_faithfulness": report_faith,
            })

        # ── alce_entailment ──────────────────────────────────────────────────
        for k, (sentence, joint_passage) in enumerate(
            _iter_alce_citation_sentences(answer, contexts)
        ):
            decision, prompt_used = judge.check_nli_entailment(sentence, joint_passage)
            records.append({
                "unit_id": f"{row_idx}-alce_entailment-{k}",
                "row_id": row_idx,
                "metric": "alce_entailment",
                "question": question,
                "unit_text": sentence,
                "context_shown": joint_passage,
                "prometheus_decision": decision,
                "prometheus_prompt": prompt_used or "",
                "report_context_recall": report_cr,
                "report_faithfulness": report_faith,
            })

    return records


def export_validation_csvs(
    records: List[dict], out_dir: Path
) -> Tuple[Path, Path]:
    df = pd.DataFrame.from_records(records)

    key_path = out_dir / "judge_validation_key.csv"
    df.to_csv(key_path, index=False)

    blind_cols = ["unit_id", "row_id", "metric", "question", "unit_text", "context_shown"]
    blind_df = df[blind_cols].copy()
    blind_df["human_label"] = ""  # to be filled in: True / False
    blind_path = out_dir / "judge_validation_blind.csv"
    blind_df.to_csv(blind_path, index=False)

    return blind_path, key_path


def run_validation(
    dataset_csv: Optional[str] = None,
    report_csv: Optional[str] = None,
    out_dir: Optional[str] = None,
    seed: int = DEFAULT_SEED,
    n_per_bucket: int = 9,
) -> None:
    dataset_csv = dataset_csv or str(_PROJECT_ROOT / "data" / "evaluation_dataset.csv")
    report_csv = report_csv or str(_PROJECT_ROOT / "data" / "evaluation_report.csv")
    out_dir_path = Path(out_dir) if out_dir else (_PROJECT_ROOT / "data")

    logging.info("Loading dataset from %s ...", dataset_csv)
    dataset_df = pd.read_csv(dataset_csv)
    logging.info("Loading report from %s ...", report_csv)
    report_df = pd.read_csv(report_csv)

    missing = [c for c in _REQUIRED_REPORT_COLS if c not in report_df.columns]
    if missing:
        raise RuntimeError(
            f"{report_csv} is missing column(s) {missing}. This usually means "
            "it's the ALCE-only intermediate save (evaluate_rag.py writes this "
            "right after the ALCE pass, before Prometheus metrics run) rather "
            "than a completed report. Run `bash run_evaluation.sh` (or "
            "`python -m src.evaluation.evaluate_rag`) to completion first, "
            "then re-run this script."
        )

    if len(dataset_df) != len(report_df):
        raise RuntimeError(
            f"Row-count mismatch: {dataset_csv} has {len(dataset_df)} rows, "
            f"{report_csv} has {len(report_df)}. These files are merged "
            "positionally (no join key) — they must come from the same run."
        )

    # Idempotent — matches ALCEEvaluator.__init__'s download pattern. `_score_
    # context_recall_items`/`_score_faithfulness_items` (evaluate_rag.py) and
    # `_iter_alce_citation_sentences` (below) all call nltk.sent_tokenize,
    # which needs both 'punkt' and 'punkt_tab' present or it raises.
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception as exc:
        logging.warning("NLTK download failed: %s", exc)

    row_indices = stratified_sample(report_df, seed=seed, n_per_bucket=n_per_bucket)
    if not row_indices:
        raise RuntimeError(
            "No eligible rows found (all rows have NaN context_recall and "
            "faithfulness). Nothing to sample."
        )

    logging.info("Initializing Prometheus 2 judge ...")
    judge, _is_gpu = get_prometheus_judge()

    records = expand_to_atomic_units(judge, dataset_df, report_df, row_indices)
    logging.info(
        "Expanded %d sampled rows into %d atomic units.",
        len(row_indices), len(records),
    )

    out_dir_path.mkdir(parents=True, exist_ok=True)
    blind_path, key_path = export_validation_csvs(records, out_dir_path)
    logging.info("Wrote blind labeling file: %s", blind_path)
    logging.info("Wrote key file (Prometheus decisions): %s", key_path)
    logging.info(
        "Next: fill `human_label` (True/False) in %s in a spreadsheet, then "
        "join back to %s on `unit_id` to compute agreement.",
        blind_path.name, key_path.name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample rows and export blind/key CSVs for human "
                     "validation of Prometheus 2's atomic judgments."
    )
    parser.add_argument("--dataset-csv", type=str, default=None)
    parser.add_argument("--report-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-per-bucket", type=int, default=9)
    args = parser.parse_args()
    run_validation(
        dataset_csv=args.dataset_csv,
        report_csv=args.report_csv,
        out_dir=args.out_dir,
        seed=args.seed,
        n_per_bucket=args.n_per_bucket,
    )
