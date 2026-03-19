"""
CRAG Threshold Calibration
--------------------------
Two calibration modes for the bge-reranker-v2-m3 CRAG gate in run_rag.py.

Mode 1 — "f1"  (default, when context_recall is available):
    Runs retrieval + reranking on the evaluation set, records each query’s
    maximum reranker logit, plots distributions for relevant vs. irrelevant
    retrievals, and finds the F1-maximising threshold.
    Requires evaluation_report.csv with a non-NaN context_recall column.

    Reference:
        Yan et al. (2024). CRAG: Corrective Retrieval Augmented Generation.
        arXiv:2401.15884. §3.3: "We calibrate the retrieval evaluator threshold
        empirically on the score distribution of training samples to maximise
        retrieval F1."

Mode 2 — "percentile" (fallback when context_recall is all-NaN):
    Reads best_rerank_score directly from evaluation_dataset.csv (written by
    generate_predictions.py) and sets the threshold at the P-th percentile
    of the observed distribution.  No RAGAS metrics required.

    Rationale (Karpukhin et al. 2020, DPR §5.4; Asai et al. 2023, Self-RAG §3.2):
        A score at the 10th percentile separates the bottom 10% of retrievals
        (where even the scoped paper has no relevant passage) from the majority,
        without discarding valid queries.  With paper-ID filtering already in
        place the distribution is expected to be right-skewed, so P=10 is apt.

Usage (from project root):
    # Mode 1 — F1-max calibration (needs context_recall):
    python -m src.evaluation.calibrate_crag

    # Mode 2 — percentile calibration (needs best_rerank_score):
    python -m src.evaluation.calibrate_crag --mode percentile
    python -m src.evaluation.calibrate_crag --mode percentile --percentile 5

Output:
    • Console: per-query table + recommended CRAG_THRESHOLD value
    • Plot saved to reports/figures/crag_calibration.png
"""

import argparse
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_eval_data(eval_csv: str) -> pd.DataFrame:
    """
    Load the evaluation CSV and derive binary relevance labels.

    Accepts evaluation_report.csv (written by evaluate_rag.py), which contains
    the RAGAS context_recall column needed to derive relevance labels.
    Required columns: one of {'question', 'user_input'} and 'context_recall'.
    """
    df = pd.read_csv(eval_csv)

    # Normalise the question column name
    if 'question' not in df.columns and 'user_input' in df.columns:
        df = df.rename(columns={'user_input': 'question'})

    missing = {'question', 'context_recall'} - set(df.columns)
    if missing:
        raise ValueError(
            f"{eval_csv} is missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df_labelled = df.dropna(subset=['context_recall']).copy()
    df_labelled['relevant'] = (df_labelled['context_recall'] > 0).astype(int)

    logging.info(
        "Loaded %d labelled samples  (%d relevant, %d irrelevant).",
        len(df_labelled),
        df_labelled['relevant'].sum(),
        (df_labelled['relevant'] == 0).sum(),
    )
    return df_labelled


# ── Score collection ──────────────────────────────────────────────────────────

def _collect_reranker_scores(questions, dense_index, dense_meta,
                             sparse_index) -> np.ndarray:
    """
    Runs retrieval + reranking for each question and returns the per-query
    maximum bge-reranker-v2-m3 logit as a float array.
    """
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import Reranker

    logging.info("Loading HybridRetriever …")
    retriever = HybridRetriever(
        dense_index_path=dense_index,
        dense_meta_path=dense_meta,
        sparse_index_path=sparse_index,
    )
    logging.info("Loading Reranker …")
    reranker = Reranker(model_name='BAAI/bge-reranker-v2-m3')

    max_logits = []
    for i, query in enumerate(questions):
        logging.info("[%d/%d] %s", i + 1, len(questions), query[:70])
        try:
            broad = retriever.search(query, k=50)
            if not broad:
                max_logits.append(float('nan'))
                continue
            reranked = reranker.rerank(query, broad, top_k=7)
            best = max(d.get('rerank_score', float('-inf')) for d in reranked)
            max_logits.append(float(best))
        except Exception as exc:
            logging.warning("Query %d failed: %s", i, exc)
            max_logits.append(float('nan'))

    return np.array(max_logits, dtype=float)


# ── Threshold search ──────────────────────────────────────────────────────────

def _find_f1_threshold(scores: np.ndarray,
                       labels: np.ndarray) -> tuple:
    """
    Grid-searches 300 candidate thresholds to find the one that maximises
    binary F1 (relevant vs. irrelevant).

    Returns (best_threshold, best_f1, candidate_array, f1_array).
    """
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    valid = ~np.isnan(scores)
    s, l = scores[valid], labels[valid]

    if len(s) < 2:
        warnings.warn("Too few valid samples for reliable threshold search.")
        return 0.0, float('nan'), np.array([0.0]), np.array([float('nan')])

    candidates = np.linspace(s.min() - 0.5, s.max() + 0.5, 300)
    f1_scores = [
        f1_score(l, (s >= t).astype(int), zero_division=0)
        for t in candidates
    ]

    best_idx = int(np.argmax(f1_scores))
    return candidates[best_idx], f1_scores[best_idx], candidates, np.array(f1_scores)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot(scores: np.ndarray, labels: np.ndarray,
          best_threshold: float, candidates, f1_curve,
          output_plot: str) -> None:
    """
    Two-panel figure:
      Left  — histogram of max reranker logits, relevant vs. irrelevant.
      Right — F1 score as a function of threshold.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed — skipping plot.")
        return

    rel = scores[(labels == 1) & ~np.isnan(scores)]
    irr = scores[(labels == 0) & ~np.isnan(scores)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: distributions
    bins = 15
    if len(rel):
        ax1.hist(rel, bins=bins, alpha=0.7, color='steelblue',
                 label='Relevant  (context_recall > 0)')
    if len(irr):
        ax1.hist(irr, bins=bins, alpha=0.7, color='salmon',
                 label='Irrelevant (context_recall = 0)')
    ax1.axvline(best_threshold, color='black', linestyle='--', linewidth=2,
                label=f'F1-max threshold = {best_threshold:.3f}')
    ax1.axvline(0.0, color='grey', linestyle=':', linewidth=1,
                label='Default threshold (0.0)')
    ax1.set_xlabel('Max Reranker Logit (bge-reranker-v2-m3)')
    ax1.set_ylabel('Count')
    ax1.set_title('CRAG Gate – Reranker Logit Distributions')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: F1 curve
    ax2.plot(candidates, f1_curve, color='steelblue', linewidth=2)
    ax2.axvline(best_threshold, color='black', linestyle='--', linewidth=2,
                label=f'F1-max = {max(f1_curve):.3f} @ {best_threshold:.3f}')
    ax2.axvline(0.0, color='grey', linestyle=':', linewidth=1,
                label='Default (0.0)')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 vs CRAG Threshold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    logging.info("Plot saved → %s", output_plot)
    plt.close()


# ── Main calibration function ─────────────────────────────────────────────────

def calibrate(eval_csv: str, dense_index: str, dense_meta: str,
              sparse_index: str, output_plot: str) -> float:
    """Full calibration pipeline. Returns the recommended threshold."""

    # 1. Load labelled data
    df = _load_eval_data(eval_csv)
    questions = df['question'].tolist()
    labels    = df['relevant'].values

    # 2. Collect max reranker logit per query
    scores = _collect_reranker_scores(
        questions, dense_index, dense_meta, sparse_index
    )

    # Attach scores to df for display
    df = df.copy()
    df['max_rerank_logit'] = scores

    print("\n── Per-query reranker logits ─────────────────────────────────")
    print(df[['question', 'context_recall', 'relevant', 'max_rerank_logit']]
          .to_string(index=False))

    # 3. Find F1-maximising threshold
    best_thresh, best_f1, candidates, f1_curve = _find_f1_threshold(
        scores, labels
    )

    print("\n── Calibration result ────────────────────────────────────────")
    print(f"  Current default threshold : 0.000  (sigmoid = 0.50)")
    print(f"  F1-maximising threshold   : {best_thresh:.4f}"
          f"  (F1 = {best_f1:.4f})")
    print()
    print(f"  → To apply: edit CRAG_THRESHOLD in src/run_rag.py:")
    print(f"      CRAG_THRESHOLD: float = {best_thresh:.4f}")
    print(f"    or pass --crag-threshold {best_thresh:.4f} at the CLI.")
    print()
    n = len(df)
    print(f"  ⚠  Note: estimated from only {n} samples — re-run with a larger")
    print(f"     evaluation set (≥50 samples) for a robust threshold.")

    # 4. Plot
    _plot(scores, labels, best_thresh, candidates, f1_curve, output_plot)

    return best_thresh


# ── Percentile calibration (Mode 2 — no RAGAS labels needed) ────────────────

def calibrate_percentile(dataset_csv: str, percentile: float,
                         output_plot: str) -> float:
    """
    Derive a CRAG threshold from the empirical distribution of
    best_rerank_score values saved by generate_predictions.py.

    Sets threshold at the `percentile`-th percentile (default 10), following
    Karpukhin et al. (2020) DPR §5.4 and Asai et al. (2023) Self-RAG §3.2:
    retain the top (100-P)% of retrievals, suppress only the lowest P%.
    With paper-ID filtering the distribution is right-skewed, so P=10 is
    a conservative but principled starting point.
    """
    df = pd.read_csv(dataset_csv)

    if "best_rerank_score" not in df.columns:
        raise ValueError(
            f"{dataset_csv} has no 'best_rerank_score' column. "
            "Re-run generate_predictions.py to regenerate the dataset "
            "(the column is saved from run_rag.py's pipeline output)."
        )

    scores = pd.to_numeric(df["best_rerank_score"], errors="coerce").dropna()

    # Remove -inf sentinel values (used when retrieval returned no documents)
    n_sentinel = int((np.isinf(scores) & (scores < 0)).sum())
    scores = scores[np.isfinite(scores)]
    if n_sentinel > 0:
        logging.warning(
            "%d rows had best_rerank_score=-inf (empty retrieval) "
            "— excluded from calibration.",
            n_sentinel,
        )

    if len(scores) == 0:
        raise ValueError(
            "No finite best_rerank_score values found — all queries returned "
            "empty retrieval results."
        )

    threshold = float(np.percentile(scores, percentile))
    suppressed = int((scores < threshold).sum())
    pct_supp = 100.0 * suppressed / len(scores)

    print(f"\n── Score-distribution calibration (percentile={percentile}) ────")
    print(f"  Samples          : {len(scores)}  (excluded {n_sentinel} empty-retrieval rows)")
    print(f"  Min / Mean / Max : {scores.min():.4f} / {scores.mean():.4f} / {scores.max():.4f}")
    print(f"  {percentile}th percentile → threshold = {threshold:.4f}")
    print(f"  Queries suppressed by CRAG gate: {suppressed}/{len(scores)} ({pct_supp:.1f}%)")
    print()
    print(f"  → To apply: edit CRAG_THRESHOLD in src/run_rag.py:")
    print(f"      CRAG_THRESHOLD: float = {threshold:.4f}")
    print(f"    or pass --crag-threshold {threshold:.4f} at the CLI.")
    print()
    print("  After the next evaluation run (with context_recall available),")
    print("  re-run with --mode f1 for a tighter F1-maximising threshold.")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(scores, bins=30, color="steelblue", alpha=0.8,
                label=f"best_rerank_score (n={len(scores)})")
        ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
                   label=f"{percentile}th pct threshold = {threshold:.3f}")
        ax.axvline(0.0, color="grey", linestyle=":", linewidth=1,
                   label="Default threshold (0.0)")
        ax.set_xlabel("Max Reranker Logit (bge-reranker-v2-m3)")
        ax.set_ylabel("Count")
        ax.set_title("CRAG Threshold — Score Distribution (percentile mode)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)
        plt.savefig(output_plot, dpi=150, bbox_inches="tight")
        logging.info("Plot saved → %s", output_plot)
        plt.close()
    except ImportError:
        logging.warning("matplotlib not installed — skipping plot.")

    return threshold


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the CRAG threshold from the evaluation dataset."
    )
    parser.add_argument(
        '--mode',
        choices=['f1', 'percentile'],
        default='f1',
        help='"f1": F1-maximising threshold (needs evaluation_report.csv with context_recall). '
             '"percentile": percentile-cut from best_rerank_score (needs evaluation_dataset.csv). '
             'Use "percentile" when context_recall is all-NaN.',
    )
    parser.add_argument(
        '--eval-csv',
        default=str(_PROJECT_ROOT / 'data' / 'evaluation_report.csv'),
        help='Path to evaluation_report.csv (needs context_recall). Used by --mode f1.',
    )
    parser.add_argument(
        '--dataset-csv',
        default=str(_PROJECT_ROOT / 'data' / 'evaluation_dataset.csv'),
        help='Path to evaluation_dataset.csv with best_rerank_score. Used by --mode percentile.',
    )
    parser.add_argument(
        '--percentile',
        type=float,
        default=10.0,
        help='Percentile cutoff for --mode percentile (default 10). Suppresses bottom P%% of retrievals.',
    )
    parser.add_argument(
        '--dense-index',
        default=str(_PROJECT_ROOT / 'data' / 'indices' / 'dense.index'),
    )
    parser.add_argument(
        '--dense-meta',
        default=str(_PROJECT_ROOT / 'data' / 'indices' / 'dense.index.meta'),
    )
    parser.add_argument(
        '--sparse-index',
        default=str(_PROJECT_ROOT / 'data' / 'indices' / 'sparse.pkl'),
    )
    parser.add_argument(
        '--output-plot',
        default=str(_PROJECT_ROOT / 'reports' / 'figures' / 'crag_calibration.png'),
        help='Where to save the calibration plot.',
    )
    args = parser.parse_args()

    if args.mode == 'percentile':
        calibrate_percentile(
            dataset_csv=args.dataset_csv,
            percentile=args.percentile,
            output_plot=args.output_plot,
        )
    else:
        calibrate(
            eval_csv=args.eval_csv,
            dense_index=args.dense_index,
            dense_meta=args.dense_meta,
            sparse_index=args.sparse_index,
            output_plot=args.output_plot,
        )


if __name__ == '__main__':
    main()
