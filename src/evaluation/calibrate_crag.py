"""
CRAG Threshold Calibration
--------------------------
Runs retrieval + reranking on the evaluation set to record each query's
maximum reranker logit, then plots logit distributions for relevant vs.
irrelevant retrievals and identifies the F1-maximising threshold.

Usage (from project root):
    python -m src.evaluation.calibrate_crag
    python -m src.evaluation.calibrate_crag --eval-csv data/evaluation_dataset.csv

Output:
    • Console table: per-query max reranker logit vs. context_recall label
    • Console: recommended threshold value
    • Plot saved to reports/figures/crag_calibration.png

How "relevance" is defined:
    context_recall > 0  →  label = 1 (retrieval found useful evidence)
    context_recall = 0  →  label = 0 (retrieval missed the gold evidence)
    context_recall = NaN→  excluded from calibration

Reference:
    Yan et al. (2024). CRAG: Corrective Retrieval Augmented Generation.
    arXiv:2401.15884. §3.3 Knowledge Refinement:
        "We calibrate the retrieval evaluator threshold empirically on the
         score distribution of training samples to maximise retrieval F1."
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

    Accepts either evaluation_dataset.csv (which already includes RAGAS metric
    columns written by evaluate_rag.py) or evaluation_report.csv.
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the CRAG threshold from the evaluation dataset."
    )
    parser.add_argument(
        '--eval-csv',
        default=str(_PROJECT_ROOT / 'data' / 'evaluation_dataset.csv'),
        help='Path to evaluation_dataset.csv (needs question + context_recall).',
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

    calibrate(
        eval_csv=args.eval_csv,
        dense_index=args.dense_index,
        dense_meta=args.dense_meta,
        sparse_index=args.sparse_index,
        output_plot=args.output_plot,
    )


if __name__ == '__main__':
    main()
