"""
DSPy compiler for the RAG generation prompt (key_metrics_improvements.md,
Finding #5).

Compiles src/generation/dspy_module.py::ScientificRAGModule's few-shot
demonstrations with dspy.BootstrapFewShot — DSPy's own recommended
starting-point optimizer for small datasets with no separate strong "prompt
model" available (see https://dspy.ai/diving-deeper/choosing-an-optimizer/).

Metric (Finding #5's original recommendation, now the default): reuses
ALCEEvaluator.calculate_metrics (evaluate_rag.py:701-767) as-is as the DSPy
metric function — it returns exactly the (precision, recall) pair, reduced
here to their F1 since BootstrapFewShot/MIPROv2 expect one scalar per call.
Citation-stuffing mitigation (also per Finding #5): wrap_with_refine() wraps
a compiled module in dspy.Refine with a reward_fn that penalizes citing more
than 3 distinct docs in one sentence, retrying up to N times at temperature
1.0 for a compliant rollout — this is a runtime wrapper applied when the
compiled module is *used*, not baked into the saved .save() artifact (DSPy's
save/load round-trip is defined for the base Module, not a Refine wrapper
around it).

GPU cost (stated explicitly, not hidden): each ALCE metric call runs one or
more live Prometheus NLI calls (evaluate_rag.py::check_nli_entailment) per
answer sentence, so a real --metric alce compile needs BOTH the Llama vLLM
server (dspy.LM generating candidate answers) and a reachable Prometheus
backend (get_prometheus_judge() — vLLM/HF/Ollama) available *at the same
time*. CLAUDE.md documents Llama and Prometheus as normally sequential on a
single GPU specifically to avoid OOM (run_evaluation.sh stops Llama before
starting Prometheus) — running both together needs enough combined VRAM, or
pointing PROMETHEUS_PORT at a Prometheus instance on a separate GPU/process.
Check headroom before a real compile; --metric citation-format remains
available as a no-judge-call fallback (format/stuffing only, not semantic
claim-to-citation correctness) if that headroom isn't there.

Usage:
    # Real compile, Finding #5's ALCE/Prometheus metric (needs Llama vLLM
    # AND a Prometheus backend both reachable at once — see GPU cost above):
    python -m src.evaluation.compile_dspy_prompt

    # Cheap fallback: deterministic citation-format metric, no judge calls,
    # only needs the Llama vLLM server up:
    python -m src.evaluation.compile_dspy_prompt --metric citation-format

    # Smoke test (no vLLM/GPU generation call — uses DummyLM, real retrieval):
    python -m src.evaluation.compile_dspy_prompt --smoke-test
"""
import argparse
import logging
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import dspy
import nltk

from src.evaluation.evaluate_rag import ALCEEvaluator, get_prometheus_judge
from src.evaluation.generate_predictions import fetch_qasper_sample
from src.generation.dspy_module import (
    DEFAULT_PAPER_FOCUS_HINT,
    ScientificRAGModule,
    format_context_blocks,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# generate_predictions.py's default eval-sample seed/size — used here only to
# compute what the eval sample looked like, so the train sample can be
# checked for disjointness against it. Not re-scored, just re-enumerated.
EVAL_SEED = 20260902
EVAL_NUM_SAMPLES = 150

# A distinct seed for the DSPy training sample. Must differ from EVAL_SEED;
# disjointness is verified explicitly below, not assumed from the seeds
# differing.
DEFAULT_TRAIN_SEED = 20260903

_MALFORMED_CITATION_RE = re.compile(r"\((?:Doc|Document)\s*\d+\)|\[Document\s*\d+\]|\[\d+\]")
_CITATION_RE = re.compile(r"\[Doc (\d+)\]")
_REFUSAL_TEXT = "The retrieved documents do not contain enough information to answer this."
_MAX_DISTINCT_CITES_PER_SENTENCE = 3


# ── Disjoint train/eval sampling ────────────────────────────────────────────

def fetch_disjoint_train_sample(
    num_samples: int,
    train_seed: int = DEFAULT_TRAIN_SEED,
    eval_seed: int = EVAL_SEED,
    eval_num_samples: int = EVAL_NUM_SAMPLES,
) -> List[Dict]:
    """
    Draws a QASPER train-split sample for DSPy compilation and verifies it
    shares no (question, paper_id) pair with generate_predictions.py's eval
    sample — explicitly checked, not just assumed from the seeds differing.
    """
    eval_pool = fetch_qasper_sample(num_samples=eval_num_samples, seed=eval_seed)
    eval_keys = {(qa["question"], qa["paper_id"]) for qa in eval_pool}

    # Over-sample slightly so we still hit num_samples after dropping overlap.
    train_pool = fetch_qasper_sample(num_samples=num_samples + 10, seed=train_seed)
    train_pool = [
        qa for qa in train_pool if (qa["question"], qa["paper_id"]) not in eval_keys
    ][:num_samples]

    overlap_found = any(
        (qa["question"], qa["paper_id"]) in eval_keys for qa in train_pool
    )
    assert not overlap_found, "Disjointness check failed after filtering — bug."
    logging.info(
        "Train sample: %d questions, disjoint from the %d-question eval "
        "sample (seed=%d vs eval seed=%d).",
        len(train_pool), len(eval_pool), train_seed, eval_seed,
    )
    return train_pool


# ── Real-pipeline retrieval (stages 1-3, no generation) ─────────────────────

def build_retrieval_components(
    dense_index_path: str, dense_meta_path: str, sparse_index_path: str,
    crag_correct_threshold: float, crag_ambiguous_threshold: float,
    crag_consistency_ratio: float,
):
    """
    Instantiates the same retrieval/rerank/CRAG stack run_rag.py's
    ScientificRAGPipeline uses (stages 1-3 only, no generator) — kept
    separate from ScientificRAGPipeline itself so this stays purely additive
    (no changes to run_rag.py needed for this script to work). Mirrors the
    pattern already used by calibrate_crag.py::_collect_reranker_scores.
    """
    from src.retrieval.crag_evaluator import CRAGEvaluator
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import ColBERTv2Reranker

    retriever = HybridRetriever(
        dense_index_path=dense_index_path,
        dense_meta_path=dense_meta_path,
        sparse_index_path=sparse_index_path,
    )
    reranker = ColBERTv2Reranker(model_name="colbert-ir/colbertv2.0")
    crag_evaluator = CRAGEvaluator(
        correct_threshold=crag_correct_threshold,
        ambiguous_threshold=crag_ambiguous_threshold,
        consistency_ratio=crag_consistency_ratio,
    )
    return retriever, reranker, crag_evaluator


def retrieve_docs_for_question(
    retriever, reranker, crag_evaluator, question: str, paper_id: str
) -> List[Dict]:
    """Stages 1-3 of run_rag.py::ScientificRAGPipeline.ask, no generation.

    Mirrors run_rag.py's own 'Incorrect' fallback (top-5 by rerank score)
    exactly, so training examples see the same context shape live queries do.
    Deliberately skips HyDE (documented simplification — see module
    docstring) to avoid needing a live LM for the retrieval stage itself.
    """
    broad = retriever.search(question, k=100, filter_paper_id=paper_id)
    if not broad:
        return []
    top_docs = reranker.rerank(question, broad, top_k=10)
    action, refined, _details = crag_evaluator.evaluate_and_refine(question, top_docs)
    if action == "Incorrect":
        refined = sorted(
            top_docs, key=lambda d: d.get("rerank_score", 0.0), reverse=True
        )[:5]
    return refined


def build_trainset(qa_pairs: List[Dict], retriever, reranker, crag_evaluator) -> List[dspy.Example]:
    examples = []
    n_empty = 0
    for i, qa in enumerate(qa_pairs):
        docs = retrieve_docs_for_question(
            retriever, reranker, crag_evaluator, qa["question"], qa.get("paper_id")
        )
        if not docs:
            n_empty += 1
            continue
        ex = dspy.Example(
            context=format_context_blocks(docs),
            paper_focus_hint=DEFAULT_PAPER_FOCUS_HINT,
            question=qa["question"],
        ).with_inputs("context", "paper_focus_hint", "question")
        examples.append(ex)
        logging.info(
            "[%d/%d] train example built (%d retrieved docs): %.60s",
            i + 1, len(qa_pairs), len(docs), qa["question"],
        )
    if n_empty:
        logging.warning(
            "%d/%d training questions had 0 retrieved docs — excluded.",
            n_empty, len(qa_pairs),
        )
    return examples


# ── Deterministic citation-format metric (fallback / Refine reward) ─────────

def _citation_format_score(answer: str, n_docs: int) -> float:
    """
    Scores `answer` for citation format compliance and anti-stuffing — no
    LLM call. See module docstring for the explicit limitation
    (format/stuffing only, not semantic correctness).

    Per-sentence pass requires ALL of:
      (a) at least one well-formed [Doc N] tag,
      (b) no malformed variant — (Doc 1), [Document 1], [1],
      (c) every cited N within 1..n_docs (no hallucinated doc indices —
          mirrors the bounds-check ALCEEvaluator/_iter_alce_citation_sentences
          already use),
      (d) <=3 distinct docs cited (the citation-stuffing guard Finding #5
          proposes for dspy.Refine, below).
    Returns the fraction of sentences that pass (1.0 if the whole answer is
    the exact refusal string — a valid, format-compliant output).
    """
    answer = (answer or "").strip()
    if not answer:
        return 0.0
    if answer == _REFUSAL_TEXT:
        return 1.0

    sentences = [s for s in nltk.sent_tokenize(answer) if len(s.strip()) >= 10]
    if not sentences:
        return 0.0

    passed = 0
    for sentence in sentences:
        if _MALFORMED_CITATION_RE.search(sentence):
            continue
        cited = _CITATION_RE.findall(sentence)
        if not cited:
            continue
        cited_ints = [int(c) for c in cited]
        if not all(1 <= c <= n_docs for c in cited_ints):
            continue
        if len(set(cited_ints)) > _MAX_DISTINCT_CITES_PER_SENTENCE:
            continue
        passed += 1

    return passed / len(sentences)


def citation_format_metric(example: dspy.Example, pred, trace=None) -> float:
    """DSPy metric shape: (example, pred, trace) -> float. See
    _citation_format_score for the actual scoring logic."""
    return _citation_format_score(getattr(pred, "cited_answer", "") or "", len(example.context))


def citation_format_reward(kwargs: dict, pred) -> float:
    """dspy.Refine reward_fn shape: (forward_kwargs, pred) -> float — kwargs
    is whatever the wrapped module's forward() was called with (context,
    paper_focus_hint, question), not a dspy.Example."""
    return _citation_format_score(getattr(pred, "cited_answer", "") or "", len(kwargs.get("context", [])))


# ── ALCE/Prometheus metric (Finding #5's original recommendation) ───────────

def build_alce_metric(judge_evaluator: ALCEEvaluator) -> Callable[..., float]:
    """
    Binds one ALCEEvaluator (and the Prometheus judge/model it wraps) into a
    DSPy metric function, so the judge is loaded once and reused across every
    compile-time metric call rather than reloaded per call.

    ALCEEvaluator.calculate_metrics(answer, contexts) -> (precision, recall)
    is used as-is (Finding #5's own text: "it already returns exactly the
    (precision, recall) pair DSPy optimizers need"); BootstrapFewShot/MIPROv2
    want one scalar per call, so this returns their F1 (0.0 if both are 0).
    """
    def alce_metric(example: dspy.Example, pred, trace=None) -> float:
        answer = (getattr(pred, "cited_answer", "") or "").strip()
        if not answer:
            return 0.0
        precision, recall = judge_evaluator.calculate_metrics(answer, example.context)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    return alce_metric


# ── dspy.Refine citation-stuffing wrapper (Finding #5) ───────────────────────

def wrap_with_refine(module: dspy.Module, n: int = 3, threshold: float = 1.0) -> dspy.Refine:
    """
    Wraps a (typically already-compiled) module in dspy.Refine using the
    citation-count reward, per Finding #5: "wrap the compiled module in
    dspy.Refine with a reward_fn that checks citation count against the
    format regex \\[Doc \\d+\\] and penalizes citing more than ~3 distinct
    docs per sentence." Runs the module up to `n` times at temperature=1.0
    and keeps the first rollout at/above `threshold`, else the highest-reward
    one — this targets the row-43 citation-stuffing pattern at generation
    time, on top of whatever the compile-time metric already selected for.

    Applied when the compiled module is *used*, not saved: dspy.Module.save()
    serializes named predictors/demos, which is defined for the base module,
    not a Refine wrapper around it — load the base module, then call this.
    """
    return dspy.Refine(module=module, N=n, reward_fn=citation_format_reward, threshold=threshold)


# ── Compilation ───────────────────────────────────────────────────────────────

def compile_program(
    trainset: List[dspy.Example],
    lm,
    metric: Callable[..., float],
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
) -> ScientificRAGModule:
    """Compiles ScientificRAGModule with BootstrapFewShot. `lm` is injected
    (a real dspy.LM for production runs, a DummyLM for the smoke test);
    `metric` is build_alce_metric(...)'s output by default (Finding #5), or
    citation_format_metric as a no-judge-call fallback."""
    dspy.settings.configure(lm=lm)
    module = ScientificRAGModule()
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )
    return optimizer.compile(module, trainset=trainset)


def _ensure_nltk():
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception as exc:
        logging.warning("NLTK download failed: %s", exc)


# ── Smoke test (no vLLM, no network beyond the already-cached QASPER data) ──

def run_smoke_test() -> None:
    """
    Exercises the full plumbing without a live vLLM call:
      - citation_format_metric against hand-crafted (well-formed / malformed
        / stuffed / out-of-bounds-index) cases,
      - fetch_disjoint_train_sample's overlap assertion,
      - real retrieval (HybridRetriever + ColBERTv2Reranker + CRAGEvaluator,
        needs the local indices + GPU, but no LLM) building 2 real training
        examples,
      - compile_program driven by dspy.utils.dummies.DummyLM instead of a
        real LM,
      - save/load round-trip of the compiled program.
    """
    _ensure_nltk()

    logging.info("[smoke] citation_format_metric unit cases...")
    # 5 docs so a 4-distinct-citation "stuffing" sentence stays in-bounds
    # (otherwise it would also trip the out-of-bounds check, testing the
    # wrong thing) while [Doc 9] below still correctly exercises OOB.
    ctx5 = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    ex = dspy.Example(context=ctx5, paper_focus_hint="", question="").with_inputs(
        "context", "paper_focus_hint", "question"
    )

    class _Pred:
        def __init__(self, cited_answer):
            self.cited_answer = cited_answer

    cases = [
        ("Well-formed", "The value is X [Doc 1]. Also Y [Doc 2].", 1.0),
        ("Malformed paren", "The value is X (Doc 1).", 0.0),
        ("Malformed bracket-num", "The value is X [1].", 0.0),
        ("Out-of-bounds index", "The value is X [Doc 9].", 0.0),
        # 4 distinct docs (1,2,3,4), all in-bounds for a 5-doc context —
        # exceeds the <=3-distinct-docs-per-sentence stuffing guard.
        ("Stuffing", "The value is X [Doc 1][Doc 2][Doc 3][Doc 4].", 0.0),
        ("Exact refusal", _REFUSAL_TEXT, 1.0),
        ("No citation", "The value is X.", 0.0),
        ("Mixed", "Good claim [Doc 1]. Bad claim [Doc 9].", 0.5),
    ]
    for name, text, expected in cases:
        got = citation_format_metric(ex, _Pred(text))
        status = "OK" if got == expected else "FAIL"
        logging.info("  [%s] %-22s expected=%.2f got=%.2f", status, name, expected, got)
        assert got == expected, f"citation_format_metric case '{name}' failed: {got} != {expected}"

    logging.info("[smoke] fetch_disjoint_train_sample (small samples, uses cached QASPER)...")
    train_sample = fetch_disjoint_train_sample(
        num_samples=5, train_seed=DEFAULT_TRAIN_SEED, eval_seed=EVAL_SEED,
        eval_num_samples=20,
    )
    assert len(train_sample) == 5

    logging.info("[smoke] real retrieval for %d training questions (no LLM)...", 2)
    retriever, reranker, crag_evaluator = build_retrieval_components(
        dense_index_path=str(_PROJECT_ROOT / "data" / "indices" / "dense.index"),
        dense_meta_path=str(_PROJECT_ROOT / "data" / "indices" / "dense.index.meta"),
        sparse_index_path=str(_PROJECT_ROOT / "data" / "indices" / "sparse.pkl"),
        crag_correct_threshold=14.0, crag_ambiguous_threshold=8.0,
        crag_consistency_ratio=0.3,
    )
    trainset = build_trainset(train_sample[:2], retriever, reranker, crag_evaluator)
    assert len(trainset) >= 1, "Expected at least 1 real training example with non-empty retrieval."

    logging.info("[smoke] compiling with DummyLM (no live vLLM call)...")
    from dspy.utils.dummies import DummyLM
    stub_lm = DummyLM([
        {"reasoning": "stub reasoning", "cited_answer": f"The answer is X [Doc 1]."}
        for _ in range(50)
    ])
    compiled = compile_program(
        trainset, lm=stub_lm, metric=citation_format_metric,
        max_bootstrapped_demos=2, max_labeled_demos=2,
    )

    logging.info("[smoke] build_alce_metric F1 math (stub evaluator, no Prometheus)...")
    class _StubALCEEvaluator:
        def __init__(self, precision, recall):
            self.precision, self.recall = precision, recall

        def calculate_metrics(self, answer, contexts):
            return self.precision, self.recall

    alce_cases = [
        (1.0, 1.0, 1.0), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    ]
    for precision, recall, expected_f1 in alce_cases:
        metric_fn = build_alce_metric(_StubALCEEvaluator(precision, recall))
        got = metric_fn(ex, _Pred("The value is X [Doc 1]."))
        assert abs(got - expected_f1) < 1e-9, (
            f"alce_metric F1 mismatch for P={precision} R={recall}: {got} != {expected_f1}"
        )
    logging.info("  [OK] build_alce_metric F1 matches expected for %d P/R cases", len(alce_cases))

    logging.info("[smoke] wrap_with_refine construction (no live LM call)...")
    refined = wrap_with_refine(compiled, n=2, threshold=1.0)
    assert isinstance(refined, dspy.Refine)
    logging.info("  [OK] dspy.Refine wraps the compiled module")

    logging.info("[smoke] save/load round-trip...")
    out_path = _PROJECT_ROOT / "data" / "dspy_compiled_prompt.smoketest.json"
    compiled.save(str(out_path))
    reloaded = ScientificRAGModule()
    reloaded.load(str(out_path))
    out_path.unlink()  # smoke-test artifact only, not the real compiled output

    logging.info("SMOKE TEST PASSED")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile the RAG generation prompt with DSPy BootstrapFewShot."
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run plumbing checks with DummyLM instead of a real compile.")
    parser.add_argument("--n-train", type=int, default=50)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--eval-num-samples", type=int, default=EVAL_NUM_SAMPLES)
    parser.add_argument("--dense-index", type=str,
                        default=str(_PROJECT_ROOT / "data" / "indices" / "dense.index"))
    parser.add_argument("--dense-meta", type=str,
                        default=str(_PROJECT_ROOT / "data" / "indices" / "dense.index.meta"))
    parser.add_argument("--sparse-index", type=str,
                        default=str(_PROJECT_ROOT / "data" / "indices" / "sparse.pkl"))
    parser.add_argument("--crag-correct", type=float, default=14.0)
    parser.add_argument("--crag-ambiguous", type=float, default=8.0)
    parser.add_argument("--crag-consistency", type=float, default=0.3)
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--api-base", type=str,
                        default=os.environ.get("VLLM_API_URL", "http://localhost:8000/v1"))
    parser.add_argument("--max-bootstrapped-demos", type=int, default=4)
    parser.add_argument("--max-labeled-demos", type=int, default=4)
    parser.add_argument(
        "--metric", type=str, choices=["alce", "citation-format"], default="alce",
        help="'alce' (default, Finding #5): ALCEEvaluator/Prometheus F1 — "
             "needs Llama vLLM AND a Prometheus backend up simultaneously. "
             "'citation-format': deterministic, no judge calls, only needs "
             "Llama vLLM up — use when GPU headroom won't hold both models.",
    )
    parser.add_argument(
        "--refine", dest="refine", action="store_true", default=True,
        help="Wrap the compiled module in dspy.Refine with the citation-count "
             "reward (Finding #5's anti-stuffing recommendation). On by default.",
    )
    parser.add_argument("--no-refine", dest="refine", action="store_false")
    parser.add_argument("--refine-n", type=int, default=3)
    parser.add_argument("--refine-threshold", type=float, default=1.0)
    parser.add_argument("--output-path", type=str,
                        default=str(_PROJECT_ROOT / "data" / "dspy_compiled_prompt.json"))
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    _ensure_nltk()
    train_sample = fetch_disjoint_train_sample(
        num_samples=args.n_train, train_seed=args.train_seed,
        eval_seed=args.eval_seed, eval_num_samples=args.eval_num_samples,
    )
    retriever, reranker, crag_evaluator = build_retrieval_components(
        dense_index_path=args.dense_index, dense_meta_path=args.dense_meta,
        sparse_index_path=args.sparse_index,
        crag_correct_threshold=args.crag_correct,
        crag_ambiguous_threshold=args.crag_ambiguous,
        crag_consistency_ratio=args.crag_consistency,
    )
    trainset = build_trainset(train_sample, retriever, reranker, crag_evaluator)
    if not trainset:
        raise RuntimeError("No training examples with non-empty retrieval — aborting compile.")

    if args.metric == "alce":
        logging.info(
            "Using ALCE/Prometheus metric (Finding #5) — requires a reachable "
            "Prometheus backend alongside the Llama vLLM server. Loading judge..."
        )
        judge, _is_gpu = get_prometheus_judge()
        metric_fn = build_alce_metric(ALCEEvaluator(judge))
    else:
        logging.info("Using deterministic citation-format metric (no judge calls).")
        metric_fn = citation_format_metric

    lm = dspy.LM(f"openai/{args.model_id}", api_base=args.api_base, api_key="EMPTY")
    compiled = compile_program(
        trainset, lm=lm, metric=metric_fn,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
    )
    compiled.save(args.output_path)
    logging.info("Compiled program saved to %s", args.output_path)

    if args.refine:
        wrap_with_refine(compiled, n=args.refine_n, threshold=args.refine_threshold)
        logging.info(
            "dspy.Refine wrapper verified (N=%d, threshold=%.2f) — apply it "
            "at inference time around the loaded module: "
            "wrap_with_refine(ScientificRAGModule().load('%s')). Not baked "
            "into the saved artifact; see module docstring.",
            args.refine_n, args.refine_threshold, args.output_path,
        )


if __name__ == "__main__":
    main()
