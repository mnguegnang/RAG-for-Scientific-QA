# Evaluation Metric Improvements — RAG for Scientific QA

Review scope: `src/retrieval/*`, `src/generation/llm_generator.py`, `src/evaluation/*`,
`configs/prompts.yaml`, `data/evaluation_dataset.csv` (150 rows), `data/evaluation_report.csv`
(150 rows), `Project_Note.md`, and the uncommitted working-tree diff (`git diff`).
No source files were modified; this is a findings-and-recommendations report only.

## Current Metric State (data/evaluation_report.csv, n=150, 128 scorable rows)

| Metric | Mean | Zero-rate | NaN rows |
|---|---|---|---|
| ALCE citation precision | 0.670 | 18.0% | 22 |
| ALCE citation recall | 0.670 | 18.0% | 22 |
| ALCE citation F1 | 0.670 | 18.0% | 22 |
| Context precision | 0.595 | 3.1% | 22 |
| **Context recall** | **0.300** | **68.1%** | 31 |
| Faithfulness | 0.672 | 17.2% | 22 |
| Answer relevancy | 0.559 | 23.4% | 22 |
| **Answer correctness** | **0.393** | **43.0%** | 22 |

**These numbers should not be treated as current ground truth.** Finding #1 below shows the
on-disk ALCE precision/recall/F1 columns are a known-stale artifact of a bug the working tree
already fixes but has not been re-run against. Context recall (0.30, 68% zero-rate) and answer
correctness (0.39, 43% zero-rate) are the two weakest metrics and, per the error analysis, are
driven overwhelmingly by one root cause (Finding #2) rather than by generation quality.

## Error Analysis

Method: loaded `evaluation_report.csv` + `evaluation_dataset.csv` with pandas, computed the
score distribution above, then read the full question/ground-truth/contexts/answer for every
row scoring 0 on `context_recall`, `answer_correctness`, or `alce_citation_precision` (30 rows
inspected, since these three columns are the most zero-heavy and most rows fail on more than
one). Failure modes were open-coded from the actual text, not inferred from code alone.

| Failure mode | Rows observed | Evidence |
|---|---|---|
| Cross-paper retrieval contamination — all 10 retrieved chunks are from a paper with a different title than the question's ground-truth paper | 4/30 sampled, all with `crag_triggered=False` and rerank scores 19.7–25.7 (mid-to-high) | Rows 3, 11, 30, 36. E.g. row 11 ("How many questions are in the dataset?", GT `2,714`) retrieves chunks titled *Towards a Robust Deep Neural Network in Text Domain* and *Question Answering from Unstructured Text*, neither of which is the source paper — model answers "200K... 8.8k... 87,361", scoring `answer_correctness=0`. |
| ALCE precision == recall == F1 for every row (measurement bug, not a content failure) | 128/128 valid rows | `df["alce_citation_precision"].equals(df["alce_citation_recall"])` → `True` for all 128. This is the exact symptom the current uncommitted diff to `ALCEEvaluator.calculate_metrics` (`src/evaluation/evaluate_rag.py:701-767`) fixes — the report on disk predates that fix. See Finding #1. |
| Judge false negatives on correctly-grounded, correctly-cited claims | 1/30 sampled directly confirmed, likely under-counted | Dataset row 0 ("What is the seed lexicon?"): all 10 retrieved chunks are from the correct paper; `contexts[0]` (`Doc 1`) states verbatim "The seed lexicon consists of positive and negative predicates" — a near word-for-word match to both the GT and the generated answer's first sentence, cited `[Doc 2]`. `score_context_recall` still returned 0.0 and ALCE precision returned 0.0 on this row. See Finding #3. |
| Citation misattribution by the generator — claim's content lives in a different retrieved doc than the one cited | 1/30 confirmed (row 0's third sentence) | Answer sentence "constructed with 15 positive and 15 negative words `[Doc 3]`" — but `contexts[2]` (Doc 3) is about "CO (Concession Pairs)"; the "15 positive and 15 negative words" text is actually `contexts[4]` (Doc 5). Prompt-building (`llm_generator.py:194-201`) and dataset-saving (`generate_predictions.py:188`) use the identical `retrieved_docs` ordering, so this is not a pipeline reordering bug — it is Llama-3.1-8B citing the wrong document index. |
| Citation stuffing on broad questions — every retrieved doc gets cited regardless of paper match | 1/30 confirmed, compounds with cross-paper contamination | Row 43 ("What dataset do they use?") retrieves 10 chunks from **10 different papers** (verified by title), and the answer dutifully lists and cites all 10 `[Doc N]` tags. Root cause is retrieval contamination, not a generation-prompt defect. |
| CRAG-triggered rows score worse than non-triggered on the metric CRAG exists to protect | 15/150 rows (`crag_triggered=True`) | Mean `context_recall` = **0.0** on CRAG-triggered rows vs. 0.324 on non-triggered; mean `answer_correctness` 0.20 vs 0.41. The Ambiguous/Incorrect fallback paths (`src/run_rag.py:180-198`, `src/retrieval/crag_evaluator.py:181-239`) are not recovering usable context once retrieval itself has failed. |

**Reprioritization:** the two dominant failure modes — cross-paper contamination and the stale
ALCE bug — are structural pipeline/measurement issues, not generation-quality issues. This
matters for where to spend effort next: prompt engineering or DSPy-style optimization of the
*generator* will not move `context_recall` or `answer_correctness` until the *retrieval and
measurement* fixes already sitting in the working tree are validated by a full re-run.

---

## Improvement Points (ranked by expected impact)

### 1. [Critical] Re-run the full pipeline — the current report is measuring a bug the code no longer has

- **Target metric(s):** ALCE citation precision/recall/F1 (all), plus any metric whose baseline
  is being read off this file.
- **Evidence:** `git diff -- src/evaluation/evaluate_rag.py` shows `ALCEEvaluator.calculate_metrics`
  was rewritten to compute precision over `total_citations` and recall over `len(sentences)` —
  two different denominators, per the reference ALCE implementation (Gao et al. 2023, EMNLP,
  §4.1). The version that produced `data/evaluation_report.csv` used `sentences_with_citations`
  for both (see the diff's own comment: "a previous revision used `sentences_with_citations` for
  both, which made recall algebraically identical to precision"). Confirmed empirically:
  `alce_citation_precision.equals(alce_citation_recall)` is `True` for all 128 valid rows.
- **Recommendation:** Run `bash run_evaluation.sh` (full regenerate: predictions + eval) after
  committing the current diff, not `evaluate_rag.py --skip-alce`. The retrieval-layer changes in
  the same diff (Finding #2) also change `contexts`, so `evaluation_dataset.csv` must be
  regenerated too, not just re-scored.
- **Expected mechanism:** Directly removes a measurement artifact. Also expected to raise the
  *true* ALCE F1 relative to the current 0.670 mean, since precision and recall are no longer
  forced identical — some rows currently being penalized by the shared-denominator bug will
  separate into (high recall, lower precision) or vice versa rather than being averaged together.
- **Source grounding:** Per Huyen's *AI Engineering* framing of eval-driven development — before
  optimizing a system against a metric, verify the metric implementation itself is correct; an
  eval bug silently caps how much any downstream improvement can show up in the numbers.
- **Effort/risk:** Low effort (already implemented, just needs a run — ~1-2 hrs GPU time per
  `Project_Note.md`). No risk; this is a correctness fix already written.

### 2. [Critical] Validate the paper-scoped retrieval fix against the contamination failure mode before trusting any other number

- **Target metric(s):** Context recall (currently 0.300, 68% zero-rate), answer correctness
  (0.393, 43% zero-rate), ALCE recall (indirectly, since uncited/uncitable claims can't be
  supported by the wrong paper's chunks).
- **Evidence:** Dataset rows 3, 11, 30, 36 (data/evaluation_dataset.csv) each retrieve 10/10
  chunks from a paper with a different title than the question's source paper, despite
  `crag_triggered=False` and rerank scores in the 19.7–25.7 range (not near the CRAG Incorrect
  threshold of 8.0, so CRAG's own signal doesn't catch this). This is exactly the failure mode
  the uncommitted diff to `src/retrieval/hybrid_retriever.py` and `src/run_rag.py` targets: the
  diff replaces "retrieve top-k globally, filter to `paper_id`, pad with an unfiltered retry if
  short" with "restrict the FAISS/BM25 candidate pool to `paper_id` *before* ranking, no
  unfiltered retry." The diff's own commit message on `run_rag.py` cites a measured gap
  (context_recall 0.038 / answer_correctness 0.125 for cross-paper rows vs. 0.373 / 0.468 for
  paper-scoped rows) that matches the shape of this report's aggregate numbers.
- **Recommendation:** After re-running per Finding #1, specifically check whether rows like 3,
  11, 30, 36 (re-identifiable by question text) now retrieve from the correct paper. If any still
  don't, the paper_id used for filtering (`row["id"]` from QASPER, `generate_predictions.py:117`)
  may not match the `paper_id` key stored in the dense/sparse index metadata — verify with a
  direct equality check between `qa_pairs[i]["paper_id"]` and a sample of `metadata_list[j]["paper_id"]`
  from `data/indices/dense.index.meta` before assuming the fix is sufficient.
- **Expected mechanism:** Paper-scoped pre-filtering removes the possibility of a top-100 global
  ranking being dominated by the other ~887 indexed papers, which is what let contamination
  through even at good rerank scores — a well-matched chunk from the *wrong* paper can still
  outscore a poorly-matched chunk from the *right* paper.
- **Source grounding:** Huyen's *AI Engineering* treatment of RAG failure analysis separates
  retrieval failures from generation failures precisely so that a generation-quality intervention
  (prompt tuning, few-shot curation) isn't applied to a problem retrieval created — which is the
  trap this project would otherwise fall into by tuning `configs/prompts.yaml` against a
  contaminated dataset.
- **Effort/risk:** Already implemented in the working tree; effort is validation, not new code.
  Risk: the `IDSelectorBatch`/`SearchParameters` FAISS API path is new and untested against a
  paper with very few indexed chunks — confirm `min(k, len(rows))` doesn't silently starve short
  papers below what CRAG's `consistency_ratio=0.3` needs to trigger `Correct`.

### 3. [High] Calibrate the CRAG thresholds after Finding #2 lands — the Ambiguous/Incorrect paths currently make quality worse, not better

- **Target metric(s):** Context recall, answer correctness, on the ~10% of rows that trigger CRAG.
- **Evidence:** `crag_triggered=True` rows (15/150) average `context_recall=0.0` and
  `answer_correctness=0.20`, vs. 0.324 and 0.41 for non-triggered rows
  (`data/evaluation_dataset.csv` × `data/evaluation_report.csv`, grouped by `crag_triggered`).
  The thresholds (`correct_threshold=14.0`, `ambiguous_threshold=8.0`,
  `consistency_ratio=0.3` — `src/retrieval/crag_evaluator.py:70-93`) were set by inspection
  ("calibrated for scientific text... tune empirically") rather than by the calibration script
  that already exists in this repo (`src/evaluation/calibrate_crag.py`).
- **Recommendation:** Once Finding #2's retrieval fix is validated, run
  `python -m src.evaluation.calibrate_crag` against a fresh evaluation report to re-derive
  `correct_threshold`/`ambiguous_threshold` from the corrected score distribution — the current
  thresholds were tuned (implicitly, by inspection) against a distribution that included
  cross-paper contamination, so they may no longer separate Correct/Ambiguous/Incorrect
  correctly once that noise is removed.
- **Expected mechanism:** CRAG's own paper (Yan et al. 2024, AAAI, §3.1) assumes threshold
  calibration against the deployed retriever's actual score distribution; thresholds copied from
  a different corpus/reranker combination, or tuned pre-fix, will misclassify.
- **Source grounding:** Huyen's *AI Engineering* discusses recalibrating pipeline gates whenever
  an upstream component changes — a threshold tuned for one retriever configuration is not
  guaranteed to transfer to a materially different one (pre- vs. post-paper-scoped filtering).
- **Effort/risk:** Low — the calibration script already exists; this is a re-run + threshold
  update in `run_rag.py`'s `--crag-correct`/`--crag-ambiguous` defaults. Low risk.

### 4. [High] The Prometheus 2 judge has not been validated against human labels — treat current scores as directional, not as ground truth

- **Target metric(s):** All Prometheus-scored metrics (context precision/recall, faithfulness,
  answer relevancy, answer correctness) and the ALCE NLI entailment check.
- **Evidence:** Dataset row 0 is a clean counter-example: retrieval is correct (all 10 chunks
  from the right paper), the claim "The seed lexicon is a vocabulary of positive and negative
  predicates" is nearly verbatim in `contexts[0]` ("The seed lexicon consists of positive and
  negative predicates"), and the answer cites `[Doc 2]` correctly for the same fact — yet
  `score_context_recall` (`evaluate_rag.py:402-463`) and the ALCE NLI check both scored this 0.
  `Project_Note.md` documents two prior NLI-prompt bugs found by exactly this kind of manual
  trace inspection (Issues 5 and 6), which is evidence the judge has produced silent false
  negatives before and there is no mechanism currently in place to catch a third one other than
  ad hoc row inspection.
- **Recommendation:** Sample ~20-30 rows stratified across the score range (not just the zeros),
  have a human — or a second, independent, stronger judge such as Claude or GPT-4 at temperature
  0 — label context_recall/faithfulness/ALCE-entailment for each, and compute agreement
  (Cohen's κ or Pearson r) against the current Prometheus 2 scores. Kim et al. (2024, §4) report
  Prometheus 2 achieving r=0.897 with GPT-4 on FeedbackBench's *original* rubrics — that
  correlation has not been re-established for this project's QASPER-specific custom rubrics
  (`evaluate_rag.py:100-228`), which is a materially different distribution.
- **Expected mechanism:** Without this, it's impossible to distinguish "the pipeline got worse"
  from "the judge got noisier" when comparing before/after numbers for Findings #1-#3 — the
  validation gates every other number in this report.
- **Source grounding:** This is close to verbatim Huyen's *AI Engineering* "criteria for a good
  eval" chapter: an AI judge must itself be evaluated against ground truth before its scores are
  used as a proxy for quality; an unvalidated judge's output is not evidence, only a hypothesis.
- **Effort/risk:** Medium — one afternoon of human labeling plus a correlation script. No code
  risk; this doesn't touch the pipeline, only the evaluation harness's credibility.

### 5. [Medium] Migrate `configs/prompts.yaml` generation prompt to a DSPy-compiled module, once Findings #1-#2 give a trustworthy dataset to optimize against

- **Target metric(s):** ALCE citation precision (citation misattribution, citation stuffing),
  faithfulness, answer relevancy.
- **Evidence:** The current prompt (`configs/prompts.yaml:43-83`) is a single hand-written
  template with one static synthetic exemplar, tuned by manual trial-and-error against citation
  format compliance (the file's own comments cite three papers to justify format choices by
  hand). Two concrete generation-side failure modes were found in the error analysis that a
  static one-shot exemplar cannot self-correct: citation misattribution (row 0, `[Doc 3]` cited
  for content actually in Doc 5) and citation stuffing (row 43, all 10 retrieved docs cited
  regardless of relevance).
- **Recommendation:** Define a `dspy.Signature` with typed fields
  (`context: list[str], paper_focus_hint: str, question: str -> reasoning: str, cited_answer: str`)
  and wrap the existing prompt logic in a `dspy.ChainOfThought` module (the current template
  already asks for a `<Reasoning>` block, so this is a direct port, not a redesign). Reuse
  `ALCEEvaluator.calculate_metrics` (`evaluate_rag.py:701-767`) as-is as the DSPy metric function
  — it already returns exactly the (precision, recall) pair DSPy optimizers need. Compile with
  `dspy.MIPROv2` or `dspy.BootstrapFewShot` against a training split of QASPER questions that is
  disjoint from the eval sample (`generate_predictions.py`'s `fetch_qasper_sample(seed=...)` is
  already seeded, so drawing a second disjoint seed for a train split is a one-line change). For
  the citation-stuffing failure mode specifically, wrap the compiled module in `dspy.Refine` with
  a `reward_fn` that checks citation count against the format regex `\[Doc \d+\]` and penalizes
  citing more than ~3 distinct docs per sentence — this directly targets the row-43 pattern
  without hand-writing more negative prompt instructions (the current prompt's "Do NOT use any
  other format such as..." approach is exactly the kind of manual constraint DSPy's reward-based
  `Refine`/`BestOfN` replaces; `dspy.Assert`/`dspy.Suggest` are deprecated as of DSPy 2.6 in favor
  of this reward-function pattern).
- **Expected mechanism:** Optimizing the few-shot exemplar and instruction phrasing directly
  against the ALCE metric (rather than hand-guessing what phrasing improves citation format
  adherence) should raise ALCE precision specifically; `Refine` with a citation-count reward
  should reduce the stuffing pattern seen in row 43.
- **Source grounding:** DSPy (dspy.ai) — Signatures replace hand-written prompt strings with a
  typed contract; teleprompters (`MIPROv2`, `BootstrapFewShot`) compile few-shot exemplars and
  instructions against a metric function instead of manual iteration; `dspy.Refine`/`BestOfN`
  (replacing the deprecated `Assert`/`Suggest`) enforce output constraints via a reward function
  with up to N retries, which fits the citation-format and citation-count constraints already
  expressed as prose in `configs/prompts.yaml`.
- **Effort/risk:** Medium-high — requires adding `dspy` to `requirements.txt` (project already
  isolates dependency conflicts across two venvs per `CLAUDE.md`; DSPy talks to the same vLLM
  OpenAI-compatible endpoint already used for Llama 3.1, so it likely belongs in `.venv`, not
  `.venv-vllm` — verify no `transformers`/`huggingface_hub` pin conflict with the SPECTER2 stack
  before installing), a disjoint train/eval split, and compute budget for compilation (each
  MIPROv2 trial calls the LLM multiple times). Do this **after** Findings #1-#2, since optimizing
  against a contamination-poisoned dataset would just compile citation behavior that
  over-fits to unanswerable, wrong-paper contexts.

### 6. [Low] Re-validate `max_precision_contexts=3` and the ColBERT top-10 rerank cutoff once retrieval is fixed

- **Target metric(s):** Context precision.
- **Evidence:** `run_prometheus_metrics` (`evaluate_rag.py:851-873`) scores context precision on
  only the top-3 reranked chunks (of the 10 that reach generation) to bound API calls, justified
  by "Lost in the Middle" (Liu et al. 2023). This is a reasonable cost/coverage trade-off, but it
  was tuned against the same contamination-affected dataset as everything else in this report.
- **Recommendation:** Not urgent — re-check only after Finding #2 lands, since a correct
  paper-scoped top-3 will look different from a contamination-affected top-3. No change needed
  unless the post-fix numbers show top-3 precision diverging materially from a full top-10 spot
  check.
- **Expected mechanism:** N/A until re-measured.
- **Source grounding:** Liu et al. (2023) as already cited in the code; no new grounding needed.
- **Effort/risk:** Low effort, low priority — listed for completeness, not because it's currently
  the binding constraint on any metric.

---

## Summary of Priority Order

1. Re-run the pipeline end-to-end (Finding #1) — the current ALCE numbers are provably an
   artifact, not a measurement of the system.
2. Validate the paper-scoped retrieval fix (Finding #2) — this is the single largest lever on
   context recall (0.30 → likely much higher) and answer correctness (0.39 → likely much higher),
   and it's already written, just unvalidated.
3. Recalibrate CRAG thresholds against the post-fix distribution (Finding #3).
4. Establish judge validity before trusting any of the above deltas as real (Finding #4) — this
   should ideally run in parallel with #1-#3, not strictly after.
5. Only then invest in DSPy-based prompt optimization (Finding #5) — optimizing generation
   against a retrieval-contaminated or judge-unvalidated dataset risks compiling a program that
   looks better on paper without being better.
