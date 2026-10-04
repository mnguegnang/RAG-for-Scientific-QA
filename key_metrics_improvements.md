# Evaluation Metric Improvements — RAG for Scientific QA

This document records the reviews done to raise the evaluation metrics of the RAG pipeline.
Newest work comes first. Each part is dated and states what it does.

| Part | Date | What it does | Status |
|---|---|---|---|
| [Part 1](#part-1--2026-10-03-literature-grounded-component-review-second-review) | 2026-10-03 | Second review: diagnoses each pipeline component against QASPER gold evidence, proposes six source-backed improvements (P1–P6), implements them, measures their effect, and explains how to test them | Implemented. §1.6 Steps 0–7a run on 2026-10-04 (results in §1.7); RRF, text-only and old-selection variants and the Step 8 decision pending (§1.8) |
| [Part 2](#part-2--2026-09-02-error-analysis-and-evaluation-correctness-review-first-review) | 2026-09-02 | First review: error analysis of the 2026-09-02 evaluation run, measurement bugs, and six ranked fixes (Findings #1–#6) | Findings #4–#6 implemented 2026-09-05 (commit `e76017e`); its metric table predates the paper-scoping fix |

---

## Part 1 — 2026-10-03: Literature-grounded component review (second review)

### 1.1 Scope and baseline

**Scope.** Every pipeline component was re-read (`chunking.py`, `hybrid_retriever.py`,
`reranker.py`, `crag_evaluator.py`, `llm_generator.py`, `configs/prompts.yaml`,
`evaluate_rag.py`). The current outputs (`data/evaluation_dataset.csv`,
`data/evaluation_report.csv`, both from the 2026-09-02 run) were joined to QASPER's own
gold evidence annotations. The join works because `fetch_qasper_sample(150, seed=20260902)`
over the cached train split reproduces the CSV's question order exactly (verified). Proposals
already made in Part 2 (the 2026-09-02 review) are not repeated: the re-run, paper-scoping
validation, CRAG *threshold* recalibration, judge validation, DSPy compilation and the
top-3 precision cutoff. Each proposal below was kept only if (a) the data shows the
component is a binding constraint here, and (b) a source shows that the fix works in a
setting close to this one: single-document scientific QA with an 8B generator. Methods that
need an external corpus or web search, or that fine-tune the generator, were left out on
purpose.

**Baseline (current `evaluation_report.csv`, 128 scored rows):** context precision 0.564 ·
context recall 0.390 · faithfulness 0.716 · answer relevancy 0.588 · answer correctness
0.467 · ALCE P/R/F1 0.654 / 0.674 / 0.656. (The metric table in Part 2 predates the
paper-scoping fix and is stale.)

### 1.2 Diagnostic evidence (D1–D6) from the 2026-09-02 evaluation run

| # | Finding | Numbers (n = 150 sampled questions) |
|---|---|---|
| D1 | **40% of questions need a table or figure, and neither is indexed.** `QasperChunker.process_paper` (`chunking.py:49-58`) reads only `full_text`. `figures_and_tables` (and `abstract`) never reach the index. | 60/150 have `FLOAT SELECTED` gold evidence (26 float-only, 34 float+text). Context recall: float-only **0.094**, float+text 0.241, text-only 0.494. 7 of the 22 refusals are float-only. |
| D2 | **Stage 1 has no effect on what reaches the reranker.** With paper-scoped search and `k=100` (`run_rag.py:143`), 94% of sampled papers have ≤ 100 chunks (median 47.5), so dense + BM25 + RRF return the *whole paper*. The final order is ColBERT alone (`reranker.py:105`). The RRF scores (`hybrid_retriever.py:240`) are discarded, and so is HyDE, an extra LLM call on 75% of questions. | Gold text evidence reaches the final context in 72% of short (< 10 words) questions vs 87% of longer ones. |
| D3 | **CRAG removes gold evidence instead of correcting it.** | Gold text evidence in final context: 85% when CRAG is not triggered vs **42%** when triggered (39/150 rows). Context recall 0.231 for rows left with ≤ 3 docs vs 0.441 with 10. 13/22 refusals are CRAG-triggered (median 3 docs left). |
| D4 | **The faithfulness judge sees only `contexts[:5]` (`evaluate_rag.py:495`), but the generator sees and cites up to 10.** | 27% of answers cite Doc 6-10. Faithfulness 0.518 on those rows vs 0.803 on the rest, even though their answer correctness is *higher* (0.526 vs 0.441). |
| D5 | **Prometheus' True/False calls agree only weakly with the second judge** (`judge_validation_blind_sonnet.csv` vs `judge_validation_key.csv`, 132 units). | Cohen's κ: ALCE entailment **0.229** (11 false negatives vs 4 false positives), faithfulness **0.292** (2 FN vs 10 FP), context recall 0.565. |
| D6 | **Answers are about 3× longer than QASPER references, and longer answers score lower on relevancy.** | Median answer 44 words vs median GT 12. Answer relevancy by length quartile: 0.727 / 0.537 / 0.554 / 0.586. 8/22 refusals had the gold paragraph *in context*. |

How D1-D3 were measured: an evidence paragraph counts as "in context" when its first 120
normalized characters appear in the saved contexts. This is a conservative proxy. It misses
paragraphs split across chunks and strips that CRAG rewrote.

---

### 1.3 Proposals P1–P6 (ranked, with sources)

#### P1. [Critical, evaluation] Make the grounding metrics see what the generator saw, and check them with a purpose-built verifier

- **Targets:** faithfulness, ALCE precision/recall/F1 (and context recall through the same checker).
- **Change:** (a) one line: score faithfulness over *all* contexts passed to the generator,
  not `contexts[:5]` (`evaluate_rag.py:495`). (b) Replace the Prometheus True/False prompt in
  `check_nli_entailment`, `_score_faithfulness_items` and `_score_context_recall_items` with
  **MiniCheck** (`MiniCheck-Flan-T5-Large`, 770M; or `Bespoke-MiniCheck-7B` served by the
  existing vLLM setup). Keep Prometheus 2 for the rubric metrics (relevancy, correctness,
  precision); those use the ABSOLUTE_PROMPT format it was trained on.
- **Why this is right here:** (a) RAGAS defines faithfulness against the retrieved context
  the answer was generated from. Truncating it to 5 docs turns every correct citation of
  Doc 6-10 into an "unsupported" claim, which is what D4 shows. (b) The project's κ = 0.23 on
  entailment is far below what ALCE's own NLI reached against humans (κ = 0.698 recall /
  0.525 precision). Prometheus 2 was trained for rubric grading, not binary NLI. MiniCheck is
  trained specifically to check whether a sentence is grounded in a document. It handles
  claims that combine several sentences, splits long documents into chunks itself, and
  matches GPT-4 on LLM-AggreFact at about 400× lower cost. It also beats the T5-11B TRUE
  model that ALCE uses (74.7 vs 61.0 BAcc).
- **Expected effect:** ALCE recall and precision rise, because the current judge's errors
  are mostly false negatives (11 FN vs 4 FP). Faithfulness rises on the 39 rows that cite
  Doc 6-10. Its net change elsewhere may be downward, because the current judge
  *over*-accepts faithfulness claims (10 FP), and that correction is the point. Re-run
  `compute_agreement.py` with the new checker. Adopt it only if κ improves.
- **Effort/risk:** Low (a) / medium (b). Check MiniCheck's dependency pins against the
  `.venv` constraints in `CLAUDE.md` before installing. The 7B variant belongs behind vLLM,
  like Prometheus.
- **Sources:** RAGAS: Es et al. 2023, §3 "Faithfulness" ([arXiv:2309.15217](https://arxiv.org/abs/2309.15217)).
  ALCE: Gao et al. 2023, §3.3 (citation recall/precision via NLI), App. C (TRUE model),
  §6 (κ vs humans) ([arXiv:2305.14627](https://arxiv.org/abs/2305.14627)).
  MiniCheck: Tang, Laban & Durrett, EMNLP 2024, Abstract, Table 2 (BAcc per dataset,
  including the RAG sets ClaimVerify / ExpertQA / LFQA), Table 3 (size and cost)
  ([arXiv:2404.10774](https://arxiv.org/abs/2404.10774), code
  [github.com/Liyan06/MiniCheck](https://github.com/Liyan06/MiniCheck), leaderboard
  [llm-aggrefact.github.io](https://llm-aggrefact.github.io)).
  TRUE: Honovich et al. 2022 ([arXiv:2204.04991](https://arxiv.org/abs/2204.04991)).

#### P2. [Critical, ingestion] Index tables and figures (captions first, then table content)

- **Targets:** context recall, answer correctness and refusal rate on the 60/150
  table/figure questions.
- **Change:** in `QasperChunker.process_paper`, add one chunk per entry in
  `paper_data['figures_and_tables']['caption']`, using the existing contextual prefix
  (`Title: … Section: Table 3.`). Then add **table content**. QASPER papers were selected
  *because* they have arXiv LaTeX sources, so `tabular` environments can be pulled from the
  arXiv source, turned into Markdown rows, and stored with their caption (≤ 500 tokens, same
  as text chunks). Index `abstract` as one more chunk while there. Rebuild dense and sparse
  indices together (`CLAUDE.md`).
- **Why this is right here:** 72% of the float-question references are numeric. Only 15% of
  their reference tokens appear in the caption, and only 6.7% have ≥ half their tokens there.
  So captions alone give the generator something to cite but rarely the number. The table
  body is what moves answer correctness. QASPER's own baselines *excluded* these questions
  ("leave multimodal QA to future work"), so no text-only pipeline has a claim on them yet.
  SPIQA built a QASPER-derived test set of exactly these questions and shows that captions
  and accompanying text carry much of the signal.
- **Expected effect:** largest single lift available for context recall. Float-only rows
  sit at 0.094, against 0.494 for text rows.
- **Effort/risk:** Captions: low (the field is already in the cached dataset). Table bodies:
  medium (LaTeX parsing; respect arXiv's access rate limits for 887 papers). Risk: caption
  chunks are short and may rank highly under MaxSim for generic questions. Check this with
  the P5 harness.
- **Sources:** QASPER: Dasigi et al., NAACL 2021, §1 (13% of questions need tables/figures,
  55.5% need multiple paragraphs), Table 1 (Table/Figure evidence 11.6%), §2.1 (papers
  restricted to arXiv with LaTeX source; figure/table images crawled separately), §5 (table
  questions excluded from baselines) ([arXiv:2105.03011](https://arxiv.org/abs/2105.03011)).
  SPIQA: Pramanick et al., NeurIPS 2024 D&B, §3.2 (test-C: 493 QASPER questions where
  figures/tables are essential), Fig. 3 / §5.3 (captions matter)
  ([arXiv:2407.09413](https://arxiv.org/abs/2407.09413)).

#### P3. [High, CRAG] Under paper-scoped retrieval, make CRAG flag weak evidence instead of deleting it

- **Targets:** context recall, answer correctness and the refusal rate on the 39 CRAG-triggered rows.
- **Change:** this is not threshold recalibration (already done, Finding #3 below). It
  changes what the actions *do*:
  1. Use CRAG's own action rule: `Correct` if **at least one** document clears the upper
     threshold. Drop the extra `correct_ratio >= 0.3` requirement
     (`crag_evaluator.py:161`), which pushes rows into `Ambiguous`.
  2. Never drop in-paper chunks for scoring below a ColBERT threshold. Keep the top-k and
     move low-confidence ones to the end (or drop them only beyond k).
  3. If strip refinement is kept, score strips with the **same ColBERT model** and
     recompose them in their original order. Do not use whitespace word overlap without
     stemming (`crag_evaluator.py:195, 267`); it drops "models" when the query says "model".
- **Why this is right here:** in CRAG, `Ambiguous` means *combine* refined internal
  knowledge with *external* (web) knowledge, and `Incorrect` means *replace* it with web
  results. This pipeline has no external source and is correctly scoped to the answer's
  paper, so every removal is a pure recall loss, which D3 shows (gold evidence kept in 42% vs
  85%). CRAG also reports that the method's efficacy "was easily affected by the accuracy of
  the retrieval evaluator": its fine-tuned T5 judged relevance at 84.3% vs 58-65% for
  ChatGPT. The ColBERT-threshold evaluator here reached only F1 = 0.554 at calibration
  (`run_rag.py` CLI comment), too weak to drive deletion.
- **Caveat:** CRAG triggers on hard questions, so part of the gap is confounding. Measure
  the ablation (CRAG off / signal-only / current) with the P5 harness before the full
  Prometheus run.
- **Sources:** CRAG: Yan et al. 2024, §4.3 (action definitions; "Discussion" paragraph on
  evaluator dependence), §4.4 (strips scored by the evaluator, recomposed in order), Table 4
  (evaluator accuracy) ([arXiv:2401.15884](https://arxiv.org/abs/2401.15884)).

#### P4. [High, retrieval → generation] Order-preserving, larger context (OP-RAG)

- **Targets:** context recall, answer correctness, faithfulness on multi-paragraph questions.
- **Change:** after ColBERT ranking, keep the top-k for k ∈ {10, 20, 30} and pass them to
  the generator **in paper order** (section and paragraph index are already in
  `original_para_id`), not in score order. At about 180 tokens per chunk, k = 30 is about
  5.5K tokens.
- **Why this is right here:** QASPER needs evidence from multiple paragraphs for 55.5% of
  text-evidence questions, and only 57% of such rows here have *all* gold paragraphs in
  context. QASPER's oracle experiments show that "the majority of the large headroom … can
  be closed with better evidence selection". OP-RAG shows that keeping document order lets
  answer quality keep rising with more chunks. For **Llama-3.1-8B, this project's generator**,
  the peak was at about 16K tokens, far above the current ~1.8K. Order preservation matters
  most at large k (small gain at k = 8).
- **Counter-evidence (why sweep k rather than jump):** ALCE found that more passages did not
  help ChatGPT, and that correctness plateaued early. The gain depends on the model, so pick
  k empirically.
- **Evaluation coupling:** context recall is scored on `contexts[:10]`
  (`evaluate_rag.py:416`) and precision on the first 3 by position. Persist the rerank rank
  per context so precision stays "top-3 by ColBERT", and score recall over all contexts.
  `[Doc N]` numbering must follow the presented (paper) order.
- **Sources:** OP-RAG: Yu, Xu & Akkiraju 2024, §3 (Eq. 2), §4.3 (context-length sweep,
  Llama-3.1-8B peak at 16K; OP vs vanilla, Fig. 4)
  ([arXiv:2409.01666](https://arxiv.org/abs/2409.01666)). QASPER §3 "Evidence types", §5.2
  "Answer prediction from gold evidence", Table 4. ALCE §5.3 and Table 7 (counter-evidence).

#### P5. [High, retrieval] Make Stage 1 count again: fuse ColBERT with BM25/SPECTER2, tuned on a gold-evidence harness

- **Targets:** context recall and context precision, especially short questions.
- **Change:** (1) Build an offline harness that computes **Evidence Recall@k** against
  QASPER's gold `evidence` paragraphs for the 150 questions, plus a disjoint tuning seed. It
  needs no LLM calls, runs in minutes, and is the same evidence-selection signal QASPER
  scores officially. Use it to settle P2-P4 before any GPU judging. (2) Within the paper,
  fuse ranks from ColBERT, BM25 and SPECTER2 with RRF (`rrf_k=60`, already implemented) for
  the final ranking, instead of ColBERT alone. Keep it only if the harness shows a gain.
  (3) Skip HyDE when the paper has ≤ k chunks (it cannot change the candidate set), and drop
  the arXiv ID from the HyDE prompt (`llm_generator.py:561`); the model cannot map an ID to
  content.
- **Why this is right here:** D2 shows that the hybrid machinery is computed and then
  thrown away. ColBERTv2 is MS MARCO-trained, and on BEIR's scientific sets it only matches
  BM25: SciFact 69.3 vs 66.5, SCIDOCS 15.4 vs 15.8 nDCG@10. When no single ranker dominates,
  rank fusion is the established remedy. Cormack et al. show RRF beats the individual rankers
  it combines.
- **Sources:** QASPER §4.1 (Evidence-F1 as the official evidence metric).
  ColBERTv2: Santhanam et al., NAACL 2022, Table 5a ([arXiv:2112.01488](https://arxiv.org/abs/2112.01488)).
  BEIR: Thakur et al., NeurIPS 2021 D&B, Table 2 (BM25 SciFact 0.665, SCIDOCS 0.158)
  ([arXiv:2104.08663](https://arxiv.org/abs/2104.08663)).
  RRF: Cormack, Clarke & Büttcher, SIGIR 2009, §1 and Tables 2-3 (fusion beats every input run)
  ([PDF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)).

#### P6. [Medium, generation] De-clutter the prompt and target QASPER-length answers

- **Targets:** answer relevancy, ALCE citation recall, refusal rate.
- **Change (in `configs/prompts.yaml` / `_build_prompt`):**
  1. Remove the "Paper Focus: anchor to a single document" step (`prompts.yaml:12`). It was a
     cross-paper-contamination patch that pre-filtering made obsolete, and it tells the model
     to ignore the multi-paragraph evidence most QASPER answers need.
  2. Drop `Source: <paper_id>` and `[relevance: x]` from each block (`llm_generator.py:237`).
     After scoping, every block carries the same ID, and raw MaxSim scores invite the model to
     anchor on the rank.
  3. Ask for the specific fact first, in 1-2 sentences, with names and numbers copied exactly.
     Every sentence must carry a citation.
  4. Narrow the refusal rule: refuse only when no document mentions the entity asked about.
- **Why this is right here:** QASPER reference answers average 14.4 words (extractive) and
  15.6 words (abstractive); ours have a median of 44. The project's own relevancy rubric
  gives a 5 only to "nothing more", and the shortest-answer quartile scores 0.727 vs about
  0.55 for the others. ALCE recall is "supported sentences / all sentences", so uncited
  filler sentences lower it directly. ALCE also reports that comprehensive instructions
  improve citation quality for instruction-tuned models. 8 refusals happened with the gold
  paragraph in context. Note: `evaluate_rag.py:1003` excludes refusals from all means, so
  report answered-coverage next to the means, or fewer refusals can look like a regression.
- **Relation to Finding #5 (DSPy):** this sets the seed program that DSPy then optimizes. It
  is not a substitute.
- **Sources:** QASPER §3 "Answer types" ([arXiv:2105.03011](https://arxiv.org/abs/2105.03011)).
  ALCE §3.3 (citation-recall definition), §5.4 / App. G.2 (instruction ablation).

### 1.4 Recommended implementation order

1. **P1(a)** (one line) and the **P5 harness**. Together they make the next numbers
   trustworthy and cheap to obtain.
2. **P2 captions + P3 + P4.** Decide all three on Evidence Recall@k first, then run a single
   full Prometheus/ALCE evaluation.
3. **P2 table bodies** and **P1(b) MiniCheck**. Each is medium effort and each is validated
   by its own check: evidence recall for P2(b), κ for P1(b).
4. **P6**, then the existing DSPy step (#5) on top of it.

### 1.5 Implementation status and measured effects (2026-10-03)

All six proposals are implemented. Unit tests are in `tests/` (35 passing). End-to-end smoke
tests ran on 5 and 12 questions with the real vLLM Llama and Prometheus servers. Two testing
aids were added: `generate_predictions.py --indices-dir` (run against another index) and
`src/evaluation/compare_reports.py` (paired per-metric deltas with bootstrap CIs).

| # | What was built | Where |
|---|---|---|
| P1 | `GroundingJudge` base class. Recall and faithfulness use **all** contexts; Prometheus prompts are batched so they fit the server window. `MiniCheckJudge` re-implements MiniCheck's reference inference. `--grounding-checker hybrid\|minicheck\|prometheus`. | `src/evaluation/grounding.py`, `evaluate_rag.py`, `compare_grounding_checkers.py` |
| P2 | Abstract, caption and table chunks with `chunk_type` / `position`. Table bodies come from arXiv LaTeX: 809/888 papers have source, and 3,259/3,631 table captions were matched to a body. | `src/retrieval/chunking.py`, `src/data/arxiv_tables.py`, `pipeline_ingest.py` |
| P3 | CRAG modes `signal` (default) / `refine` (ColBERT-scored strips) / `legacy`. The action rule now follows CRAG §4.3. | `src/retrieval/crag_evaluator.py` |
| P4 | `context_k` (default 20), paper-order presentation, token budget, `context_ranks` saved, precision taken on the top-3 *by rank*. | `src/run_rag.py`, `context_selection.py`, `generate_predictions.py` |
| P5 | Offline gold-evidence harness, opt-in RRF final ranking, HyDE skipped when it cannot change the candidates, arXiv ID removed from the HyDE prompt. | `src/evaluation/evaluate_retrieval.py`, `run_rag.py`, `llm_generator.py` |
| P6 | New prompt: no paper-focus step, no ID/score lines, short cited answers, narrow refusal rule, and (added after the smoke test) cite-only-what-states-it. | `configs/prompts.yaml`, `llm_generator.py`, `dspy_module.py` |

**Measured effects so far** (`data/retrieval_eval.csv`, 150 evaluation questions, no LLM):

| Configuration | Evidence recall | Text-evidence hit | Table/figure hit |
|---|---|---|---|
| Old pipeline, old text-only index | 0.485 | 0.766 | 0.000 |
| New defaults, old index (P3+P4) | 0.646 | 0.944 | 0.000 |
| **New defaults, new index (P2+P3+P4)** | **0.829** | 0.944 | **0.700** |
| k=30, new index | 0.897 | 0.968 | 0.817 |

**Where the evidence changed the plan:**

- **P1 became a per-metric split.** Measured against the 132 labelled units, MiniCheck agrees better
  on faithfulness (κ 0.391 vs 0.292) and ALCE entailment (0.284 vs 0.229), but worse on context
  recall (0.430 vs 0.565, n=25). The default checker is therefore `hybrid`: Prometheus for recall,
  MiniCheck for faithfulness and ALCE.
- **P5's RRF stays opt-in.** On the old index it is neutral to slightly worse. On the new index it
  raises the table/figure hit rate but lowers text recall (k=20: 0.831 vs 0.884), so ColBERT-only
  remains the default.
- **CRAG `refine` is not better than `signal`.** It scores 0.4–1.1 points lower on evidence recall,
  using about 15% fewer tokens.
- **k=30 beats k=20 on retrieval** (+6.8 points), but whether Llama-3.1-8B uses the extra context is
  untested (see the ALCE §5.3 counter-evidence). The default stays 20 until an end-to-end
  k=20 vs k=30 comparison is run.
- **New finding from the smoke test: citation stuffing.** With 20 excerpts in view, the model cited
  4–5 docs per sentence and ALCE precision fell to 0.30. A cite-only-what-states-it instruction
  was added to the prompt. In the 12-question re-run the median fell to 2 citations per answer
  (from 4) and the median answer length to 32 words.
- **Two robustness fixes found in the smoke tests.**
  - Repetition loops: when the answer is absent from the excerpts, Llama sometimes loops until
    max_tokens and never writes `<Final Answer>`, and the whole loop was then scored as the answer.
    The vLLM backend now retries such outputs once with `frequency_penalty=0.5`.
  - Prose verdicts: Prometheus sometimes answers in prose ("…is directly supported by Passage 3…"),
    which the parser silently scored as False. The parser now reads supported / not-supported
    phrasing.
- **CUDA out-of-memory in the first full run (2026-10-04).** Step 4 of §1.6 crashed after ALCE,
  at the first faithfulness check. Faithfulness gives MiniCheck all 20 contexts at once; with
  table chunks these inputs reach 2,048 tokens, and batches of 16 needed several GB for
  attention. Meanwhile the Prometheus server held 26.1 of 31.4 GiB. Fixes:
  - MiniCheck batches by `batch_size × length²`, so long inputs go 2 at a time. It halves the
    batch and retries instead of crashing on an out-of-memory error.
  - MiniCheck runs in bfloat16 by default (`MINICHECK_DTYPE`). Validated on the same data: 0 of
    131 labelled decisions flip (max probability change 0.025), and faithfulness is identical on
    all 150 rows. Peak memory is 3.2 GiB reserved, and the check runs 2.3× faster.
  - `run_evaluation.sh` now reserves 7 GiB next to the Prometheus server (was 6 GiB).
  - `SKIP_GENERATION=1` reuses the saved predictions, and `EVAL_ARGS` passes flags such as
    `--skip-alce` to `evaluate_rag`, so a crashed evaluation can resume without regenerating
    or re-running ALCE.


### 1.6 How to test and see the effect

The aim is to tell *system* improvements apart from *measurement* changes. P1 changed the
evaluator itself, so old and new predictions must be scored by the same evaluator before they
are compared. Every run below uses the same 150 questions (fixed seed), so reports can be
paired question by question. Each full run (`run_evaluation.sh`) takes about 1–2 GPU-hours,
mostly judging.

**Step 0 — Environment and unit tests (about 3 min).**
```bash
cd /workspace/RAG-for-Scientific-QA && source /workspace/.bashrc_custom
export HF_TOKEN=hf_xxx                       # run_evaluation.sh checks gated-model access
.venv/bin/python -m pytest tests -q          # expect 35 passed
```

**Step 1 — Freeze the baseline.** `run_evaluation.sh` overwrites `data/evaluation_dataset.csv`
and `data/evaluation_report.csv`, which still hold the 2026-09-02 baseline.
```bash
mkdir -p data/runs
cp data/evaluation_dataset.csv data/runs/baseline_dataset.csv
cp data/evaluation_report.csv  data/runs/baseline_report.csv
```

**Step 2 — Retrieval effect, no LLM (about 5 min per index).** Results for the evaluation
questions are already in `data/retrieval_eval.csv` (tags `text_only` and `tables`). Confirm the
choices on 150 *disjoint* questions, so that settings are not tuned on the evaluation set:
```bash
.venv/bin/python -m src.evaluation.evaluate_retrieval --split tune --tag tables_tune
```
Look at `all_recall` (all gold evidence), `text_hit` and `float_hit` (tables/figures). Expect
the same ordering as in §1.5: new defaults far above `baseline`, k=30 above k=20, and
`legacy` CRAG lowest.

**Step 3 — Judge check (about 5 min).** Reproduces the κ table behind the `hybrid` checker:
```bash
.venv/bin/python -m src.evaluation.compare_grounding_checkers   # -> data/grounding_checker_comparison.csv
```
If you relabel units by hand (`data/judge_validation_blind.csv`), pass `--labels` with that file
and revisit the per-metric split in `evaluate_rag.py::build_grounding_judges`.

**Step 4 — Full run with the new defaults.**
```bash
bash run_evaluation.sh 2>&1 | tee logs/run_new_k20.log
cp data/evaluation_dataset.csv data/runs/new_k20_dataset.csv
cp data/evaluation_report.csv  data/runs/new_k20_report.csv
```
If the evaluation stops after predictions were generated (e.g. the 2026-10-04 out-of-memory
error), resume it without regenerating. If the log shows "ALCE complete — intermediate results
saved", also skip the ALCE pass:
```bash
SKIP_GENERATION=1 EVAL_ARGS="--skip-alce" bash run_evaluation.sh 2>&1 | tee logs/run_new_k20_resume.log
```

**Step 5 — Re-score the baseline predictions with the new evaluator.** Without this step, a
difference could come from the evaluator rather than the system.
```bash
PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model prometheus-eval/prometheus-7b-v2.0 --port 8011 --max-model-len 8192 \
  --gpu-memory-utilization 0.6 > logs/prometheus_rescore.log 2>&1 &
until curl -s localhost:8011/v1/models | grep -q prometheus; do sleep 10; done
PROMETHEUS_PORT=8011 .venv/bin/python -m src.evaluation.evaluate_rag \
  --input-csv data/runs/baseline_dataset.csv --output-csv data/runs/baseline_report_neweval.csv
kill %1                                      # stop the Prometheus server
```

**Step 6 — Compare.** `compare_reports` pairs the questions both runs answered and gives a 95%
bootstrap confidence interval for each metric's change.
```bash
# System effect (same evaluator on both sides):
.venv/bin/python -m src.evaluation.compare_reports \
  data/runs/baseline_report_neweval.csv data/runs/new_k20_report.csv --labels baseline new
# Measurement effect of P1 alone (same predictions, old vs new evaluator):
.venv/bin/python -m src.evaluation.compare_reports \
  data/runs/baseline_report.csv data/runs/baseline_report_neweval.csv --labels old_eval new_eval
```
How to read the output:
- Trust a change only where `resolved=True`, i.e. its interval excludes 0.
- Read the "questions scored" lines and each report's "Answered coverage" log line alongside the
  means. Refusals are excluded from every mean, so turning refusals into answers can lower a mean
  while improving the system.
- Expected direction: context recall and answer correctness should rise the most (P2 + P4).
  Watch ALCE citation precision, which is sensitive to citation stuffing.

**Step 7 — Attribute the gain to each change.** Run one full evaluation per variant, copying the
outputs as in Step 4, then compare each variant with `new_k20_report.csv`:
```bash
PIPELINE_ARGS="--context-k 30" bash run_evaluation.sh                          # k=30 vs k=20 (P4)
PIPELINE_ARGS="--indices-dir data/indices_text_only" bash run_evaluation.sh   # without tables (P2)
PIPELINE_ARGS="--context-k 10 --context-order rank --crag-mode legacy --max-context-tokens 0" \
  bash run_evaluation.sh                                                       # old selection (P3/P4)
PIPELINE_ARGS="--final-ranking rrf" bash run_evaluation.sh                     # optional: RRF (P5)
```
Each run stores its settings in the `pipeline_config` column of the dataset CSV, so every saved
file records exactly what produced it.

**Step 8 — Decide the defaults.** If k=30 improves answer correctness without hurting ALCE
citation precision, change the `--context-k` default in `run_rag.py::add_pipeline_args` (and
the docs). Then layer the remaining Part 2 work on top: CRAG threshold recalibration
(`calibrate_crag.py`, Finding #3) and DSPy compilation (`compile_dspy_prompt.py`, Finding #5).
Pass it the same context flags you settled on.

---

## Part 2 — 2026-09-02: Error analysis and evaluation-correctness review (first review)

> **Status (as of 2026-10-03).** Kept for history. Findings #4–#6 were implemented on
> 2026-09-05 (commit `e76017e`: judge validation, DSPy compilation, precision-cutoff check), and
> the CRAG thresholds of Finding #3 were recalibrated on 2026-09-02 (see `run_rag.py`). The
> metric table in §2.2 was measured *before* the paper-scoping fix; Part 1 §1.1 has the
> baseline that the 2026-10-03 work starts from.

### 2.1 Scope

Review scope: `src/retrieval/*`, `src/generation/llm_generator.py`, `src/evaluation/*`,
`configs/prompts.yaml`, `data/evaluation_dataset.csv` (150 rows), `data/evaluation_report.csv`
(150 rows), `Project_Note.md`, and the uncommitted working-tree diff (`git diff`).
No source files were modified; this is a findings-and-recommendations report only.

### 2.2 Metric state at the time (data/evaluation_report.csv, n=150, 128 scorable rows)

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

### 2.3 Error analysis

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

### 2.4 Improvement points — Findings #1–#6 (ranked by expected impact)

#### Finding #1 [Critical] Re-run the full pipeline — the current report is measuring a bug the code no longer has

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

#### Finding #2 [Critical] Validate the paper-scoped retrieval fix against the contamination failure mode before trusting any other number

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

#### Finding #3 [High] Calibrate the CRAG thresholds after Finding #2 lands — the Ambiguous/Incorrect paths currently make quality worse, not better

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

#### Finding #4 [High] The Prometheus 2 judge has not been validated against human labels — treat current scores as directional, not as ground truth

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

#### Finding #5 [Medium] Migrate `configs/prompts.yaml` generation prompt to a DSPy-compiled module, once Findings #1-#2 give a trustworthy dataset to optimize against

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

#### Finding #6 [Low] Re-validate `max_precision_contexts=3` and the ColBERT top-10 rerank cutoff once retrieval is fixed

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

### 2.5 Summary of priority order

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
