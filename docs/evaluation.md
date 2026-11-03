# Evaluation details

Metric definitions, pipeline steps and supporting scripts. For the commands and results, see the
[README](../README.md#evaluation).

## Metrics

- **Prometheus 2** (Kim et al. 2024, [arXiv:2405.01535](https://arxiv.org/abs/2405.01535)) — an
  LLM judge fine-tuned specifically for evaluation, scored with the ABSOLUTE_PROMPT rubric format
  (raw 1–5 score normalized to `[0, 1]`). Auto-selects a backend: vLLM server → HF `transformers`
  pipeline → Ollama (CPU fallback).
  - *Context Precision* — relevance of the 3 best-ranked retrieved chunks
  - *Answer Relevancy* — how completely the answer addresses the question
  - *Answer Correctness* — factual match against the QASPER reference answer

- **Sentence-level support checks** (`src/evaluation/grounding.py`, `--grounding-checker`):
  - *Context Recall* — fraction of ground-truth sentences supported by the retrieved chunks
  - *Faithfulness* — fraction of answer sentences supported by the retrieved chunks
  - The default `hybrid` checker uses Prometheus True/False prompts for context recall and
    **MiniCheck-Flan-T5-Large** ([Tang et al. 2024](https://arxiv.org/abs/2404.10774)) for
    faithfulness and ALCE — the per-metric split that agreed best with independent labels
    (`compare_grounding_checkers.py`). Both metrics see every chunk the generator saw.

- **[ALCE](https://github.com/princeton-nlp/ALCE)** (Gao et al. 2023, EMNLP) — citation-level
  grounding via NLI entailment (decided by the support checker above):
  - *Citation Precision* — fraction of citations that are correct (sufficient or necessary)
  - *Citation Recall* — fraction of answer sentences entailed by their cited documents

- **Retrieval harness** (`python -m src.evaluation.evaluate_retrieval`) — evidence recall against
  QASPER's gold evidence (text paragraphs and tables/figures) for a grid of context-selection
  configurations, with no LLM or judge calls. Use it to choose `--context-k`, `--final-ranking`
  and `--crag-mode` before a full evaluation run.

## What `run_evaluation.sh` does

The script auto-detects available GPUs (multi-GPU tensor-parallel, single-GPU with a computed
`--gpu-memory-utilization`, or CPU-only via Ollama) and runs two phases sequentially so a single
24 GB GPU (e.g. RTX 4090) is enough:

1. Start Llama 3.1-8B on a vLLM server.
2. `generate_predictions.py` runs RAG over QASPER → `data/evaluation_dataset.csv`.
3. Stop the Llama server; start `prometheus-eval/prometheus-7b-v2.0` on a vLLM server.
4. `evaluate_rag.py` scores with Prometheus 2 + ALCE → `data/evaluation_report.csv`.
5. Stop the Prometheus server.

Logs: `logs/eval_<jobid>.log` / `logs/eval_error_<jobid>.log` under SLURM.

## CRAG threshold calibration

Finds the F1-maximizing ColBERT score boundary from a completed evaluation report and writes a
diagnostic plot.

```bash
python -m src.evaluation.calibrate_crag                                     # uses data/evaluation_report.csv
python -m src.evaluation.calibrate_crag --eval-csv data/evaluation_report.csv
```

Output: recommended threshold on stdout + `reports/figures/crag_calibration.png`.

> Note: `calibrate_crag.py` collects ColBERT scores without paper scoping, while the pipeline
> ranks within one paper, and it uses the evaluation questions. The current threshold (14.44)
> therefore comes from a different score distribution. `reports/figures/crag_calibration.png`
> dates from the earlier bge-reranker version and does not match the current code.

## Supporting scripts

`src/evaluation/` also has standalone scripts for validating the judges, comparing runs and
tuning the prompt:

| Script | Purpose |
|--------|---------|
| `validate_judge.py` / `compute_agreement.py` | Human-vs-Prometheus agreement (Cohen's kappa) at the atomic-decision level |
| `second_judge_label.py` | Auto-labels the blind validation set with Claude as an independent second judge |
| `validate_retrieval_scoping.py` | Regression check that retrieved contexts stay within the queried paper |
| `validate_precision_cutoff.py` | Checks whether scoring context precision on top-3 vs. top-10 chunks diverges |
| `compile_dspy_prompt.py` | Compiles the generation prompt's few-shot demonstrations with DSPy |
| `compare_grounding_checkers.py` | Kappa of MiniCheck vs. Prometheus 2 against the labelled validation units |
| `evaluate_retrieval.py` | Offline evidence recall vs. QASPER gold evidence (no LLM) |
| `rescore_grounding.py` | Re-scores the support metrics of an existing report with a chosen checker, without re-running generation |
| `compare_reports.py` | Paired per-metric comparison of two evaluation reports, with bootstrap CIs |
