# Project Notes — RAG for Scientific QA (QASPER)

Record of bugs found, fixes applied, and evaluation design decisions.
Environment: Lightning Studio, NVIDIA L40S (46 GB VRAM), Python 3.10.19 (.venv).

---

## Issue 1 — SyntaxError in `chunking.py`: git merge conflict markers

**Date:** 2026-05-05
**File:** `src/retrieval/chunking.py`, line 16
**Symptom:**
```
SyntaxError: invalid syntax
  <<<<<<< HEAD
```
`python -m src.pipeline_ingest` aborted immediately on import.

**Root cause:** A git merge between HEAD and commit `cc6e01a` left unresolved
conflict markers (`<<<<<<< HEAD`, `=======`, `>>>>>>> cc6e01a...`) inside the
`QasperChunker.__init__` method. Both sides of the conflict were identical
(only a trailing blank line differed), so the merge could have been
auto-resolved but was not.

**Fix:** Removed all three conflict markers and kept a single clean version of
the two-line block:
```python
# Prevent HuggingFace from throwing console warnings when calculating length
self.tokenizer.model_max_length = int(1e30)
```

---

## Issue 2 — PermissionError in `run_pipeline_ingest.sh`: `/scratch` does not exist

**Date:** 2026-05-05
**File:** `src/run_pipeline_ingest.sh`, line 88 (original)
**Symptom:**
```
PermissionError: [Errno 13] Permission denied: '/scratch'
```
Raised at `datasets/builder.py` when `load_dataset("allenai/qasper")` tried
to create its cache directory.

**Root cause:** The script unconditionally set:
```bash
export HF_HOME="/scratch/$USER/.cache/huggingface"
```
`/scratch` is a SLURM cluster path. It does not exist in Lightning Studios, so
`os.makedirs` failed before any dataset I/O could happen.

**Fix:** Replaced the hardcoded path with a runtime check:
```bash
if [ -d "/scratch/$USER" ]; then
    export HF_HOME="/scratch/$USER/.cache/huggingface"
else
    export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
fi
mkdir -p "${HF_HOME}"
```
The same pattern was applied to `run_evaluation.sh`, where the existing
free-space check correctly fell through to `${HOME}` but was changed to
`${PROJECT_ROOT}/.cache/huggingface` for consistency and to avoid writing to
the home quota on shared clusters.

---

## Issue 3 — `run_evaluation.sh`: venv not activated before Python calls

**Date:** 2026-05-05
**File:** `run_evaluation.sh`
**Symptom:**
```
ModuleNotFoundError: No module named 'datasets'
```
(and similar for other project packages)

**Root cause:** The original script activated the Conda environment
(`conda activate qasper-rag`) but the project now uses a `.venv` managed by
`uv`. When run interactively (not via `sbatch`), the conda activation silently
failed if conda was not initialized in the current shell, leaving the system
Python in use.

Additionally, `PROJECT_ROOT` was computed *after* `source .venv/bin/activate`
in `run_pipeline_ingest.sh`, meaning the activate path was relative to
whatever directory the shell was in — not reliably the project root.

**Fix (both scripts):**
1. Moved `PROJECT_ROOT` detection to the top of the script (before any
   environment activation).
2. Replaced `conda activate` with an absolute-path venv activation:
```bash
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"
if [ ! -f "${VENV_ACTIVATE}" ]; then
    echo "ERROR: virtualenv not found at ${VENV_ACTIVATE}"
    echo "       Run: uv venv && uv pip install -r requirements.txt"
    exit 1
fi
source "${VENV_ACTIVATE}"
```

---

## Issue 4 — `run_evaluation.sh`: `vllm` not installed, server launch fails

**Date:** 2026-05-05
**File:** `run_evaluation.sh`
**Symptom:**
```
ModuleNotFoundError: No module named 'vllm'
ERROR: Llama 3.1-8B server (PID 43476) exited unexpectedly.
```

**Root cause:** `vllm` is not installed in `.venv`. Installing it via `uv`
would force-upgrade `transformers` from 4.51.3 → 5.8.0, breaking the rest of
the pipeline (SPECTER2 embeddings, ColBERT reranker, Prometheus 2 loader all
depend on the pinned version).

**Fix:** Added a runtime `vllm` availability check and fallback:
```bash
HAVE_VLLM=false
if python -c "import vllm" 2>/dev/null; then
    HAVE_VLLM=true
fi
```
When `HAVE_VLLM=false`, `GENERATOR_BACKEND` is set to `transformers` and all
vLLM server start/stop phases are skipped. The L40S (46 GB VRAM) can run
Llama 3.1-8B (≈16 GB) via HuggingFace `transformers` directly with no
throughput penalty for a 150-question evaluation set. All three
server-dependent blocks (`PHASE 1`, `PHASE 3`, cleanup) are gated on
`USE_GPU=true AND HAVE_VLLM=true`.

---

## Issue 5 — `evaluate_rag.py` Fix 1: ALCE collapses to 0.012 (NLI prompt bug)

**Date:** 2026-05-06
**File:** `src/evaluation/evaluate_rag.py`, `check_nli_entailment()`
**Symptom:** ALCE Citation Precision/Recall/F1 all reported as ~0.012 (135/138
non-refusal rows at 0.000). Previous evaluation reported ALCE F1 = 0.853.
131/150 answers contained valid `[Doc N]` citations; the citations exist but
every NLI entailment check returned False.

**Root cause:** The method used the Prometheus 2 `ABSOLUTE_PROMPT` format
(Kim et al. 2024, arXiv:2405.01535, Figure 1) with the `response` field set
to a meta-statement:
```python
response=f'Evaluating whether the document entails: "{clean_claim[:200]}"'
```
The ABSOLUTE_PROMPT format expects `response` to be **the student model's
actual answer to be graded**. Passing a restatement of the question confused
Prometheus 2, which consistently returned scores of 1–2. With
`NLI_THRESHOLD = 4`, every check returned `False`, making every cited sentence
appear unsupported and collapsing ALCE to zero.

**Fix:** Replaced the ABSOLUTE_PROMPT-based call entirely with a direct
binary True/False NLI prompt sent via `_generate()`:
```python
prompt = (
    "You are an NLI judge for scientific text. "
    "Determine whether the document passage below supports the claim. ..."
    "Does the document passage support the claim? "
    "Answer with exactly one word: True or False."
)
text = self._generate(prompt).strip()
if re.search(r"\bTrue\b", text, re.IGNORECASE): return True
if re.search(r"\bFalse\b", text, re.IGNORECASE): return False
```
`NLI_THRESHOLD` class attribute removed (no longer used).

**Justification:**
- Gao et al. (2023) EMNLP, §4.1: ALCE treats entailment as binary —
  a cited passage either supports the claim or it does not.
- Honovich et al. (2022) "TRUE: Re-evaluating Factual Consistency Evaluation"
  (NAACL 2022): the canonical ALCE NLI model is a binary classifier, not a
  5-point rubric grader.
- Kim et al. (2024) §3: ABSOLUTE_PROMPT is designed for grading a response
  against a reference answer, not for binary entailment decisions.

---

## Issue 6 — `evaluate_rag.py` Fix 2: Context truncated to 500 chars causes Context Recall = 0.062

**Date:** 2026-05-06
**File:** `src/evaluation/evaluate_rag.py`, `score_context_recall()`,
`score_faithfulness()`, `check_nli_entailment()`
**Symptom:** Context Recall dropped from 0.4715 (RAGAS) to 0.0616 (Prometheus).
102/134 valid rows scored 0.0 (raw Prometheus score = 1: "fact completely
absent"). Faithfulness dropped from 0.6583 to 0.3097; 69/134 scored 0.0.

**Root cause:** All three methods truncated each context passage to 500
characters (`c[:500]`). Measurement of the actual `evaluation_dataset.csv`:
- Mean chunk length: 660 chars, median: 594 chars, max: 2936 chars
- **64% of all chunks exceeded 500 chars** — the truncation cut the majority
  of passages mid-way, discarding the second half where the ground-truth fact
  often appears
- Only 2.4% of chunks exceed 1500 chars

When Prometheus 2 received a truncated passage without the target fact, it
correctly scored the passage as unhelpful (score 1 → normalized to 0.0). This
was an evaluation artefact, not a retrieval quality issue.

**Fix:** Raised the per-passage character limit from 500 → 1500:
- `score_context_recall`: `c[:500]` → `c[:1500]`
- `score_faithfulness`: `c[:500]` → `c[:1500]`
- `check_nli_entailment`: `cited_text[:1000]` → `cited_text[:1500]`

**Justification:**
- Es et al. (2023) "RAGAS" arXiv:2309.15217, §3.2–3.3: RAGAS passes full
  context strings to the evaluator LLM without truncation.
- Liu et al. (2023) "Lost in the Middle" ACL 2023, §3: LLM recall degrades
  for facts positioned in the middle or end of a truncated window — exactly
  what `c[:500]` caused.
- At 1500 chars, 97.6% of chunks are captured completely.

---

## Issue 7 — `evaluate_rag.py` Fix 3: Context Recall uses only 6 of 10 reranked passages

**Date:** 2026-05-06
**File:** `src/evaluation/evaluate_rag.py`, `score_context_recall()`
**Symptom:** Minor contributor to low Context Recall. The reranker returns 10
passages but only 6 were sent to Prometheus for recall evaluation.

**Root cause:** `contexts[:6]` in `score_context_recall`. If the passage
containing the ground-truth fact ranked 7th–10th, it was silently excluded
from the recall computation.

**Fix:** Changed `contexts[:6]` → `contexts[:10]`.

**Justification:** Es et al. (2023) §3.2: "we check whether the ground truth
can be attributed to the union of all retrieved passages." The full reranked
set should be used.

---

## Evaluation Design Decision: Prometheus 2 vs RAGAS

**Date:** 2026-05-06

**Decision:** Prometheus 2 (`prometheus-eval/prometheus-7b-v2.0`) is the
appropriate evaluator for this QASPER-based project. RAGAS was the wrong tool.

**Reasoning:**

1. **RAGAS is reference-free by design.** Es et al. (2023) §1 explicitly
   target scenarios with no ground-truth answers. QASPER (Dasigi et al. 2021,
   NAACL) provides annotator-verified reference answers for every question.
   Using a reference-free evaluator on a benchmark with gold references
   discards the most valuable signal.

2. **RAGAS Answer Relevancy uses cosine similarity, not semantic reasoning.**
   It generates synthetic questions from the answer and measures embedding
   cosine similarity to the original question. It cannot distinguish
   "the model achieves ~90%" from "the model achieves 93.2%" — a critical
   failure for QASPER's short numerical answers.

3. **Prometheus 2 is reference-based and can reason.** Its ABSOLUTE_PROMPT
   format takes the QASPER reference answer as `reference_answer` and grades
   the generated answer against it. Kim et al. (2024) Table 2: Prometheus 2
   achieves Pearson r = 0.897 with GPT-4 judgments on reference-based
   evaluation tasks (FeedbackBench).

4. **Answer Correctness is the primary QASPER metric.** Dasigi et al. (2021)
   §4 define evaluation as token-level F1 and exact match against reference
   answers. This metric did not exist in the previous RAGAS evaluation.
   Prometheus 2 adds it explicitly.

5. **RAGAS components worth preserving (future work):**
   - Statement-level faithfulness decomposition (Es et al. 2023 §3.3):
     decompose answer into atomic claims, check each individually. More
     reliable than the current holistic Prometheus call.
   - Binary NLI for ALCE (already fixed in Issue 5 above).

---

## How to Re-run Evaluation Without Regenerating Predictions

The existing `data/evaluation_dataset.csv` (generated 2026-05-05 23:28:19)
is clean: 150 rows, 0 errors, 0 empty-context rows. Since all three fixes
(Issues 5, 6, 7) are in `evaluate_rag.py` only, predictions do not need to
be regenerated. Running just the evaluator saves ~1–2 hours of Llama inference.

```bash
# Re-evaluate existing predictions with fixed evaluate_rag.py:
.venv/bin/python -m src.evaluation.evaluate_rag

# If the evaluation crashes mid-way (e.g. OOM), resume without re-running ALCE:
.venv/bin/python -m src.evaluation.evaluate_rag --skip-alce

# Full re-run (regenerates predictions + evaluates) — only needed if
# the RAG pipeline itself changes:
bash run_evaluation.sh
```
