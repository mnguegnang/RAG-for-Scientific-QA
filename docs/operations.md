# Operations

Deployment notes and recovery steps. For installation, see the
[README](../README.md#installation).

## RunPod and HuggingFace downloads

> **RunPod note:** `/` is a small container overlay and `/workspace` is the persistent volume.
> `run_evaluation.sh` pins `HF_HOME`, `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`,
> `TRITON_CACHE_DIR`, and `NLTK_DATA` under `/workspace` so caches survive a pod restart, and
> sources `/workspace/.bashrc_custom` (skipped by non-interactive shells otherwise).

> **`hf_transfer` is required** wherever `HF_HUB_ENABLE_HF_TRANSFER=1` is set (the RunPod image
> sets it by default) — without the package, downloads fail with a misleading
> `Unrecognized model ... no model_type key` error. It's pinned in `requirements.txt`; install it
> into `.venv-vllm` too if that env downloads models directly.

## Fixing failed prediction rows

Re-generates only the rows in `evaluation_dataset.csv` whose answer is a `System Error:`/`Error:`
string (e.g. after a transient Ollama/vLLM outage), scoped to the same `paper_id` as the original
row so retries can't pull in another paper's passages:

```bash
python -m src.evaluation.generate_predictions --fix-errors
```
