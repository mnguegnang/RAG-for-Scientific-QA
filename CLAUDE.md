# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Two virtualenvs are required — they cannot be merged.** vLLM 0.28 depends on
`transformers>=5.5` and `huggingface_hub>=1.27`; the SPECTER2 encoder depends on
`adapters`, which pins `transformers~=4.51` and needs the pre-1.0
`huggingface_hub` API (it imports `HfFolder`). No `adapters` release supports
transformers 5.x, so a resolver rejects the combination outright. They never
need to share an environment: the pipeline reaches vLLM only over HTTP and
never imports it. Installing vLLM into `.venv` silently upgrades transformers
and breaks SPECTER2 with a misleading "adapters package is required" error.

```bash
# 1. Project env — RAG pipeline, evaluation, ingestion
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. Server env — vLLM only (Llama 3.1 generation + Prometheus 2 judging)
uv venv --python 3.10 .venv-vllm
uv pip install --python .venv-vllm/bin/python vllm==0.28.0

# FAISS must be installed via conda for GPU support (pip is CPU-only):
#   CPU: conda install -c pytorch faiss-cpu=1.9.0
#   GPU: conda install -c pytorch -c nvidia faiss-gpu=1.9.0 pytorch-cuda=12.1
.venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
export HF_TOKEN="hf_XXXX"   # required for meta-llama/Llama-3.1-8B-Instruct
```

`run_evaluation.sh` picks the interpreter per phase automatically (`RAG_PY` /
`VLLM_PY`) and falls back to `.venv` when `.venv-vllm` is absent.

**Cache placement (RunPod).** `/` is a small container overlay; `/workspace` is
the persistent network volume. `run_evaluation.sh` pins `HF_HOME`,
`VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` and
`NLTK_DATA` under `/workspace` so the container disk never fills and the
torch.compile artifacts (~20 min to rebuild on a new GPU arch) survive a pod
restart. It also sources `/workspace/.bashrc_custom`, which non-interactive
shells skip.

**`hf_transfer` is mandatory here.** The RunPod image exports
`HF_HUB_ENABLE_HF_TRANSFER=1`; if the module is missing from an env, every
HuggingFace download raises — and transformers reports it as an unrelated
`Unrecognized model ... no model_type key` error. It is pinned in
`requirements.txt`; install it in `.venv-vllm` too.

## Key Commands

**Build indices (required before first use):**
```bash
python -m src.data.arxiv_tables        # optional, ~45 min: table bodies from arXiv LaTeX -> data/table_bodies.json
python -m src.pipeline_ingest          # local GPU (uses data/table_bodies.json when present)
sbatch src/run_pipeline_ingest.sh      # SLURM
```

**Ask a question:**
```bash
python -m src.run_rag --query "What encoder architecture does the paper use?"
python -m src.run_rag --query "..." --backend ollama   # CPU / laptop
python -m src.run_rag --query "..." --backend transformers --hf-model meta-llama/Llama-3.1-8B-Instruct
```

**Run evaluation:**
```bash
python -m src.evaluation.evaluate_retrieval     # offline evidence recall vs QASPER gold evidence (no LLM)
python -m src.evaluation.generate_predictions   # writes data/evaluation_dataset.csv (accepts --context-k, --crag-mode, ...)
python -m src.evaluation.evaluate_rag           # writes data/evaluation_report.csv (--grounding-checker hybrid|minicheck|prometheus)
bash run_evaluation.sh                          # end-to-end (RunPod / interactive); PIPELINE_ARGS / GROUNDING_CHECKER env
sbatch run_evaluation.sh                        # end-to-end SLURM job
python -m src.evaluation.calibrate_crag         # CRAG threshold calibration + plot
python -m src.evaluation.compare_grounding_checkers  # kappa of MiniCheck vs Prometheus against labels
python -m src.evaluation.compare_reports A.csv B.csv # paired per-metric deltas with bootstrap CIs
.venv/bin/python -m pytest tests                # unit tests (HF_HOME must point at the model cache)
```

**Environment variables:**
- `HF_TOKEN` — required for the gated LLaMA model (`meta-llama/Llama-3.1-8B-Instruct`)
- `GENERATOR_BACKEND` — override LLM backend: `vllm`, `transformers`, or `ollama`
- `VLLM_API_URL` — Llama vLLM server endpoint (default: `http://localhost:8000/v1`)
- `PROMETHEUS_PORT` — Prometheus 2 vLLM port for evaluation (default: `8001`)
- `GROUNDING_CHECKER` — support-check judge for evaluation: `hybrid` (default), `minicheck`, `prometheus`
- `PIPELINE_ARGS` — extra context-selection flags `run_evaluation.sh` passes to `generate_predictions`
- `SKIP_GENERATION=1` / `EVAL_ARGS` — `run_evaluation.sh` resume controls (reuse predictions; extra `evaluate_rag` flags)
- `MINICHECK_DTYPE` — `bfloat16` (default on GPU) or `float32`

## Architecture

The pipeline is in `src/run_rag.py::ScientificRAGPipeline` and runs four sequential stages:

`ScientificRAGPipeline.retrieve_context()` runs stages 1-3 (usable without an LLM via `load_generator=False`, as the retrieval harness and the DSPy compiler do); `ask()` adds generation. Context-selection settings (`--context-k`, `--context-order`, `--final-ranking`, `--crag-mode`, `--max-context-tokens`) are shared by `run_rag.py`, `generate_predictions.py` and `compile_dspy_prompt.py` via `add_pipeline_args`, and are saved per run in the `pipeline_config` column.

**Stage 1 — Hybrid Retrieval (`src/retrieval/hybrid_retriever.py`)**
Fetches up to 100 candidates (paper-scoped by `filter_paper_id`) by fusing dense FAISS search (SPECTER2 embeddings, `allenai/specter2_base`) and BM25 sparse search using Reciprocal Rank Fusion (`rrf_k=60`). Results carry chunk metadata (`chunk_id`, `section_name`, `chunk_type`, `position`) and each leg's rank (`dense_rank`, `sparse_rank`). 94% of QASPER papers have ≤ 100 chunks, so under paper scoping stage 1 returns the whole paper. Short queries (< 10 words) trigger HyDE for the dense leg only when it can change the outcome (`final_ranking="rrf"` or a paper larger than the candidate pool).

**Stage 2 — Ranking (`src/retrieval/reranker.py`, `src/retrieval/context_selection.py`)**
ColBERT v2 late-interaction (`colbert-ir/colbertv2.0`) via RAGatouille scores every candidate (MaxSim). The final ranking is ColBERT alone (default) or RRF of ColBERT + BM25 + SPECTER2 ranks (`--final-ranking rrf`); the top `context_k` (default 20) are kept, each tagged with `rerank_rank`.

**Stage 3 — CRAG Evaluation (`src/retrieval/crag_evaluator.py`)**
Labels each document `{Correct, Ambiguous, Incorrect}` from its ColBERT score (`correct_threshold=14.44`, `ambiguous_threshold=8.0`); action `Correct` as soon as one document clears the upper threshold (CRAG §4.3). Modes: `signal` (default — label and report, never delete; with paper-scoped retrieval and no web fallback, deletion is pure recall loss), `refine` (Ambiguous docs reduced to the sentence strips that ColBERT scores ≥ the ambiguous threshold, recomposed in order), `legacy` (old filtering + lexical strips + top-5 fallback, for ablations).

**Stage 4 — Generation (`src/generation/llm_generator.py`)**
Kept chunks are trimmed to `--max-context-tokens` (lowest-ranked dropped first) and presented in paper order (order-preserving RAG; `--context-order rank` for the old behaviour). Blocks are `[Doc N] <chunk text>` — no paper ID or scores. `LocalLLMGenerator` supports `transformers`, `ollama` and `vllm` backends. Prompt template from `configs/prompts.yaml` (short cited answers, multi-excerpt synthesis, narrow refusal rule); inline fallback if missing.

**Ingestion (`src/pipeline_ingest.py`, `src/retrieval/`, `src/data/arxiv_tables.py`)**
`QasperChunker` emits, in reading order (`position`): the abstract, every full-text paragraph (500-token chunks, 10% overlap), and one chunk per table/figure caption (`chunk_type` table/figure). When `data/table_bodies.json` exists (built by `src/data/arxiv_tables.py` from the papers' arXiv LaTeX `tabular` environments, cached in `data/arxiv_src/`), table chunks carry the Markdown table body, split row-wise with the caption and header repeated. All chunks keep the contextual prefix (`Title: ... Section: ...`). `DenseIndexer` builds a FAISS `IndexFlatIP` and saves the chunk metadata pickle. `SparseIndexer` builds a BM25 model with NLTK tokenization (LaTeX stripping + stop-word removal + Porter stemming). Both indices must always be rebuilt together to keep metadata aligned.

**Evaluation (`src/evaluation/`)**
`generate_predictions.py` runs the RAG pipeline over QASPER questions and writes `data/evaluation_dataset.csv` (including `context_ranks`, `context_types`, `crag_action`, `pipeline_config`). `evaluate_rag.py` scores with **Prometheus 2** (`prometheus-eval/prometheus-7b-v2.0`, Kim et al. 2024 arXiv:2405.01535) for the rubric metrics (context precision on the top-3 *by rank*, answer relevancy, answer correctness). Sentence-level support checks go through `grounding.py::GroundingJudge` — `--grounding-checker hybrid` (default) uses Prometheus True/False prompts for context recall and **MiniCheck-Flan-T5-Large** (in-process) for faithfulness and ALCE citation precision/recall, the split that agreed best with independent labels (`compare_grounding_checkers.py`). Recall and faithfulness consider every context the generator saw. `evaluate_retrieval.py` measures evidence recall against QASPER gold evidence for a grid of context-selection configs without any LLM. `calibrate_crag.py` finds optimal CRAG thresholds from a completed evaluation report.

## Important Implementation Details

**Pickle security:** `hybrid_retriever.py::_load_pickle_verified()` checks for path traversal and validates a SHA-256 sidecar (`<file>.sha256`) before deserializing any index file. Sidecar files are generated during ingestion. If indices were built before this check existed, a warning is logged and the file loads anyway.

**RAGatouille / LangChain shim:** `src/retrieval/reranker.py` injects stub modules at `langchain.retrievers.*` before importing RAGatouille 0.0.9.x, which expects a pre-1.0 LangChain path removed in LangChain 1.0. This shim must be imported before any RAGatouille usage.

**Evaluation GPU strategy (two-phase sequential):** `run_evaluation.sh` starts Llama 3.1-8B on port 8000 for prediction generation (`--max-model-len ${GEN_MAX_MODEL_LEN:-12288}`), kills it after `generate_predictions.py` completes, then starts Prometheus 2-7B on port 8001 for evaluation (`${EVAL_MAX_MODEL_LEN:-8192}`; 7 GiB of VRAM is reserved for in-process MiniCheck — bfloat16, length-aware batching, peak ≈ 3.2 GiB — unless `GROUNDING_CHECKER=prometheus`). `SKIP_GENERATION=1` reuses `data/evaluation_dataset.csv`; `EVAL_ARGS="--skip-alce"` resumes an evaluation whose ALCE pass already finished. The script never passes `--hf-token` to vLLM (vLLM logs its arguments); the token comes from the environment. This sequential approach is required on single-GPU deployments (RTX 4090, 24 GB VRAM — each 7-8B model takes ~14-16 GB). `evaluate_rag.py` auto-selects backend: vLLM server (port `PROMETHEUS_PORT`) → HF transformers pipeline → Ollama (CPU fallback).

**Index format:** `dense.index.meta` and `sparse.pkl` are pickled Python objects. `sparse.pkl` is `{'model': BM25Okapi, 'metadata': [{'text': str, 'paper_id': str, 'chunk_id': str, 'chunk_type': str, 'position': int, ...}]}`. Both share the same chunk ordering — never swap one without rebuilding the other. Indices built before 2026-10 lack `chunk_type`/`position`; paper order then falls back to parsing `chunk_id`. `data/indices_text_only/` keeps the pre-2026-10 text-only index for before/after comparisons.

**FAISS:** not installable via pip for GPU support; must use the conda channel. The `requirements.txt` pip entry is kept only as a reference.
