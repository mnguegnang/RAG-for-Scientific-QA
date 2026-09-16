# RAG System for Answering Scientific Questions in NLP

A **Retrieval-Augmented Generation (RAG)** pipeline for open-domain question answering over scientific papers. Given a natural-language question, the system retrieves relevant passages from a local FAISS + BM25 index, reranks them with ColBERT v2 late interaction, applies a Corrective RAG relevance gate, and prompts a local LLM to produce a cited, well-supported answer.

The pipeline is built on the [QASPER](https://huggingface.co/datasets/allenai/qasper) benchmark (Dasigi et al. 2021, NAACL) but is dataset-agnostic and can be reindexed against any document collection.

---

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data and Knowledge Base](#data-and-knowledge-base)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [License](#license)

---

## Architecture

```
User Query
    │
    ├─ [< 10 words] ──► HyDE: LLM generates a hypothetical passage
    │                         used as the dense query instead of the raw question
    ▼
┌───────────────────────────────────────────────┐
│  Stage 1 — Hybrid Retrieval  (top-100)        │
│  SPECTER2 dense search   (FAISS flat index)   │
│  BM25 sparse search      (NLTK-tokenized)     │
│       └── Reciprocal Rank Fusion (k=60)       │
│       └── [optional] paper_id filter          │
└───────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│  Stage 2 — ColBERT v2 Late Interaction │
│  colbert-ir/colbertv2.0 → top-10       │
│  MaxSim scoring (via RAGatouille)      │
└───────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Stage 3 — CRAG Relevance Gate (Yan et al. 2024) │
│  Per-doc classify: {Correct, Ambiguous, Incorrect}│
│  + self-consistency ratio across the top-10       │
│  Ambiguous → sentence-level knowledge refinement  │
│  Incorrect → fall back to top-5 by rerank score   │
│              (no hard refusal)                    │
└─────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Stage 4 — Generation                                │
│  Llama-3.1-8B-Instruct (vLLM / HuggingFace / Ollama) │
│  Chain-of-Thought + [Doc N] citation prompt          │
└──────────────────────────────────────────────────────┘
    │
    ▼
 Cited Answer
```

| Component | Model / Library |
|-----------|----------------|
| **Document ingestion** | HuggingFace `datasets` — `allenai/qasper` |
| **Chunking** | Custom `QasperChunker` — 500-token chunks, 10% overlap, contextual prefix |
| **Dense embedding** | `allenai/specter2_base` + retrieval adapter (768-dim) |
| **Vector store** | FAISS flat index (`IndexFlatIP`) |
| **Sparse index** | BM25 (`rank-bm25`) with NLTK tokenization, stop-word removal, Porter stemming |
| **Retrieval fusion** | Reciprocal Rank Fusion (Cormack et al. 2009, `rrf_k=60`) |
| **Reranker** | ColBERT v2 late interaction (`colbert-ir/colbertv2.0`, Santhanam et al. 2022) via RAGatouille |
| **Relevance gate** | Corrective RAG (Yan et al. 2024) — three-way classification + knowledge refinement |
| **LLM** | `meta-llama/Llama-3.1-8B-Instruct` via vLLM, HuggingFace Transformers, or Ollama |
| **Orchestration** | Custom Python (`src/run_rag.py::ScientificRAGPipeline`) |

---

## Prerequisites

- Python 3.10
- CUDA-capable GPU for local encoding/reranking (CPU works via the `ollama` backend, just slower)
- [uv](https://docs.astral.sh/uv/) for environment management, and [Conda](https://docs.conda.io/en/latest/) for FAISS
- **HuggingFace account with access to `meta-llama/Llama-3.1-8B-Instruct`**
  Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

---

## Installation

**Two virtualenvs are required and cannot be merged.** vLLM 0.28 needs `transformers>=5.5` /
`huggingface_hub>=1.27`; the SPECTER2 encoder depends on `adapters`, which pins
`transformers~=4.51` and the pre-1.0 `huggingface_hub` API. No `adapters` release supports
transformers 5.x, so the two can't share an environment. They never need to: the pipeline talks
to vLLM only over HTTP and never imports it.

```bash
# 1. Project env — RAG pipeline, evaluation, ingestion
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. Server env — vLLM only (Llama 3.1 generation + Prometheus 2 judging)
uv venv --python 3.10 .venv-vllm
uv pip install --python .venv-vllm/bin/python vllm==0.28.0

# 3. FAISS — GPU support requires the conda build (pip is CPU-only)
conda install -c pytorch -c nvidia faiss-gpu=1.9.0 pytorch-cuda=12.1   # GPU
conda install -c pytorch faiss-cpu=1.9.0                              # CPU

# 4. NLTK data (required by the BM25 tokenizer)
.venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# 5. HuggingFace token (required for the gated Llama model)
export HF_TOKEN="hf_XXXX"
```

`run_evaluation.sh` selects the interpreter per phase (`RAG_PY` / `VLLM_PY`) automatically and
falls back to `.venv` when `.venv-vllm` is absent.

> **RunPod note:** `/` is a small container overlay and `/workspace` is the persistent volume.
> `run_evaluation.sh` pins `HF_HOME`, `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`,
> `TRITON_CACHE_DIR`, and `NLTK_DATA` under `/workspace` so caches survive a pod restart, and
> sources `/workspace/.bashrc_custom` (skipped by non-interactive shells otherwise).

> **`hf_transfer` is required** wherever `HF_HUB_ENABLE_HF_TRANSFER=1` is set (the RunPod image
> sets it by default) — without the package, downloads fail with a misleading
> `Unrecognized model ... no model_type key` error. It's pinned in `requirements.txt`; install it
> into `.venv-vllm` too if that env downloads models directly.

---

## Data and Knowledge Base

### What data is used

The system indexes the training split of QASPER (`allenai/qasper`), which contains full texts of
NLP research papers. Each paper is split into overlapping 500-token chunks, each prefixed with:

```
Title: <paper title>. Section: <section heading>.
<chunk text>
```

This prefix anchors each chunk to its source paper for both BM25 keyword matching and dense/rerank
scoring.

### Building the index (required before first use)

```bash
# Local GPU
python -m src.pipeline_ingest

# SLURM cluster
sbatch src/run_pipeline_ingest.sh
```

**Outputs (`data/indices/`):**

| File | Description |
|------|-------------|
| `dense.index` | FAISS flat index (`IndexFlatIP`) |
| `dense.index.meta` | Pickled chunk metadata — text, paper_id, section |
| `sparse.pkl` | BM25 model + tokenized corpus, as `{'model': BM25Okapi, 'metadata': [...]}` |
| `*.sha256` | Sidecar hash for each index, checked before deserialization |

`dense.index.meta` and `sparse.pkl` share the same chunk ordering and must always be rebuilt
together — never swap one without rebuilding the other.

### Updating the knowledge base

1. Point `src/data/make_dataset.py::load_and_inspect_qasper` at a different dataset/split.
2. Re-run `python -m src.pipeline_ingest` to rebuild both indices from scratch.

The indices are append-free — there is no incremental update path.

---

## Usage

### Ask a single question

```bash
python -m src.run_rag --query "What encoder architecture does the paper use?"
python -m src.run_rag --query "..." --backend ollama         # CPU / laptop
python -m src.run_rag --query "..." --backend transformers --hf-model meta-llama/Llama-3.1-8B-Instruct
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `auto` | `auto` (transformers on GPU, ollama on CPU), `ollama`, or `transformers`. Use the `GENERATOR_BACKEND=vllm` env var to route to a vLLM server instead. |
| `--crag-correct` | `14.44` | ColBERT MaxSim threshold for the CRAG `Correct` label |
| `--crag-ambiguous` | `8.0` | ColBERT MaxSim threshold for the CRAG `Ambiguous` label |
| `--crag-consistency` | `0.3` | Minimum fraction of top-10 docs labeled `Correct` for the gate to pass |
| `--dense-index` | `data/indices/dense.index` | Path to the FAISS index |
| `--dense-meta` | `data/indices/dense.index.meta` | Path to the chunk metadata pickle |
| `--sparse-index` | `data/indices/sparse.pkl` | Path to the BM25 index |

**Example output:**

```
ANSWER:
The model uses a bidirectional LSTM encoder with 256 hidden units [Doc 2].
Attention is computed over the encoder states using a learned query vector [Doc 2].

[CRAG] Action: Correct
[CRAG] Correct: 8, Ambiguous: 1, Incorrect: 1 | Consistency: 0.80
```

### Run CRAG threshold calibration

Finds the F1-maximizing ColBERT score boundary from a completed evaluation report and writes a
diagnostic plot.

```bash
python -m src.evaluation.calibrate_crag                                     # uses data/evaluation_report.csv
python -m src.evaluation.calibrate_crag --eval-csv data/evaluation_report.csv
```

Output: recommended threshold on stdout + `reports/figures/crag_calibration.png`.

### Run the full evaluation pipeline

```bash
export HF_TOKEN="hf_XXXX"
bash run_evaluation.sh      # interactive / RunPod
sbatch run_evaluation.sh    # SLURM
```

The script auto-detects available GPUs (multi-GPU tensor-parallel, single-GPU with a computed
`--gpu-memory-utilization`, or CPU-only via Ollama) and runs two phases sequentially so a single
24 GB GPU (e.g. RTX 4090) is enough:

1. Start Llama 3.1-8B on a vLLM server.
2. `generate_predictions.py` runs RAG over QASPER → `data/evaluation_dataset.csv`.
3. Stop the Llama server; start `prometheus-eval/prometheus-7b-v2.0` on a vLLM server.
4. `evaluate_rag.py` scores with Prometheus 2 + ALCE → `data/evaluation_report.csv`.
5. Stop the Prometheus server.

Logs: `logs/eval_<jobid>.log` / `logs/eval_error_<jobid>.log` under SLURM.

### Fixing failed prediction rows

Re-generates only the rows in `evaluation_dataset.csv` whose answer is a `System Error:`/`Error:`
string (e.g. after a transient Ollama/vLLM outage), scoped to the same `paper_id` as the original
row so retries can't pull in another paper's passages:

```bash
python -m src.evaluation.generate_predictions --fix-errors
```

---

## Configuration

### Environment variables

| Variable | Required | Description |
|----------|----------|--------------|
| `HF_TOKEN` | **Yes** | HuggingFace access token for `meta-llama/Llama-3.1-8B-Instruct` |
| `GENERATOR_BACKEND` | No | Override LLM backend: `vllm`, `transformers`, or `ollama` |
| `VLLM_API_URL` | No | Llama vLLM server endpoint (default: `http://localhost:8000/v1`) |
| `PROMETHEUS_PORT` | No | Prometheus 2 vLLM port for evaluation (default: `8001`) |
| `HF_HOME` | No | Model weight cache — pinned to `/workspace` by `run_evaluation.sh` on RunPod |

### Key hyperparameters

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `retrieval/chunking.py` | `500` | Chunk size in tokens (SPECTER2 max: 512) |
| `overlap_pct` | `retrieval/chunking.py` | `0.1` | Overlap fraction (50 tokens) |
| retrieval `k` | `run_rag.py` | `100` | Candidates fetched by hybrid retrieval before reranking |
| `rrf_k` | `retrieval/hybrid_retriever.py` | `60` | RRF constant (Cormack et al. 2009) |
| rerank `top_k` | `run_rag.py` | `10` | Documents kept after ColBERT reranking |
| `HYDE_QUERY_WORD_THRESHOLD` | `run_rag.py` | `10` | Queries shorter than this (in words) trigger HyDE |
| `crag_correct_threshold` | `run_rag.py` | `14.44` | ColBERT MaxSim floor for CRAG `Correct` |
| `crag_ambiguous_threshold` | `run_rag.py` | `8.0` | ColBERT MaxSim floor for CRAG `Ambiguous` |
| `crag_consistency_ratio` | `run_rag.py` | `0.3` | Min. fraction of `Correct`-labeled docs required |
| `gpu_memory_utilization` | `run_evaluation.sh` | auto-computed | vLLM GPU memory fraction (single-GPU mode) |

### `configs/default.yaml`

Retrieval and generation hyperparameters, overridable without editing source code.

### `configs/prompts.yaml`

The generation system prompt, few-shot exemplar, and output-format instructions.

---

## Evaluation

### Framework

- **Prometheus 2** (Kim et al. 2024, [arXiv:2405.01535](https://arxiv.org/abs/2405.01535)) — an
  LLM judge fine-tuned specifically for evaluation, scored with the ABSOLUTE_PROMPT rubric format
  (raw 1–5 score normalized to `[0, 1]`). Auto-selects a backend: vLLM server → HF `transformers`
  pipeline → Ollama (CPU fallback).
  - *Context Precision* — fraction of retrieved chunks relevant to the question
  - *Context Recall* — fraction of gold evidence covered by retrieved chunks
  - *Faithfulness* — fraction of answer claims supported by retrieved context
  - *Answer Relevancy* — how completely the answer addresses the question
  - *Answer Correctness* — factual match against the QASPER reference answer

- **[ALCE](https://github.com/princeton-nlp/ALCE)** (Gao et al. 2023, EMNLP) — citation-level
  grounding via NLI entailment:
  - *Citation Precision* — fraction of cited sentences entailed by the cited document
  - *Citation Recall* — fraction of answer sentences that have a supporting citation

### Running evaluation

```bash
python -m src.evaluation.generate_predictions   # requires indices → data/evaluation_dataset.csv
python -m src.evaluation.evaluate_rag           # requires vLLM/Ollama running → data/evaluation_report.csv
bash run_evaluation.sh                          # both, end-to-end
```

### Current results

150 QASPER questions, 128 scored (the rest suppressed by the CRAG gate or dropped as judge
errors); reproduce with the commands above:

| Metric | Mean |
|--------|------|
| Context Precision | 0.5645 |
| Context Recall | 0.3902 |
| Faithfulness | 0.7160 |
| Answer Relevancy | 0.5879 |
| Answer Correctness | 0.4668 |
| ALCE Citation Precision | 0.6542 |
| ALCE Citation Recall | 0.6740 |
| ALCE Citation F1 | 0.6556 |

### Judge validation utilities

`src/evaluation/` also has standalone scripts used to validate the evaluation pipeline itself
rather than the RAG system:

| Script | Purpose |
|--------|---------|
| `validate_judge.py` / `compute_agreement.py` | Human-vs-Prometheus agreement (Cohen's kappa) at the atomic-decision level |
| `second_judge_label.py` | Auto-labels the blind validation set with Claude as an independent second judge |
| `validate_retrieval_scoping.py` | Regression check that retrieved contexts stay within the queried paper |
| `validate_precision_cutoff.py` | Checks whether scoring context precision on top-3 vs. top-10 chunks diverges |
| `compile_dspy_prompt.py` | Compiles the generation prompt's few-shot demonstrations with DSPy |

---

## Project Structure

```
RAG-for-Scientific-QA/
├── configs/
│   ├── default.yaml                # Retrieval and generation hyperparameters
│   └── prompts.yaml                # System prompt and output-format templates
├── data/
│   ├── evaluation_dataset.csv      # RAG predictions (input to evaluate_rag.py)
│   ├── evaluation_report.csv       # Prometheus 2 + ALCE scores per question
│   └── indices/
│       ├── dense.index             # FAISS flat index
│       ├── dense.index.meta        # Chunk metadata (text, paper_id, section)
│       └── sparse.pkl              # BM25 model + tokenized corpus
├── reports/
│   └── figures/
│       └── crag_calibration.png    # CRAG threshold calibration plot
├── src/
│   ├── pipeline_ingest.py          # Builds dense + sparse indices
│   ├── run_rag.py                  # ScientificRAGPipeline orchestrator
│   ├── data/
│   │   └── make_dataset.py         # Loads the QASPER dataset for ingestion
│   ├── retrieval/
│   │   ├── chunking.py             # 500-token overlapping chunks + metadata prefix
│   │   ├── encoders.py             # SPECTER2 encoder
│   │   ├── vector_store.py         # FAISS index builder
│   │   ├── sparse_store.py         # BM25 index builder (NLTK tokenization)
│   │   ├── hybrid_retriever.py     # RRF fusion + paper_id filter + pickle verification
│   │   └── reranker.py             # ColBERT v2 late-interaction reranker
│   ├── generation/
│   │   ├── llm_generator.py        # vLLM / HuggingFace / Ollama backends
│   │   └── dspy_module.py          # DSPy-compiled generation prompt
│   └── evaluation/
│       ├── generate_predictions.py # Run RAG over QASPER → evaluation_dataset.csv
│       ├── evaluate_rag.py         # Prometheus 2 + ALCE scorer → evaluation_report.csv
│       └── calibrate_crag.py       # CRAG threshold calibration
├── run_evaluation.sh               # End-to-end evaluation (SLURM or interactive)
└── src/run_pipeline_ingest.sh      # SLURM batch job for index building
```

---

## License

MIT — see [LICENSE](LICENSE).
