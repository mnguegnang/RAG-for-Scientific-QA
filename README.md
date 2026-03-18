# QASPER Scientific Question Answering — RAG System

This **Retrieval-Augmented Generation (RAG)** pipeline enables open-domain question answering over scientific papers. When provided with a natural-language question, the system retrieves relevant passages from a local FAISS and BM25 index, reranks them using a cross-encoder, and employs a local LLM to generate a cited, well-supported answer.

The pipeline is built on the [QASPER](https://huggingface.co/datasets/allenai/qasper) benchmark, which includes 5,049 NLP research-paper question-answer pairs. However, it remains dataset-agnostic and can be reindexed for use with any document collection.

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

The pipeline runs in four stages:

```
User Query
    │
    ├─ [< 10 words] ──► HyDE: LLM generates a hypothetical passage
    │                         used as the dense query instead of the raw question
    ▼
┌─────────────────────────────────────────────┐
│  Stage 1 — Hybrid Retrieval  (top-50)       │
│  SPECTER2 dense search  (FAISS flat index)  │
│  BM25 sparse search     (NLTK-tokenized)    │
│       └── Reciprocal Rank Fusion (k=60)     │
│       └── [optional] paper_id filter        │
└─────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Stage 2 — BGE Cross-Encoder         │
│  BAAI/bge-reranker-v2-m3 → top-7    │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Stage 3 — CRAG Relevance Gate       │
│  if best logit < threshold:          │
│      return "insufficient context"   │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Stage 4 — Generation                                │
│  Llama-3.1-8B-Instruct (vLLM / HuggingFace / Ollama)│
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
| **Dense embedding** | `allenai/specter2_base` + retrieval adapter (768-dim, FP16) |
| **Vector store** | FAISS flat index (`IndexFlatIP`) |
| **Sparse index** | BM25 (`rank-bm25`) with NLTK tokenization, stop-word removal, Porter stemming |
| **Retrieval fusion** | Reciprocal Rank Fusion (Cormack et al. 2009) |
| **Reranker** | `BAAI/bge-reranker-v2-m3` (XLM-RoBERTa cross-encoder) |
| **LLM** | `meta-llama/Llama-3.1-8B-Instruct` via vLLM, HuggingFace Transformers, or Ollama |
| **Orchestration** | Custom Python (`src/run_rag.py`) |

---

## Prerequisites

- Python 3.10
- CUDA 12.x and at least one GPU (FAISS and SPECTER2 encoding are GPU-accelerated)
- [Conda](https://docs.conda.io/en/latest/) or equivalent environment manager
- **HuggingFace account with access to `meta-llama/Llama-3.1-8B-Instruct`**
  Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

> For SLURM cluster usage, 3× A100-SXM4-80GB GPUs are used: GPUs 0–1 for vLLM (tensor-parallel), GPU 2 for encoders and reranker.

---

## Installation

```bash
# 1. Clone
git clone <repo-url>
cd qasper-rag-scientific-qa

# 2. Create environment
conda create -n qasper-rag python=3.10
conda activate qasper-rag

# 3. Install
pip install -e .
pip install -r requirements.txt

# 4. NLTK data (required by BM25 tokenizer)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# 5. Set your HuggingFace token (required for the gated LLM)
export HF_TOKEN="hf_XXXX"
# Add to ~/.bashrc for persistence
```

---

## Data and Knowledge Base

### What data is used

The system indexes the training split of QASPER (`allenai/qasper`), which contains full texts of NLP research papers. Each paper is split into overlapping 500-token chunks, each prefixed with:

```
Title: <paper title>. Section: <section heading>.
<chunk text>
```

This prefix anchors each chunk to its source paper in both BM25 keyword matching and cross-encoder scoring.

### Building the index (required before first use)

```bash
# Interactive (GPU node)
export HF_TOKEN="hf_XXXX"
python -m src.pipeline_ingest

# SLURM cluster
export HF_TOKEN="hf_XXXX"
sbatch src/run_pipeline_ingest.sh
```

**Outputs:**

| File | Description |
|------|-------------|
| `data/indices/dense.index` | FAISS flat index (47,810 vectors × 768-dim) |
| `data/indices/dense.index.meta` | Pickled chunk metadata — text, paper_id, section |
| `data/indices/sparse.pkl` | BM25 model + tokenized corpus |

### Updating the knowledge base

1. Add new paper texts to the raw data loader in `src/data/make_dataset.py`.
2. Re-run `python -m src.pipeline_ingest` to rebuild both indices.

The indices are append-free — a full rebuild is required when adding documents. Both `dense.index` and `sparse.pkl` must always be rebuilt together to keep metadata aligned.

---

## Usage

### Ask a single question

```bash
python -m src.run_rag --query "What encoder architecture does the paper use?"
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `auto` | `vllm`, `transformers`, or `ollama` |
| `--crag-threshold` | `0.0` | BGE logit gate; lower = more permissive |
| `--dense-index` | `data/indices/dense.index` | Path to FAISS index |
| `--sparse-index` | `data/indices/sparse.pkl` | Path to BM25 index |

**Example output:**

```
Answer:
The model uses a bidirectional LSTM encoder with 256 hidden units [Doc 2].
Attention is computed over the encoder states using a learned query vector [Doc 2].
```

### Run CRAG threshold calibration

Calibrates the CRAG relevance gate against the evaluation report and outputs a recommended threshold + a diagnostic plot.

```bash
# Uses data/evaluation_report.csv by default
python -m src.evaluation.calibrate_crag

# Explicit path
python -m src.evaluation.calibrate_crag --eval-csv data/evaluation_report.csv
```

Output: threshold recommendation on stdout + `reports/figures/crag_calibration.png`

### Run the full evaluation pipeline

Requires a running SLURM allocation with 3× A100 GPUs. The script starts a vLLM server, generates RAG predictions, runs RAGAS + ALCE scoring, and writes results to CSV.

```bash
cd /path/to/qasper-rag-scientific-qa
export HF_TOKEN="hf_XXXX"
sbatch run_evaluation.sh
```

Steps performed:
1. vLLM server starts on GPUs 0–1 (`--tensor-parallel-size 2`)
2. `generate_predictions.py` runs on GPU 2 → writes `data/evaluation_dataset.csv`
3. `evaluate_rag.py` runs RAGAS + ALCE → writes `data/evaluation_report.csv`
4. vLLM server is shut down
5. Logs: `logs/eval_<jobid>.log` / `logs/eval_error_<jobid>.log`

---

## Configuration

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | **Yes** | HuggingFace access token for `meta-llama/Llama-3.1-8B-Instruct` |
| `GENERATOR_BACKEND` | No | Override LLM backend: `vllm`, `transformers`, or `ollama` |
| `HF_HOME` | No | Cache directory for model weights (auto-selected by `run_evaluation.sh` based on free disk space) |

### Key hyperparameters

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `retrieval/chunking.py` | `500` | Chunk size in tokens (SPECTER2 max: 512) |
| `overlap_pct` | `retrieval/chunking.py` | `0.1` | Overlap fraction (50 tokens) |
| `CRAG_THRESHOLD` | `run_rag.py` | `0.0` | BGE logit below which generation is suppressed |
| `HYDE_QUERY_WORD_THRESHOLD` | `run_rag.py` | `10` | Queries shorter than this trigger HyDE |
| `retrieval k` | `run_rag.py` | `50` | Broad candidates from hybrid retrieval |
| `rerank top_k` | `run_rag.py` | `7` | Documents passed to the LLM |
| `rrf_k` | `retrieval/hybrid_retriever.py` | `60` | RRF constant (Cormack et al. 2009) |
| `gpu_memory_utilization` | `run_evaluation.sh` | `0.85` | vLLM GPU memory fraction |
| `num_samples` | `evaluation/generate_predictions.py` | `200` | Questions to evaluate |

### `configs/default.yaml`

Override retrieval and generation hyperparameters without editing source code.

### `configs/prompts.yaml`

Override the system prompt and output-format instructions for the LLM.

---

## Evaluation

### Framework

Evaluation combines two frameworks:

- **[RAGAS](https://github.com/explodinggradients/ragas)** (Es et al. 2023, arXiv:2309.15217) — measures retrieval and generation quality using an LLM judge:
  - *Context Precision* — fraction of retrieved chunks relevant to the question
  - *Context Recall* — fraction of gold evidence covered by retrieved chunks
  - *Faithfulness* — fraction of answer claims supported by retrieved context
  - *Answer Relevancy* — cosine similarity between the answer and the original question

- **[ALCE](https://github.com/princeton-nlp/ALCE)** (Gao et al. 2023, EMNLP) — citation-level grounding via NLI:
  - *Citation Precision* — fraction of cited sentences entailed by the cited document
  - *Citation Recall* — fraction of answer sentences that have a supporting citation

### Running evaluation

```bash
# Generate predictions (requires indices)
python -m src.evaluation.generate_predictions

# Score predictions (requires vLLM or Ollama running)
python -m src.evaluation.evaluate_rag

# Or run both end-to-end via SLURM
sbatch run_evaluation.sh
```

Results are written to `data/evaluation_report.csv` and summarised in the job log.

### Current results (job 27970)

| Metric | Score | Notes |
|--------|-------|-------|
| Context Recall | 0.6667 | n=3 valid rows |
| Faithfulness | 0.9000 | n=2 valid rows |
| Context Precision | N/A | vLLM judge timeout during eval |
| Answer Relevancy | N/A | Fixed in current codebase (nomic prefix) |
| ALCE Citation Precision | 0.0000 | Fixed in current codebase (regex + paper-ID filter) |
| ALCE Citation Recall | 0.0000 | Fixed in current codebase (regex + paper-ID filter) |

> The low row counts (n=2–3) reflect the CRAG gate suppressing answers for questions where the retrieved context is below threshold. Increase `num_samples` to ≥200 and tune `CRAG_THRESHOLD` via `calibrate_crag` for statistically reliable scores.

---

## Project Structure

```
qasper-rag-scientific-qa/
├── configs/
│   ├── default.yaml                # Retrieval and generation hyperparameters
│   └── prompts.yaml                # System prompt and output-format templates
├── data/
│   ├── evaluation_dataset.csv      # RAG predictions (input to evaluate_rag.py)
│   ├── evaluation_report.csv       # RAGAS + ALCE scores per question
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
│   ├── retrieval/
│   │   ├── chunking.py             # 500-token overlapping chunks + metadata prefix
│   │   ├── encoders.py             # SPECTER2 encoder (DataParallel multi-GPU)
│   │   ├── vector_store.py         # FAISS index builder
│   │   ├── sparse_store.py         # BM25 index builder (NLTK tokenization)
│   │   ├── hybrid_retriever.py     # RRF fusion + paper_id filter
│   │   └── reranker.py             # BGE cross-encoder reranker
│   ├── generation/
│   │   └── llm_generator.py        # vLLM / HuggingFace / Ollama backends
│   └── evaluation/
│       ├── generate_predictions.py # Run RAG over QASPER → evaluation_dataset.csv
│       ├── evaluate_rag.py         # RAGAS + ALCE scorer → evaluation_report.csv
│       └── calibrate_crag.py       # CRAG threshold calibration
├── run_evaluation.sh               # SLURM batch job (3× A100-SXM4-80GB)
└── src/run_pipeline_ingest.sh      # SLURM batch job for index building
```

---

## License

MIT — see [LICENSE](LICENSE).
