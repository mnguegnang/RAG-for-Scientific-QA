# RAG System for Answering Scientific Questions in NLP

A **Retrieval-Augmented Generation (RAG)** pipeline for open-domain question answering over scientific papers. Given a natural-language question, the system retrieves relevant passages from a local FAISS + BM25 index, reranks them with ColBERT v2 late interaction, labels their relevance with Corrective RAG, and prompts a local LLM to produce a cited answer.


The pipeline is built on the [QASPER](https://huggingface.co/datasets/allenai/qasper) benchmark (Dasigi et al. 2021, NAACL) but is dataset-agnostic and can be reindexed against any document collection.

**Status:** research project under active development. The current pipeline defaults have not
yet been evaluated end to end; see [Evaluation](#evaluation).

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
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)

---

## Architecture

```
User Query (+ paper_id)
    │
    ├─ [< 10 words, and only when it can change the result] ──► HyDE passage for the dense leg
    ▼
┌──────────────────────────────────────────────────┐
│  Stage 1 — Hybrid Retrieval (≤ 100, paper-scoped) │
│  SPECTER2 dense search   (FAISS flat index)       │
│  BM25 sparse search      (NLTK-tokenized)         │
│       └── Reciprocal Rank Fusion (k=60)           │
│  Index: abstract + paragraphs + table/figure      │
│         captions + table bodies (arXiv LaTeX)     │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Stage 2 — Ranking                                │
│  ColBERT v2 MaxSim over every candidate           │
│  [--final-ranking rrf] fuse with BM25/SPECTER2    │
│  keep top context_k (default 20)                  │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Stage 3 — CRAG (Yan et al. 2024)                 │
│  Per-doc label: {Correct, Ambiguous, Incorrect}   │
│  signal (default): label + report, never delete   │
│  refine: ColBERT-scored strip refinement          │
│  legacy: pre-2026-10 filtering (ablations)        │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Stage 4 — Generation                             │
│  token budget → chunks in paper order (OP-RAG)    │
│  Llama-3.1-8B-Instruct (vLLM / HF / Ollama)       │
│  short answers, every sentence cited [Doc N]      │
└──────────────────────────────────────────────────┘
    │
    ▼
 Cited Answer
```

In paper-scoped mode, stage 1 returns every chunk of the paper for most papers (94% of
sampled papers had ≤ 100 chunks on the text-only index), so ColBERT ranking decides what the
model reads. CRAG in `signal` mode labels documents without changing the context.

| Component | Model / Library |
|-----------|----------------|
| **Document ingestion** | HuggingFace `datasets` — `allenai/qasper` (pinned to datasets==2.21.0: 3.x rejects QASPER’s loading script) |
| **Chunking** | Custom `QasperChunker` — abstract, 500-token paragraph chunks (10% overlap), table/figure captions, Markdown table bodies; contextual prefix |
| **Dense embedding** | `allenai/specter2_base` + retrieval adapter (768-dim) |
| **Sparse index** | BM25 (`rank-bm25`) with NLTK tokenization, stop-word removal, Porter stemming |
| **Retrieval fusion** | Reciprocal Rank Fusion (Cormack et al. 2009, `rrf_k=60`) |
| **Reranker** | ColBERT v2 late interaction (`colbert-ir/colbertv2.0`, Santhanam et al. 2022) via RAGatouille; optional RRF with the stage-1 ranks |
| **Relevance labelling** | Corrective RAG (Yan et al. 2024) — three-way classification; changes the context only in `refine` and `legacy` modes |
| **Context order** | Order-preserving RAG (Yu et al. 2024, arXiv:2409.01666) |
| **LLM** | `meta-llama/Llama-3.1-8B-Instruct` via vLLM, HuggingFace Transformers, or Ollama |

---

## Prerequisites

- Python 3.10
- CUDA-capable GPU for local encoding/reranking (CPU works via the `ollama` backend, just slower)
- [uv](https://docs.astral.sh/uv/) for environment management, and [Conda](https://docs.conda.io/en/latest/) for FAISS
- **HuggingFace account with access to `meta-llama/Llama-3.1-8B-Instruct`**
  Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

---

## Installation

**Two virtualenvs are required and cannot be merged.** vLLM 0.28 needs `transformers>=5.5`, while
the SPECTER2 encoder's `adapters` dependency pins `transformers~=4.51`. The pipeline reaches vLLM
only over HTTP, so the two never share an environment.

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

On RunPod, or wherever `HF_HUB_ENABLE_HF_TRANSFER=1` is set, see
[Project_Note.md](Project_Note.md#deployment-notes-runpod-and-huggingface-downloads).

---

## Data and Knowledge Base

### What data is used

The system indexes the training split of QASPER (`allenai/qasper`), which contains full texts of
NLP research papers. Each paper yields, in reading order: its abstract, its paragraphs (split into
overlapping 500-token chunks when longer), and one chunk per table/figure caption. When
`data/table_bodies.json` exists, table chunks also carry the table body as Markdown, extracted from
the paper's arXiv LaTeX source (60 of the 150 evaluation questions have table or figure gold
evidence; 26 rely on it alone. Figures get captions only.) Every chunk is prefixed with:

```
Title: <paper title>. Section: <section heading>.
<chunk text>
```

This prefix anchors each chunk to its source paper for both BM25 keyword matching and dense/rerank
scoring.

### Building the index (required before first use)

```bash
# Optional (~45 min, rate-limited to 1 request / 3 s): table bodies from arXiv LaTeX
python -m src.data.arxiv_tables            # -> data/table_bodies.json, cache in data/arxiv_src/

# Local GPU
python -m src.pipeline_ingest              # uses data/table_bodies.json when present

# SLURM cluster
sbatch src/run_pipeline_ingest.sh
```

**Outputs (`data/indices/`):**

| File | Description |
|------|-------------|
| `dense.index` | FAISS flat index (`IndexFlatIP`) |
| `dense.index.meta` | Pickled chunk metadata — text, paper_id, section, chunk_id, chunk_type, position |
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
python -m src.run_rag --query "..." --paper-id 1909.00694    # restrict to one paper
python -m src.run_rag --query "..." --backend ollama         # CPU / laptop
python -m src.run_rag --query "..." --backend transformers --hf-model meta-llama/Llama-3.1-8B-Instruct
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `auto` | `auto` (transformers on GPU, ollama on CPU), `ollama`, `transformers`, or `vllm` |
| `--paper-id` | none | Restrict retrieval to one QASPER paper |
| `--context-k` | `20` | Chunks passed to the generator |
| `--context-order` | `paper` | `paper` (order-preserving) or `rank` (by relevance) |
| `--final-ranking` | `colbert` | `colbert`, or `rrf` to fuse ColBERT with the BM25/SPECTER2 ranks |
| `--max-context-tokens` | `6000` | Approximate context budget; lowest-ranked chunks dropped beyond it |
| `--crag-mode` | `signal` | `signal` (label only), `refine` (ColBERT strip refinement), `legacy` |
| `--crag-correct` | `14.44` | ColBERT MaxSim threshold for the CRAG `Correct` label |
| `--crag-ambiguous` | `8.0` | ColBERT MaxSim threshold for the CRAG `Ambiguous` label |
| `--crag-consistency` | `0.0` (`0.3` in legacy mode) | Minimum fraction of docs labeled `Correct` for action `Correct` |

**Example output:**

```
ANSWER:
The model uses a bidirectional LSTM encoder with 256 hidden units [Doc 2].
Attention is computed over the encoder states using a learned query vector [Doc 2].
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
| `GROUNDING_CHECKER` | No | Support-check judge: `hybrid` (default), `minicheck`, `prometheus` |
| `PIPELINE_ARGS` | No | Extra context-selection flags `run_evaluation.sh` passes to `generate_predictions` |
| `HF_HOME` | No | Model weight cache — pinned to `/workspace` by `run_evaluation.sh` on RunPod |

### Key hyperparameters

Parameters with a CLI flag are listed under [Usage](#usage).

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `retrieval/chunking.py` | `500` | Chunk size in tokens (SPECTER2 max: 512) |
| `overlap_pct` | `retrieval/chunking.py` | `0.1` | Overlap fraction (50 tokens) |
| retrieval `k` | `run_rag.py` | `100` | Candidates fetched by hybrid retrieval before reranking |
| `rrf_k` | `retrieval/hybrid_retriever.py` | `60` | RRF constant (Cormack et al. 2009) |
| `HYDE_QUERY_WORD_THRESHOLD` | `run_rag.py` | `10` | Queries shorter than this (in words) trigger HyDE |
| `gpu_memory_utilization` | `run_evaluation.sh` | auto-computed | vLLM GPU memory fraction (single-GPU mode) |

### `configs/prompts.yaml`

The generation system prompt, few-shot exemplar, and output-format instructions.

---

## Evaluation

### Metrics

- **[Prometheus 2](https://arxiv.org/abs/2405.01535) rubric judge:** context precision, answer
  relevancy and answer correctness.
- **Sentence-level support checks:** context recall and faithfulness.
- **[ALCE](https://github.com/princeton-nlp/ALCE) citation checks:** citation precision and recall.
- **Retrieval harness:** evidence recall against QASPER's gold evidence, with no LLM.

Definitions, scoring details, CRAG calibration and supporting scripts are in
[Project_Note.md](Project_Note.md).

### Running evaluation

```bash
export HF_TOKEN="hf_XXXX"
python -m src.evaluation.evaluate_retrieval     # no LLM: evidence recall per config → data/retrieval_eval.csv
python -m src.evaluation.generate_predictions   # requires indices → data/evaluation_dataset.csv
python -m src.evaluation.evaluate_rag           # requires vLLM/Ollama running → data/evaluation_report.csv
bash run_evaluation.sh                          # both, end-to-end
sbatch run_evaluation.sh                        # both, on SLURM
PIPELINE_ARGS="--context-k 30 --crag-mode refine" bash run_evaluation.sh   # an alternative config
```

`run_evaluation.sh` runs generation and scoring one after the other, so a single 24 GB GPU is enough.

### Baseline results (2026-09-02, before the P1–P6 changes)

These numbers come from the 2026-09-02 pipeline: text-only index, top 10 contexts, CRAG in
legacy mode, previous prompt. The current defaults are not yet evaluated end to end
(see `key_metrics_improvements.md`). 150 QASPER questions from the train split, free-form
answers only. 128 were scored; 22 refusals and errors are excluded from every mean.
Counting them as wrong gives an answer correctness of about 0.40. Context recall also
excludes reference answers shorter than 10 characters.

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

**Retrieval only, current code** (150 questions, gold evidence, no LLM judge):

| Configuration | Evidence recall | Table/figure caption hit |
|---|---|---|
| Old pipeline, text-only index | 0.485 | 0.000 |
| Current defaults, text-only index | 0.646 | 0.000 |
| Current defaults, current index | 0.829 | 0.700 |

A table/figure hit counts the caption reaching the context, not the row holding the answer.
These settings were chosen on the same 150 questions.

---

## Project Structure

- `src/pipeline_ingest.py` builds the dense and sparse indices.
- `src/run_rag.py` holds `ScientificRAGPipeline` and the query CLI.
- `src/evaluation/` holds prediction generation, scoring and calibration.
- `configs/prompts.yaml` holds the generation prompt.

---

## Contributing

Issues and pull requests are welcome. Before opening a pull request:

1. Set up the environment as described in [Installation](#installation).
2. Run the unit tests with `.venv/bin/python -m pytest tests` (`HF_HOME` must point at the model cache).
3. If you change retrieval or context selection, include the before and after results of
   `python -m src.evaluation.evaluate_retrieval`.
4. If you change chunking or indexing, rebuild both indices together.

---

## Support

Report bugs and ask questions through
[GitHub Issues](https://github.com/mnguegnang/RAG-for-Scientific-QA/issues).
Maintained by Gabin Maxime Nguegnang.

---

## License

MIT — see [LICENSE](LICENSE).
