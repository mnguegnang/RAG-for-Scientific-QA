# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda create -n qasper-rag python=3.10 && conda activate qasper-rag
pip install -e .
pip install -r requirements.txt
# FAISS must be installed via conda (pip version has no GPU support):
#   CPU: conda install -c pytorch faiss-cpu=1.9.0
#   GPU: conda install -c pytorch -c nvidia faiss-gpu=1.9.0 pytorch-cuda=12.1
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
export HF_TOKEN="hf_XXXX"   # required for meta-llama/Llama-3.1-8B-Instruct
```

## Key Commands

**Build indices (required before first use):**
```bash
python -m src.pipeline_ingest          # local GPU
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
python -m src.evaluation.generate_predictions   # writes data/evaluation_dataset.csv
python -m src.evaluation.evaluate_rag           # writes data/evaluation_report.csv
bash run_evaluation.sh                          # end-to-end (RunPod / interactive)
sbatch run_evaluation.sh                        # end-to-end SLURM job
python -m src.evaluation.calibrate_crag         # CRAG threshold calibration + plot
```

**Environment variables:**
- `HF_TOKEN` — required for the gated LLaMA model (`meta-llama/Llama-3.1-8B-Instruct`)
- `GENERATOR_BACKEND` — override LLM backend: `vllm`, `transformers`, or `ollama`
- `VLLM_API_URL` — Llama vLLM server endpoint (default: `http://localhost:8000/v1`)
- `PROMETHEUS_PORT` — Prometheus 2 vLLM port for evaluation (default: `8001`)

## Architecture

The pipeline is in `src/run_rag.py::ScientificRAGPipeline` and runs four sequential stages:

**Stage 1 — Hybrid Retrieval (`src/retrieval/hybrid_retriever.py`)**
Fetches top-100 candidates by fusing dense FAISS search (SPECTER2 embeddings, `allenai/specter2_base`) and BM25 sparse search using Reciprocal Rank Fusion (`rrf_k=60`). Short queries (< 10 words) trigger HyDE: the LLM generates a hypothetical passage that is encoded for dense search in place of the raw question; BM25 always uses the original query.

**Stage 2 — Reranking (`src/retrieval/reranker.py`)**
ColBERT v2 late-interaction (`colbert-ir/colbertv2.0`) via RAGatouille narrows to top-10 using MaxSim scoring. Replaces the previous `BAAI/bge-reranker-v2-m3` cross-encoder.

**Stage 3 — CRAG Evaluation (`src/retrieval/crag_evaluator.py`)**
Classifies each document as `{Correct, Ambiguous, Incorrect}` using ColBERT MaxSim scores against two thresholds (`correct_threshold=14.0`, `ambiguous_threshold=8.0`). A self-consistency ratio determines the overall action. Ambiguous documents are refined via sentence-level strip filtering. Incorrect action falls back to top-3 by rerank score (no hard refusal).

**Stage 4 — Generation (`src/generation/llm_generator.py`)**
`LocalLLMGenerator` supports three backends resolved at init: `transformers` (HuggingFace pipeline, GPU), `ollama` (local HTTP daemon, CPU), and `vllm` (OpenAI-compatible server). Prompt template is loaded from `configs/prompts.yaml`; inline fallback if missing. Tracks per-call latency and approximate token counts.

**Ingestion (`src/pipeline_ingest.py`, `src/retrieval/`)**
`QasperChunker` splits papers into 500-token chunks with 10% overlap and a contextual prefix (`Title: ... Section: ...`). `DenseIndexer` builds a FAISS `IndexFlatIP` and saves the chunk metadata pickle. `SparseIndexer` builds a BM25 model with NLTK tokenization (LaTeX stripping + stop-word removal + Porter stemming). Both indices must always be rebuilt together to keep metadata aligned.

**Evaluation (`src/evaluation/`)**
`generate_predictions.py` runs the RAG pipeline over QASPER questions and writes `data/evaluation_dataset.csv`. `evaluate_rag.py` scores with **Prometheus 2** (`prometheus-eval/prometheus-7b-v2.0`, Kim et al. 2024 arXiv:2405.01535) using the ABSOLUTE_PROMPT rubric format — replaces `nomic-ai/nomic-embed-text-v1.5` and RAGAS. Four metrics are computed (context precision/recall, faithfulness, answer relevancy) plus ALCE citation precision/recall/F1. `calibrate_crag.py` finds optimal CRAG thresholds from a completed evaluation report.

## Important Implementation Details

**Pickle security:** `hybrid_retriever.py::_load_pickle_verified()` checks for path traversal and validates a SHA-256 sidecar (`<file>.sha256`) before deserializing any index file. Sidecar files are generated during ingestion. If indices were built before this check existed, a warning is logged and the file loads anyway.

**RAGatouille / LangChain shim:** `src/retrieval/reranker.py` injects stub modules at `langchain.retrievers.*` before importing RAGatouille 0.0.9.x, which expects a pre-1.0 LangChain path removed in LangChain 1.0. This shim must be imported before any RAGatouille usage.

**Evaluation GPU strategy (two-phase sequential):** `run_evaluation.sh` starts Llama 3.1-8B on port 8000 for prediction generation, kills it after `generate_predictions.py` completes, then starts Prometheus 2-7B on port 8001 for evaluation. This sequential approach is required on single-GPU deployments (RTX 4090, 24 GB VRAM — each 7-8B model takes ~14-16 GB). `evaluate_rag.py` auto-selects backend: vLLM server (port `PROMETHEUS_PORT`) → HF transformers pipeline → Ollama (CPU fallback).

**Index format:** `dense.index.meta` and `sparse.pkl` are pickled Python objects. `sparse.pkl` is `{'model': BM25Okapi, 'metadata': [{'text': str, 'paper_id': str, ...}]}`. Both share the same chunk ordering — never swap one without rebuilding the other.

**FAISS:** not installable via pip for GPU support; must use the conda channel. The `requirements.txt` pip entry is kept only as a reference.
