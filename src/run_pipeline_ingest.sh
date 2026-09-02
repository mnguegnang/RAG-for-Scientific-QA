#!/bin/bash
#SBATCH --job-name=rag_index
#SBATCH --output=logs/ingest_%j.out
#SBATCH --error=logs/ingest_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --partition=all               #cluster's partition name

# ============================================================
# QASPER ingestion — chunk, dense (FAISS/SPECTER2) + sparse (BM25) indices
#
# Models loaded here: allenai/specter2_base + allenai/specter2 (retrieval
# adapter). Both are PUBLIC — this job needs no HuggingFace token, and never
# loads Llama. It runs entirely from .venv (the RAG environment); .venv-vllm
# is only for the evaluation servers.
#
# Storage policy (RunPod): / is a small container overlay, /workspace is the
# persistent network volume. HF_HOME / NLTK_DATA / torch caches are pinned
# under /workspace so the container disk never fills and nothing is
# re-downloaded after a pod restart.
#
# Usage (always from the project root):
#   bash src/run_pipeline_ingest.sh      # interactive / RunPod
#   sbatch src/run_pipeline_ingest.sh    # SLURM
# ============================================================

# ============================================================
# PROJECT ROOT — SLURM-safe detection (must happen first)
# ============================================================
# SLURM copies the batch script to its spool directory before running it,
# so BASH_SOURCE[0] resolves to /var/spool/slurmd/jobXXX/slurm_script —
# NOT to the original file.  SLURM_SUBMIT_DIR is the directory from which
# `sbatch` was called, which is the project root (= reliable in SLURM jobs).
# For interactive use (`bash src/run_pipeline_ingest.sh`), fall back to the
# script's own directory parent.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

# ============================================================
# PERSISTENT ENV — /workspace/.bashrc_custom is not sourced by
# non-interactive shells (sbatch, nohup, ssh -c), so load it here.
# Anything already exported by the caller wins.
# ============================================================
_PRESET_HF_HOME="${HF_HOME:-}"
_PRESET_NLTK_DATA="${NLTK_DATA:-}"
if [ -f /workspace/.bashrc_custom ]; then
    # shellcheck disable=SC1091
    source /workspace/.bashrc_custom
    echo "Env         : sourced /workspace/.bashrc_custom"
fi
[ -n "${_PRESET_HF_HOME}" ]   && export HF_HOME="${_PRESET_HF_HOME}"
[ -n "${_PRESET_NLTK_DATA}" ] && export NLTK_DATA="${_PRESET_NLTK_DATA}"

# ============================================================
# ENVIRONMENT — activated after PROJECT_ROOT is known
# ============================================================
# Ingestion needs SPECTER2, which needs `adapters` — that lives in .venv.
# It must NOT run from .venv-vllm: vLLM forces transformers>=5.5, which
# `adapters` cannot work with (see CLAUDE.md, Environment Setup).
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"
if [ ! -f "${VENV_ACTIVATE}" ]; then
    echo "ERROR: virtualenv not found at ${VENV_ACTIVATE}"
    echo "       Run: uv venv && uv pip install -r requirements.txt"
    exit 1
fi
# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"
RAG_PY="${PROJECT_ROOT}/.venv/bin/python"

if ! "${RAG_PY}" -c "import adapters" 2>/dev/null; then
    echo "ERROR: the 'adapters' package is not importable in ${RAG_PY}."
    echo "       SPECTER2 cannot load without it. Most likely vLLM was"
    echo "       installed into this environment and upgraded transformers."
    echo "       Repair with: uv pip install --python .venv/bin/python -r requirements.txt"
    echo "       and keep vLLM in .venv-vllm (see CLAUDE.md)."
    exit 1
fi

# Prepend the project root to PYTHONPATH so that `python -m src.*` always
# resolves imports from THIS copy of the project, not any stale editable
# install that may exist elsewhere in the environment.
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ============================================================
# CACHES — keep everything off the container overlay
# ============================================================
_WORKSPACE_ROOT="/workspace"
[ -d "${_WORKSPACE_ROOT}" ] || _WORKSPACE_ROOT="${PROJECT_ROOT}"

# Honour an exported HF_HOME; otherwise use the persistent volume. The old
# behaviour overwrote HF_HOME unconditionally with a project-local path,
# which split the cache and re-downloaded models already on disk.
if [ -z "${HF_HOME:-}" ]; then
    if [ -d "/scratch/${USER:-nobody}" ]; then
        export HF_HOME="/scratch/${USER}/.cache/huggingface"
    else
        export HF_HOME="${_WORKSPACE_ROOT}/hf_cache"
    fi
fi
# NLTK bootstraps punkt/punkt_tab/stopwords on import of sparse_store. Without
# NLTK_DATA those land in ~/nltk_data on the container disk (~64 MB).
export NLTK_DATA="${NLTK_DATA:-${_WORKSPACE_ROOT}/nltk_data}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${_WORKSPACE_ROOT}/.inductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${_WORKSPACE_ROOT}/.triton_cache}"
mkdir -p "${HF_HOME}" "${NLTK_DATA}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

echo "HF cache    : ${HF_HOME}"
echo "NLTK data   : ${NLTK_DATA}"

# The RunPod image exports HF_HUB_ENABLE_HF_TRANSFER=1. If hf_transfer is not
# installed, every HuggingFace download raises instead of falling back — and
# transformers reports it as an unrelated "Unrecognized model in
# allenai/specter2_base ... no model_type key" error.
if [ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" = "1" ] && ! "${RAG_PY}" -c "import hf_transfer" 2>/dev/null; then
    echo "WARNING: HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is not installed"
    echo "         — disabling it for this run (pip install hf_transfer to restore fast downloads)."
    export HF_HUB_ENABLE_HF_TRANSFER=0
fi

# ── HuggingFace authentication (OPTIONAL for this job) ──────────────────────
# Ingestion only loads the public allenai/specter2* models and the QASPER
# dataset, so no token is required. Export it when present anyway: it raises
# the anonymous rate limit and is needed if the models are ever swapped for
# gated ones.
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    echo "HF token    : provided (not required for ingestion)"
else
    echo "HF token    : not set (fine — no gated model is used here)"
fi

# SLURM already sets CUDA_VISIBLE_DEVICES to the allocated GPU indices
# (e.g. "0,1,2" for --gres=gpu:3).  We intentionally do NOT override it
# so the allocation is always fully utilised regardless of GPU count.
# Specter2Encoder detects all visible GPUs via torch.cuda.device_count()
# and wraps the model in DataParallel automatically. On a single-GPU RunPod
# box this is simply a one-GPU run.

# PyTorch / parallelism
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
echo "========================================"
echo "Job ID    : ${SLURM_JOB_ID:-interactive}"
echo "Node      : ${SLURM_JOB_NODELIST:-local}"
echo "Project   : ${PROJECT_ROOT}"
echo "Start     : $(date +%T)"
echo "========================================"

mkdir -p logs

# GPU report — skipped gracefully if nvidia-smi is unavailable (CPU node)
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free \
               --format=csv,noheader
else
    echo "nvidia-smi not found — running in CPU-only mode"
fi

echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-unset}"
"${RAG_PY}" -c "
import torch
n = torch.cuda.device_count()
if n:
    print(f'GPUs visible for DataParallel: {n}')
    for i in range(n):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('No CUDA GPUs detected — encoding will run on CPU')
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
"

# Rebuilding replaces both indices. They share one chunk ordering, so they are
# always written together — warn before clobbering a usable pair.
if [ -f data/indices/dense.index ] && [ -f data/indices/sparse.pkl ]; then
    echo "NOTE: existing indices in data/indices will be overwritten."
fi

# ============================================================
# RUN
# ============================================================
echo "Working directory: ${PROJECT_ROOT}"
echo "--- Starting pipeline_ingest at $(date +%T) ---"

"${RAG_PY}" -m src.pipeline_ingest

EXIT_CODE=$?
echo "--- Finished at $(date +%T) with exit code ${EXIT_CODE} ---"
exit ${EXIT_CODE}
