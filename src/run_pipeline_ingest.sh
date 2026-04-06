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
# ENVIRONMENT
# ============================================================
source ~/miniconda3/bin/activate
conda activate qasper-rag

# ============================================================
# PROJECT ROOT — SLURM-safe detection
# ============================================================
# SLURM copies the batch script to its spool directory before running it,
# so BASH_SOURCE[0] resolves to /var/spool/slurmd/jobXXX/slurm_script —
# NOT to the original file.  SLURM_SUBMIT_DIR is the directory from which
# `sbatch` was called, which is the project root (= reliable in SLURM jobs).
# For interactive use (`bash src/run_pipeline_ingest.sh`), fall back to the
# script's own directory parent.
#
# IMPORTANT: always submit this job from the project root:
#   cd /home/math/nguegnang/RAG_project/qasper-rag-scientific-qa
#   sbatch src/run_pipeline_ingest.sh
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

# Prepend the project root to PYTHONPATH so that `python -m src.*` always
# resolves imports from THIS copy of the project, not any stale editable
# install that may exist elsewhere in the conda environment.
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ============================================================
# VARIABLES
# ============================================================
# ── HuggingFace authentication ──────────────────────────────────────────────
# Required for gated models such as meta-llama/Llama-3.1-8B-Instruct.
# Set HF_TOKEN in ONE of these ways (most secure first):
#   1. Before submitting: export HF_TOKEN="hf_XXXX"  (shell variable)
#   2. Persistent:        echo "HF_TOKEN=hf_XXXX" >> ~/.bashrc
#   3. Direct edit below (least secure — avoid committing the token):
#      export HF_TOKEN="hf_XXXX"
if [ -z "${HF_TOKEN}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "       Please export HF_TOKEN=hf_XXXX before submitting this job."
    echo "       The model meta-llama/Llama-3.1-8B-Instruct requires it."
    exit 1
fi
# vLLM reads HUGGING_FACE_HUB_TOKEN; HuggingFace CLI / transformers read HF_TOKEN.
# Exporting both ensures every downstream tool is authenticated.
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# Quick gated-access pre-flight: verify the token has access to the model
# before starting vLLM, so we fail in seconds rather than 15 minutes.
check_hf_access() {
    local model_id="$1"
    local url="https://huggingface.co/${model_id}/resolve/main/config.json"
    local http_code
    http_code=$(curl -sf -o /dev/null -w "%{http_code}"         -H "Authorization: Bearer ${HF_TOKEN}" "${url}" 2>/dev/null || echo "000")
    if [ "${http_code}" != "200" ]; then
        echo "ERROR: HuggingFace token does not have access to '${model_id}'."
        echo "       HTTP status: ${http_code}"
        echo "       Request access at https://huggingface.co/${model_id}"
        exit 1
    fi
    echo "HuggingFace access confirmed for '${model_id}' (HTTP ${http_code})"
}

# Point HF cache to scratch space — avoids filling your home quota
export HF_HOME="/scratch/$USER/.cache/huggingface"

# SLURM already sets CUDA_VISIBLE_DEVICES to the allocated GPU indices
# (e.g. "0,1,2" for --gres=gpu:3).  We intentionally do NOT override it
# so the allocation is always fully utilised regardless of GPU count.
# Specter2Encoder detects all visible GPUs via torch.cuda.device_count()
# and wraps the model in DataParallel automatically.

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
python -c "
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

# ============================================================
# RUN
# ============================================================
echo "Working directory: ${PROJECT_ROOT}"
echo "--- Starting pipeline_ingest at $(date +%T) ---"

python -m src.pipeline_ingest

EXIT_CODE=$?
echo "--- Finished at $(date +%T) with exit code ${EXIT_CODE} ---"
exit ${EXIT_CODE}
