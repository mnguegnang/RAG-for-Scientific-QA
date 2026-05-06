#!/bin/bash
# ============================================================
# Evaluation pipeline — Generation (Llama) + Evaluation (Prometheus 2)
#
# Two-phase GPU strategy:
#   Phase 1 — RAG prediction generation
#     vLLM: meta-llama/Llama-3.1-8B-Instruct  (port 8000)
#     RAG:  SPECTER2 + ColBERT v2 reranker
#   Phase 2 — Prometheus 2 evaluation (Kim et al. 2024, arXiv:2405.01535)
#     Llama vLLM is stopped first (frees VRAM on small GPUs like RTX 4090)
#     vLLM: prometheus-eval/prometheus-7b-v2.0 (port 8001)
#
# GPU / CPU auto-selection:
#   N_GPU >= 2 : tensor-parallel across N-1 GPUs for vLLM; last GPU for RAG
#   N_GPU == 1 : single GPU; vLLM uses 60% VRAM for generation, 80% for eval
#   N_GPU == 0 : CPU-only; generation via Ollama; evaluation via Ollama+llama3
#
# Supported environments:
#   RunPod  RTX 4090 (24 GB VRAM)  — single-GPU mode
#   SLURM   A100-SXM4-80GB         — multi-GPU mode
#
# Usage:
#   # SLURM:
#   sbatch run_evaluation.sh
#   # Interactive / RunPod terminal:
#   bash run_evaluation.sh
# ============================================================
#SBATCH --job-name=rag_eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_error_%j.log
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --time=09:00:00

set -e

echo "Starting Evaluation Job on Node: ${HOSTNAME}"

# ============================================================
# PROJECT ROOT — SLURM-safe detection (must happen first)
# ============================================================
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ============================================================
# ENVIRONMENT — activated after PROJECT_ROOT is known
# ============================================================
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"
if [ ! -f "${VENV_ACTIVATE}" ]; then
    echo "ERROR: virtualenv not found at ${VENV_ACTIVATE}"
    echo "       Run: uv venv && uv pip install -r requirements.txt"
    exit 1
fi
source "${VENV_ACTIVATE}"

# ============================================================
# HuggingFace token
# ============================================================
if [ -z "${HF_TOKEN}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "       export HF_TOKEN=hf_XXXX before running this script."
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

check_hf_access() {
    local model_id="$1"
    local url="https://huggingface.co/${model_id}/resolve/main/config.json"
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${HF_TOKEN}" "${url}" 2>/dev/null)
    [ -z "${http_code}" ] && http_code="000"
    if [ "${http_code}" != "200" ]; then
        echo "ERROR: HuggingFace access check failed for '${model_id}' (HTTP ${http_code})."
        [ "${http_code}" = "401" ] && echo "       Token missing or expired."
        [ "${http_code}" = "403" ] && echo "       Token lacks model permission. Request access at https://huggingface.co/${model_id}"
        [ "${http_code}" = "000" ] && echo "       Network unreachable."
        exit 1
    fi
    echo "HuggingFace access confirmed: ${model_id} (HTTP ${http_code})"
}

# HuggingFace cache — prefer /scratch if >= 30 GB free (both models together
# need ~30 GB on disk: Llama 3.1-8B ≈ 16 GB + Prometheus 2-7B ≈ 14 GB).
_REQUIRED_GB=30
_SCRATCH_DIR="/scratch/${USER}"
if [ -d "${_SCRATCH_DIR}" ]; then
    _FREE_KB=$(df -k "${_SCRATCH_DIR}" 2>/dev/null | awk 'NR==2 {print $4}')
    _FREE_GB=$(( ${_FREE_KB:-0} / 1024 / 1024 ))
else
    _FREE_GB=0
fi
if [ "${_FREE_GB}" -ge "${_REQUIRED_GB}" ]; then
    export HF_HOME="${_SCRATCH_DIR}/.cache/huggingface"
    echo "HF cache    : ${HF_HOME} (${_FREE_GB} GB free on /scratch)"
else
    export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
    echo "HF cache    : ${HF_HOME} (scratch has ${_FREE_GB} GB — using project cache)"
fi
mkdir -p "${HF_HOME}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

mkdir -p logs

echo "========================================"
echo "Job ID    : ${SLURM_JOB_ID:-interactive}"
echo "Node      : ${SLURM_JOB_NODELIST:-local}"
echo "Project   : ${PROJECT_ROOT}"
echo "Start     : $(date +%T)"
echo "========================================"

# ============================================================
# GPU ASSIGNMENT
# ============================================================
N_GPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "Available GPUs: ${N_GPU}"

VLLM_PID=""
PROMETHEUS_PID=""
PROMETHEUS_PORT=8001

# Detect whether vllm is importable; fall back to transformers if not.
HAVE_VLLM=false
if python -c "import vllm" 2>/dev/null; then
    HAVE_VLLM=true
    echo "vLLM        : available"
else
    echo "vLLM        : not installed — using 'transformers' backend"
fi

if [ "${N_GPU}" -ge 2 ]; then
    IFS="," read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N_GPU - 1)))}"
    N_ALLOC=${#_GPUS[@]}
    RAG_GPU="${_GPUS[$((N_ALLOC - 1))]}"
    VLLM_GPUS=$(IFS=,; echo "${_GPUS[*]:0:$((N_ALLOC - 1))}")
    TP_SIZE=$((N_ALLOC - 1))
    USE_GPU=true
    GEN_MEM_UTIL="0.85"
    EVAL_MEM_UTIL="0.85"
    if [ "${HAVE_VLLM}" = true ]; then
        GENERATOR_BACKEND_VALUE=vllm
        echo "GPU layout  : vLLM on [${VLLM_GPUS}] (TP=${TP_SIZE}) | RAG/Eval on [${RAG_GPU}]"
    else
        GENERATOR_BACKEND_VALUE=transformers
        echo "GPU layout  : transformers on all GPUs [${CUDA_VISIBLE_DEVICES:-all}] | RAG/Eval on [${RAG_GPU}]"
    fi

elif [ "${N_GPU}" -eq 1 ]; then
    RAG_GPU="${CUDA_VISIBLE_DEVICES:-0}"
    VLLM_GPUS="${RAG_GPU}"
    TP_SIZE=1
    USE_GPU=true
    GEN_MEM_UTIL="0.60"
    EVAL_MEM_UTIL="0.80"
    if [ "${HAVE_VLLM}" = true ]; then
        GENERATOR_BACKEND_VALUE=vllm
        echo "GPU layout  : single GPU [${RAG_GPU}] (vLLM mode)"
    else
        GENERATOR_BACKEND_VALUE=transformers
        echo "GPU layout  : single GPU [${RAG_GPU}] (transformers mode)"
    fi

else
    USE_GPU=false
    RAG_GPU=""
    GENERATOR_BACKEND_VALUE=ollama
    echo "No GPU detected — CPU/Ollama mode."
    echo "Ensure 'ollama serve' is running and 'llama3' model is pulled."
fi

# Helper — wait for an HTTP server to become ready
wait_for_server() {
    local port="$1"
    local label="$2"
    local max_wait="${3:-600}"
    local elapsed=0
    local pid_var="${4:-}"

    until curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
        if [ -n "${pid_var}" ]; then
            local pid="${!pid_var}"
            if ! kill -0 "${pid}" 2>/dev/null; then
                echo "ERROR: ${label} server (PID ${pid}) exited unexpectedly."
                exit 1
            fi
        fi
        if [ "${elapsed}" -ge "${max_wait}" ]; then
            echo "ERROR: ${label} server did not become ready within ${max_wait}s."
            [ -n "${pid_var}" ] && kill "${!pid_var}" 2>/dev/null || true
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "  ...${label} not ready yet (${elapsed}s)"
    done
    echo "${label} server ready on port ${port}."
}

# ============================================================
# PHASE 1 — Start Llama vLLM server (generation, vLLM only)
# ============================================================
if [ "${USE_GPU}" = true ] && [ "${HAVE_VLLM}" = true ]; then
    check_hf_access "meta-llama/Llama-3.1-8B-Instruct"

    echo "Starting Llama 3.1-8B vLLM server (port 8000, TP=${TP_SIZE})..."
    CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dtype auto \
        --port 8000 \
        --tensor-parallel-size "${TP_SIZE}" \
        --gpu-memory-utilization "${GEN_MEM_UTIL}" \
        --max-model-len 8192 \
        --disable-custom-all-reduce \
        --hf-token "${HF_TOKEN}" &
    VLLM_PID=$!

    wait_for_server 8000 "Llama 3.1-8B" 900 VLLM_PID
fi

# ============================================================
# PHASE 2 — Generate RAG predictions
# ============================================================
echo "Generating RAG predictions (backend=${GENERATOR_BACKEND_VALUE})..."
CUDA_VISIBLE_DEVICES="${RAG_GPU}" \
GENERATOR_BACKEND="${GENERATOR_BACKEND_VALUE}" \
python -m src.evaluation.generate_predictions

# ============================================================
# PHASE 3 — Stop Llama, start Prometheus 2 vLLM (evaluation, vLLM only)
# ============================================================
if [ "${USE_GPU}" = true ] && [ "${HAVE_VLLM}" = true ]; then
    echo "Stopping Llama vLLM (PID ${VLLM_PID}) before starting Prometheus 2..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    VLLM_PID=""
    sleep 5   # allow GPU VRAM to be fully released

    # Prometheus 2 is not a gated model — no HF access check needed.
    echo "Starting Prometheus 2 vLLM server (port ${PROMETHEUS_PORT}, TP=${TP_SIZE})..."
    CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" \
    python -m vllm.entrypoints.openai.api_server \
        --model prometheus-eval/prometheus-7b-v2.0 \
        --dtype auto \
        --port "${PROMETHEUS_PORT}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --gpu-memory-utilization "${EVAL_MEM_UTIL}" \
        --max-model-len 4096 \
        --hf-token "${HF_TOKEN}" &
    PROMETHEUS_PID=$!

    wait_for_server "${PROMETHEUS_PORT}" "Prometheus 2" 600 PROMETHEUS_PID
fi

# ============================================================
# PHASE 4 — Run Prometheus 2 + ALCE evaluation
# ============================================================
echo "Running Prometheus 2 + ALCE evaluation..."
CUDA_VISIBLE_DEVICES="${RAG_GPU}" \
PROMETHEUS_PORT="${PROMETHEUS_PORT}" \
python -m src.evaluation.evaluate_rag

# ============================================================
# CLEANUP
# ============================================================
if [ -n "${PROMETHEUS_PID}" ]; then
    echo "Shutting down Prometheus 2 server (PID ${PROMETHEUS_PID})..."
    kill "${PROMETHEUS_PID}" 2>/dev/null || true
fi
if [ -n "${VLLM_PID}" ]; then
    echo "Shutting down Llama vLLM server (PID ${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
fi

echo "Evaluation Complete! — $(date +%T)"
