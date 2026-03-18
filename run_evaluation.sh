#!/bin/bash
#SBATCH --job-name=rag_eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_error_%j.log
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --time=07:00:00

set -e  # Exit immediately on any error

echo "Starting Evaluation Job on Node: ${HOSTNAME}"

# ============================================================
# ENVIRONMENT
# ============================================================
source ~/miniconda3/bin/activate
conda activate qasper-rag

# ============================================================
# PROJECT ROOT — SLURM-safe detection
# ============================================================
# Same logic as run_pipeline_ingest.sh: use SLURM_SUBMIT_DIR in SLURM jobs,
# fall back to script location for interactive use.
#
# IMPORTANT: always submit from the project root:
#   cd /home/math/nguegnang/RAG_project/qasper-rag-scientific-qa
#   sbatch run_evaluation.sh
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${PROJECT_ROOT}"
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
# Choose HuggingFace cache: prefer /scratch if it has >=20 GB free,
# otherwise fall back to ~/.cache/huggingface (home directory).
_REQUIRED_GB=20
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
    export HF_HOME="${HOME}/.cache/huggingface"
    echo "HF cache    : ${HF_HOME} (scratch only has ${_FREE_GB} GB free -- using home)"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "========================================"
echo "Job ID    : ${SLURM_JOB_ID:-interactive}"
echo "Node      : ${SLURM_JOB_NODELIST:-local}"
echo "Project   : ${PROJECT_ROOT}"
echo "Start     : $(date +%T)"
echo "========================================"

mkdir -p logs

# ============================================================
# GPU ASSIGNMENT — fully dynamic, no hardcoded count
# ============================================================
# Detect how many GPUs are actually allocated to this job.
N_GPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "Available GPUs: ${N_GPU}"

if [ "${N_GPU}" -ge 2 ]; then
    # Two or more GPUs:
    #   - All GPUs except the last → vLLM (tensor-parallel)
    #   - Last GPU → SPECTER2 + BGE reranker + nomic-embed (RAG pipeline)
    #
    # SLURM sets CUDA_VISIBLE_DEVICES to the allocated physical GPU IDs.
    # We parse that list so we never assume logical indices 0,1,2.
    IFS="," read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N_GPU - 1)))}"
    N_ALLOC=${#_GPUS[@]}
    RAG_GPU="${_GPUS[$((N_ALLOC - 1))]}"
    # Build VLLM_GPUS = all elements except the last
    VLLM_GPUS=$(IFS=,; echo "${_GPUS[*]:0:$((N_ALLOC - 1))}")
    TP_SIZE=$((N_ALLOC - 1))
    USE_GPU=true
    echo "GPU layout  : vLLM on [${VLLM_GPUS}] (tensor-parallel-size=${TP_SIZE})"
    echo "              RAG/Eval on [${RAG_GPU}]"
    GENERATOR_BACKEND_VALUE=vllm

elif [ "${N_GPU}" -eq 1 ]; then
    # Single GPU: run everything on it.
    # vLLM and RAG share the one GPU; tensor-parallel-size must be 1.
    RAG_GPU="${CUDA_VISIBLE_DEVICES:-0}"
    VLLM_GPUS="${RAG_GPU}"
    TP_SIZE=1
    USE_GPU=true
    echo "GPU layout  : single GPU [${RAG_GPU}] for vLLM + RAG/Eval"
    GENERATOR_BACKEND_VALUE=vllm

else
    # CPU-only mode: skip vLLM entirely, fall back to Ollama for generation.
    USE_GPU=false
    RAG_GPU=""
    GENERATOR_BACKEND_VALUE=ollama
    echo "No GPU detected — running in CPU/Ollama mode"
    echo "Make sure 'ollama serve' is running before this step."
fi

# ============================================================
# 2. Start the vLLM server (GPU mode only)
# ============================================================
VLLM_PID=""
if [ "${USE_GPU}" = true ]; then
    # Verify token access before spending time downloading — fail fast.
    check_hf_access "meta-llama/Llama-3.1-8B-Instruct"

    echo "Starting vLLM server (tensor-parallel-size=${TP_SIZE} on GPUs [${VLLM_GPUS}])..."
    CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dtype auto \
        --port 8000 \
        --tensor-parallel-size "${TP_SIZE}" \
        --gpu-memory-utilization 0.85 \
        --disable-custom-all-reduce \
        --hf-token "${HF_TOKEN}" &
    VLLM_PID=$!

    # 3. Wait for vLLM to become ready
    echo "Waiting for vLLM server to be ready..."
    MAX_WAIT=900
    ELAPSED=0
    until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
        # Abort immediately if vLLM exited (e.g. OOM, disk full, auth error)
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM server process exited unexpectedly. Check logs above."
            exit 1
        fi
        if [ "${ELAPSED}" -ge "${MAX_WAIT}" ]; then
            echo "ERROR: vLLM did not become ready within ${MAX_WAIT}s. Aborting."
            kill "${VLLM_PID}" 2>/dev/null
            exit 1
        fi
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        echo "  ...still waiting (${ELAPSED}s elapsed)"
    done
    echo "vLLM server is ready."
fi

# ============================================================
# 4. Generate RAG predictions
# ============================================================
# CUDA_VISIBLE_DEVICES restricts this process to the RAG GPU (or unset on CPU).
# GENERATOR_BACKEND routes LLM calls to vLLM server (GPU) or Ollama (CPU).
echo "Generating RAG predictions (backend=${GENERATOR_BACKEND_VALUE})..."
CUDA_VISIBLE_DEVICES="${RAG_GPU}" \
GENERATOR_BACKEND="${GENERATOR_BACKEND_VALUE}" \
python -m src.evaluation.generate_predictions

# ============================================================
# 5. Run RAGAS + ALCE evaluation
# ============================================================
echo "Starting RAGAS Evaluation (backend=${GENERATOR_BACKEND_VALUE})..."
CUDA_VISIBLE_DEVICES="${RAG_GPU}" \
python -m src.evaluation.evaluate_rag

# ============================================================
# CLEANUP
# ============================================================
if [ -n "${VLLM_PID}" ]; then
    echo "Shutting down vLLM server (PID ${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
fi

echo "Evaluation Complete!"
