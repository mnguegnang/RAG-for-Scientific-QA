#!/bin/bash
# ============================================================
# Evaluation pipeline — Generation (Llama) + Evaluation (Prometheus 2)
#
# Two-phase GPU strategy:
#   Phase 1 — RAG prediction generation
#     vLLM: meta-llama/Llama-3.1-8B-Instruct  (auto-selected free port)
#     RAG:  SPECTER2 + ColBERT v2 reranker
#   Phase 2 — Prometheus 2 evaluation (Kim et al. 2024, arXiv:2405.01535)
#     Llama vLLM is stopped first (frees VRAM on small GPUs like RTX 4090)
#     vLLM: prometheus-eval/prometheus-7b-v2.0 (auto-selected free port)
#
# GPU / CPU auto-selection:
#   N_GPU >= 2 : tensor-parallel across N-1 GPUs for vLLM; last GPU for RAG
#   N_GPU == 1 : single GPU; --gpu-memory-utilization computed from real VRAM
#   N_GPU == 0 : CPU-only; generation via Ollama; evaluation via Ollama+llama3
#
# Storage policy (RunPod: / is a small container overlay, /workspace is the
# persistent network volume). Every cache that can grow or that is expensive
# to rebuild lives on /workspace so the container disk never fills and so a
# pod restart does not throw the caches away:
#   HF_HOME                  model weights           (~30 GB for both models)
#   VLLM_CACHE_ROOT          torch.compile + FlashInfer autotune artifacts
#   TORCHINDUCTOR_CACHE_DIR  Inductor codegen cache
#   TRITON_CACHE_DIR         Triton kernel cache
#   NLTK_DATA                tokenizer corpora
# These are picked up from /workspace/.bashrc_custom when it exists; any value
# already exported in the environment always wins.
#
# Supported environments:
#   RunPod  RTX 4090 (24 GB) / RTX PRO 4500 Blackwell (32 GB) — single-GPU mode
#   SLURM   A100-SXM4-80GB                                    — multi-GPU mode
#
# Usage:
#   # SLURM:
#   sbatch run_evaluation.sh
#   # Interactive / RunPod terminal:
#   bash run_evaluation.sh
#
# Tunables (environment):
#   VLLM_STARTUP_TIMEOUT   readiness budget per server, seconds (default 2700)
#   VLLM_ENFORCE_EAGER=1   skip CUDA-graph capture — much faster cold start,
#                          slower generation. Useful for a smoke test.
#   SKIP_PREFETCH=1        do not pre-download weights before starting a server
# ============================================================
#SBATCH --job-name=rag_eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_error_%j.log
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --time=09:00:00

set -euo pipefail

echo "Starting Evaluation Job on Node: ${HOSTNAME:-unknown}"

# ============================================================
# PROJECT ROOT — SLURM-safe detection (must happen first)
# ============================================================
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ============================================================
# PERSISTENT ENV — /workspace/.bashrc_custom is not sourced by
# non-interactive shells (sbatch, nohup, ssh -c), so load it here.
# Values already exported by the caller take precedence.
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
# Two interpreters, on purpose. vLLM 0.28 requires transformers>=5.5 and
# huggingface_hub>=1.27; the SPECTER2 encoder needs `adapters`, which pins
# transformers~=4.51 and the pre-1.0 huggingface_hub API. uv proves those are
# unsatisfiable together, and no `adapters` release supports transformers 5.x.
# They do not need to share an environment: the pipeline only ever reaches
# vLLM over HTTP, never by importing it.
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"
if [ ! -f "${VENV_ACTIVATE}" ]; then
    echo "ERROR: virtualenv not found at ${VENV_ACTIVATE}"
    echo "       Run: uv venv && uv pip install -r requirements.txt"
    exit 1
fi
# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"

RAG_PY="${PROJECT_ROOT}/.venv/bin/python"
VLLM_PY="${PROJECT_ROOT}/.venv-vllm/bin/python"
if [ ! -x "${VLLM_PY}" ]; then
    # No dedicated vLLM env — fall back to the project env and let the
    # HAVE_VLLM probe below decide whether vLLM is usable at all.
    VLLM_PY="${RAG_PY}"
    echo "NOTE: .venv-vllm not found — looking for vLLM in .venv instead."
    echo "      To create it:  uv venv --python 3.10 .venv-vllm &&"
    echo "                     uv pip install --python .venv-vllm/bin/python vllm==0.28.0"
fi
echo "Interpreters: RAG=${RAG_PY}"
echo "              vLLM=${VLLM_PY}"

# The RunPod image exports HF_HUB_ENABLE_HF_TRANSFER=1. If the hf_transfer
# module is missing from an environment, every single HuggingFace download
# raises instead of falling back — and transformers reports that as an
# unrelated "Unrecognized model ... no model_type key" error.
for _py in "${RAG_PY}" "${VLLM_PY}"; do
    if [ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" = "1" ] && ! "${_py}" -c "import hf_transfer" 2>/dev/null; then
        echo "WARNING: HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is missing from ${_py}"
        echo "         — disabling it for this run (pip install hf_transfer to keep fast downloads)."
        export HF_HUB_ENABLE_HF_TRANSFER=0
    fi
done

# ============================================================
# HuggingFace token
# ============================================================
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "       export HF_TOKEN=hf_XXXX before running this script."
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

LLAMA_MODEL="meta-llama/Llama-3.1-8B-Instruct"
PROMETHEUS_MODEL="prometheus-eval/prometheus-7b-v2.0"

check_hf_access() {
    local model_id="$1"
    local url="https://huggingface.co/${model_id}/resolve/main/config.json"
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${HF_TOKEN}" "${url}" 2>/dev/null || true)
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

# ============================================================
# CACHE PLACEMENT — keep the container disk empty, keep the
# expensive compile artifacts across pod restarts.
# ============================================================
_WORKSPACE_ROOT="/workspace"
[ -d "${_WORKSPACE_ROOT}" ] || _WORKSPACE_ROOT="${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-${_WORKSPACE_ROOT}/hf_cache}"
export NLTK_DATA="${NLTK_DATA:-${_WORKSPACE_ROOT}/nltk_data}"
# vLLM keeps torch.compile + FlashInfer autotune artifacts here. Rebuilding
# them costs ~20 min on a new GPU architecture; persisting them is the single
# biggest cold-start win, and they are only ~25 MB.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${_WORKSPACE_ROOT}/.vllm_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${_WORKSPACE_ROOT}/.inductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${_WORKSPACE_ROOT}/.triton_cache}"
mkdir -p "${HF_HOME}" "${NLTK_DATA}" "${VLLM_CACHE_ROOT}" \
         "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# One-time migration: earlier revisions of this script cached weights inside
# the project tree. Both paths live on the same volume, so moving the model
# directories is a metadata rename, not a 15 GB copy or re-download.
_LEGACY_HUB="${PROJECT_ROOT}/.cache/huggingface/hub"
_TARGET_HUB="${HF_HOME}/hub"
if [ -d "${_LEGACY_HUB}" ] && [ "${_LEGACY_HUB}" != "${_TARGET_HUB}" ]; then
    mkdir -p "${_TARGET_HUB}"
    for _d in "${_LEGACY_HUB}"/models--*; do
        [ -d "${_d}" ] || continue
        if [ ! -d "${_TARGET_HUB}/$(basename "${_d}")" ]; then
            echo "Migrating cached model $(basename "${_d}") -> ${_TARGET_HUB}"
            mv "${_d}" "${_TARGET_HUB}/" 2>/dev/null || \
                echo "  (migration skipped — different filesystem; will re-download)"
        fi
    done
fi

# The torch.compile / FlashInfer artifacts are what make a first start cost
# ~20 extra minutes on a new GPU architecture. If a previous run left them in
# the default ~/.cache/vllm on the container overlay, carry them across so the
# work is not repeated and does not sit on the small container disk.
_LEGACY_VLLM_CACHE="${HOME:-/root}/.cache/vllm"
if [ -d "${_LEGACY_VLLM_CACHE}" ] && [ "${_LEGACY_VLLM_CACHE}" != "${VLLM_CACHE_ROOT}" ] \
   && [ -z "$(ls -A "${VLLM_CACHE_ROOT}" 2>/dev/null)" ]; then
    echo "Carrying over vLLM compile cache from ${_LEGACY_VLLM_CACHE}"
    cp -a "${_LEGACY_VLLM_CACHE}/." "${VLLM_CACHE_ROOT}/" 2>/dev/null || true
fi

# Disk guard — both models together need ~30 GB in HF_HOME.
_FREE_GB=$(df -PBG "${HF_HOME}" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
_FREE_GB="${_FREE_GB:-0}"
_CACHED_GB=$(du -sBG "${HF_HOME}" 2>/dev/null | awk '{gsub("G","",$1); print $1}')
_CACHED_GB="${_CACHED_GB:-0}"
echo "HF cache    : ${HF_HOME} (${_CACHED_GB} GB cached, ${_FREE_GB} GB free)"
echo "vLLM cache  : ${VLLM_CACHE_ROOT}"
if [ "${_FREE_GB}" -lt 35 ] && [ "${_CACHED_GB}" -lt 28 ]; then
    echo "WARNING: only ${_FREE_GB} GB free on ${HF_HOME}; both models need ~30 GB."
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

mkdir -p logs
RUN_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

echo "========================================"
echo "Job ID    : ${SLURM_JOB_ID:-interactive}"
echo "Node      : ${SLURM_JOB_NODELIST:-local}"
echo "Project   : ${PROJECT_ROOT}"
echo "Start     : $(date +%T)"
echo "========================================"

# ============================================================
# SERVER LIFECYCLE HELPERS
# ============================================================
VLLM_PID=""
PROMETHEUS_PID=""

# Kill a server by its process-group leader: SIGTERM the whole group, escalate
# to SIGKILL if it does not go. vLLM runs its engine in a child process
# (VLLM::EngineCore) that holds the VRAM, and the front end cannot service a
# signal while the engine is inside torch.compile — signalling only the leader
# PID is how a run leaks a server that keeps ~20 GB of VRAM allocated.
stop_server() {
    local pid="$1" label="$2"
    [ -n "${pid}" ] || return 0
    kill -0 "${pid}" 2>/dev/null || return 0

    echo "Stopping ${label} (PID ${pid})..."
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true

    local waited=0
    while kill -0 "${pid}" 2>/dev/null && [ "${waited}" -lt 60 ]; do
        sleep 2; waited=$((waited + 2))
    done
    if kill -0 "${pid}" 2>/dev/null; then
        echo "  ${label} ignored SIGTERM after ${waited}s — sending SIGKILL to the process group."
        kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
        sleep 3
    fi
    wait "${pid}" 2>/dev/null || true
    echo "  ${label} stopped."
}

# List PIDs of vLLM servers/engines belonging to an earlier run.
#
# `pgrep -f vllm.entrypoints...` is not safe here: -f matches any command line
# that merely *contains* the string, so a shell, an editor or a `tail` of the
# server log gets matched too. Select from /proc instead, require the process
# to actually be a Python interpreter (or the EngineCore worker), and never
# match anything in this script's own process tree.
list_stale_vllm() {
    python - "$$" <<'PYEOF'
import os, sys

self_pid = int(sys.argv[1])
try:
    self_pgid = os.getpgid(self_pid)
except OSError:
    self_pgid = -1

found = []
for entry in os.listdir("/proc"):
    if not entry.isdigit():
        continue
    pid = int(entry)
    if pid in (self_pid, os.getpid()):
        continue
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as fh:
            argv = [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
        if not argv:
            continue
        with open("/proc/%d/comm" % pid) as fh:
            comm = fh.read().strip()
        if os.getpgid(pid) == self_pgid:      # our own process tree
            continue
    except (OSError, ProcessLookupError):
        continue

    is_engine = comm.startswith("VLLM::")
    is_server = (os.path.basename(argv[0]).startswith("python")
                 and "vllm.entrypoints.openai.api_server" in argv)
    if is_engine or is_server:
        found.append(pid)

print(" ".join(str(p) for p in found))
PYEOF
}

# Reap anything left behind by an earlier aborted run. A stale server holding
# VRAM is the reason a fresh vLLM cannot allocate its KV pool.
kill_stale_vllm() {
    local stale
    stale=$(list_stale_vllm 2>/dev/null || true)
    [ -n "${stale// /}" ] || return 0

    echo "Found stale vLLM process(es) from a previous run: ${stale}"
    for p in ${stale}; do
        kill -TERM "${p}" 2>/dev/null || true
    done
    sleep 10

    # The front end cannot service SIGTERM while its engine is inside
    # torch.compile, and EngineCore is re-parented to init when the front end
    # dies — so escalate unconditionally on whatever is still there.
    stale=$(list_stale_vllm 2>/dev/null || true)
    for p in ${stale}; do
        echo "  force-killing ${p}"
        kill -KILL "${p}" 2>/dev/null || true
    done
    sleep 5
    return 0
}

# Advisory: give the driver a moment to actually release the freed VRAM.
wait_for_vram_release() {
    [ "${USE_GPU:-false}" = true ] || return 0
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local waited=0 used
    while [ "${waited}" -lt 60 ]; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        used="${used:-0}"
        if [ "${used}" -lt 2000 ]; then
            echo "GPU VRAM released (${used} MiB in use)."
            return 0
        fi
        sleep 3; waited=$((waited + 3))
    done
    echo "WARNING: ${used} MiB of VRAM still in use after ${waited}s — the next server may fail to allocate."
    return 0
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    stop_server "${PROMETHEUS_PID}" "Prometheus 2 server"
    stop_server "${VLLM_PID}" "Llama vLLM server"
    exit "${rc}"
}
trap cleanup EXIT INT TERM

# Launch a vLLM server in its own session and echo its REAL pid.
#
# `setsid cmd &` plus `$!` is not reliable. Whether setsid forks depends on
# whether the backgrounded child is already a process-group leader, which
# depends on whether the calling shell has job control enabled. When it does
# fork, `$!` is the short-lived setsid wrapper: the liveness check sees a dead
# pid and declares a phantom crash while the real server keeps running and
# holding VRAM — and the next launch then competes with it for the GPU and
# dies with "No available memory for the cache blocks".
#
# Instead, let the new session leader record its own pid and exec the server
# over itself, so the pid stays valid and equals the process-group id.
start_vllm_server() {
    local logfile="$1" pidfile="$2"
    shift 2
    rm -f "${pidfile}"
    # The server runs from its own virtualenv's interpreter, which does NOT put
    # that venv's bin/ on PATH. FlashInfer and torch.compile shell out to
    # `ninja` (and friends) during JIT, so without this the engine dies with
    # FileNotFoundError: 'ninja' several minutes into startup.
    local bindir
    bindir="$(cd "$(dirname "$1")" 2>/dev/null && pwd)" || bindir=""
    setsid bash -c '
        [ -n "$1" ] && export PATH="$1:${PATH}"
        echo $$ > "$2"
        shift 2
        exec "$@"' _ "${bindir}" "${pidfile}" "$@" \
        > "${logfile}" 2>&1 &
    local waited=0
    while [ ! -s "${pidfile}" ] && [ "${waited}" -lt 30 ]; do
        sleep 1
        waited=$((waited + 1))
    done
    if [ ! -s "${pidfile}" ]; then
        echo "ERROR: the server never reported its pid (see ${logfile})." >&2
        return 1
    fi
    cat "${pidfile}"
}

# Pick a port nothing is listening on AND that we can actually bind.
# Probing with a plain connect is not enough: on RunPod nginx listens on 8001
# and answers /health with HTTP 200, which makes a naive readiness check pass
# instantly while vLLM never started.
find_free_port() {
    python - "$1" <<'PY'
import socket, sys
start = int(sys.argv[1])
for port in range(start, start + 200):
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", port))
    except OSError:
        continue
    finally:
        s.close()
    print(port)
    sys.exit(0)
sys.exit(1)
PY
}

# Readiness probe that cannot be satisfied by an unrelated web server:
# /v1/models must come back listing the model we asked vLLM to serve.
server_is_serving() {
    local port="$1" model="$2"
    curl -sf --max-time 5 "http://localhost:${port}/v1/models" 2>/dev/null \
        | grep -Fq "\"${model}\""
}

# Wait for a vLLM server, failing fast and loudly when it dies.
wait_for_server() {
    local port="$1" label="$2" model="$3" max_wait="$4" pid="$5" logfile="$6"
    local elapsed=0

    echo "Waiting for ${label} on port ${port} (budget ${max_wait}s, log: ${logfile})"
    until server_is_serving "${port}" "${model}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: ${label} server (PID ${pid}) exited after ${elapsed}s."
            echo "----- last 40 lines of ${logfile} -----"
            tail -n 40 "${logfile}" 2>/dev/null || echo "(no log)"
            echo "---------------------------------------"
            return 1
        fi
        if [ "${elapsed}" -ge "${max_wait}" ]; then
            echo "ERROR: ${label} server did not become ready within ${max_wait}s."
            echo "----- last 40 lines of ${logfile} -----"
            tail -n 40 "${logfile}" 2>/dev/null || echo "(no log)"
            echo "---------------------------------------"
            echo "HINT: a first start on a new GPU architecture pays torch.compile +"
            echo "      FlashInfer autotune (~20 min). Those artifacts persist in"
            echo "      ${VLLM_CACHE_ROOT}, so the next start is far quicker."
            echo "      Raise VLLM_STARTUP_TIMEOUT, or set VLLM_ENFORCE_EAGER=1 to"
            echo "      skip CUDA-graph capture entirely."
            return 1
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        # Echo vLLM's own progress instead of a content-free countdown.
        if [ $((elapsed % 60)) -eq 0 ]; then
            echo "  [${elapsed}s] ${label} still starting — $(tail -n 1 "${logfile}" 2>/dev/null | cut -c1-140)"
        fi
    done
    echo "${label} server ready on port ${port} after ${elapsed}s."
}

# Download weights before the server starts, so the readiness budget covers
# model load and compilation only — not a 15 GB transfer.
prefetch_model() {
    local model_id="$1"
    [ "${SKIP_PREFETCH:-0}" = "1" ] && return 0
    echo "Prefetching ${model_id} into ${HF_HOME}..."
    python - "${model_id}" <<'PYEOF'
import os, sys
from huggingface_hub import HfApi, snapshot_download

model_id = sys.argv[1]
token = os.environ.get("HF_TOKEN")

# Never pull the duplicate original/*.pth checkpoint Meta ships alongside the
# safetensors shards (another full copy of the weights), nor repo docs.
ignore = ["*.pth", "*.h5", "*.msgpack", "*.gguf", "original/*",
          "*.md", "LICENSE*", ".gitattributes"]
try:
    files = HfApi().list_repo_files(model_id, token=token)
    if any(f.endswith(".safetensors") for f in files):
        ignore.append("*.bin")
except Exception as exc:                      # offline / transient API failure
    print(f"  (could not list repo files: {exc})")

try:
    path = snapshot_download(model_id, token=token, ignore_patterns=ignore,
                             max_workers=8)
except Exception as exc:
    # A network blip must not abort a run whose weights are already on disk.
    print(f"  download failed: {exc}")
    try:
        path = snapshot_download(model_id, ignore_patterns=ignore,
                                 local_files_only=True)
    except Exception:
        print("  and the model is not fully cached locally — cannot continue.")
        sys.exit(1)
    print("  continuing with the already-cached copy.")

print(f"  cached at {path}")
PYEOF
}

# ============================================================
# GPU ASSIGNMENT
# ============================================================
N_GPU=$("${RAG_PY}" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "Available GPUs: ${N_GPU}"

# Detect whether vllm is importable; fall back to transformers if not.
HAVE_VLLM=false
if "${VLLM_PY}" -c "import vllm" 2>/dev/null; then
    HAVE_VLLM=true
    echo "vLLM        : available"
else
    echo "vLLM        : not installed — using 'transformers' backend"
fi

# During generation the RAG stack (SPECTER2 + ColBERT v2 + torch context) shares
# the GPU with vLLM, so its memory must be carved out of the utilisation budget.
# During evaluation the judge talks to vLLM over HTTP and needs almost nothing.
RAG_RESERVE_MIB=5120
EVAL_RESERVE_MIB=2048
LLAMA_WEIGHTS_MIB=16000       # 8.03 B params in bf16

# util = (total - reserve) / total, clamped — a fixed fraction is wrong because
# it does not know the card. The old hard-coded 0.60 gave vLLM 14.4 GB on a
# 24 GB 4090, which is less than the 16 GB of weights it must load.
compute_mem_util() {
    local total="$1" reserve="$2"
    awk -v t="${total}" -v r="${reserve}" 'BEGIN {
        u = (t - r) / t;
        if (u > 0.92) u = 0.92;
        if (u < 0.50) u = 0.50;
        printf "%.2f", u;
    }'
}

if [ "${N_GPU}" -ge 2 ]; then
    IFS="," read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N_GPU - 1)))}"
    N_ALLOC=${#_GPUS[@]}
    RAG_GPU="${_GPUS[$((N_ALLOC - 1))]}"
    VLLM_GPUS=$(IFS=,; echo "${_GPUS[*]:0:$((N_ALLOC - 1))}")
    TP_SIZE=$((N_ALLOC - 1))
    USE_GPU=true
    # vLLM has its own GPUs here — nothing else competes for that VRAM.
    GEN_MEM_UTIL="0.90"
    EVAL_MEM_UTIL="0.90"
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

    TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    TOTAL_MIB="${TOTAL_MIB:-24000}"
    GEN_MEM_UTIL=$(compute_mem_util "${TOTAL_MIB}" "${RAG_RESERVE_MIB}")
    EVAL_MEM_UTIL=$(compute_mem_util "${TOTAL_MIB}" "${EVAL_RESERVE_MIB}")

    GEN_POOL_MIB=$(awk -v t="${TOTAL_MIB}" -v u="${GEN_MEM_UTIL}" 'BEGIN {printf "%d", t * u}')
    if [ "${GEN_POOL_MIB}" -lt "${LLAMA_WEIGHTS_MIB}" ]; then
        echo "ERROR: this GPU has ${TOTAL_MIB} MiB total; after reserving"
        echo "       ${RAG_RESERVE_MIB} MiB for the RAG models only ${GEN_POOL_MIB} MiB is"
        echo "       left for vLLM, which is below the ~${LLAMA_WEIGHTS_MIB} MiB of bf16 weights."
        echo "       Use a larger GPU, or run generation with GENERATOR_BACKEND=ollama."
        exit 1
    fi
    if [ "${HAVE_VLLM}" = true ]; then
        GENERATOR_BACKEND_VALUE=vllm
        echo "GPU layout  : single GPU [${RAG_GPU}], ${TOTAL_MIB} MiB (vLLM mode)"
        echo "              gen util=${GEN_MEM_UTIL} (~${GEN_POOL_MIB} MiB pool, ${RAG_RESERVE_MIB} MiB reserved for RAG)"
        echo "              eval util=${EVAL_MEM_UTIL}"
    else
        GENERATOR_BACKEND_VALUE=transformers
        echo "GPU layout  : single GPU [${RAG_GPU}] (transformers mode)"
    fi

else
    USE_GPU=false
    RAG_GPU=""
    GENERATOR_BACKEND_VALUE=ollama
    GEN_MEM_UTIL=""
    EVAL_MEM_UTIL=""
    echo "No GPU detected — CPU/Ollama mode."
    echo "Ensure 'ollama serve' is running and 'llama3' model is pulled."
fi

STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-2700}"
EAGER_FLAG=()
if [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ]; then
    EAGER_FLAG=(--enforce-eager)
    echo "vLLM        : --enforce-eager (CUDA-graph capture skipped)"
fi

# ============================================================
# PREFLIGHT — the retrieval indices must exist. Without this the
# failure surfaces as a traceback deep inside HybridRetriever after
# the LLM server has already been started and warmed up.
# ============================================================
_MISSING_INDEX=""
for _f in data/indices/dense.index data/indices/dense.index.meta data/indices/sparse.pkl; do
    [ -f "${_f}" ] || _MISSING_INDEX="${_MISSING_INDEX} ${_f}"
done
if [ -n "${_MISSING_INDEX}" ]; then
    echo "ERROR: retrieval indices are missing:${_MISSING_INDEX}"
    echo "       Build them first (dense and sparse must always be built together"
    echo "       so their chunk ordering stays aligned):"
    echo "         ${RAG_PY} -m src.pipeline_ingest"
    exit 1
fi
echo "Indices     : data/indices present"

# ============================================================
# PHASE 0 — Reclaim anything a previous run leaked
# ============================================================
if [ "${USE_GPU}" = true ] && [ "${HAVE_VLLM}" = true ]; then
    kill_stale_vllm
    wait_for_vram_release
fi

# ============================================================
# PHASE 1 — Start Llama vLLM server (generation, vLLM only)
# ============================================================
LLAMA_PORT=8000
if [ "${USE_GPU}" = true ] && [ "${HAVE_VLLM}" = true ]; then
    check_hf_access "${LLAMA_MODEL}"
    prefetch_model "${LLAMA_MODEL}"

    LLAMA_PORT=$(find_free_port 8000) || { echo "ERROR: no free port for the Llama server."; exit 1; }
    [ "${LLAMA_PORT}" != "8000" ] && echo "NOTE: port 8000 is taken — using ${LLAMA_PORT} instead."
    LLAMA_LOG="logs/vllm_llama_${RUN_TAG}.log"

    echo "Starting Llama 3.1-8B vLLM server (port ${LLAMA_PORT}, TP=${TP_SIZE})..."
    VLLM_PID=$(CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" \
        start_vllm_server "${LLAMA_LOG}" "logs/.llama_${RUN_TAG}.pid" \
        "${VLLM_PY}" -m vllm.entrypoints.openai.api_server \
        --model "${LLAMA_MODEL}" \
        --dtype auto \
        --port "${LLAMA_PORT}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --gpu-memory-utilization "${GEN_MEM_UTIL}" \
        --max-model-len 8192 \
        --disable-custom-all-reduce \
        --hf-token "${HF_TOKEN}" \
        "${EAGER_FLAG[@]}")
    echo "  server pid ${VLLM_PID}"

    wait_for_server "${LLAMA_PORT}" "Llama 3.1-8B" "${LLAMA_MODEL}" \
        "${STARTUP_TIMEOUT}" "${VLLM_PID}" "${LLAMA_LOG}"
fi
export VLLM_API_URL="http://localhost:${LLAMA_PORT}/v1"

# ============================================================
# PHASE 2 — Generate RAG predictions
# ============================================================
echo "Generating RAG predictions (backend=${GENERATOR_BACKEND_VALUE}, endpoint=${VLLM_API_URL})..."
CUDA_VISIBLE_DEVICES="${RAG_GPU}" \
GENERATOR_BACKEND="${GENERATOR_BACKEND_VALUE}" \
VLLM_API_URL="${VLLM_API_URL}" \
"${RAG_PY}" -m src.evaluation.generate_predictions

# ============================================================
# PHASE 3 — Stop Llama, start Prometheus 2 vLLM (evaluation, vLLM only)
# ============================================================
PROMETHEUS_PORT=8001
if [ "${USE_GPU}" = true ] && [ "${HAVE_VLLM}" = true ]; then
    echo "Stopping Llama vLLM before starting Prometheus 2..."
    stop_server "${VLLM_PID}" "Llama vLLM server"
    VLLM_PID=""
    wait_for_vram_release

    # Prometheus 2 is not a gated model — no HF access check needed.
    prefetch_model "${PROMETHEUS_MODEL}"

    # 8001 is nginx's port on RunPod, and it answers /health with 200. Pick a
    # port we can genuinely bind so the judge cannot end up talking to nginx.
    PROMETHEUS_PORT=$(find_free_port 8001) || { echo "ERROR: no free port for the Prometheus server."; exit 1; }
    [ "${PROMETHEUS_PORT}" != "8001" ] && echo "NOTE: port 8001 is taken (nginx on RunPod) — using ${PROMETHEUS_PORT} instead."
    PROM_LOG="logs/vllm_prometheus_${RUN_TAG}.log"

    echo "Starting Prometheus 2 vLLM server (port ${PROMETHEUS_PORT}, TP=${TP_SIZE})..."
    PROMETHEUS_PID=$(CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" \
        start_vllm_server "${PROM_LOG}" "logs/.prometheus_${RUN_TAG}.pid" \
        "${VLLM_PY}" -m vllm.entrypoints.openai.api_server \
        --model "${PROMETHEUS_MODEL}" \
        --dtype auto \
        --port "${PROMETHEUS_PORT}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --gpu-memory-utilization "${EVAL_MEM_UTIL}" \
        --max-model-len 4096 \
        --hf-token "${HF_TOKEN}" \
        "${EAGER_FLAG[@]}")
    echo "  server pid ${PROMETHEUS_PID}"

    wait_for_server "${PROMETHEUS_PORT}" "Prometheus 2" "${PROMETHEUS_MODEL}" \
        "${STARTUP_TIMEOUT}" "${PROMETHEUS_PID}" "${PROM_LOG}"
fi

# ============================================================
# PHASE 4 — Run Prometheus 2 + ALCE evaluation
# ============================================================
echo "Running Prometheus 2 + ALCE evaluation (judge on port ${PROMETHEUS_PORT})..."
CUDA_VISIBLE_DEVICES="${RAG_GPU}" \
PROMETHEUS_PORT="${PROMETHEUS_PORT}" \
"${RAG_PY}" -m src.evaluation.evaluate_rag

# ============================================================
# CLEANUP — handled by the EXIT trap (stops both servers and
# their EngineCore children on every exit path).
# ============================================================
echo "Evaluation Complete! — $(date +%T)"
