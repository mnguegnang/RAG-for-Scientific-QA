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
# VARIABLES
# ============================================================
#export HF_TOKEN="your_hf_token_here"

# Point HF cache to scratch space — avoids filling your home quota
export HF_HOME="/scratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# PyTorch / parallelism
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
echo "========================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_JOB_NODELIST"
echo "Start     : $(date +%T)"
echo "========================================"

mkdir -p logs

nvidia-smi --query-gpu=index,name,memory.total,memory.free \
           --format=csv,noheader

python -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
print(f'GPUs    : {torch.cuda.device_count()}')
"

# ============================================================
# RUN
# ============================================================
echo "--- Starting pipeline_ingest.py at $(date +%T) ---"

python pipeline_ingest.py

EXIT_CODE=$?
echo "--- Finished at $(date +%T) with exit code $EXIT_CODE ---"
exit $EXIT_CODE