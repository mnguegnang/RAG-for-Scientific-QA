#!/bin/bash
#SBATCH --job-name=rag_eval                 # Name of the job
#SBATCH --output=logs/eval_%j.log           # Save output
#SBATCH --error=logs/eval_error_%j.log      # Save errors
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24                   # CPU cores
#SBATCH --mem=40G                           # System RAM
#SBATCH --time=07:00:00                     # 7 Hours limit

set -e  # Exit immediately on any error

echo "Starting Evaluation Job on Node: $HOSTNAME"

# 1. Load your environment
source ~/miniconda3/bin/activate
conda activate qasper-rag

# Ensure you are in your project folder
cd /home/math/nguegnang/RAG_project/qasper-rag-scientific-qa

# 2. Start the vLLM server in the BACKGROUND
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype auto \
    --port 8000 \
    --gpu-memory-utilization 0.7 &
VLLM_PID=$!

# 3. Wait for vLLM to become ready (poll health endpoint instead of blind sleep)
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=300  # 5-minute timeout
ELAPSED=0
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: vLLM did not become ready within ${MAX_WAIT}s. Aborting."
        kill "$VLLM_PID" 2>/dev/null
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  ...still waiting (${ELAPSED}s elapsed)"
done
echo "vLLM server is ready."

# 4. Generate RAG predictions (creates data/evaluation_dataset.csv)
#echo "Generating RAG predictions..."
#python -m src.evaluation.generate_predictions

# 5. Run the RAGAS + ALCE evaluation
echo "Starting RAGAS Evaluation..."
python -m src.evaluation.evaluate_rag

echo "Evaluation Complete!"
