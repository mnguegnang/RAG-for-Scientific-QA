#!/bin/bash
#SBATCH --job-name=rag_eval                 # Name of the job
#SBATCH --output=logs/eval_%j.log           # Save output
#SBATCH --error=logs/eval_error_%j.log      # Save errors
#SBATCH --gres=gpu:2                        
#SBATCH --cpus-per-task=24                   # CPU cores
#SBATCH --mem=40G                           # System RAM
#SBATCH --time=07:00:00                     # 7 Hours limit

echo "Starting Evaluation Job on Node: $HOSTNAME"

# 1. Load your environment
source ~/miniconda3/bin/activate
conda activate qasper-rag

# Ensure you are in your project folder
cd /home/math/nguegnang/RAG_project/qasper-rag-scientific-qa  # Actual project path

# 2. Start the vLLM server in the BACKGROUND (Notice the & at the end)
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype auto \
    --port 8000 \
    --gpu-memory-utilization 0.7 &

# 3. Wait for the A100 to load the weights into VRAM
echo "Waiting 60 seconds for vLLM to boot..."
sleep 60

# 4. Run the Python evaluation script
echo "Starting Ragas Evaluation..."
python -m src.evaluation.evaluate_rag

echo "Evaluation Complete!"