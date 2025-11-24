#!/bin/bash
#SBATCH --job-name=euroeval_qwen
#SBATCH --account=berzelius-2024-XX  # REPLACE WITH YOUR PROJECT ID
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Load Apptainer module
module load Apptainer

# Define paths (Using absolute paths recommended on HPC)
PROJECT_ROOT=$(pwd)
CONTAINER_PATH="${PROJECT_ROOT}/containers/euroeval.sif"
RESULTS_PATH="${PROJECT_ROOT}/results"

# Ensure results dir exists
mkdir -p "$RESULTS_PATH"

# Export HF Home to project storage to avoid filling up home directory
export HF_HOME="${PROJECT_ROOT}/models"

# Run the evaluation
# We map the project root to /workspace in the container
apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --env HF_HOME=/workspace/models \
    "$CONTAINER_PATH" \
    scandeval --model Qwen/Qwen2.5-1.5B-Instruct --language de --output_dir /workspace/results --device cuda

