#!/bin/bash
#SBATCH --job-name=qwen_eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --account=berzelius-aiics-real

# Load modules
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.7.0

# Activate virtual environment
source .venv/bin/activate

echo "Starting evaluation pipeline"

# 1. Evaluate on Training Split (Custom)
echo "Running Training Split Evaluation..."
python scripts/eval_train.py

# 2. Evaluate on Validation Split (EuroEval)
echo "Running Validation Split Evaluation..."
python scripts/eval_validation.py

# 3. Evaluate on Test Split (EuroEval)
echo "Running Test Split Evaluation..."
python scripts/eval_test.py

echo "Evaluation pipeline finished"

