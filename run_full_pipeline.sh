#!/bin/bash

# Exit on error
set -e

echo "Submitting full training and evaluation pipeline..."

# 1. Submit Training Job
echo "Submitting training job..."
TRAIN_JOB_ID=$(sbatch --parsable slurm/train_qwen.sh)
echo "Training job submitted with ID: $TRAIN_JOB_ID"

# 2. Submit Evaluation Job (Dependent on Training)
echo "Submitting evaluation job (depends on $TRAIN_JOB_ID)..."
EVAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB_ID slurm/eval_euroeval.sh)
echo "Evaluation job submitted with ID: $EVAL_JOB_ID"

echo "Pipeline submitted successfully."
echo "Monitor status with 'squeue -u $USER'"

