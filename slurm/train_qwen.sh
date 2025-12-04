#!/bin/bash
#SBATCH --job-name=qwen_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --account=berzelius-aiics-real
#SBATCH --partition=gpu

# Load modules
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.7.0

# Activate virtual environment
source .venv/bin/activate

# Set distributed training environment variables
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=4
export RANK=$SLURM_PROCID

echo "Starting training on $HOSTNAME"
echo "Job ID: $SLURM_JOBID"

# Run training script
# We use torchrun for distributed training
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT scripts/train.py

echo "Training finished"

