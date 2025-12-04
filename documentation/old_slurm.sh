#!/bin/bash
#SBATCH -A berzelius-2025-264
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -J frodi-rl-only
#SBATCH -o /home/x_anbue/frodi/scripts/logs/frodi_rl_only_par%j.out
#SBATCH -e /home/x_anbue/frodi/scripts/logs/frodi_rl_only_par%j.err
#SBATCH --signal=TERM@90

set -euo pipefail

# Environment
export PYTHONPATH="${PYTHONPATH:-}:/home/x_anbue/frodi"
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache/datasets"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
# SECURITY: Do not hardcode secrets. Expect HF_TOKEN to be provided in the environment.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set; access to private/gated models may fail."
fi
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6"
export TRANSFORMERS_VERBOSITY="info"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"
FRODI_PROJECT_DIR="/home/x_anbue/frodi"
DATA_DIR="/proj/berzelius-aiics-real/users/x_anbue/frodi_data"
LOGS_DIR="${FRODI_PROJECT_DIR}/logs"
OUTPUTS_DIR="${FRODI_PROJECT_DIR}/outputs"

mkdir -p "$LOGS_DIR" "$OUTPUTS_DIR" /home/x_anbue/frodi/scripts/logs

echo "=================================================="
echo "FRODI RL-ONLY Training Job Started"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Container: $APPTAINER_ENV"
echo "Project directory: $FRODI_PROJECT_DIR"
echo "Data directory: $DATA_DIR"
echo "HF Cache: $HF_HOME"
echo "=================================================="

echo "GPU Information:"
nvidia-smi || true

export FRODI_RL_ONLY=1

echo "=================================================="
echo "Starting FRODI Phase 2 (RL self-play only)"
echo "=================================================="

cd "$FRODI_PROJECT_DIR"

# Optional: Start periodic GPU power/util logging (every 10s)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Starting GPU power/util logging (10s interval)"
  nvidia-smi dmon -s pucmt -d 10 -o DT -f "${LOGS_DIR}/gpu_dmon_${SLURM_JOB_ID}.csv" >/dev/null 2>&1 &
  GPU_DMON_PID=$!
fi

# Launch with srun so Slurm tracks the step and signals propagate cleanly
srun apptainer exec --nv \
  --env HF_TOKEN="$HF_TOKEN" \
  --env HF_HOME="$HF_HOME" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  --env PYTHONPATH="$PYTHONPATH" \
  --env CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING" \
  --env TORCH_CUDNN_V8_API_ENABLED="$TORCH_CUDNN_V8_API_ENABLED" \
  --env FRODI_RL_ONLY="$FRODI_RL_ONLY" \
  "$APPTAINER_ENV" \
  python run_pipeline.py

EXIT_CODE=$?

echo "=================================================="
echo "FRODI RL-ONLY Training Completed"
echo "=================================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ RL-only training completed successfully!"
  echo "Check outputs in: $OUTPUTS_DIR"
  echo "Training logs: $LOGS_DIR/frodi_rl_only_${SLURM_JOB_ID}.out"
else
  echo "❌ RL-only training failed with exit code: $EXIT_CODE"
  echo "Check error logs: $LOGS_DIR/frodi_rl_only_${SLURM_JOB_ID}.err"
fi

# Stop GPU monitoring if running
if [ -n "${GPU_DMON_PID:-}" ]; then
  kill "$GPU_DMON_PID" 2>/dev/null || true
fi

exit $EXIT_CODE


