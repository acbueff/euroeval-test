#!/bin/bash
#SBATCH -A berzelius-2025-264
#SBATCH --job-name=euroeval
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/%j.out
#SBATCH --error=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/%j.err
#SBATCH --signal=TERM@90

set -euo pipefail

# ==================== Configuration ====================
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
LANGUAGE="${LANGUAGE:-de}"
EUROEVAL_BASE="/proj/berzelius-aiics-real/users/x_anbue/euroeval"
RESULTS_DIR="${EUROEVAL_BASE}/results"
LOGS_DIR="${EUROEVAL_BASE}/logs"
CACHE_DIR="${EUROEVAL_BASE}/.euroeval_cache"
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/euroeval.sif"

# ==================== Environment ====================
# Point HF_HOME to euroeval's model cache so cached models are found
export HF_HOME="${CACHE_DIR}/model_cache"
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export EUROEVAL_CACHE_DIR="$CACHE_DIR"
# Online mode needed for euroeval to fetch model metadata from HF Hub
# Model weights will still use local cache if available
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
# Note: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF
export PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6"

# HF_TOKEN check
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set; access to private/gated models may fail."
fi

# Create directories
mkdir -p "$RESULTS_DIR" "$LOGS_DIR" "$CACHE_DIR"

# ==================== Job Info ====================
echo "=================================================="
echo "EuroEval Benchmark Job Started"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Model: $MODEL"
echo "Language: $LANGUAGE"
echo "Container: $APPTAINER_ENV"
echo "Results directory: $RESULTS_DIR"
echo "EuroEval cache: $CACHE_DIR"
echo "HF Cache: $HF_HOME"
echo "=================================================="

echo "GPU Information:"
nvidia-smi || true

# ==================== Run EuroEval ====================
echo ""
echo "Starting EuroEval benchmark..."
echo ""

# Change to results directory so euroeval saves results there
cd "$RESULTS_DIR"

apptainer exec --nv \
  --pwd "$RESULTS_DIR" \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env HF_HOME="$HF_HOME" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env EUROEVAL_CACHE_DIR="$CACHE_DIR" \
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  --env TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
  --env PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF" \
  --env SSL_CERT_FILE="" \
  --env REQUESTS_CA_BUNDLE="" \
  --env CURL_CA_BUNDLE="" \
  "$APPTAINER_ENV" \
  euroeval \
    --model "$MODEL" \
    --language "$LANGUAGE" \
    --cache-dir "$CACHE_DIR" \
    --evaluate-val-split \
    --trust-remote-code \
    --save-results \
    --num-iterations 3 \
    --batch-size 16 \
    --verbose

EXIT_CODE=$?

# List any results that were saved
echo ""
echo "Contents of results directory:"
ls -la "$RESULTS_DIR" || true

# ==================== Completion ====================
echo ""
echo "=================================================="
echo "EuroEval Benchmark Completed"
echo "=================================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ Evaluation completed successfully!"
  echo "Results saved to: $RESULTS_DIR"
  echo "Logs: $LOGS_DIR/${SLURM_JOB_ID}.out"
else
  echo "❌ Evaluation failed with exit code: $EXIT_CODE"
  echo "Check error logs: $LOGS_DIR/${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE

