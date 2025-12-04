#!/bin/bash
# Submit EuroEval jobs for Swedish finetuned models

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_euroeval.sh"

# Swedish finetuned models to evaluate
MODELS=(
  "/proj/berzelius-aiics-real/users/x_anbue/frodi_data/distillation_swedish/cpt_checkpoint_epoch_1"
  "/proj/berzelius-aiics-real/users/x_anbue/frodi_data/distillation_swedish/sft_checkpoint_epoch_3"
  "/proj/berzelius-aiics-real/users/x_anbue/frodi_data/distillation_swedish/frodi-swedish-student-kd"
)

# Language for evaluation (Swedish)
LANGUAGE="sv"

echo "=================================================="
echo "Submitting EuroEval jobs for Swedish models"
echo "=================================================="
echo ""

for MODEL in "${MODELS[@]}"; do
  MODEL_NAME=$(basename "$MODEL")
  echo "Submitting job for: $MODEL_NAME"
  echo "  Full path: $MODEL"
  
  JOB_ID=$(sbatch \
    --job-name="euroeval-${MODEL_NAME}" \
    --export=ALL,MODEL="$MODEL",LANGUAGE="$LANGUAGE" \
    "$RUN_SCRIPT" | awk '{print $4}')
  
  echo "  Submitted job ID: $JOB_ID"
  echo ""
done

echo "=================================================="
echo "All jobs submitted!"
echo "=================================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "View logs in: /proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/"

