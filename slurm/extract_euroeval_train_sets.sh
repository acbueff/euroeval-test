#!/bin/bash
#SBATCH -A berzelius-2025-264
#SBATCH --job-name=extract_train
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/extract_train_%j.out
#SBATCH --error=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/extract_train_%j.err

set -euo pipefail

# ==================== Configuration ====================
REPO_DIR="/home/x_anbue/euroeval-test"
EUROEVAL_BASE="/proj/berzelius-aiics-real/users/x_anbue/euroeval"
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/euroeval.sif"
LOGS_DIR="${EUROEVAL_BASE}/logs"

mkdir -p "$LOGS_DIR"

# ==================== Job Info ====================
echo "============================================================"
echo "EuroEval Training Set Extraction"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================"

# ==================== Run Extraction ====================
apptainer exec \
    "$APPTAINER_ENV" \
    python3 "$REPO_DIR/scripts/extract_euroeval_train_sets.py"

# ==================== List Output ====================
echo ""
echo "============================================================"
echo "Extracted Training Sets"
echo "============================================================"

OUTPUT_DIR="${EUROEVAL_BASE}/train_sets"

for lang in is de sv; do
    echo ""
    echo "=== ${lang^^} ==="
    if [ -d "$OUTPUT_DIR/$lang" ]; then
        ls -lh "$OUTPUT_DIR/$lang"/*.json 2>/dev/null || echo "No JSON files found"
    else
        echo "Directory not created"
    fi
done

echo ""
cat "$OUTPUT_DIR/extraction_summary.json" 2>/dev/null || echo "No summary file"

# ==================== Completion ====================
echo ""
echo "============================================================"
echo "Extraction Completed"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Output: $OUTPUT_DIR"
echo "Done!"
