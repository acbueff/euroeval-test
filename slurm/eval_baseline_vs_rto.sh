#!/bin/bash
#SBATCH -A berzelius-2025-264
#SBATCH --job-name=eval_rto_exp
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/rto_exp_%j.out
#SBATCH --error=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/rto_exp_%j.err
#SBATCH --signal=TERM@90

set -euo pipefail

# ==================== Configuration ====================
REPO_DIR="/home/x_anbue/euroeval-test"
RESULTS_JSON="${REPO_DIR}/results/checkpoint_exp_progress.json"
OUTPUT_PLOTS_DIR="${REPO_DIR}/results/checkpoint_exp_plots"

# RTO checkpoint path (only these need evaluation - baseline already done)
RTO_DIR="/proj/berzelius-aiics-real/users/x_anbue/frodi_data/exp_checkpoints/exp_loss_rto/eval_checkpoints"

# EuroEval paths
EUROEVAL_BASE="/proj/berzelius-aiics-real/users/x_anbue/euroeval"
CACHE_DIR="${EUROEVAL_BASE}/.euroeval_cache"
RESULTS_DIR="${EUROEVAL_BASE}/results"
LOGS_DIR="${EUROEVAL_BASE}/logs"
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/euroeval.sif"

# ==================== Environment ====================
export HF_HOME="${CACHE_DIR}/model_cache"
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export EUROEVAL_CACHE_DIR="$CACHE_DIR"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6"

# Create directories
mkdir -p "$LOGS_DIR" "$RESULTS_DIR" "$OUTPUT_PLOTS_DIR"

# ==================== Job Info ====================
echo "============================================================"
echo "RTO Experiment Evaluation (Baseline already evaluated)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "RTO Checkpoints to evaluate:"
echo "  1. RTO iter_100: $RTO_DIR/iter_100"
echo "  2. RTO iter_200: $RTO_DIR/iter_200"
echo "  3. RTO iter_300: $RTO_DIR/iter_300"
echo ""
echo "Baseline data already exists in JSON (iterations 0, 100, 200, 300)"
echo "============================================================"

nvidia-smi || true

# ==================== Run RTO Evaluations ====================
cd "$RESULTS_DIR"

for iteration in 100 200 300; do
    checkpoint_path="$RTO_DIR/iter_$iteration"
    
    echo ""
    echo "============================================================"
    echo "Evaluating: RTO iteration $iteration"
    echo "Path: $checkpoint_path"
    echo "Time: $(date)"
    echo "============================================================"
    
    # Check if checkpoint exists
    if [ ! -d "$checkpoint_path" ]; then
        echo "ERROR: Checkpoint not found: $checkpoint_path"
        continue
    fi
    
    # Run euroeval (validation set)
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
            --model "$checkpoint_path" \
            --language sv \
            --cache-dir "$CACHE_DIR" \
            --evaluate-val-split \
            --trust-remote-code \
            --save-results \
            --num-iterations 3 \
            --batch-size 16 \
            --verbose
    
    EVAL_EXIT=$?
    echo "Evaluation exit code: $EVAL_EXIT"
done

# ==================== Parse Results and Update JSON ====================
echo ""
echo "============================================================"
echo "Parsing RTO results and updating JSON..."
echo "============================================================"

cd "$REPO_DIR"

python3 << 'PYTHON_SCRIPT'
import json
import os
from datetime import datetime

RESULTS_JSON = '/home/x_anbue/euroeval-test/results/checkpoint_exp_progress.json'
EUROEVAL_RESULTS = '/proj/berzelius-aiics-real/users/x_anbue/euroeval/results/euroeval_benchmark_results.jsonl'
RTO_DIR = '/proj/berzelius-aiics-real/users/x_anbue/frodi_data/exp_checkpoints/exp_loss_rto/eval_checkpoints'

SWEDISH_DATASETS = {
    "swerec": {"metric": "test_mcc"},
    "scala-sv": {"metric": "test_mcc"},
    "suc3": {"metric": "test_micro_f1_no_misc"},
    "scandiqa-sv": {"metric": "test_f1"},
    "mmlu-sv": {"metric": "test_mcc"},
    "hellaswag-sv": {"metric": "test_mcc"},
}

# Load existing JSON
with open(RESULTS_JSON, 'r') as f:
    progress_data = json.load(f)

# Parse euroeval results for RTO checkpoints
if not os.path.exists(EUROEVAL_RESULTS):
    print(f"Warning: EuroEval results file not found: {EUROEVAL_RESULTS}")
else:
    all_results = []
    with open(EUROEVAL_RESULTS, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    for iteration in [100, 200, 300]:
        checkpoint_path = f"{RTO_DIR}/iter_{iteration}"
        scores = {}
        
        for record in all_results:
            model_path = record.get("model", "")
            
            # Check if this result matches our RTO checkpoint
            if "exp_loss_rto" not in model_path:
                continue
            if f"iter_{iteration}" not in model_path:
                continue
            
            # Check if validation split
            if not record.get("validation_split", False):
                continue
            
            # Check if Swedish
            if "sv" not in record.get("dataset_languages", []):
                continue
            
            dataset = record.get("dataset", "")
            if dataset in SWEDISH_DATASETS:
                metric_key = SWEDISH_DATASETS[dataset]["metric"]
                metric_value = record.get("results", {}).get("total", {}).get(metric_key)
                metric_se = record.get("results", {}).get("total", {}).get(f"{metric_key}_se", 0)
                
                if metric_value is not None:
                    scores[dataset] = {
                        "value": metric_value,
                        "se": metric_se,
                        "metric": metric_key
                    }
        
        if scores:
            aggregate = sum(s["value"] for s in scores.values()) / len(scores)
            
            eval_data = {
                "iteration": iteration,
                "checkpoint_path": checkpoint_path,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "scores": scores,
                "aggregate_score": aggregate
            }
            
            progress_data["runs"]["rto"]["evaluations"][str(iteration)] = eval_data
            print(f"Updated RTO iter_{iteration}: {len(scores)} datasets, aggregate={aggregate:.2f}")
        else:
            print(f"No validation results found for RTO iter_{iteration}")

# Save updated JSON
progress_data["metadata"]["last_updated"] = datetime.now().isoformat()
with open(RESULTS_JSON, 'w') as f:
    json.dump(progress_data, f, indent=2)

print(f"\nSaved results to: {RESULTS_JSON}")
PYTHON_SCRIPT

# ==================== Generate Plots ====================
echo ""
echo "============================================================"
echo "Generating experiment comparison plots..."
echo "============================================================"

apptainer exec "$APPTAINER_ENV" python3 "$REPO_DIR/scripts/plot_checkpoint_progress.py" \
    --input "$RESULTS_JSON" \
    --output-dir "$OUTPUT_PLOTS_DIR"

# ==================== Completion ====================
echo ""
echo "============================================================"
echo "RTO Experiment Evaluation Completed"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  JSON: $RESULTS_JSON"
echo "  Plots: $OUTPUT_PLOTS_DIR"
echo ""

ls -la "$RESULTS_JSON" 2>/dev/null || true
ls -la "$OUTPUT_PLOTS_DIR" 2>/dev/null || true

echo ""
echo "✅ Experiment evaluation completed!"
