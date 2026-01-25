#!/bin/bash
#SBATCH -A berzelius-2025-264
#SBATCH --job-name=eval_adp_400_900
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/adp_400_900_%j.out
#SBATCH --error=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/adp_400_900_%j.err
#SBATCH --signal=TERM@90

set -euo pipefail

# ==================== Configuration ====================
REPO_DIR="/home/x_anbue/euroeval-test"
RESULTS_JSON="${REPO_DIR}/results/checkpoint_exp_progress.json"
OUTPUT_PLOTS_DIR="${REPO_DIR}/results/checkpoint_exp_plots"

# Adaptive Power checkpoint directory
ADP_PWR_DIR="/proj/berzelius-aiics-real/users/x_anbue/frodi_data/exp_checkpoints/exp_neurips_adaptive_power/eval_checkpoints"

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

mkdir -p "$LOGS_DIR" "$RESULTS_DIR" "$OUTPUT_PLOTS_DIR"

# ==================== Job Info ====================
echo "============================================================"
echo "GRPO + Adaptive Power Evaluation: iter_400 - iter_900"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Checkpoints to evaluate (6 total):"
echo "  - iter_400: $ADP_PWR_DIR/iter_400"
echo "  - iter_500: $ADP_PWR_DIR/iter_500"
echo "  - iter_600: $ADP_PWR_DIR/iter_600"
echo "  - iter_700: $ADP_PWR_DIR/iter_700"
echo "  - iter_800: $ADP_PWR_DIR/iter_800"
echo "  - iter_900: $ADP_PWR_DIR/iter_900"
echo ""
echo "Already evaluated: iter_200, iter_300"
echo "============================================================"

nvidia-smi || true

# ==================== Run Evaluations ====================
cd "$RESULTS_DIR"

for iteration in 400 500 600 700 800 900; do
    checkpoint_path="$ADP_PWR_DIR/iter_$iteration"
    
    echo ""
    echo "============================================================"
    echo "Evaluating: adp_pwr iteration $iteration"
    echo "Path: $checkpoint_path"
    echo "Time: $(date)"
    echo "============================================================"
    
    if [ ! -d "$checkpoint_path" ]; then
        echo "ERROR: Checkpoint not found: $checkpoint_path"
        continue
    fi
    
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
    
    echo "Evaluation exit code: $?"
done

# ==================== Parse Results and Update JSON ====================
echo ""
echo "============================================================"
echo "Parsing results and updating JSON..."
echo "============================================================"

cd "$REPO_DIR"

python3 << 'PYTHON_SCRIPT'
import json
import os
from datetime import datetime

RESULTS_JSON = '/home/x_anbue/euroeval-test/results/checkpoint_exp_progress.json'
EUROEVAL_RESULTS = '/proj/berzelius-aiics-real/users/x_anbue/euroeval/results/euroeval_benchmark_results.jsonl'
ADP_PWR_DIR = "/proj/berzelius-aiics-real/users/x_anbue/frodi_data/exp_checkpoints/exp_neurips_adaptive_power/eval_checkpoints"

SWEDISH_DATASETS = {
    "swerec": {"metric": "test_mcc"},
    "scala-sv": {"metric": "test_mcc"},
    "suc3": {"metric": "test_micro_f1_no_misc"},
    "scandiqa-sv": {"metric": "test_f1"},
    "mmlu-sv": {"metric": "test_mcc"},
    "hellaswag-sv": {"metric": "test_mcc"},
}

with open(RESULTS_JSON, 'r') as f:
    progress_data = json.load(f)

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
    
    for iteration in [400, 500, 600, 700, 800, 900]:
        checkpoint_path = f"{ADP_PWR_DIR}/iter_{iteration}"
        scores = {}
        
        for record in all_results:
            model_path = record.get("model", "")
            
            if checkpoint_path not in model_path:
                continue
            
            if not record.get("validation_split", False):
                continue
            
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
            
            progress_data["runs"]["adp_pwr"]["evaluations"][str(iteration)] = eval_data
            print(f"Updated adp_pwr iter_{iteration}: {len(scores)} datasets, aggregate={aggregate:.2f}")
        else:
            print(f"No validation results found for adp_pwr iter_{iteration}")

progress_data["metadata"]["last_updated"] = datetime.now().isoformat()
with open(RESULTS_JSON, 'w') as f:
    json.dump(progress_data, f, indent=2)

print(f"\nSaved results to: {RESULTS_JSON}")
PYTHON_SCRIPT

# ==================== Generate Plots ====================
echo ""
echo "============================================================"
echo "Generating plots..."
echo "============================================================"

apptainer exec "$APPTAINER_ENV" python3 "$REPO_DIR/scripts/plot_checkpoint_progress.py" \
    --input "$RESULTS_JSON" \
    --output-dir "$OUTPUT_PLOTS_DIR"

# ==================== Completion ====================
echo ""
echo "============================================================"
echo "Adaptive Power 400-900 Evaluation Completed"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results: $RESULTS_JSON"
echo "Plots: $OUTPUT_PLOTS_DIR"
echo ""
echo "✅ Evaluation completed!"
