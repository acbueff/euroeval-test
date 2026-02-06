#!/bin/bash
#SBATCH -A berzelius-2025-264
#SBATCH --job-name=eval_is_checkpts
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/is_checkpts_%j.out
#SBATCH --error=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/is_checkpts_%j.err
#SBATCH --signal=TERM@90

set -euo pipefail

# ==================== Configuration ====================
REPO_DIR="/home/x_anbue/euroeval-test"
RESULTS_JSON="${REPO_DIR}/results/icelandic/icelandic_checkpoint_progress.json"
OUTPUT_PLOTS_DIR="${REPO_DIR}/results/icelandic/plots"

# Icelandic GRPO checkpoint directory
ICELANDIC_CHECKPOINTS_DIR="/proj/berzelius-aiics-real/users/x_anbue/frodi_data_icelandic/self_play_rl_icelandic/eval_checkpoints"

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
echo "Icelandic GRPO Checkpoint Evaluation"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Checkpoints to evaluate (6 total):"
echo "  - iter_100: $ICELANDIC_CHECKPOINTS_DIR/iter_100"
echo "  - iter_200: $ICELANDIC_CHECKPOINTS_DIR/iter_200"
echo "  - iter_300: $ICELANDIC_CHECKPOINTS_DIR/iter_300"
echo "  - iter_400: $ICELANDIC_CHECKPOINTS_DIR/iter_400"
echo "  - iter_500: $ICELANDIC_CHECKPOINTS_DIR/iter_500"
echo "  - iter_600: $ICELANDIC_CHECKPOINTS_DIR/iter_600"
echo ""
echo "Language: Icelandic (is)"
echo "Split: Validation"
echo "============================================================"

nvidia-smi || true

# ==================== Run Evaluations ====================
cd "$RESULTS_DIR"

for iteration in 100 200 300 400 500 600; do
    checkpoint_path="$ICELANDIC_CHECKPOINTS_DIR/iter_$iteration"
    
    echo ""
    echo "============================================================"
    echo "Evaluating: Icelandic GRPO iteration $iteration"
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
            --language is \
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
echo "Parsing Icelandic results and updating JSON..."
echo "============================================================"

cd "$REPO_DIR"

python3 << 'PYTHON_SCRIPT'
import json
import os
from datetime import datetime

RESULTS_JSON = '/home/x_anbue/euroeval-test/results/icelandic/icelandic_checkpoint_progress.json'
EUROEVAL_RESULTS = '/proj/berzelius-aiics-real/users/x_anbue/euroeval/results/euroeval_benchmark_results.jsonl'
ICELANDIC_CHECKPOINTS_DIR = "/proj/berzelius-aiics-real/users/x_anbue/frodi_data_icelandic/self_play_rl_icelandic/eval_checkpoints"

# Icelandic datasets with their primary metrics
ICELANDIC_DATASETS = {
    "hotter-and-colder-sentiment": {"metric": "test_mcc"},
    "scala-is": {"metric": "test_mcc"},
    "mim-gold-ner": {"metric": "test_micro_f1_no_misc"},
    "nqii": {"metric": "test_f1"},
    "icelandic-knowledge": {"metric": "test_mcc"},
    "winogrande-is": {"metric": "test_mcc"},
}

# Load existing progress JSON
with open(RESULTS_JSON, 'r') as f:
    progress_data = json.load(f)

if not os.path.exists(EUROEVAL_RESULTS):
    print(f"Warning: EuroEval results file not found: {EUROEVAL_RESULTS}")
else:
    # Load all euroeval results
    all_results = []
    with open(EUROEVAL_RESULTS, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(all_results)} total euroeval records")
    
    # Process each checkpoint iteration
    for iteration in [100, 200, 300, 400, 500, 600]:
        checkpoint_path = f"{ICELANDIC_CHECKPOINTS_DIR}/iter_{iteration}"
        scores = {}
        
        for record in all_results:
            model_path = record.get("model", "")
            
            # Match on checkpoint path
            if checkpoint_path not in model_path:
                continue
            
            # Check if validation split
            if not record.get("validation_split", False):
                continue
            
            # Check if Icelandic
            if "is" not in record.get("dataset_languages", []):
                continue
            
            dataset = record.get("dataset", "")
            if dataset in ICELANDIC_DATASETS:
                metric_key = ICELANDIC_DATASETS[dataset]["metric"]
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
            
            progress_data["runs"]["icelandic_grpo"]["evaluations"][str(iteration)] = eval_data
            print(f"Updated icelandic_grpo iter_{iteration}: {len(scores)} datasets, aggregate={aggregate:.2f}")
        else:
            print(f"No validation results found for icelandic_grpo iter_{iteration}")

# Update timestamp
progress_data["metadata"]["last_updated"] = datetime.now().isoformat()

# Save updated JSON
with open(RESULTS_JSON, 'w') as f:
    json.dump(progress_data, f, indent=2)

print(f"\nSaved results to: {RESULTS_JSON}")
PYTHON_SCRIPT

# ==================== Generate Plots ====================
echo ""
echo "============================================================"
echo "Generating Icelandic checkpoint progress plots..."
echo "============================================================"

apptainer exec "$APPTAINER_ENV" python3 "$REPO_DIR/scripts/plot_icelandic_progress.py" \
    --input "$RESULTS_JSON" \
    --output-dir "$OUTPUT_PLOTS_DIR"

# ==================== Completion ====================
echo ""
echo "============================================================"
echo "Icelandic Checkpoint Evaluation Completed"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results: $RESULTS_JSON"
echo "Plots: $OUTPUT_PLOTS_DIR"
echo ""
echo "✅ Evaluation completed!"

