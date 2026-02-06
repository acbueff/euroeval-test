#!/bin/bash
#SBATCH -A berzelius-2025-264
#SBATCH --job-name=eval_is_baseline
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/is_baseline_%j.out
#SBATCH --error=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/is_baseline_%j.err
#SBATCH --signal=TERM@90

set -euo pipefail

# ==================== Configuration ====================
REPO_DIR="/home/x_anbue/euroeval-test"
RESULTS_DIR_REPO="${REPO_DIR}/results"
OUTPUT_FILE="${RESULTS_DIR_REPO}/icelandic_baseline_results.json"

# Baseline Qwen3 model (CPT Epoch 1 - before GRPO training)
BASELINE_MODEL="/proj/berzelius-aiics-real/users/x_anbue/frodi_data/distillation_swedish/cpt_checkpoint_epoch_1"

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
# IMPORTANT: Allow downloads for Icelandic data (first time)
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6"

mkdir -p "$LOGS_DIR" "$RESULTS_DIR" "$RESULTS_DIR_REPO"

# ==================== Job Info ====================
echo "============================================================"
echo "Icelandic Baseline Evaluation (Qwen3 CPT Epoch 1)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Model: $BASELINE_MODEL"
echo "Language: Icelandic (is)"
echo "Split: Validation"
echo ""
echo "NOTE: Icelandic datasets will be downloaded on first run"
echo "============================================================"

nvidia-smi || true

# ==================== Run Evaluation ====================
cd "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "Evaluating baseline Qwen3 on Icelandic validation set"
echo "Path: $BASELINE_MODEL"
echo "Time: $(date)"
echo "============================================================"

if [ ! -d "$BASELINE_MODEL" ]; then
    echo "ERROR: Baseline model not found: $BASELINE_MODEL"
    exit 1
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
        --model "$BASELINE_MODEL" \
        --language is \
        --cache-dir "$CACHE_DIR" \
        --evaluate-val-split \
        --trust-remote-code \
        --save-results \
        --num-iterations 3 \
        --batch-size 16 \
        --verbose

EVAL_EXIT=$?
echo "Evaluation exit code: $EVAL_EXIT"

# ==================== Parse Results ====================
echo ""
echo "============================================================"
echo "Parsing Icelandic results..."
echo "============================================================"

cd "$REPO_DIR"

python3 << 'PYTHON_SCRIPT'
import json
import os
from datetime import datetime

EUROEVAL_RESULTS = '/proj/berzelius-aiics-real/users/x_anbue/euroeval/results/euroeval_benchmark_results.jsonl'
OUTPUT_FILE = '/home/x_anbue/euroeval-test/results/icelandic_baseline_results.json'
BASELINE_MODEL = "/proj/berzelius-aiics-real/users/x_anbue/frodi_data/distillation_swedish/cpt_checkpoint_epoch_1"

# Icelandic datasets in EuroEval (common ones)
# Note: These are typical Icelandic datasets - adjust based on what euroeval actually evaluates
ICELANDIC_DATASETS = {
    "mim-gold-ner": {"metric": "test_micro_f1_no_misc"},  # NER
    "scala-is": {"metric": "test_mcc"},                   # Linguistic acceptability
    "nqii": {"metric": "test_f1"},                        # QA
    "wikinews-is": {"metric": "test_mcc"},               # Sentiment
    "mmlu-is": {"metric": "test_mcc"},                   # Knowledge
    "hellaswag-is": {"metric": "test_mcc"},              # Common sense
}

results = {
    "metadata": {
        "created": datetime.now().isoformat(),
        "language": "is",
        "description": "Icelandic EuroEval validation set evaluation - Baseline Qwen3",
        "model": BASELINE_MODEL
    },
    "scores": {},
    "all_datasets": []
}

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
    
    for record in all_results:
        model_path = record.get("model", "")
        
        # Match on our baseline model path
        if BASELINE_MODEL not in model_path:
            continue
        
        # Check if validation split
        if not record.get("validation_split", False):
            continue
        
        # Check if Icelandic
        if "is" not in record.get("dataset_languages", []):
            continue
        
        dataset = record.get("dataset", "")
        results["all_datasets"].append(dataset)
        
        # Try to extract the metric value
        total_results = record.get("results", {}).get("total", {})
        
        # Check known metrics
        metric_value = None
        metric_key = None
        for key in ["test_mcc", "test_f1", "test_micro_f1_no_misc", "test_accuracy"]:
            if key in total_results:
                metric_value = total_results[key]
                metric_key = key
                break
        
        if metric_value is not None:
            metric_se = total_results.get(f"{metric_key}_se", 0)
            results["scores"][dataset] = {
                "value": metric_value,
                "se": metric_se,
                "metric": metric_key
            }
            print(f"Found: {dataset} = {metric_value:.2f} ({metric_key})")
    
    if results["scores"]:
        aggregate = sum(s["value"] for s in results["scores"].values()) / len(results["scores"])
        results["aggregate_score"] = aggregate
        print(f"\nAggregate score: {aggregate:.2f} across {len(results['scores'])} datasets")
    else:
        print("No Icelandic validation results found for baseline model")
        print(f"Datasets found (not validation or not Icelandic): {results['all_datasets']}")

# Save results
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to: {OUTPUT_FILE}")
PYTHON_SCRIPT

# ==================== Completion ====================
echo ""
echo "============================================================"
echo "Icelandic Baseline Evaluation Completed"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results: $OUTPUT_FILE"
echo ""
echo "✅ Evaluation completed!"

