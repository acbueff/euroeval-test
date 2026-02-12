#!/bin/bash
#SBATCH -A berzelius-2025-264
#SBATCH --job-name=eval_sv_test_32b
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/sv_test_32b_%j.out
#SBATCH --error=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/sv_test_32b_%j.err
#SBATCH --signal=TERM@90

set -euo pipefail

# ==================== Configuration ====================
REPO_DIR="/home/x_anbue/euroeval-test"
RESULTS_JSON="${REPO_DIR}/results/swedish_testset_results.json"
OUTPUT_PLOTS_DIR="${REPO_DIR}/results/swedish_testset_plots"

# Judge model (Qwen3-32B)
JUDGE_MODEL_ID="Qwen/Qwen3-32B"

# EuroEval paths
EUROEVAL_BASE="/proj/berzelius-aiics-real/users/x_anbue/euroeval"
CACHE_DIR="${EUROEVAL_BASE}/.euroeval_cache"
RESULTS_DIR="${EUROEVAL_BASE}/results"
LOGS_DIR="${EUROEVAL_BASE}/logs"
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/euroeval.sif"

# HuggingFace cache for model downloads
HF_CACHE_DIR="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"

# ==================== Environment ====================
export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export EUROEVAL_CACHE_DIR="$CACHE_DIR"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6"

mkdir -p "$LOGS_DIR" "$RESULTS_DIR" "$OUTPUT_PLOTS_DIR"

# ==================== Job Info ====================
echo "============================================================"
echo "Swedish EuroEval TEST SET Evaluation: Qwen3-32B Judge Model"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Model to evaluate:"
echo "  Judge Model: $JUDGE_MODEL_ID"
echo ""
echo "HuggingFace Cache: $HF_CACHE_DIR"
echo ""
echo "NOTE: This evaluates on the TEST SET (not validation)"
echo "NOTE: Using 4 GPUs and reduced batch size for 32B model"
echo "============================================================"

nvidia-smi || true

# ==================== Run Evaluation ====================
cd "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "Evaluating: Qwen3-32B Judge Model"
echo "Model ID: $JUDGE_MODEL_ID"
echo "Time: $(date)"
echo "============================================================"

# Run euroeval on TEST SET (using --evaluate-test-split flag)
# Using smaller batch size (4) for the 32B model due to memory constraints
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
        --model "$JUDGE_MODEL_ID" \
        --language sv \
        --cache-dir "$CACHE_DIR" \
        --evaluate-test-split \
        --trust-remote-code \
        --save-results \
        --num-iterations 3 \
        --batch-size 4 \
        --verbose

echo "Evaluation exit code: $?"

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

RESULTS_JSON = '/home/x_anbue/euroeval-test/results/swedish_testset_results.json'
EUROEVAL_RESULTS = '/proj/berzelius-aiics-real/users/x_anbue/euroeval/results/euroeval_benchmark_results.jsonl'

# Judge model config
JUDGE_MODEL_CONFIG = {
    "model_id": "qwen3_32b_judge",
    "name": "Qwen3-32B Judge/Teacher Model",
    "description": "Large Qwen3-32B model used as judge/teacher for distillation and GRPO training",
    "path": "Qwen/Qwen3-32B",
    "color": "#E63946",
    "model_size": "32B"
}

# Swedish datasets and their metrics (TEST SET)
SWEDISH_DATASETS = {
    "swerec": {"metric": "test_mcc", "display": "SweRec (Sentiment)"},
    "scala-sv": {"metric": "test_mcc", "display": "ScaLA-SV (Linguistic)"},
    "suc3": {"metric": "test_micro_f1_no_misc", "display": "SUC3 (NER)"},
    "scandiqa-sv": {"metric": "test_f1", "display": "ScandiQA-SV (Reading)"},
    "mmlu-sv": {"metric": "test_mcc", "display": "MMLU-SV (Knowledge)"},
    "hellaswag-sv": {"metric": "test_mcc", "display": "HellaSwag-SV (Reasoning)"},
}

# Load existing results JSON
if os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON, 'r') as f:
        results_data = json.load(f)
else:
    results_data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "experiment_name": "Swedish EuroEval TEST SET Evaluation",
            "language": "sv",
            "split": "test",
            "description": "Evaluation of Qwen3 models on Swedish EuroEval TEST set - comparing baseline, trained model, and judge model",
            "datasets": list(SWEDISH_DATASETS.keys())
        },
        "models": {}
    }

# Parse euroeval results for judge model
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
    
    model_path = JUDGE_MODEL_CONFIG["path"]
    model_id = JUDGE_MODEL_CONFIG["model_id"]
    scores = {}
    
    for record in all_results:
        record_model_path = record.get("model", "")
        
        # Match model path (Qwen/Qwen3-32B)
        if model_path not in record_model_path:
            continue
        
        # Only TEST set results (validation_split should be False or not present)
        if record.get("validation_split", False):
            continue
        
        # Check for Swedish language
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
                    "metric": metric_key,
                    "display_name": SWEDISH_DATASETS[dataset]["display"]
                }
    
    if scores:
        aggregate = sum(s["value"] for s in scores.values()) / len(scores)
        
        results_data["models"][model_id] = {
            "name": JUDGE_MODEL_CONFIG["name"],
            "description": JUDGE_MODEL_CONFIG["description"],
            "model_path": model_path,
            "model_size": JUDGE_MODEL_CONFIG["model_size"],
            "color": JUDGE_MODEL_CONFIG["color"],
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "split": "test",
            "scores": scores,
            "aggregate_score": aggregate,
            "num_datasets": len(scores)
        }
        print(f"Updated {model_id}: {len(scores)} datasets, aggregate={aggregate:.2f}")
    else:
        print(f"No test results found for {model_id}")

# Update metadata
results_data["metadata"]["last_updated"] = datetime.now().isoformat()

with open(RESULTS_JSON, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"\nSaved results to: {RESULTS_JSON}")
PYTHON_SCRIPT

# ==================== Generate Plots ====================
echo ""
echo "============================================================"
echo "Generating comparison plots..."
echo "============================================================"

# Check if plotting script exists and run it
if [ -f "$REPO_DIR/scripts/plot_swedish_testset.py" ]; then
    apptainer exec "$APPTAINER_ENV" python3 "$REPO_DIR/scripts/plot_swedish_testset.py" \
        --input "$RESULTS_JSON" \
        --output-dir "$OUTPUT_PLOTS_DIR"
else
    echo "Note: Plot script not found. Skipping plot generation."
fi

# ==================== Completion ====================
echo ""
echo "============================================================"
echo "Swedish EuroEval TEST SET Evaluation (32B Model) Completed"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results: $RESULTS_JSON"
echo "Plots: $OUTPUT_PLOTS_DIR"
echo ""
echo "✅ Evaluation completed!"

