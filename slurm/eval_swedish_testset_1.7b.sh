#!/bin/bash
#SBATCH -A berzelius-2025-264
#SBATCH --job-name=eval_sv_test_1.7b
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/sv_test_1.7b_%j.out
#SBATCH --error=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/sv_test_1.7b_%j.err
#SBATCH --signal=TERM@90

set -euo pipefail

# ==================== Configuration ====================
REPO_DIR="/home/x_anbue/euroeval-test"
RESULTS_JSON="${REPO_DIR}/results/swedish_testset_results.json"
OUTPUT_PLOTS_DIR="${REPO_DIR}/results/swedish_testset_plots"

# Model paths
BASELINE_QWEN_PATH="/proj/berzelius-aiics-real/users/x_anbue/frodi_data/distillation_swedish/cpt_checkpoint_epoch_1"
TRAINED_ITER_300_PATH="/proj/berzelius-aiics-real/users/x_anbue/frodi_data/self_play_rl_swedish/eval_checkpoints/iter_300"

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
echo "Swedish EuroEval TEST SET Evaluation: 1.7B Models"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Models to evaluate:"
echo "  1. Baseline Qwen3 1.7B (no training): $BASELINE_QWEN_PATH"
echo "  2. Trained iter_300: $TRAINED_ITER_300_PATH"
echo ""
echo "NOTE: This evaluates on the TEST SET (not validation)"
echo "============================================================"

nvidia-smi || true

# ==================== Run Evaluations ====================
cd "$RESULTS_DIR"

# Define models to evaluate
declare -a MODELS=(
    "qwen_baseline|Qwen3 1.7B Baseline (No Training)|$BASELINE_QWEN_PATH"
    "grpo_iter_300|GRPO Trained iter_300|$TRAINED_ITER_300_PATH"
)

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id model_name model_path <<< "$model_entry"
    
    echo ""
    echo "============================================================"
    echo "Evaluating: $model_name"
    echo "Model ID: $model_id"
    echo "Path: $model_path"
    echo "Time: $(date)"
    echo "============================================================"
    
    if [ ! -d "$model_path" ]; then
        echo "ERROR: Model not found: $model_path"
        continue
    fi
    
    # Run euroeval on TEST SET (using --evaluate-test-split flag)
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
            --model "$model_path" \
            --language sv \
            --cache-dir "$CACHE_DIR" \
            --evaluate-test-split \
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

RESULTS_JSON = '/home/x_anbue/euroeval-test/results/swedish_testset_results.json'
EUROEVAL_RESULTS = '/proj/berzelius-aiics-real/users/x_anbue/euroeval/results/euroeval_benchmark_results.jsonl'

# Model paths to match
MODEL_CONFIGS = {
    "qwen_baseline": {
        "name": "Qwen3 1.7B Baseline (No Training)",
        "description": "Base Qwen3 1.7B model before any GRPO training - CPT Epoch 1",
        "path": "/proj/berzelius-aiics-real/users/x_anbue/frodi_data/distillation_swedish/cpt_checkpoint_epoch_1",
        "color": "#888888",
        "model_size": "1.7B"
    },
    "grpo_iter_300": {
        "name": "GRPO Trained iter_300",
        "description": "Best performing GRPO checkpoint at iteration 300",
        "path": "/proj/berzelius-aiics-real/users/x_anbue/frodi_data/self_play_rl_swedish/eval_checkpoints/iter_300",
        "color": "#3A86FF",
        "model_size": "1.7B"
    }
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

# Initialize or load results JSON
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

# Parse euroeval results
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
    
    for model_id, config in MODEL_CONFIGS.items():
        model_path = config["path"]
        scores = {}
        
        for record in all_results:
            record_model_path = record.get("model", "")
            
            # Match model path
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
                "name": config["name"],
                "description": config["description"],
                "model_path": model_path,
                "model_size": config["model_size"],
                "color": config["color"],
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

# ==================== Completion ====================
echo ""
echo "============================================================"
echo "Swedish EuroEval TEST SET Evaluation (1.7B Models) Completed"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results: $RESULTS_JSON"
echo ""
echo "NOTE: The 32B judge model requires separate evaluation with more resources."
echo "      Run: sbatch slurm/eval_swedish_testset_32b.sh"
echo ""
echo "✅ Evaluation completed!"

