#!/bin/bash
#SBATCH -A berzelius-2025-264
#SBATCH --job-name=eval_ckpt_progress
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/ckpt_progress_%j.out
#SBATCH --error=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/ckpt_progress_%j.err
#SBATCH --signal=TERM@90

set -euo pipefail

# ==================== Configuration ====================
# Override these via sbatch --export or environment variables
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/proj/berzelius-aiics-real/users/x_anbue/frodi_data/self_play_rl_swedish/eval_checkpoints}"
RESULTS_JSON="${RESULTS_JSON:-/home/x_anbue/euroeval-test/results/checkpoint_progress.json}"
REPO_DIR="${REPO_DIR:-/home/x_anbue/euroeval-test}"
DRY_RUN="${DRY_RUN:-0}"

# EuroEval paths
EUROEVAL_BASE="/proj/berzelius-aiics-real/users/x_anbue/euroeval"
CACHE_DIR="${EUROEVAL_BASE}/.euroeval_cache"
LOGS_DIR="${EUROEVAL_BASE}/logs"
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/euroeval.sif"

# ==================== Environment ====================
export HF_HOME="${CACHE_DIR}/model_cache"
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export EUROEVAL_CACHE_DIR="$CACHE_DIR"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6"

# Check HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN is not set; access to private/gated models may fail."
fi

# Create directories
mkdir -p "$LOGS_DIR" "$(dirname "$RESULTS_JSON")"

# ==================== Job Info ====================
echo "============================================================"
echo "GRPO Checkpoint Evaluation Pipeline"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Checkpoint directory: $CHECKPOINT_DIR"
echo "  Results JSON: $RESULTS_JSON"
echo "  Repository: $REPO_DIR"
echo "  Container: $APPTAINER_ENV"
echo "  Cache: $CACHE_DIR"
echo "  Dry run: $DRY_RUN"
echo "============================================================"

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi || true

# ==================== Find Unevaluated Checkpoints ====================
echo ""
echo "Scanning for checkpoint iterations..."

# Get available iterations
ITERATIONS=()
for dir in "$CHECKPOINT_DIR"/iter_*; do
    if [ -d "$dir" ] && [ -f "$dir/config.json" ]; then
        iter_num=$(basename "$dir" | sed 's/iter_//')
        ITERATIONS+=("$iter_num")
    fi
done

# Sort iterations
IFS=$'\n' ITERATIONS=($(sort -n <<<"${ITERATIONS[*]}")); unset IFS

echo "Available iterations: ${ITERATIONS[*]}"

# Check which are already evaluated
UNEVALUATED=()
if [ -f "$RESULTS_JSON" ]; then
    echo "Found existing results file: $RESULTS_JSON"
    for iter in "${ITERATIONS[@]}"; do
        # Check if iteration exists in JSON with completed status
        if ! grep -q "\"$iter\".*\"completed\"" "$RESULTS_JSON" 2>/dev/null; then
            UNEVALUATED+=("$iter")
        fi
    done
else
    echo "No existing results file found, will evaluate all iterations"
    UNEVALUATED=("${ITERATIONS[@]}")
fi

echo "Iterations to evaluate: ${UNEVALUATED[*]:-none}"

if [ ${#UNEVALUATED[@]} -eq 0 ]; then
    echo ""
    echo "No new iterations to evaluate!"
    echo "Generating plots from existing results..."
    
    # Still generate plots
    cd "$REPO_DIR"
    if [ -f "$RESULTS_JSON" ]; then
        python scripts/plot_checkpoint_progress.py --input "$RESULTS_JSON"
    fi
    
    echo ""
    echo "============================================================"
    echo "Pipeline completed (no new evaluations)"
    echo "============================================================"
    exit 0
fi

# ==================== Run Evaluations ====================
echo ""
echo "Starting evaluation pipeline..."

RESULTS_DIR="${EUROEVAL_BASE}/results"
mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR"

for iter in "${UNEVALUATED[@]}"; do
    CHECKPOINT_PATH="$CHECKPOINT_DIR/iter_$iter"
    
    echo ""
    echo "============================================================"
    echo "Evaluating: iter_$iter"
    echo "Path: $CHECKPOINT_PATH"
    echo "Time: $(date)"
    echo "============================================================"
    
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY RUN] Would evaluate $CHECKPOINT_PATH"
        continue
    fi
    
    # Run euroeval on this checkpoint (VALIDATION SET - no --evaluate-test-split)
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
            --model "$CHECKPOINT_PATH" \
            --language sv \
            --cache-dir "$CACHE_DIR" \
            --trust-remote-code \
            --save-results \
            --num-iterations 3 \
            --batch-size 16 \
            --verbose
    
    EVAL_EXIT=$?
    
    if [ $EVAL_EXIT -ne 0 ]; then
        echo "Warning: Evaluation for iter_$iter exited with code $EVAL_EXIT"
    fi
    
    echo ""
    echo "Completed iter_$iter at $(date)"
done

# ==================== Parse Results and Update JSON ====================
echo ""
echo "============================================================"
echo "Parsing results and updating JSON..."
echo "============================================================"

cd "$REPO_DIR"

# Create/update the JSON file by parsing euroeval results
python3 << 'PYTHON_SCRIPT'
import json
import os
import re
from datetime import datetime
from pathlib import Path

CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', '/proj/berzelius-aiics-real/users/x_anbue/frodi_data/self_play_rl_swedish/eval_checkpoints')
RESULTS_JSON = os.environ.get('RESULTS_JSON', '/home/x_anbue/euroeval-test/results/checkpoint_progress.json')
EUROEVAL_RESULTS = '/proj/berzelius-aiics-real/users/x_anbue/euroeval/results/euroeval_benchmark_results.jsonl'

SWEDISH_DATASETS = {
    "swerec": {"metric": "test_mcc"},
    "scala-sv": {"metric": "test_mcc"},
    "suc3": {"metric": "test_micro_f1_no_misc"},
    "scandiqa-sv": {"metric": "test_f1"},
    "swedn": {"metric": "test_bertscore"},
    "mmlu-sv": {"metric": "test_mcc"},
    "hellaswag-sv": {"metric": "test_mcc"},
}

# Load existing data
if os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON, 'r') as f:
        progress_data = json.load(f)
else:
    progress_data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "checkpoint_base_dir": CHECKPOINT_DIR,
            "language": "sv",
            "description": "GRPO training checkpoint evaluation on Swedish EuroEval validation set"
        },
        "evaluations": {}
    }

# Find all checkpoint iterations
checkpoint_iters = []
for item in Path(CHECKPOINT_DIR).iterdir():
    if item.is_dir() and item.name.startswith("iter_"):
        match = re.match(r"iter_(\d+)", item.name)
        if match and (item / "config.json").exists():
            checkpoint_iters.append(int(match.group(1)))

checkpoint_iters.sort()
print(f"Found checkpoints: {checkpoint_iters}")

# Parse euroeval results
if not os.path.exists(EUROEVAL_RESULTS):
    print(f"Warning: EuroEval results file not found: {EUROEVAL_RESULTS}")
else:
    euroeval_data = {}
    with open(EUROEVAL_RESULTS, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                model_path = record.get("model", "")
                
                # Check if Swedish
                if "sv" not in record.get("dataset_languages", []):
                    continue
                
                dataset = record.get("dataset", "")
                if dataset not in SWEDISH_DATASETS:
                    continue
                
                # Check if this is one of our checkpoints
                for iter_num in checkpoint_iters:
                    iter_path = f"iter_{iter_num}"
                    if iter_path in model_path:
                        if iter_num not in euroeval_data:
                            euroeval_data[iter_num] = {"scores": {}}
                        
                        metric_key = SWEDISH_DATASETS[dataset]["metric"]
                        metric_value = record.get("results", {}).get("total", {}).get(metric_key)
                        metric_se = record.get("results", {}).get("total", {}).get(f"{metric_key}_se", 0)
                        
                        if metric_value is not None:
                            euroeval_data[iter_num]["scores"][dataset] = {
                                "value": metric_value,
                                "se": metric_se,
                                "metric": metric_key
                            }
                        break
            except json.JSONDecodeError:
                continue
    
    # Update progress data with parsed results
    for iter_num, data in euroeval_data.items():
        scores = [s["value"] for s in data["scores"].values()]
        aggregate = sum(scores) / len(scores) if scores else 0.0
        
        progress_data["evaluations"][str(iter_num)] = {
            "iteration": iter_num,
            "checkpoint_path": os.path.join(CHECKPOINT_DIR, f"iter_{iter_num}"),
            "timestamp": datetime.now().isoformat(),
            "status": "completed" if scores else "failed",
            "scores": data["scores"],
            "aggregate_score": aggregate
        }
        
        print(f"Updated iter_{iter_num}: {len(scores)} datasets, aggregate={aggregate:.2f}")

# Save updated JSON
progress_data["metadata"]["last_updated"] = datetime.now().isoformat()
os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
with open(RESULTS_JSON, 'w') as f:
    json.dump(progress_data, f, indent=2)

print(f"\nSaved results to: {RESULTS_JSON}")
PYTHON_SCRIPT

# ==================== Generate Plots ====================
echo ""
echo "============================================================"
echo "Generating progress plots..."
echo "============================================================"

if [ -f "$RESULTS_JSON" ]; then
    # Run plotting inside container (has numpy/matplotlib)
    apptainer exec "$APPTAINER_ENV" python3 "$REPO_DIR/scripts/plot_checkpoint_progress.py" --input "$RESULTS_JSON"
else
    echo "Warning: Results JSON not found, skipping plots"
fi

# ==================== Completion ====================
echo ""
echo "============================================================"
echo "Checkpoint Evaluation Pipeline Completed"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  JSON: $RESULTS_JSON"
echo "  Plots: $REPO_DIR/results/checkpoint_plots/"
echo ""

# List output files
echo "Generated files:"
ls -la "$REPO_DIR/results/checkpoint_progress.json" 2>/dev/null || true
ls -la "$REPO_DIR/results/checkpoint_plots/" 2>/dev/null || true

echo ""
echo "✅ Pipeline completed successfully!"

