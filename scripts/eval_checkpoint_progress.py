#!/usr/bin/env python3
"""
Evaluate GRPO Training Checkpoints on Swedish EuroEval Validation Set

This script:
1. Finds all checkpoint iterations in the eval_checkpoints directory
2. Checks which iterations have already been evaluated (from JSON)
3. Runs euroeval on unevaluated checkpoints (validation set only)
4. Saves results to a JSON file in the repo
5. Generates progress plots

Usage:
    python scripts/eval_checkpoint_progress.py [--dry-run] [--force-eval ITER]
"""

import json
import os
import sys
import re
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# ==================== Configuration ====================
CHECKPOINT_BASE_DIR = "/proj/berzelius-aiics-real/users/x_anbue/frodi_data/self_play_rl_swedish/eval_checkpoints"
RESULTS_JSON_PATH = "/home/x_anbue/euroeval-test/results/checkpoint_progress.json"
EUROEVAL_CACHE_DIR = "/proj/berzelius-aiics-real/users/x_anbue/euroeval/.euroeval_cache"
APPTAINER_ENV = "/proj/berzelius-aiics-real/users/x_anbue/euroeval.sif"
EUROEVAL_RESULTS_DIR = "/proj/berzelius-aiics-real/users/x_anbue/euroeval/results"

# Swedish datasets and their metrics (validation set uses same datasets)
SWEDISH_DATASETS = {
    "swerec": {"task": "sentiment-classification", "metric": "test_mcc", "display": "SweRec"},
    "scala-sv": {"task": "linguistic-acceptability", "metric": "test_mcc", "display": "ScaLA-SV"},
    "suc3": {"task": "named-entity-recognition", "metric": "test_micro_f1_no_misc", "display": "SUC3"},
    "scandiqa-sv": {"task": "reading-comprehension", "metric": "test_f1", "display": "ScandiQA-SV"},
    "swedn": {"task": "summarization", "metric": "test_bertscore", "display": "SweDN"},
    "mmlu-sv": {"task": "knowledge", "metric": "test_mcc", "display": "MMLU-SV"},
    "hellaswag-sv": {"task": "common-sense-reasoning", "metric": "test_mcc", "display": "HellaSwag-SV"},
}


def load_progress_json(json_path: str) -> Dict[str, Any]:
    """Load existing progress data from JSON file."""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing JSON ({e}), starting fresh")
    
    return {
        "metadata": {
            "created": datetime.now().isoformat(),
            "checkpoint_base_dir": CHECKPOINT_BASE_DIR,
            "language": "sv",
            "description": "GRPO training checkpoint evaluation on Swedish EuroEval validation set"
        },
        "evaluations": {}
    }


def save_progress_json(data: Dict[str, Any], json_path: str) -> None:
    """Save progress data to JSON file."""
    data["metadata"]["last_updated"] = datetime.now().isoformat()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved progress to: {json_path}")


def find_checkpoint_iterations(base_dir: str) -> List[int]:
    """Find all iteration checkpoints in the base directory."""
    iterations = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Warning: Checkpoint directory does not exist: {base_dir}")
        return []
    
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("iter_"):
            match = re.match(r"iter_(\d+)", item.name)
            if match:
                iteration = int(match.group(1))
                # Verify it has model files
                if (item / "config.json").exists():
                    iterations.append(iteration)
                else:
                    print(f"Warning: {item} missing config.json, skipping")
    
    return sorted(iterations)


def get_evaluated_iterations(progress_data: Dict[str, Any]) -> List[int]:
    """Get list of iterations that have already been evaluated."""
    evaluated = []
    for iter_key, eval_data in progress_data.get("evaluations", {}).items():
        if eval_data.get("status") == "completed":
            iteration = int(iter_key)
            evaluated.append(iteration)
    return evaluated


def run_euroeval_on_checkpoint(
    checkpoint_path: str,
    iteration: int,
    dry_run: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Run euroeval on a checkpoint and return results.
    
    Uses the validation set (not test set) by omitting --evaluate-test-split.
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating checkpoint: iter_{iteration}")
    print(f"Path: {checkpoint_path}")
    print(f"{'=' * 60}\n")
    
    if dry_run:
        print("[DRY RUN] Would run euroeval here")
        return None
    
    # Build euroeval command
    # NOTE: We use validation set by NOT including --evaluate-test-split
    cmd = [
        "apptainer", "exec", "--nv",
        "--pwd", EUROEVAL_RESULTS_DIR,
        "--env", f"HF_HOME={EUROEVAL_CACHE_DIR}/model_cache",
        "--env", f"HF_DATASETS_CACHE={EUROEVAL_CACHE_DIR}/datasets",
        "--env", f"EUROEVAL_CACHE_DIR={EUROEVAL_CACHE_DIR}",
        "--env", "HF_HUB_OFFLINE=0",
        "--env", "TRANSFORMERS_OFFLINE=0",
        "--env", "SSL_CERT_FILE=",
        "--env", "REQUESTS_CA_BUNDLE=",
        "--env", "CURL_CA_BUNDLE=",
        APPTAINER_ENV,
        "euroeval",
        "--model", checkpoint_path,
        "--language", "sv",
        "--cache-dir", EUROEVAL_CACHE_DIR,
        "--trust-remote-code",
        "--save-results",
        "--num-iterations", "3",
        "--batch-size", "16",
        "--verbose"
    ]
    
    # Add HF_TOKEN if available
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        cmd.insert(cmd.index("--env") + 1, f"HF_TOKEN={hf_token}")
    
    print(f"Running: {' '.join(cmd[:10])}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per checkpoint
        )
        
        if result.returncode != 0:
            print(f"Error running euroeval: {result.stderr}")
            return None
        
        print(result.stdout)
        
    except subprocess.TimeoutExpired:
        print(f"Timeout evaluating checkpoint iter_{iteration}")
        return None
    except Exception as e:
        print(f"Exception running euroeval: {e}")
        return None
    
    # Parse results from the euroeval output file
    return parse_euroeval_results(checkpoint_path, iteration)


def parse_euroeval_results(checkpoint_path: str, iteration: int) -> Optional[Dict[str, Any]]:
    """
    Parse euroeval results from the JSONL output file.
    
    EuroEval writes results to euroeval_benchmark_results.jsonl
    """
    results_file = Path(EUROEVAL_RESULTS_DIR) / "euroeval_benchmark_results.jsonl"
    
    if not results_file.exists():
        print(f"Warning: Results file not found: {results_file}")
        return None
    
    results = {
        "iteration": iteration,
        "checkpoint_path": checkpoint_path,
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
        "scores": {},
        "aggregate_score": 0.0
    }
    
    scores = []
    
    # Read the JSONL file and find results for this model
    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Check if this result is for our checkpoint
            model_path = record.get("model", "")
            if checkpoint_path not in model_path and model_path not in checkpoint_path:
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
                    results["scores"][dataset] = {
                        "value": metric_value,
                        "se": metric_se,
                        "metric": metric_key
                    }
                    scores.append(metric_value)
    
    # Calculate aggregate score
    if scores:
        results["aggregate_score"] = sum(scores) / len(scores)
    
    return results


def generate_progress_plots(json_path: str) -> None:
    """Generate progress plots using the plotting script."""
    plot_script = Path(__file__).parent / "plot_checkpoint_progress.py"
    
    if not plot_script.exists():
        print(f"Warning: Plot script not found: {plot_script}")
        return
    
    print("\nGenerating progress plots...")
    
    try:
        result = subprocess.run(
            [sys.executable, str(plot_script), "--input", json_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Warning: Plot generation failed: {result.stderr}")
            
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GRPO checkpoints on Swedish EuroEval validation set"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be evaluated without running"
    )
    parser.add_argument(
        "--force-eval",
        type=int,
        help="Force re-evaluation of a specific iteration"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating progress plots"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=CHECKPOINT_BASE_DIR,
        help="Override checkpoint base directory"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=RESULTS_JSON_PATH,
        help="Override output JSON path"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GRPO Checkpoint Evaluation Pipeline")
    print("=" * 60)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Results JSON: {args.output_json}")
    print(f"Language: Swedish (sv)")
    print(f"Evaluation set: VALIDATION")
    print()
    
    # Load existing progress
    progress_data = load_progress_json(args.output_json)
    
    # Find available checkpoints
    available_iterations = find_checkpoint_iterations(args.checkpoint_dir)
    print(f"Found {len(available_iterations)} checkpoint iterations: {available_iterations}")
    
    # Get already evaluated iterations
    evaluated_iterations = get_evaluated_iterations(progress_data)
    print(f"Already evaluated: {evaluated_iterations}")
    
    # Determine what to evaluate
    if args.force_eval:
        iterations_to_eval = [args.force_eval] if args.force_eval in available_iterations else []
        if not iterations_to_eval:
            print(f"Error: Iteration {args.force_eval} not found in available checkpoints")
            sys.exit(1)
    else:
        iterations_to_eval = [i for i in available_iterations if i not in evaluated_iterations]
    
    print(f"Iterations to evaluate: {iterations_to_eval}")
    print()
    
    if not iterations_to_eval:
        print("No new iterations to evaluate!")
        
        if not args.skip_plots and evaluated_iterations:
            generate_progress_plots(args.output_json)
        
        return
    
    if args.dry_run:
        print("[DRY RUN MODE]")
        for iteration in iterations_to_eval:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"iter_{iteration}")
            print(f"  Would evaluate: {checkpoint_path}")
        return
    
    # Run evaluations
    new_results = []
    for iteration in iterations_to_eval:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"iter_{iteration}")
        
        results = run_euroeval_on_checkpoint(
            checkpoint_path=checkpoint_path,
            iteration=iteration,
            dry_run=args.dry_run
        )
        
        if results:
            progress_data["evaluations"][str(iteration)] = results
            new_results.append(results)
            
            # Save after each evaluation (in case of interruption)
            save_progress_json(progress_data, args.output_json)
        else:
            # Mark as failed
            progress_data["evaluations"][str(iteration)] = {
                "iteration": iteration,
                "checkpoint_path": checkpoint_path,
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }
            save_progress_json(progress_data, args.output_json)
    
    # Generate plots
    if not args.skip_plots and new_results:
        generate_progress_plots(args.output_json)
    
    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Evaluated {len(new_results)} new checkpoints")
    
    for result in new_results:
        if result.get("status") == "completed":
            print(f"  iter_{result['iteration']}: aggregate={result['aggregate_score']:.2f}")
    
    print(f"\nResults saved to: {args.output_json}")
    print("Done!")


if __name__ == "__main__":
    main()

