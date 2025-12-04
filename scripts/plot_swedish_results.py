#!/usr/bin/env python3
"""
Plot Swedish EuroEval Benchmark Results
Creates bar charts, radar charts, and baseline difference plots.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# Configuration
INPUT_FILE = "/proj/berzelius-aiics-real/users/x_anbue/euroeval/results/euroeval_benchmark_results.jsonl"
OUTPUT_DIR = "/home/x_anbue/euroeval-test/results"
# New baseline path from local cache (3 iterations run)
BASELINE_MODEL = "/proj/berzelius-aiics-real/users/x_anbue/euroeval/.euroeval_cache/model_cache/Qwen--Qwen3-1.7B/models--Qwen--Qwen3-1.7B/snapshots/"
LANGUAGE_FILTER = "sv"

# Swedish datasets and their primary metrics
SWEDISH_DATASETS = {
    "swerec": {"task": "sentiment-classification", "metric": "test_mcc", "display": "SweRec (Sentiment)"},
    "scala-sv": {"task": "linguistic-acceptability", "metric": "test_mcc", "display": "ScaLA-SV (Ling. Accept.)"},
    "suc3": {"task": "named-entity-recognition", "metric": "test_micro_f1_no_misc", "display": "SUC3 (NER)"},
    "scandiqa-sv": {"task": "reading-comprehension", "metric": "test_f1", "display": "ScandiQA-SV (Reading)"},
    "swedn": {"task": "summarization", "metric": "test_bertscore", "display": "SweDN (Summarization)"},
    "mmlu-sv": {"task": "knowledge", "metric": "test_mcc", "display": "MMLU-SV (Knowledge)"},
    "hellaswag-sv": {"task": "common-sense-reasoning", "metric": "test_mcc", "display": "HellaSwag-SV (Reasoning)"},
}

# Aesthetics - Custom color palette
COLORS = {
    "baseline": "#2E86AB",      # Steel blue
    "cpt": "#A23B72",           # Raspberry
    "sft": "#F18F01",           # Orange
    "kd": "#C73E1D",            # Rust red
    "default": "#5C8001",       # Olive green
}

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 16


def is_baseline_model(model_path):
    """Check if model path refers to the baseline Qwen3-1.7B model."""
    return (model_path == BASELINE_MODEL or 
            "Qwen--Qwen3-1.7B" in model_path or 
            model_path == "Qwen/Qwen3-1.7B")


def get_model_short_name(model_path):
    """Extract readable short name from model path."""
    # Check for baseline - match both old HF ID and new local cache path
    if is_baseline_model(model_path):
        return "Qwen3-1.7B (Baseline)"
    elif "cpt_checkpoint" in model_path:
        return "CPT Epoch 1"
    elif "sft_checkpoint" in model_path:
        return "SFT Epoch 3"
    elif "frodi-swedish-student-kd" in model_path:
        return "Frodi-KD"
    else:
        return Path(model_path).name


def get_model_color(model_name):
    """Get color for model based on its type."""
    if "Baseline" in model_name:
        return COLORS["baseline"]
    elif "CPT" in model_name:
        return COLORS["cpt"]
    elif "SFT" in model_name:
        return COLORS["sft"]
    elif "KD" in model_name:
        return COLORS["kd"]
    return COLORS["default"]


def load_swedish_results(filepath):
    """Load and filter results for Swedish language."""
    results = defaultdict(dict)
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            
            # Filter for Swedish language
            if LANGUAGE_FILTER not in record.get("dataset_languages", []):
                continue
            
            dataset = record["dataset"]
            model = record["model"]
            
            if dataset in SWEDISH_DATASETS:
                metric_key = SWEDISH_DATASETS[dataset]["metric"]
                metric_value = record["results"]["total"].get(metric_key)
                metric_se = record["results"]["total"].get(f"{metric_key}_se", 0)
                
                if metric_value is not None:
                    results[dataset][model] = {
                        "value": metric_value,
                        "se": metric_se,
                        "task": record["task"]
                    }
    
    return results


def create_bar_charts(results, output_dir):
    """Create individual bar charts for each dataset."""
    n_datasets = len(SWEDISH_DATASETS)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()
    
    # Custom background
    fig.patch.set_facecolor('#F5F5F5')
    
    for idx, (dataset, info) in enumerate(SWEDISH_DATASETS.items()):
        ax = axes[idx]
        ax.set_facecolor('#FFFFFF')
        
        if dataset not in results:
            ax.set_title(f"{info['display']}\n(No data)")
            ax.set_visible(False)
            continue
        
        dataset_results = results[dataset]
        models = sorted(dataset_results.keys(), key=lambda x: not is_baseline_model(x))
        
        model_names = [get_model_short_name(m) for m in models]
        values = [dataset_results[m]["value"] for m in models]
        errors = [dataset_results[m]["se"] for m in models]
        colors = [get_model_color(name) for name in model_names]
        
        bars = ax.bar(range(len(models)), values, yerr=errors, capsize=5,
                     color=colors, edgecolor='white', linewidth=1.5, alpha=0.9)
        
        ax.set_xlabel('')
        ax.set_ylabel(info['metric'].replace('test_', '').upper())
        ax.set_title(info['display'], fontweight='bold', pad=10)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(model_names, rotation=35, ha='right', fontsize=9)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(len(SWEDISH_DATASETS), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Swedish EuroEval Benchmark Results by Task', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "swedish_bar_charts.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved bar charts to: {output_path}")


def create_radar_chart(results, output_dir):
    """Create radar chart comparing all models across datasets."""
    # Get all unique models
    all_models = set()
    for dataset_results in results.values():
        all_models.update(dataset_results.keys())
    
    models = sorted(all_models, key=lambda x: not is_baseline_model(x))
    datasets = list(SWEDISH_DATASETS.keys())
    
    # Prepare data matrix
    data = {}
    for model in models:
        model_scores = []
        for dataset in datasets:
            if dataset in results and model in results[dataset]:
                model_scores.append(results[dataset][model]["value"])
            else:
                model_scores.append(0)
        data[model] = model_scores
    
    # Number of variables
    num_vars = len(datasets)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#F5F5F5')
    ax.set_facecolor('#FFFFFF')
    
    for model in models:
        values = data[model] + data[model][:1]  # Complete the loop
        model_name = get_model_short_name(model)
        color = get_model_color(model_name)
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model_name, color=color, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    labels = [SWEDISH_DATASETS[d]["display"].split(" (")[0] for d in datasets]
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, 
              frameon=True, facecolor='white', edgecolor='gray')
    
    plt.title('Model Comparison Across Swedish Benchmarks', fontsize=16, fontweight='bold', pad=20)
    
    output_path = os.path.join(output_dir, "swedish_radar_chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved radar chart to: {output_path}")


def create_baseline_diff_chart(results, output_dir):
    """Create chart showing differences from baseline (Qwen3 1.7B)."""
    # Get all unique models (excluding baseline)
    all_models = set()
    for dataset_results in results.values():
        all_models.update(dataset_results.keys())
    
    other_models = sorted([m for m in all_models if not is_baseline_model(m)])
    
    # Find the actual baseline model key in results
    baseline_key = None
    for model in all_models:
        if is_baseline_model(model):
            baseline_key = model
            break
    
    if not baseline_key:
        print("No baseline model found. Skipping baseline diff chart.")
        return
    
    datasets = [d for d in SWEDISH_DATASETS.keys() if d in results and baseline_key in results[d]]
    
    if not datasets:
        print("No datasets have baseline results. Skipping baseline diff chart.")
        return
    
    # Compute differences
    diff_data = {model: [] for model in other_models}
    
    for dataset in datasets:
        baseline_value = results[dataset].get(baseline_key, {}).get("value", 0)
        
        for model in other_models:
            if model in results[dataset]:
                diff = results[dataset][model]["value"] - baseline_value
            else:
                diff = 0  # No data
            diff_data[model].append(diff)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#F5F5F5')
    ax.set_facecolor('#FFFFFF')
    
    x = np.arange(len(datasets))
    width = 0.25
    multiplier = 0
    
    for model in other_models:
        model_name = get_model_short_name(model)
        color = get_model_color(model_name)
        offset = width * multiplier
        
        diffs = diff_data[model]
        colors_per_bar = [color if d >= 0 else '#888888' for d in diffs]
        
        bars = ax.bar(x + offset, diffs, width, label=model_name, color=color, 
                     edgecolor='white', linewidth=1.5, alpha=0.9)
        
        # Add value labels
        for bar, val in zip(bars, diffs):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset_pts = 3 if height >= 0 else -3
            ax.annotate(f'{val:+.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset_pts), textcoords="offset points",
                       ha='center', va=va, fontsize=8, fontweight='bold')
        
        multiplier += 1
    
    # Reference line at 0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Labels and styling
    ax.set_ylabel('Difference from Baseline (Qwen3-1.7B)', fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('Performance Difference from Baseline Model', fontsize=16, fontweight='bold', pad=15)
    
    labels = [SWEDISH_DATASETS[d]["display"].split(" (")[0] for d in datasets]
    ax.set_xticks(x + width * (len(other_models) - 1) / 2)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
    
    ax.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Color regions
    ax.axhspan(0, ax.get_ylim()[1], alpha=0.05, color='green')
    ax.axhspan(ax.get_ylim()[0], 0, alpha=0.05, color='red')
    
    output_path = os.path.join(output_dir, "swedish_baseline_diff.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved baseline diff chart to: {output_path}")


def create_summary_table(results, output_dir):
    """Create a summary visualization table."""
    # Get all unique models
    all_models = set()
    for dataset_results in results.values():
        all_models.update(dataset_results.keys())
    
    models = sorted(all_models, key=lambda x: not is_baseline_model(x))
    datasets = list(SWEDISH_DATASETS.keys())
    
    # Build data matrix
    data_matrix = []
    for model in models:
        row = []
        for dataset in datasets:
            if dataset in results and model in results[dataset]:
                row.append(results[dataset][model]["value"])
            else:
                row.append(np.nan)
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#F5F5F5')
    
    # Create heatmap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=80)
    
    # Labels
    model_labels = [get_model_short_name(m) for m in models]
    dataset_labels = [SWEDISH_DATASETS[d]["display"].split(" (")[0] for d in datasets]
    
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(dataset_labels, rotation=30, ha='right', fontsize=11)
    ax.set_yticklabels(model_labels, fontsize=11)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(datasets)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 30 or val > 60 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color=text_color, fontsize=10, fontweight='bold')
            else:
                ax.text(j, i, 'N/A', ha='center', va='center', 
                       color='gray', fontsize=9)
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Score', rotation=-90, va='bottom', fontsize=11)
    
    plt.title('Swedish Benchmark Results Summary', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "swedish_summary_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved summary heatmap to: {output_path}")


def main():
    """Main function to generate all plots."""
    print("=" * 60)
    print("Swedish EuroEval Results Visualization")
    print("=" * 60)
    print(f"Input file: {INPUT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Baseline model: {BASELINE_MODEL}")
    print()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load results
    print("Loading Swedish benchmark results...")
    results = load_swedish_results(INPUT_FILE)
    
    print(f"Found {len(results)} Swedish datasets:")
    for dataset, models in results.items():
        print(f"  - {dataset}: {len(models)} models")
    print()
    
    # Generate plots
    print("Generating plots...")
    print("-" * 40)
    
    create_bar_charts(results, OUTPUT_DIR)
    create_radar_chart(results, OUTPUT_DIR)
    create_baseline_diff_chart(results, OUTPUT_DIR)
    create_summary_table(results, OUTPUT_DIR)
    
    print("-" * 40)
    print()
    print("✅ All plots generated successfully!")
    print(f"Output files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
