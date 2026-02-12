#!/usr/bin/env python3
"""
Plot Swedish EuroEval TEST SET Results

Creates visualizations comparing model performance on the Swedish EuroEval test set:
1. Bar chart comparing all models across datasets
2. Radar chart for model comparison
3. Summary table

Usage:
    python scripts/plot_swedish_testset.py --input results/swedish_testset_results.json
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

# ==================== Configuration ====================
DEFAULT_INPUT_JSON = "/home/x_anbue/euroeval-test/results/swedish_testset_results.json"
DEFAULT_OUTPUT_DIR = "/home/x_anbue/euroeval-test/results/swedish_testset_plots"

# Plot aesthetics
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def load_results(json_path: str) -> Dict[str, Any]:
    """Load test set results from JSON file."""
    if not os.path.exists(json_path):
        print(f"Error: Input file not found: {json_path}")
        sys.exit(1)
    
    with open(json_path, 'r') as f:
        return json.load(f)


def create_bar_comparison(
    data: Dict[str, Any],
    output_dir: Path
) -> None:
    """Create grouped bar chart comparing all models across datasets."""
    models = data.get("models", {})
    metadata = data.get("metadata", {})
    
    # Filter to completed models only
    completed_models = {k: v for k, v in models.items() if v.get("status") == "completed"}
    
    if not completed_models:
        print("No completed models found for bar chart")
        return
    
    # Get datasets from first completed model
    first_model = list(completed_models.values())[0]
    datasets = list(first_model.get("scores", {}).keys())
    
    if not datasets:
        print("No scores found for bar chart")
        return
    
    # Get display names from metadata
    dataset_info = metadata.get("dataset_info", {})
    dataset_labels = [dataset_info.get(d, {}).get("display", d) for d in datasets]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    
    x = np.arange(len(datasets))
    width = 0.25
    multiplier = 0
    
    for model_id, model_data in completed_models.items():
        scores = model_data.get("scores", {})
        values = [scores.get(d, {}).get("value", 0) for d in datasets]
        errors = [scores.get(d, {}).get("se", 0) for d in datasets]
        
        color = model_data.get("color", "#333333")
        name = model_data.get("name", model_id)
        size = model_data.get("model_size", "")
        label = f"{name} ({size})" if size else name
        
        offset = width * multiplier
        bars = ax.bar(x + offset, values, width, label=label, color=color, 
                     alpha=0.85, edgecolor='white', linewidth=1)
        ax.errorbar(x + offset, values, yerr=errors, fmt='none', 
                   color='black', capsize=3, alpha=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, val),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        multiplier += 1
    
    # Styling
    ax.set_xlabel('Dataset', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Swedish EuroEval TEST SET: Model Comparison', 
                fontsize=16, fontweight='bold', pad=15)
    
    ax.set_xticks(x + width * (len(completed_models) - 1) / 2)
    ax.set_xticklabels(dataset_labels, rotation=30, ha='right', fontsize=10)
    
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
    
    plt.tight_layout()
    
    output_file = output_dir / 'testset_bar_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_file}")


def create_radar_chart(
    data: Dict[str, Any],
    output_dir: Path
) -> None:
    """Create radar chart comparing models across all datasets."""
    models = data.get("models", {})
    metadata = data.get("metadata", {})
    
    # Filter to completed models only
    completed_models = {k: v for k, v in models.items() if v.get("status") == "completed"}
    
    if not completed_models:
        print("No completed models found for radar chart")
        return
    
    # Get datasets
    first_model = list(completed_models.values())[0]
    datasets = list(first_model.get("scores", {}).keys())
    
    if not datasets:
        print("No scores found for radar chart")
        return
    
    dataset_info = metadata.get("dataset_info", {})
    num_vars = len(datasets)
    
    # Compute angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    
    for model_id, model_data in completed_models.items():
        scores = model_data.get("scores", {})
        values = [scores.get(d, {}).get("value", 0) for d in datasets]
        values += values[:1]  # Close the loop
        
        color = model_data.get("color", "#333333")
        name = model_data.get("name", model_id)
        size = model_data.get("model_size", "")
        agg = model_data.get("aggregate_score", 0)
        label = f"{name} ({size})\nAvg: {agg:.1f}" if size else f"{name}\nAvg: {agg:.1f}"
        
        ax.plot(angles, values, linewidth=2.5, label=label, color=color, marker='o', markersize=8)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Labels
    ax.set_xticks(angles[:-1])
    labels = []
    for d in datasets:
        display = dataset_info.get(d, {}).get("display", d)
        # Shorten labels for radar chart
        short = display.split(" (")[0] if " (" in display else display
        labels.append(short)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(['20', '40', '60', '80'], fontsize=9)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10, 
              frameon=True, facecolor='white', edgecolor='gray')
    
    plt.title('Swedish EuroEval TEST SET: Radar Comparison', 
              fontsize=16, fontweight='bold', pad=20)
    
    output_file = output_dir / 'testset_radar_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_file}")


def create_aggregate_bar(
    data: Dict[str, Any],
    output_dir: Path
) -> None:
    """Create bar chart of aggregate scores."""
    models = data.get("models", {})
    
    # Filter to completed models only
    completed_models = {k: v for k, v in models.items() if v.get("status") == "completed"}
    
    if not completed_models:
        print("No completed models found for aggregate bar chart")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    
    names = []
    scores = []
    colors = []
    
    for model_id, model_data in completed_models.items():
        name = model_data.get("name", model_id)
        size = model_data.get("model_size", "")
        names.append(f"{name}\n({size})" if size else name)
        scores.append(model_data.get("aggregate_score", 0))
        colors.append(model_data.get("color", "#333333"))
    
    x = np.arange(len(names))
    bars = ax.bar(x, scores, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax.annotate(f'{score:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, score),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Aggregate Score', fontweight='bold', fontsize=12)
    ax.set_title('Swedish EuroEval TEST SET: Aggregate Scores', 
                fontsize=16, fontweight='bold', pad=15)
    
    ax.set_ylim(0, max(scores) * 1.15 if scores else 100)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    output_file = output_dir / 'testset_aggregate_scores.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_file}")


def create_summary_text(
    data: Dict[str, Any],
    output_dir: Path
) -> None:
    """Create summary text file."""
    models = data.get("models", {})
    metadata = data.get("metadata", {})
    dataset_info = metadata.get("dataset_info", {})
    
    summary = f"""Swedish EuroEval TEST SET Results Summary
{'=' * 60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Configuration:
  Language: Swedish (sv)
  Split: TEST
  Experiment: {metadata.get('experiment_name', 'N/A')}
  Description: {metadata.get('description', 'N/A')}

Datasets Evaluated:
"""
    for dataset, info in dataset_info.items():
        summary += f"  - {info.get('display', dataset)} (metric: {info.get('metric', 'N/A')})\n"
    
    summary += f"\n{'=' * 60}\n"
    summary += "MODEL RESULTS\n"
    summary += f"{'=' * 60}\n"
    
    for model_id, model_data in models.items():
        status = model_data.get("status", "unknown")
        name = model_data.get("name", model_id)
        
        summary += f"\n{'-' * 50}\n"
        summary += f"{name}\n"
        summary += f"{'-' * 50}\n"
        summary += f"  Model ID: {model_id}\n"
        summary += f"  Model Size: {model_data.get('model_size', 'N/A')}\n"
        summary += f"  Path: {model_data.get('model_path', 'N/A')}\n"
        summary += f"  Status: {status}\n"
        
        if status == "completed":
            summary += f"  Timestamp: {model_data.get('timestamp', 'N/A')}\n"
            summary += f"\n  Aggregate Score: {model_data.get('aggregate_score', 0):.2f}\n"
            summary += f"  Datasets Evaluated: {model_data.get('num_datasets', 0)}\n"
            summary += f"\n  Per-Dataset Scores:\n"
            
            for dataset, score_data in model_data.get("scores", {}).items():
                display = score_data.get("display_name", dataset)
                value = score_data.get("value", 0)
                se = score_data.get("se", 0)
                summary += f"    {display}: {value:.2f} (±{se:.2f})\n"
        else:
            summary += f"  (Evaluation pending)\n"
    
    summary += f"\n{'=' * 60}\n"
    summary += "END OF SUMMARY\n"
    
    output_file = output_dir / 'testset_summary.txt'
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"Saved: {output_file}")
    print("\n" + summary)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Swedish EuroEval TEST SET results"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_INPUT_JSON,
        help="Input JSON file with test set results"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Swedish EuroEval TEST SET Visualization")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    data = load_results(args.input)
    
    models = data.get("models", {})
    completed = sum(1 for m in models.values() if m.get("status") == "completed")
    
    print(f"Found {len(models)} models ({completed} completed)")
    for model_id, model_data in models.items():
        status = model_data.get("status", "unknown")
        name = model_data.get("name", model_id)
        print(f"  - {name}: {status}")
    print()
    
    if completed == 0:
        print("No completed evaluations found. Run the evaluation scripts first.")
        return
    
    # Generate plots
    print("Generating plots...")
    print("-" * 40)
    
    create_bar_comparison(data, output_dir)
    create_radar_chart(data, output_dir)
    create_aggregate_bar(data, output_dir)
    create_summary_text(data, output_dir)
    
    print("-" * 40)
    print()
    print("✅ All plots generated successfully!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()

