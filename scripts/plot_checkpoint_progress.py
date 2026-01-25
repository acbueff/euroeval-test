#!/usr/bin/env python3
"""
Plot GRPO Checkpoint Training Progress on Swedish EuroEval Validation Set

Creates visualizations showing model performance across training iterations.
Supports multiple training runs/variants (e.g., uniform vs power sampling).

Output plots:
1. Learning curve (aggregate score vs iteration) - with multiple runs
2. Per-dataset learning curves
3. Improvement from baseline
4. Heatmap timeline of all datasets
5. Run comparison radar chart

Usage:
    python scripts/plot_checkpoint_progress.py --input results/checkpoint_progress.json
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ==================== Configuration ====================
DEFAULT_INPUT_JSON = "/home/x_anbue/euroeval-test/results/checkpoint_progress.json"
DEFAULT_OUTPUT_DIR = "/home/x_anbue/euroeval-test/results/checkpoint_plots"

# Swedish datasets with colors and display names
SWEDISH_DATASETS = {
    "swerec": {"display": "SweRec (Sentiment)", "color": "#E63946", "short": "SweRec"},
    "scala-sv": {"display": "ScaLA-SV (Ling.)", "color": "#F4A261", "short": "ScaLA"},
    "suc3": {"display": "SUC3 (NER)", "color": "#E9C46A", "short": "SUC3"},
    "scandiqa-sv": {"display": "ScandiQA (Reading)", "color": "#2A9D8F", "short": "ScandiQA"},
    "swedn": {"display": "SweDN (Summary)", "color": "#264653", "short": "SweDN"},
    "mmlu-sv": {"display": "MMLU-SV (Knowledge)", "color": "#7209B7", "short": "MMLU"},
    "hellaswag-sv": {"display": "HellaSwag (Reason)", "color": "#3A86FF", "short": "HellaSwag"},
}

# Default run colors if not specified in JSON
DEFAULT_RUN_COLORS = ["#3A86FF", "#E63946", "#2A9D8F", "#F4A261", "#7209B7"]
DEFAULT_RUN_LINESTYLES = ["-", "--", "-.", ":", "-"]

# Plot aesthetics
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def load_checkpoint_data(json_path: str) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
    """
    Load checkpoint evaluation data from JSON file.
    
    Returns:
        metadata: Global metadata
        runs: Dict of run_id -> {"config": {...}, "evaluations": [...]}
    """
    if not os.path.exists(json_path):
        print(f"Error: Input file not found: {json_path}")
        sys.exit(1)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metadata = data.get("metadata", {})
    
    # Check for new multi-run format
    if "runs" in data:
        runs = {}
        for run_id, run_data in data["runs"].items():
            evaluations = run_data.get("evaluations", {})
            eval_list = []
            for iter_key, eval_data in evaluations.items():
                if eval_data.get("status") == "completed":
                    eval_data["iteration"] = int(iter_key)
                    eval_list.append(eval_data)
            eval_list.sort(key=lambda x: x["iteration"])
            
            runs[run_id] = {
                "config": {
                    "name": run_data.get("name", run_id),
                    "description": run_data.get("description", ""),
                    "color": run_data.get("color"),
                    "linestyle": run_data.get("linestyle", "-"),
                    "branch_point": run_data.get("branch_point"),
                    "checkpoint_base_dir": run_data.get("checkpoint_base_dir", ""),
                },
                "evaluations": eval_list
            }
        return metadata, runs
    
    # Legacy format: single "evaluations" dict
    evaluations = data.get("evaluations", {})
    eval_list = []
    for iter_key, eval_data in evaluations.items():
        if eval_data.get("status") == "completed":
            eval_data["iteration"] = int(iter_key)
            eval_list.append(eval_data)
    eval_list.sort(key=lambda x: x["iteration"])
    
    runs = {
        "default": {
            "config": {
                "name": "GRPO Training",
                "color": "#3A86FF",
                "linestyle": "-",
            },
            "evaluations": eval_list
        }
    }
    
    return metadata, runs


def create_learning_curve(
    runs: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Create learning curve plot showing aggregate score over iterations for all runs."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    
    all_iterations = []
    all_scores = []
    legend_handles = []
    
    for idx, (run_id, run_data) in enumerate(runs.items()):
        config = run_data["config"]
        data = run_data["evaluations"]
        
        if not data:
            continue
        
        iterations = [d["iteration"] for d in data]
        scores = [d.get("aggregate_score", 0) for d in data]
        
        all_iterations.extend(iterations)
        all_scores.extend(scores)
        
        color = config.get("color") or DEFAULT_RUN_COLORS[idx % len(DEFAULT_RUN_COLORS)]
        linestyle = config.get("linestyle", "-")
        name = config.get("name", run_id)
        
        # Main line with markers
        line, = ax.plot(iterations, scores, 
                linestyle=linestyle, marker='o',
                color=color, linewidth=2.5, markersize=8,
                label=name, zorder=3)
        
        # Fill under curve (only for primary run)
        if idx == 0:
            ax.fill_between(iterations, scores, alpha=0.1, color=color)
        
        legend_handles.append(line)
        
        # Highlight best score for this run
        if scores:
            best_idx = np.argmax(scores)
            best_iter = iterations[best_idx]
            best_score = scores[best_idx]
            ax.scatter([best_iter], [best_score], s=150, c=color, 
                       marker='*', zorder=5, edgecolors='white', linewidths=1.5)
            ax.annotate(f'{best_score:.1f}', 
                       xy=(best_iter, best_score),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, fontweight='bold', color=color)
    
    # Styling
    ax.set_xlabel('Training Iteration', fontweight='bold', fontsize=12)
    ax.set_ylabel('Aggregate EuroEval Score (Swedish)', fontweight='bold', fontsize=12)
    ax.set_title('GRPO Training Progress: Swedish Validation Performance', 
                 fontsize=16, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Set y-axis range
    if all_scores:
        y_min = max(0, min(all_scores) - 3)
        y_max = min(100, max(all_scores) + 5)
        ax.set_ylim(y_min, y_max)
    
    # Legend
    ax.legend(handles=legend_handles, loc='lower right', fontsize=10, 
              frameon=True, facecolor='white', edgecolor='gray')
    
    plt.tight_layout()
    
    output_file = output_dir / 'checkpoint_learning_curve.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_file}")


def create_dataset_curves(
    runs: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Create multi-line plot showing each dataset's learning curve for all runs."""
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    
    legend_elements = []
    
    for run_idx, (run_id, run_data) in enumerate(runs.items()):
        config = run_data["config"]
        data = run_data["evaluations"]
        
        if not data:
            continue
        
        iterations = [d["iteration"] for d in data]
        run_linestyle = config.get("linestyle", "-")
        run_name = config.get("name", run_id)
        
        # Plot each dataset
        for dataset, ds_config in SWEDISH_DATASETS.items():
            scores = []
            for d in data:
                score_data = d.get("scores", {}).get(dataset)
                if score_data:
                    scores.append(score_data.get("value", np.nan))
                else:
                    scores.append(np.nan)
            
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                label = f"{ds_config['display']}" if run_idx == 0 else None
                ax.plot(iterations, scores, 
                        linestyle=run_linestyle, marker='o',
                        color=ds_config['color'], linewidth=2, markersize=5,
                        label=label, alpha=0.85)
        
        # Add run indicator to legend
        legend_elements.append(
            Line2D([0], [0], color='gray', linestyle=run_linestyle, linewidth=2, label=run_name)
        )
    
    # Dataset legend
    for dataset, ds_config in SWEDISH_DATASETS.items():
        legend_elements.append(
            Line2D([0], [0], color=ds_config['color'], linestyle='-', linewidth=2, 
                   marker='o', markersize=5, label=ds_config['display'])
        )
    
    # Styling
    ax.set_xlabel('Training Iteration', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('GRPO Training Progress: Per-Dataset Performance', 
                 fontsize=16, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 100)
    
    # Combined legend
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
              fontsize=9, frameon=True, facecolor='white', edgecolor='gray')
    
    plt.tight_layout()
    
    output_file = output_dir / 'checkpoint_dataset_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_file}")


def create_improvement_chart(
    runs: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Create bar chart showing improvement from baseline for all runs."""
    # Get primary run for baseline
    primary_run = list(runs.values())[0]
    primary_data = primary_run["evaluations"]
    
    if len(primary_data) < 2:
        print("Need at least 2 data points for improvement chart")
        return
    
    baseline = primary_data[0].get("aggregate_score", 0)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    
    bar_width = 0.35
    x_positions = []
    x_labels = []
    
    for run_idx, (run_id, run_data) in enumerate(runs.items()):
        config = run_data["config"]
        data = run_data["evaluations"]
        
        if not data:
            continue
        
        iterations = [d["iteration"] for d in data]
        scores = [d.get("aggregate_score", 0) for d in data]
        improvements = [s - baseline for s in scores]
        
        color = config.get("color") or DEFAULT_RUN_COLORS[run_idx % len(DEFAULT_RUN_COLORS)]
        name = config.get("name", run_id)
        
        # Create x positions
        x = np.arange(len(iterations)) * (len(runs) + 1) + run_idx * bar_width
        
        colors = [color if imp >= 0 else '#888888' for imp in improvements]
        alphas = [0.85 if imp >= 0 else 0.5 for imp in improvements]
        
        bars = ax.bar(x, improvements, bar_width, 
                      color=colors, alpha=0.85, edgecolor='white', linewidth=1,
                      label=name)
        
        # Value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 1 if height >= 0 else -1
            ax.annotate(f'{imp:+.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset), textcoords="offset points",
                       ha='center', va=va, fontsize=8, fontweight='bold')
        
        if run_idx == 0:
            x_positions = x
            x_labels = [str(i) for i in iterations]
    
    # Reference line at 0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Styling
    ax.set_xlabel('Training Iteration', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'Improvement from Baseline (iter 0: {baseline:.1f})', fontweight='bold', fontsize=12)
    ax.set_title('Score Improvement During GRPO Training', 
                 fontsize=16, fontweight='bold', pad=15)
    
    ax.set_xticks(x_positions + bar_width * (len(runs) - 1) / 2)
    ax.set_xticklabels(x_labels)
    
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='upper right', fontsize=10)
    
    # Color regions
    ylim = ax.get_ylim()
    if ylim[1] > 0:
        ax.axhspan(0, ylim[1], alpha=0.03, color='green')
    if ylim[0] < 0:
        ax.axhspan(ylim[0], 0, alpha=0.03, color='red')
    
    plt.tight_layout()
    
    output_file = output_dir / 'checkpoint_improvement.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_file}")


def create_heatmap_timeline(
    runs: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Create heatmap showing all datasets over iterations (primary run only)."""
    # Use primary run for heatmap
    primary_run = list(runs.values())[0]
    data = primary_run["evaluations"]
    
    if not data:
        print("No data for heatmap")
        return
    
    iterations = [d["iteration"] for d in data]
    datasets = list(SWEDISH_DATASETS.keys())
    
    # Build score matrix
    score_matrix = []
    for d in data:
        row = []
        for dataset in datasets:
            score_data = d.get("scores", {}).get(dataset)
            if score_data:
                row.append(score_data.get("value", np.nan))
            else:
                row.append(np.nan)
        score_matrix.append(row)
    
    score_matrix = np.array(score_matrix).T  # Datasets as rows
    
    fig, ax = plt.subplots(figsize=(max(12, len(iterations) * 1.2), 8))
    fig.patch.set_facecolor('#FAFAFA')
    
    # Heatmap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(score_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=80)
    
    # Labels
    dataset_labels = [SWEDISH_DATASETS[d]['short'] for d in datasets]
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(dataset_labels, fontsize=11)
    
    ax.set_xticks(np.arange(len(iterations)))
    ax.set_xticklabels([f'{i}' for i in iterations], fontsize=10, rotation=45, ha='right')
    
    # Text annotations
    for i in range(len(datasets)):
        for j in range(len(iterations)):
            val = score_matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 30 or val > 60 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel('Score', rotation=-90, va='bottom', fontsize=11)
    
    ax.set_xlabel('Iteration', fontweight='bold', fontsize=12)
    plt.title('Swedish Benchmark Scores Across Training', 
              fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    output_file = output_dir / 'checkpoint_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_file}")


def create_run_comparison_radar(
    runs: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Create radar chart comparing runs at their best iterations."""
    datasets = [d for d in SWEDISH_DATASETS.keys() if d != "swedn"]  # Exclude SweDN (often N/A)
    num_vars = len(datasets)
    
    # Compute angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    
    for run_idx, (run_id, run_data) in enumerate(runs.items()):
        config = run_data["config"]
        data = run_data["evaluations"]
        
        if not data:
            continue
        
        # Find best iteration for this run
        best_idx = max(range(len(data)), key=lambda i: data[i].get("aggregate_score", 0))
        best_data = data[best_idx]
        best_iter = best_data["iteration"]
        
        color = config.get("color") or DEFAULT_RUN_COLORS[run_idx % len(DEFAULT_RUN_COLORS)]
        linestyle = config.get("linestyle", "-")
        name = config.get("name", run_id)
        
        values = []
        for dataset in datasets:
            score_data = best_data.get("scores", {}).get(dataset)
            if score_data:
                values.append(score_data.get("value", 0))
            else:
                values.append(0)
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linestyle=linestyle, linewidth=2.5, 
                label=f"{name} (iter {best_iter})", color=color, markersize=8, marker='o')
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Labels
    ax.set_xticks(angles[:-1])
    labels = [SWEDISH_DATASETS[d]['short'] for d in datasets]
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10, 
              frameon=True, facecolor='white', edgecolor='gray')
    
    plt.title('Run Comparison: Best Checkpoints', 
              fontsize=16, fontweight='bold', pad=20)
    
    output_file = output_dir / 'checkpoint_radar.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_file}")


def create_summary_stats(
    runs: Dict[str, Dict],
    metadata: Dict[str, Any],
    output_dir: Path
) -> None:
    """Create summary statistics text file."""
    summary = f"""GRPO Checkpoint Training Progress Summary
{'=' * 60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Configuration:
  Language: {metadata.get('language', 'sv')}
  Evaluation Set: Validation
"""
    
    for run_id, run_data in runs.items():
        config = run_data["config"]
        data = run_data["evaluations"]
        
        if not data:
            continue
        
        scores = [d.get("aggregate_score", 0) for d in data]
        iterations = [d["iteration"] for d in data]
        best_idx = np.argmax(scores)
        
        summary += f"""
{'=' * 60}
Run: {config.get('name', run_id)}
{'=' * 60}
  Description: {config.get('description', 'N/A')}
  Checkpoint Dir: {config.get('checkpoint_base_dir', 'N/A')}
  Branch Point: {config.get('branch_point', 'N/A')}

Evaluation Statistics:
  Total Checkpoints: {len(data)}
  Iteration Range: {min(iterations)} - {max(iterations)}

Aggregate Score:
  First (iter {iterations[0]}): {scores[0]:.2f}
  Latest (iter {iterations[-1]}): {scores[-1]:.2f}
  Best (iter {iterations[best_idx]}): {scores[best_idx]:.2f}
  Mean: {np.mean(scores):.2f}
  Std Dev: {np.std(scores):.2f}

Progress:
  Improvement (Latest vs First): {scores[-1] - scores[0]:+.2f}
  Best Improvement: {scores[best_idx] - scores[0]:+.2f}

Per-Dataset Scores at Best Iteration (iter {iterations[best_idx]}):
"""
        
        best_data = data[best_idx]
        for dataset, ds_config in SWEDISH_DATASETS.items():
            score_data = best_data.get("scores", {}).get(dataset)
            if score_data:
                summary += f"  {ds_config['display']}: {score_data['value']:.2f}\n"
            else:
                summary += f"  {ds_config['display']}: N/A\n"
    
    output_file = output_dir / 'checkpoint_summary.txt'
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"Saved: {output_file}")
    print("\n" + summary)


def main():
    parser = argparse.ArgumentParser(
        description="Plot GRPO checkpoint training progress (supports multiple runs)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_INPUT_JSON,
        help="Input JSON file with checkpoint evaluation results"
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
    print("GRPO Checkpoint Progress Visualization")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    metadata, runs = load_checkpoint_data(args.input)
    
    total_evals = sum(len(r["evaluations"]) for r in runs.values())
    if total_evals == 0:
        print("No completed evaluations found!")
        sys.exit(1)
    
    print(f"Loaded {len(runs)} run(s) with {total_evals} total evaluations:")
    for run_id, run_data in runs.items():
        config = run_data["config"]
        evals = run_data["evaluations"]
        iters = [e["iteration"] for e in evals]
        print(f"  - {config.get('name', run_id)}: {len(evals)} checkpoints {iters}")
    print()
    
    # Generate all plots
    print("Generating plots...")
    print("-" * 40)
    
    create_learning_curve(runs, output_dir)
    create_dataset_curves(runs, output_dir)
    create_improvement_chart(runs, output_dir)
    create_heatmap_timeline(runs, output_dir)
    create_run_comparison_radar(runs, output_dir)
    create_summary_stats(runs, metadata, output_dir)
    
    print("-" * 40)
    print()
    print("✅ All plots generated successfully!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
