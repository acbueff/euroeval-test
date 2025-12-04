#!/usr/bin/env python3
"""
Plot GRPO Training Progress with EuroEval Swedish Validation Results

Creates incremental line plots showing model performance over training iterations.
Designed to be run periodically during training to monitor progress.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# Configuration
DEFAULT_RESULTS_DIR = "/proj/berzelius-aiics-real/users/x_anbue/frodi_data/self_play_rl_swedish/metrics"
DEFAULT_OUTPUT_DIR = "/home/x_anbue/euroeval-test/results/grpo_progress"

# Swedish datasets and their display names
SWEDISH_DATASETS = {
    "swerec": {"display": "SweRec (Sentiment)", "color": "#E63946"},
    "scala-sv": {"display": "ScaLA-SV (Ling.)", "color": "#F4A261"},
    "suc3": {"display": "SUC3 (NER)", "color": "#E9C46A"},
    "scandiqa-sv": {"display": "ScandiQA (Reading)", "color": "#2A9D8F"},
    "swedn": {"display": "SweDN (Summary)", "color": "#264653"},
    "mmlu-sv": {"display": "MMLU-SV (Knowledge)", "color": "#7209B7"},
    "hellaswag-sv": {"display": "HellaSwag (Reason)", "color": "#3A86FF"},
}

# Aesthetic settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def load_progress_data(results_file: Path) -> List[Dict[str, Any]]:
    """Load evaluation progress from JSONL file."""
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return []
    
    data = []
    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
    
    # Sort by iteration
    data.sort(key=lambda x: x.get('iteration', 0))
    return data


def load_csv_progress(csv_file: Path) -> List[Dict[str, Any]]:
    """Load evaluation progress from CSV file."""
    import csv
    
    if not csv_file.exists():
        return []
    
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            record = {'scores': {}}
            for key, value in row.items():
                if key == 'iteration':
                    record['iteration'] = int(value)
                elif key == 'timestamp':
                    record['timestamp'] = value
                elif key == 'aggregate_score':
                    record['aggregate_score'] = float(value)
                elif value:  # Dataset score
                    record['scores'][key] = float(value)
            data.append(record)
    
    data.sort(key=lambda x: x.get('iteration', 0))
    return data


def create_aggregate_plot(
    data: List[Dict[str, Any]], 
    output_dir: Path,
    show_best: bool = True
) -> None:
    """Create line plot of aggregate score over iterations."""
    if not data:
        print("No data available for aggregate plot")
        return
    
    iterations = [d.get('iteration', 0) for d in data]
    scores = [d.get('aggregate_score', 0) for d in data]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    
    # Main line
    ax.plot(iterations, scores, 'o-', 
            color='#3A86FF', linewidth=2.5, markersize=8,
            label='Aggregate Score', zorder=3)
    
    # Fill under curve
    ax.fill_between(iterations, scores, alpha=0.15, color='#3A86FF')
    
    # Best model marker
    if show_best and scores:
        best_idx = np.argmax(scores)
        best_iter = iterations[best_idx]
        best_score = scores[best_idx]
        ax.scatter([best_iter], [best_score], s=200, c='#E63946', 
                   marker='*', zorder=5, label=f'Best: {best_score:.2f} @ iter {best_iter}')
        ax.annotate(f'{best_score:.2f}', 
                   xy=(best_iter, best_score),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold', color='#E63946')
    
    # Styling
    ax.set_xlabel('Training Iteration', fontweight='bold')
    ax.set_ylabel('Aggregate EuroEval Score', fontweight='bold')
    ax.set_title('GRPO Training Progress: Swedish EuroEval Aggregate Score', 
                 fontsize=16, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='lower right', fontsize=10, frameon=True, 
              facecolor='white', edgecolor='gray')
    
    # Set reasonable y-limits
    if scores:
        y_min = max(0, min(scores) - 5)
        y_max = min(100, max(scores) + 5)
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    output_file = output_dir / 'grpo_aggregate_progress.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved aggregate plot to: {output_file}")


def create_dataset_breakdown_plot(
    data: List[Dict[str, Any]], 
    output_dir: Path
) -> None:
    """Create multi-line plot showing each dataset's progress."""
    if not data:
        print("No data available for dataset breakdown plot")
        return
    
    iterations = [d.get('iteration', 0) for d in data]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    
    # Plot each dataset
    for dataset, config in SWEDISH_DATASETS.items():
        scores = []
        for d in data:
            score = d.get('scores', {}).get(dataset)
            scores.append(score if score is not None else np.nan)
        
        # Only plot if we have data
        valid_scores = [s for s in scores if not np.isnan(s)]
        if valid_scores:
            ax.plot(iterations, scores, 'o-', 
                    color=config['color'], linewidth=2, markersize=6,
                    label=config['display'], alpha=0.8)
    
    # Styling
    ax.set_xlabel('Training Iteration', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('GRPO Training Progress: Swedish EuroEval by Dataset', 
                 fontsize=16, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
              fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
    
    # Set reasonable y-limits
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    output_file = output_dir / 'grpo_dataset_progress.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved dataset breakdown plot to: {output_file}")


def create_improvement_plot(
    data: List[Dict[str, Any]], 
    output_dir: Path,
    baseline_score: Optional[float] = None
) -> None:
    """Create plot showing improvement from baseline or first evaluation."""
    if len(data) < 2:
        print("Need at least 2 data points for improvement plot")
        return
    
    iterations = [d.get('iteration', 0) for d in data]
    scores = [d.get('aggregate_score', 0) for d in data]
    
    # Use first evaluation or provided baseline as reference
    reference = baseline_score if baseline_score is not None else scores[0]
    improvements = [s - reference for s in scores]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    
    # Color bars by positive/negative
    colors = ['#2A9D8F' if imp >= 0 else '#E63946' for imp in improvements]
    
    bars = ax.bar(iterations, improvements, color=colors, alpha=0.8, 
                  edgecolor='white', linewidth=1.5)
    
    # Reference line at 0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 1 if height >= 0 else -1
        ax.annotate(f'{imp:+.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, offset), textcoords="offset points",
                   ha='center', va=va, fontsize=9, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Training Iteration', fontweight='bold')
    ax.set_ylabel(f'Improvement from {"Baseline" if baseline_score else "First Eval"}', 
                  fontweight='bold')
    ax.set_title('GRPO Training: Score Improvement Over Time', 
                 fontsize=16, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Color regions
    ylim = ax.get_ylim()
    ax.axhspan(0, ylim[1], alpha=0.05, color='green')
    ax.axhspan(ylim[0], 0, alpha=0.05, color='red')
    
    plt.tight_layout()
    
    output_file = output_dir / 'grpo_improvement.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved improvement plot to: {output_file}")


def create_heatmap_timeline(
    data: List[Dict[str, Any]], 
    output_dir: Path
) -> None:
    """Create heatmap showing all datasets over time."""
    if not data:
        print("No data available for heatmap")
        return
    
    iterations = [d.get('iteration', 0) for d in data]
    datasets = list(SWEDISH_DATASETS.keys())
    
    # Build score matrix
    score_matrix = []
    for d in data:
        row = []
        for dataset in datasets:
            score = d.get('scores', {}).get(dataset)
            row.append(score if score is not None else np.nan)
        score_matrix.append(row)
    
    score_matrix = np.array(score_matrix).T  # Datasets as rows, iterations as columns
    
    fig, ax = plt.subplots(figsize=(max(10, len(iterations) * 0.8), 8))
    fig.patch.set_facecolor('#FAFAFA')
    
    # Create heatmap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(score_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=80)
    
    # Labels
    dataset_labels = [SWEDISH_DATASETS[d]['display'].split(' (')[0] for d in datasets]
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(dataset_labels, fontsize=11)
    
    ax.set_xticks(np.arange(len(iterations)))
    ax.set_xticklabels([str(i) for i in iterations], fontsize=10, rotation=45, ha='right')
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(iterations)):
            val = score_matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 30 or val > 60 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color=text_color, fontsize=9, fontweight='bold')
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel('Score', rotation=-90, va='bottom', fontsize=11)
    
    ax.set_xlabel('Training Iteration', fontweight='bold')
    plt.title('GRPO Training Progress: Swedish Benchmark Heatmap', 
              fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    output_file = output_dir / 'grpo_heatmap_timeline.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved heatmap timeline to: {output_file}")


def create_summary_stats(data: List[Dict[str, Any]], output_dir: Path) -> None:
    """Create summary statistics text file."""
    if not data:
        return
    
    scores = [d.get('aggregate_score', 0) for d in data]
    iterations = [d.get('iteration', 0) for d in data]
    
    best_idx = np.argmax(scores)
    
    summary = f"""GRPO Training Progress Summary
{'=' * 50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total Evaluations: {len(data)}
Iteration Range: {min(iterations)} - {max(iterations)}

Aggregate Score Statistics:
  - Initial: {scores[0]:.2f}
  - Final: {scores[-1]:.2f}
  - Best: {scores[best_idx]:.2f} (iteration {iterations[best_idx]})
  - Mean: {np.mean(scores):.2f}
  - Std: {np.std(scores):.2f}

Improvement from Initial: {scores[-1] - scores[0]:+.2f}
Best Improvement: {scores[best_idx] - scores[0]:+.2f}

Per-Dataset Best Scores:
"""
    
    # Find best score for each dataset
    for dataset, config in SWEDISH_DATASETS.items():
        dataset_scores = []
        for d in data:
            score = d.get('scores', {}).get(dataset)
            if score is not None:
                dataset_scores.append((d['iteration'], score))
        
        if dataset_scores:
            best_iter, best_score = max(dataset_scores, key=lambda x: x[1])
            summary += f"  - {config['display']}: {best_score:.2f} (iteration {best_iter})\n"
    
    output_file = output_dir / 'grpo_summary.txt'
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"Saved summary to: {output_file}")
    print("\n" + summary)


def load_training_progress(csv_file: Path) -> List[Dict[str, Any]]:
    """Load training progress (loss/reward) from CSV file."""
    import csv
    
    if not csv_file.exists():
        return []
    
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {}
            for key, value in row.items():
                if key == 'iteration':
                    record['iteration'] = int(value)
                elif key == 'timestamp':
                    record['timestamp'] = value
                elif value:
                    try:
                        record[key] = float(value)
                    except ValueError:
                        record[key] = value
            data.append(record)
    
    data.sort(key=lambda x: x.get('iteration', 0))
    return data


def create_training_progress_plot(data: List[Dict[str, Any]], output_dir: Path) -> None:
    """Create plot of training loss and reward over iterations."""
    if not data:
        print("No training progress data available")
        return
    
    iterations = [d.get('iteration', 0) for d in data]
    losses = [d.get('loss', 0) for d in data]
    rewards = [d.get('avg_reward', 0) for d in data]
    best_rewards = [d.get('best_reward', 0) for d in data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor('#FAFAFA')
    
    # Loss plot
    ax1.set_facecolor('#FFFFFF')
    ax1.plot(iterations, losses, 'o-', color='#E63946', linewidth=2, markersize=4, label='Loss')
    ax1.set_ylabel('Policy Loss', fontweight='bold')
    ax1.set_title('GRPO Training Progress', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right')
    
    # Reward plot
    ax2.set_facecolor('#FFFFFF')
    ax2.plot(iterations, rewards, 'o-', color='#3A86FF', linewidth=2, markersize=4, label='Avg Reward')
    ax2.plot(iterations, best_rewards, '--', color='#2A9D8F', linewidth=2, label='Best Reward')
    ax2.fill_between(iterations, rewards, alpha=0.15, color='#3A86FF')
    ax2.set_xlabel('Training Iteration', fontweight='bold')
    ax2.set_ylabel('Reward', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    
    output_file = output_dir / 'grpo_training_progress.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved training progress plot to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot GRPO training progress with EuroEval Swedish results'
    )
    parser.add_argument(
        '--results-dir', '-r',
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help='Directory containing euroeval_grpo_progress.jsonl'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for plots'
    )
    parser.add_argument(
        '--baseline-score', '-b',
        type=float,
        default=None,
        help='Baseline score for improvement calculation'
    )
    parser.add_argument(
        '--use-csv',
        action='store_true',
        help='Use CSV file instead of JSONL'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GRPO Training Progress Visualization")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load training progress data (loss/reward over time)
    training_csv = results_dir / 'training_progress.csv'
    if training_csv.exists():
        print(f"Loading training progress from: {training_csv}")
        training_data = load_training_progress(training_csv)
        if training_data:
            print(f"Loaded {len(training_data)} training progress records")
            create_training_progress_plot(training_data, output_dir)
    
    # Load EuroEval data
    if args.use_csv:
        csv_file = results_dir / 'euroeval_progress.csv'
        print(f"Loading EuroEval from CSV: {csv_file}")
        data = load_csv_progress(csv_file)
    else:
        jsonl_file = results_dir / 'euroeval_grpo_progress.jsonl'
        print(f"Loading EuroEval from JSONL: {jsonl_file}")
        data = load_progress_data(jsonl_file)
    
    if not data:
        print("No EuroEval data found.")
        if not training_csv.exists():
            print("Run GRPO training first, then run EuroEval evaluation.")
            sys.exit(1)
        else:
            print("Training progress plotted. Run EuroEval evaluation for benchmark scores.")
            sys.exit(0)
    
    print(f"Loaded {len(data)} EuroEval records")
    print()
    
    # Generate EuroEval plots
    print("Generating EuroEval plots...")
    print("-" * 40)
    
    create_aggregate_plot(data, output_dir)
    create_dataset_breakdown_plot(data, output_dir)
    create_improvement_plot(data, output_dir, args.baseline_score)
    create_heatmap_timeline(data, output_dir)
    create_summary_stats(data, output_dir)
    
    print("-" * 40)
    print()
    print("✅ All plots generated successfully!")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()

