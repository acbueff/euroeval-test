#!/bin/bash
#SBATCH -A berzelius-2025-264
#SBATCH --job-name=dl_euroeval_train
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/download_train_%j.out
#SBATCH --error=/proj/berzelius-aiics-real/users/x_anbue/euroeval/logs/download_train_%j.err

set -euo pipefail

# ==================== Configuration ====================
REPO_DIR="/home/x_anbue/euroeval-test"

# EuroEval paths
EUROEVAL_BASE="/proj/berzelius-aiics-real/users/x_anbue/euroeval"
CACHE_DIR="${EUROEVAL_BASE}/.euroeval_cache"
LOGS_DIR="${EUROEVAL_BASE}/logs"
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/euroeval.sif"

# Output directory for extracted training sets
TRAIN_DATA_DIR="${EUROEVAL_BASE}/train_sets"

# Languages to download
LANGUAGES=("is" "de" "sv")

# A small model to use for download-only (we just need any valid model)
DUMMY_MODEL="Qwen/Qwen2.5-0.5B"

# ==================== Environment ====================
export HF_HOME="${CACHE_DIR}/model_cache"
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export EUROEVAL_CACHE_DIR="$CACHE_DIR"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

mkdir -p "$LOGS_DIR" "$TRAIN_DATA_DIR"

# ==================== Job Info ====================
echo "============================================================"
echo "EuroEval Training Set Download"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Languages: ${LANGUAGES[*]}"
echo "Cache directory: $CACHE_DIR"
echo "Output directory: $TRAIN_DATA_DIR"
echo "============================================================"

# ==================== Download Datasets ====================
for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Downloading EuroEval datasets for: $lang"
    echo "Time: $(date)"
    echo "============================================================"

    # Download datasets (this caches all splits: train, val, test)
    # Using --raise-errors to catch any issues
    apptainer exec --nv \
        --env HF_TOKEN="${HF_TOKEN:-}" \
        --env HF_HOME="$HF_HOME" \
        --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
        --env EUROEVAL_CACHE_DIR="$CACHE_DIR" \
        --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
        --env TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
        --env SSL_CERT_FILE="" \
        --env REQUESTS_CA_BUNDLE="" \
        --env CURL_CA_BUNDLE="" \
        "$APPTAINER_ENV" \
        euroeval \
            --model "$DUMMY_MODEL" \
            --language "$lang" \
            --cache-dir "$CACHE_DIR" \
            --download-only \
            --trust-remote-code \
            --verbose || echo "Warning: Some datasets for $lang may have failed"

    echo "Download completed for: $lang"
done

# ==================== Extract Training Sets ====================
echo ""
echo "============================================================"
echo "Extracting training sets to JSON format..."
echo "============================================================"

apptainer exec \
    --env HF_HOME="$HF_HOME" \
    --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    "$APPTAINER_ENV" \
    python3 << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path
from datetime import datetime

try:
    from datasets import load_dataset
except ImportError:
    print("datasets library not available, will use cached files directly")
    exit(0)

TRAIN_DATA_DIR = "/proj/berzelius-aiics-real/users/x_anbue/euroeval/train_sets"
CACHE_DIR = "/proj/berzelius-aiics-real/users/x_anbue/euroeval/.euroeval_cache"

# EuroEval dataset configurations per language
# Format: {language: {dataset_name: {hf_path, task_type, description}}}
DATASETS = {
    "is": {
        "hotter-and-colder-sentiment": {
            "hf_path": "EuroEval/hotter-and-colder-sentiment",
            "task": "sentiment-classification",
            "description": "Icelandic blog post sentiment (positive/neutral/negative)"
        },
        "scala-is": {
            "hf_path": "EuroEval/scala-is",
            "task": "linguistic-acceptability",
            "description": "Icelandic grammatical correctness"
        },
        "mim-gold-ner": {
            "hf_path": "EuroEval/mim-gold-ner",
            "task": "named-entity-recognition",
            "description": "Icelandic NER (PER, LOC, ORG, MISC)"
        },
        "nqii": {
            "hf_path": "EuroEval/nqii",
            "task": "reading-comprehension",
            "description": "Natural Questions in Icelandic (extractive QA)"
        },
        "icelandic-knowledge": {
            "hf_path": "EuroEval/icelandic-knowledge",
            "task": "knowledge",
            "description": "Icelandic factual knowledge (4-choice MC)"
        },
        "winogrande-is": {
            "hf_path": "EuroEval/winogrande-is",
            "task": "common-sense-reasoning",
            "description": "Icelandic pronoun disambiguation"
        },
    },
    "de": {
        "german-europarl-sentiment": {
            "hf_path": "EuroEval/german-europarl-sentiment",
            "task": "sentiment-classification",
            "description": "German Europarl sentiment classification"
        },
        "scala-de": {
            "hf_path": "EuroEval/scala-de",
            "task": "linguistic-acceptability",
            "description": "German grammatical correctness"
        },
        "germeval-ner": {
            "hf_path": "EuroEval/germeval-ner",
            "task": "named-entity-recognition",
            "description": "German NER"
        },
        "germanquad": {
            "hf_path": "EuroEval/germanquad",
            "task": "reading-comprehension",
            "description": "German extractive QA"
        },
        "mmlu-de": {
            "hf_path": "EuroEval/mmlu-de",
            "task": "knowledge",
            "description": "German MMLU knowledge"
        },
        "hellaswag-de": {
            "hf_path": "EuroEval/hellaswag-de",
            "task": "common-sense-reasoning",
            "description": "German common-sense reasoning"
        },
    },
    "sv": {
        "absabank-sentiment": {
            "hf_path": "EuroEval/absabank-sentiment",
            "task": "sentiment-classification",
            "description": "Swedish financial sentiment"
        },
        "scala-sv": {
            "hf_path": "EuroEval/scala-sv",
            "task": "linguistic-acceptability",
            "description": "Swedish grammatical correctness"
        },
        "suc3-ner": {
            "hf_path": "EuroEval/suc3-ner",
            "task": "named-entity-recognition",
            "description": "Swedish NER"
        },
        "scandiqa-sv": {
            "hf_path": "EuroEval/scandiqa-sv",
            "task": "reading-comprehension",
            "description": "Swedish extractive QA"
        },
        "mmlu-sv": {
            "hf_path": "EuroEval/mmlu-sv",
            "task": "knowledge",
            "description": "Swedish MMLU knowledge"
        },
        "hellaswag-sv": {
            "hf_path": "EuroEval/hellaswag-sv",
            "task": "common-sense-reasoning",
            "description": "Swedish common-sense reasoning"
        },
    }
}

def extract_dataset(lang, name, config):
    """Extract training split from a EuroEval dataset."""
    output_dir = Path(TRAIN_DATA_DIR) / lang
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{name}_train.json"

    print(f"\nProcessing: {config['hf_path']}")

    try:
        # Load dataset from HuggingFace (will use cache)
        ds = load_dataset(config["hf_path"], trust_remote_code=True)

        # Get training split
        if "train" in ds:
            train_data = ds["train"]
        elif "training" in ds:
            train_data = ds["training"]
        else:
            print(f"  Warning: No train split found for {name}")
            print(f"  Available splits: {list(ds.keys())}")
            return None

        # Convert to list of dicts
        samples = [dict(sample) for sample in train_data]

        # Create output structure
        output = {
            "metadata": {
                "dataset": name,
                "hf_path": config["hf_path"],
                "language": lang,
                "task": config["task"],
                "description": config["description"],
                "num_samples": len(samples),
                "extracted_at": datetime.now().isoformat(),
            },
            "samples": samples
        }

        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"  Saved {len(samples)} training samples to: {output_file}")
        return len(samples)

    except Exception as e:
        print(f"  Error loading {name}: {e}")
        return None

# Process all languages and datasets
summary = {}
for lang, datasets in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"Extracting {lang.upper()} training sets")
    print(f"{'='*60}")

    summary[lang] = {}
    for name, config in datasets.items():
        count = extract_dataset(lang, name, config)
        if count:
            summary[lang][name] = count

# Save summary
summary_file = Path(TRAIN_DATA_DIR) / "extraction_summary.json"
with open(summary_file, 'w') as f:
    json.dump({
        "extracted_at": datetime.now().isoformat(),
        "datasets": summary
    }, f, indent=2)

print(f"\n{'='*60}")
print("Extraction Summary")
print(f"{'='*60}")
for lang, datasets in summary.items():
    total = sum(datasets.values())
    print(f"{lang.upper()}: {len(datasets)} datasets, {total} total training samples")
print(f"\nSummary saved to: {summary_file}")

PYTHON_SCRIPT

# ==================== List Downloaded Files ====================
echo ""
echo "============================================================"
echo "Downloaded Training Sets"
echo "============================================================"

for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "=== ${lang^^} ==="
    if [ -d "$TRAIN_DATA_DIR/$lang" ]; then
        ls -lh "$TRAIN_DATA_DIR/$lang"/*.json 2>/dev/null || echo "No JSON files found"
    else
        echo "Directory not created yet"
    fi
done

# ==================== Completion ====================
echo ""
echo "============================================================"
echo "EuroEval Training Set Download Completed"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Training sets saved to: $TRAIN_DATA_DIR"
echo "  - Icelandic (is): $TRAIN_DATA_DIR/is/"
echo "  - German (de): $TRAIN_DATA_DIR/de/"
echo "  - Swedish (sv): $TRAIN_DATA_DIR/sv/"
echo ""
echo "Each file contains:"
echo "  - metadata: dataset info, task type, description"
echo "  - samples: list of training examples"
echo ""
echo "Cache directory (raw HF datasets): $CACHE_DIR/datasets"
echo ""
echo "Done!"
