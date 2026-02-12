#!/usr/bin/env python3
"""
Extract EuroEval training sets from cached arrow files.
Reads directly from the EuroEval cache directory.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import pyarrow as pa

# Configuration
CACHE_DIR = Path("/proj/berzelius-aiics-real/users/x_anbue/euroeval/.euroeval_cache")
OUTPUT_DIR = Path("/proj/berzelius-aiics-real/users/x_anbue/euroeval/train_sets")

# Dataset configurations: {cache_folder: {language, task, description}}
DATASETS = {
    # Icelandic
    "EuroEval___hotter-and-colder-sentiment": {
        "language": "is",
        "name": "hotter-and-colder-sentiment",
        "task": "sentiment-classification",
        "description": "Icelandic blog post sentiment (positive/neutral/negative)",
    },
    "EuroEval___scala-is": {
        "language": "is",
        "name": "scala-is",
        "task": "linguistic-acceptability",
        "description": "Icelandic grammatical correctness (yes/no)",
    },
    "EuroEval___mim-gold-ner-mini": {
        "language": "is",
        "name": "mim-gold-ner",
        "task": "named-entity-recognition",
        "description": "Icelandic NER (PER, LOC, ORG, MISC)",
    },
    "EuroEval___nqii-mini": {
        "language": "is",
        "name": "nqii",
        "task": "reading-comprehension",
        "description": "Natural Questions in Icelandic (extractive QA)",
    },
    "EuroEval___icelandic-knowledge": {
        "language": "is",
        "name": "icelandic-knowledge",
        "task": "knowledge",
        "description": "Icelandic factual knowledge (4-choice MC)",
    },
    "EuroEval___winogrande-is": {
        "language": "is",
        "name": "winogrande-is",
        "task": "common-sense-reasoning",
        "description": "Icelandic pronoun disambiguation",
    },
    # German
    "EuroEval___sb10k-mini": {
        "language": "de",
        "name": "sb10k",
        "task": "sentiment-classification",
        "description": "German sentiment classification",
    },
    "EuroEval___scala-de": {
        "language": "de",
        "name": "scala-de",
        "task": "linguistic-acceptability",
        "description": "German grammatical correctness",
    },
    "EuroEval___germeval-mini": {
        "language": "de",
        "name": "germeval",
        "task": "named-entity-recognition",
        "description": "German NER",
    },
    "EuroEval___germanquad-mini": {
        "language": "de",
        "name": "germanquad",
        "task": "reading-comprehension",
        "description": "German extractive QA",
    },
    "EuroEval___mmlu-de-mini": {
        "language": "de",
        "name": "mmlu-de",
        "task": "knowledge",
        "description": "German MMLU knowledge",
    },
    "EuroEval___hellaswag-de-mini": {
        "language": "de",
        "name": "hellaswag-de",
        "task": "common-sense-reasoning",
        "description": "German common-sense reasoning",
    },
    "EuroEval___mlsum-mini": {
        "language": "de",
        "name": "mlsum",
        "task": "summarization",
        "description": "German news summarization",
    },
    # Swedish
    "EuroEval___swerec-mini": {
        "language": "sv",
        "name": "swerec",
        "task": "sentiment-classification",
        "description": "Swedish review sentiment",
    },
    "EuroEval___scala-sv": {
        "language": "sv",
        "name": "scala-sv",
        "task": "linguistic-acceptability",
        "description": "Swedish grammatical correctness",
    },
    "EuroEval___suc3-mini": {
        "language": "sv",
        "name": "suc3",
        "task": "named-entity-recognition",
        "description": "Swedish NER",
    },
    "EuroEval___scandiqa-sv-mini": {
        "language": "sv",
        "name": "scandiqa-sv",
        "task": "reading-comprehension",
        "description": "Swedish extractive QA",
    },
    "EuroEval___mmlu-sv-mini": {
        "language": "sv",
        "name": "mmlu-sv",
        "task": "knowledge",
        "description": "Swedish MMLU knowledge",
    },
    "EuroEval___hellaswag-sv-mini": {
        "language": "sv",
        "name": "hellaswag-sv",
        "task": "common-sense-reasoning",
        "description": "Swedish common-sense reasoning",
    },
    "EuroEval___swedn-mini": {
        "language": "sv",
        "name": "swedn",
        "task": "summarization",
        "description": "Swedish news summarization",
    },
}


def find_arrow_files(dataset_dir: Path) -> dict:
    """Find train/val/test arrow files in dataset directory."""
    arrow_files = {}

    # Look for files in the nested structure
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.arrow') and not f.startswith('cache-'):
                # Extract split name from filename (e.g., "dataset-train.arrow" -> "train")
                parts = f.replace('.arrow', '').split('-')
                if parts[-1] in ['train', 'val', 'test', 'validation']:
                    split = parts[-1]
                    if split == 'validation':
                        split = 'val'
                    arrow_files[split] = Path(root) / f

    return arrow_files


def read_arrow_file(filepath: Path) -> list:
    """Read samples from an arrow file (IPC stream format)."""
    try:
        # Try IPC stream format first (HuggingFace datasets format)
        with open(filepath, 'rb') as f:
            reader = pa.ipc.open_stream(f)
            table = reader.read_all()

        # Convert to list of dicts
        samples = table.to_pydict()

        # Transpose from column-oriented to row-oriented
        num_samples = len(list(samples.values())[0]) if samples else 0
        rows = []
        for i in range(num_samples):
            row = {k: v[i] for k, v in samples.items()}
            rows.append(row)

        return rows
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return []


def extract_dataset(cache_folder: str, config: dict) -> dict:
    """Extract training set from a cached dataset."""
    dataset_dir = CACHE_DIR / cache_folder

    if not dataset_dir.exists():
        print(f"  Dataset directory not found: {dataset_dir}")
        return None

    arrow_files = find_arrow_files(dataset_dir)

    if 'train' not in arrow_files:
        print(f"  No train split found for {config['name']}")
        print(f"  Available splits: {list(arrow_files.keys())}")
        return None

    print(f"  Found splits: {list(arrow_files.keys())}")

    # Read training data
    train_samples = read_arrow_file(arrow_files['train'])

    if not train_samples:
        print(f"  No samples read from train split")
        return None

    return {
        "metadata": {
            "dataset": config["name"],
            "language": config["language"],
            "task": config["task"],
            "description": config["description"],
            "num_samples": len(train_samples),
            "extracted_at": datetime.now().isoformat(),
            "source_file": str(arrow_files['train']),
        },
        "samples": train_samples
    }


def main():
    print("=" * 60)
    print("EuroEval Training Set Extraction")
    print("=" * 60)
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Create output directories
    for lang in ['is', 'de', 'sv']:
        (OUTPUT_DIR / lang).mkdir(parents=True, exist_ok=True)

    summary = {"is": {}, "de": {}, "sv": {}}

    for cache_folder, config in DATASETS.items():
        print(f"\nProcessing: {config['name']} ({config['language']})")

        result = extract_dataset(cache_folder, config)

        if result:
            # Save to JSON
            output_file = OUTPUT_DIR / config["language"] / f"{config['name']}_train.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            summary[config["language"]][config["name"]] = result["metadata"]["num_samples"]
            print(f"  Saved {result['metadata']['num_samples']} samples to {output_file}")

    # Save summary
    summary_file = OUTPUT_DIR / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "extracted_at": datetime.now().isoformat(),
            "datasets": summary,
            "totals": {
                lang: sum(datasets.values())
                for lang, datasets in summary.items()
            }
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    for lang, datasets in summary.items():
        if datasets:
            total = sum(datasets.values())
            print(f"{lang.upper()}: {len(datasets)} datasets, {total} total training samples")
            for name, count in datasets.items():
                print(f"  - {name}: {count} samples")
        else:
            print(f"{lang.upper()}: No datasets extracted")

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
