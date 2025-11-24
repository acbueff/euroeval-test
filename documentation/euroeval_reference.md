# EuroEval Reference

**Note:** "EuroEval" often refers to the broader benchmarking initiative for European languages. The primary tool implementing these benchmarks (including German) is currently **ScandEval**. This guide assumes the use of `ScandEval` to run the European evaluation tasks.

## Overview
EuroEval/ScandEval is a benchmarking framework designed to evaluate the performance of language models across European languages. It ensures comparable results by using standardized datasets and metrics.

- **Repository:** [ScandEval on GitHub](https://github.com/ScandEval/ScandEval)
- **Python Package:** `scandeval`
- **Supported Languages:** Includes German (`de`), Swedish, Danish, Norwegian, Icelandic, English, etc.

## Evaluation on German Validation Set
To evaluate a model on the German subset of the benchmark, you typically specify the language and the model.

### Key Command
```bash
scandeval --model Qwen/Qwen2.5-1.5B-Instruct --language de
```

### Parameters
- `--model`: The Hugging Face model ID (e.g., `Qwen/Qwen2.5-1.5B-Instruct`).
- `--language`: The language code to evaluate on (e.g., `de` for German).
- `--dataset`: (Optional) To specify a specific dataset (e.g., `germeval`).
- `--device`: `cuda` for GPU acceleration.

### German Datasets in EuroEval/ScandEval
Common German datasets evaluated include:
- **GermanQuAD** (Question Answering)
- **GermEval** (Sentiment/Classification)
- **XNLI** (German subset - NLI)

## Usage with Qwen Model
The `Qwen/Qwen2.5-1.5B-Instruct` is a supported Hugging Face model.
- **Context Length:** Supports up to 32k tokens.
- **Prompting:** As an instruct model, it works best with chat templates, which ScandEval handles automatically for generic HF models.

## Installation
```bash
pip install scandeval
```
*Note: On HPC, this should be installed inside a virtual environment or container.*

