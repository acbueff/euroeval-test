# Qwen 1.5B Training on Berzelius (EuroEval)

This repository contains a complete pipeline for training and evaluating the Qwen 1.5B model on the German EuroEval benchmark using the Berzelius HPC system.

## Project Structure

```
├── configs/              # Configuration files
│   └── qwen_1.5b_config.yaml
├── scripts/              # Python scripts
│   ├── train.py          # Distributed training script
│   ├── eval_train.py     # Evaluation on training split
│   ├── eval_validation.py# Evaluation on validation split (EuroEval)
│   └── eval_test.py      # Evaluation on test split (EuroEval)
├── slurm/                # SLURM submission scripts
│   ├── train_qwen.sh
│   └── eval_euroeval.sh
├── utils/                # Helper utilities
│   ├── data_utils.py
│   └── paths.py          # Parses data paths from BERZ-PATH.md
├── logs/                 # Job logs and results
├── documentation/        # Reference documentation
├── setup_environment.sh  # Environment setup script
├── run_full_pipeline.sh  # Pipeline execution script
└── README.md
```

## Prerequisites

- Access to Berzelius HPC
- Python 3.9+
- CUDA 11.7+ (Loaded via module)

## Setup

1. **Configure Environment**
   Run the setup script to create a virtual environment and install dependencies:
   ```bash
   ./setup_environment.sh
   ```

2. **Data Path**
   The system automatically reads the EuroEval cache path from `documentation/BERZ-PATH.md`. Ensure this file exists and contains the correct path.

3. **Project ID**
   Edit `configs/qwen_1.5b_config.yaml` and `slurm/*.sh` files to update the `#SBATCH --account=` with your actual Berzelius project ID if different from the default.

## Usage

### 1. Run Full Pipeline (Training + Evaluation)
To submit the training job followed automatically by the evaluation jobs:
```bash
./run_full_pipeline.sh
```

### 2. Run Individual Steps

**Training Only:**
```bash
sbatch slurm/train_qwen.sh
```

**Evaluation Only:**
```bash
sbatch slurm/eval_euroeval.sh
```
Note: Evaluation expects a trained model at `checkpoints/final_model`. If not found, it may default to the base model.

## Configuration

Modify `configs/qwen_1.5b_config.yaml` to adjust:
- **Model**: Base model name (`Qwen/Qwen2.5-1.5B-Instruct`)
- **Training**: Learning rate, batch size, epochs, mixed precision (bf16)
- **Data**: Language (`de`) and dataset list

## Output

- **Checkpoints**: Saved to `./checkpoints/`
- **Logs**: SLURM output and error logs in `./logs/`
- **Evaluation Results**: JSON reports saved to `./logs/`

## Technical Notes

- **Distributed Training**: Uses `torchrun` with `transformers.Trainer` for DDP training on A100 GPUs.
- **EuroEval**: Evaluation uses the `euroeval` library (formerly ScandEval) for standardized benchmarking on validation and test sets.
- **Training Evaluation**: A custom script `eval_train.py` calculates perplexity on the training split to assess fit.
