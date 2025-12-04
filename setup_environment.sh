#!/bin/bash

# Exit on error
set -e

echo "Setting up environment for Qwen 1.5B on Berzelius..."

# Load Python module
module load Python/3.9.6-GCCcore-11.2.0

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu117
pip install transformers datasets accelerate peft
pip install "euroeval[all]"
pip install wandb

echo "Environment setup complete."
echo "Activate with: source .venv/bin/activate"

