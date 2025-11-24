# EuroEval on Berzelius

This repository contains the setup for running EuroEval (via ScandEval) on the Berzelius HPC.

## Structure
- `scripts/`: SLURM submission scripts and setup scripts.
- `data/`: Place your German validation data here (if using custom data).
- `models/`: Hugging Face models will be cached here.
- `containers/`: Apptainer images.
- `results/`: Evaluation outputs.
- `documentation/`: Reference guides.

## Quick Start

1. **Setup Container**
   Run the setup script to build the EuroEval container:
   ```bash
   bash scripts/setup_container.sh
   ```

2. **Configure Slurm Script**
   Edit `scripts/run_eval.slurm`:
   - Update `#SBATCH --account=...` with your Berzelius project ID.

3. **Run Evaluation**
   Submit the job:
   ```bash
   sbatch scripts/run_eval.slurm
   ```

## Notes
- Ensure you are in the project root when submitting jobs.
- The `HF_HOME` is set to `./models` to avoid filling your home quota.

