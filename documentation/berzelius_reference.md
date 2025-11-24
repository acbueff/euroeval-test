# Berzelius HPC Reference

## System Overview
- **Hardware:** 752 NVIDIA A100 GPUs (80GB or 40GB).
- **OS:** Red Hat Enterprise Linux 8.8.
- **Scheduler:** Slurm.
- **File Systems:**
  - `/home`: Small storage (20GB), limited file count. **Do not store data/models here.**
  - `/proj`: Project storage (2TB+). **Store data and models here.**

## GPU Resources
- **Nodes:**
  - Thin: 8x A100 (40GB)
  - Fat: 8x A100 (80GB)
- **Access:** Via `sbatch` (batch jobs) or `interactive` (testing).

## Software & Environment
### Modules
Berzelius uses Lmod.
```bash
module avail
module load Python/3.9.6-GCCcore-11.2.0  # Example
```

### Apptainer (Containers)
Apptainer (formerly Singularity) is the recommended way to run complex Python environments (like those needing specific PyTorch/CUDA versions).

**Basic Apptainer Workflow:**
1. **Build/Pull Image:**
   ```bash
   apptainer pull pytorch.sif docker://pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
   ```
2. **Run Command:**
   ```bash
   apptainer exec --nv pytorch.sif python script.py
   ```
   *`--nv` flag is crucial for GPU access.*

### VSCode Integration
VSCode can connect via SSH.
1. Install "Remote - SSH" extension.
2. Host config:
   ```ssh
   Host berzelius
       HostName berzelius1.nsc.liu.se
       User <your-username>
       IdentityFile ~/.ssh/id_rsa
   ```
3. **Warning:** Do not run heavy computations on the login node via VSCode terminal. Use `interactive` sessions or `sbatch`.

## Job Submission (Slurm)
**Example `submit.sh`:**
```bash
#!/bin/bash
#SBATCH --job-name=euroeval_qwen
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --account=<your-project-id>

module load Apptainer
apptainer exec --nv my_container.sif scandeval --model Qwen/Qwen2.5-1.5B-Instruct --language de
```

## Storage Best Practices
- **Models:** Download HF models to `/proj/<project>/models` (set `HF_HOME` env var).
- **Data:** Store validation sets in `/proj/<project>/data`.

