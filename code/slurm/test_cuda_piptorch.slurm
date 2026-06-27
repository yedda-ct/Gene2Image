#!/bin/bash
#SBATCH --job-name=test_cuda_piptorch
#SBATCH --partition=alpha
#SBATCH --account=swtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load release/24.10 GCCcore/13.2.0 Python/3.11.5

source /data/horse/ws/chwu350f-g2i/venv_piptorch/bin/activate

echo "hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}"
nvidia-smi
nvidia-smi -q | egrep -i "Driver Version|CUDA Version|Fabric|MIG|Compute Mode" || true

python - <<'PY'
import os
import torch

print("python:", __import__("sys").executable)
print("torch:", torch.__version__, torch.__file__)
print("torch cuda:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    print("gpu tensor OK:", x.mean().item())
PY
