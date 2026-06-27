#!/usr/bin/env bash
# =============================================================================
# Gene2Image — pip-only PyTorch environment setup for Alpha / portable clusters
#
# Purpose
# -------
# Create a clean Python venv in the workspace and install PyTorch + torchvision
# from pip wheels, instead of using a site-provided PyTorch module.
#
# This is useful when the site PyTorch module either:
#   - does not ship torchvision;
#   - ships a torchvision build whose compiled ops fail to load;
#   - causes mixed torch/torchvision ABI problems.
#
# Default Alpha modules:
#   release/24.10 GCCcore/13.2.0 Python/3.11.5
#
# Usage on login node:
#   cd /data/horse/ws/chwu350f-g2i/Gene2Image/code
#   bash /path/to/setup_env_pip_alpha.sh
#
# Optional overrides:
#   PROJECT_DIR=/path/to/Gene2Image/code \
#   VENV_DIR=/path/to/venv_piptorch \
#   RELEASE_MODULE=release/24.10 \
#   GCCCORE_MODULE=GCCcore/13.2.0 \
#   PYTHON_MODULE=Python/3.11.5 \
#   TORCH_SPEC="torch==2.2.2+cu121 torchvision==0.17.2+cu121" \
#   TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" \
#   bash setup_env_pip_alpha.sh
#
# Notes
# -----
# - This script intentionally does NOT load a PyTorch module.
# - This script intentionally does NOT create the venv with --system-site-packages.
# - CUDA availability may be False on login nodes. Test CUDA inside a Slurm GPU job.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# User-tunable defaults
# -----------------------------------------------------------------------------

PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/chwu350f-g2i/Gene2Image/code}"
VENV_DIR="${VENV_DIR:-/data/horse/ws/chwu350f-g2i/venv_piptorch}"

RELEASE_MODULE="${RELEASE_MODULE:-release/24.10}"
GCCCORE_MODULE="${GCCCORE_MODULE:-GCCcore/13.2.0}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5}"

# Gene2Image was smoke-tested with this pair on Alpha:
#   torch 2.2.2 + CUDA 12.1 wheel
#   torchvision 0.17.2 + CUDA 12.1 wheel
TORCH_SPEC="${TORCH_SPEC:-torch==2.2.2+cu121 torchvision==0.17.2+cu121}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

# Keep NumPy below 2.x for scanpy/numba/skimage compatibility in this project.
NUMPY_SPEC="${NUMPY_SPEC:-numpy==1.26.4}"

# Extra packages required by training/evaluation/mask construction.
EXTRA_DEPS="${EXTRA_DEPS:-gseapy torchmetrics==1.7.1 scikit-image timm einops safetensors opencv-python-headless==4.10.0.84}"

# Set to 1 to recreate the venv from scratch.
RECREATE_VENV="${RECREATE_VENV:-0}"

# Set to 1 to skip installing PROJECT_DIR/requirements.txt.
SKIP_PROJECT_REQUIREMENTS="${SKIP_PROJECT_REQUIREMENTS:-0}"

# Temporary filtered requirements file.
FILTERED_REQ=""

cleanup() {
  if [ -n "${FILTERED_REQ}" ] && [ -f "${FILTERED_REQ}" ]; then
    rm -f "${FILTERED_REQ}"
  fi
}
trap cleanup EXIT

echo "=== Gene2Image pip-only environment setup ==="
echo "PROJECT_DIR       = ${PROJECT_DIR}"
echo "VENV_DIR          = ${VENV_DIR}"
echo "RELEASE_MODULE    = ${RELEASE_MODULE}"
echo "GCCCORE_MODULE    = ${GCCCORE_MODULE}"
echo "PYTHON_MODULE     = ${PYTHON_MODULE}"
echo "TORCH_SPEC        = ${TORCH_SPEC}"
echo "TORCH_INDEX_URL   = ${TORCH_INDEX_URL}"
echo "NUMPY_SPEC        = ${NUMPY_SPEC}"
echo "RECREATE_VENV     = ${RECREATE_VENV}"
echo

# -----------------------------------------------------------------------------
# Basic checks
# -----------------------------------------------------------------------------

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "ERROR: PROJECT_DIR does not exist: ${PROJECT_DIR}" >&2
  exit 1
fi

if [ ! -f "${PROJECT_DIR}/requirements.txt" ]; then
  echo "WARN: ${PROJECT_DIR}/requirements.txt not found; project requirements will be skipped." >&2
fi

# -----------------------------------------------------------------------------
# Load compiler/Python modules only. Do not load PyTorch module.
# -----------------------------------------------------------------------------

echo "=== [1/6] Loading site modules, without PyTorch module ==="
module --force purge >/dev/null 2>&1 || module purge || true
module load "${RELEASE_MODULE}" "${GCCCORE_MODULE}" "${PYTHON_MODULE}"

echo "Loaded modules:"
module list 2>&1 | sed 's/^/  /'
echo

echo "Python from module:"
which python
python --version
echo

# -----------------------------------------------------------------------------
# Create clean venv without system site packages.
# -----------------------------------------------------------------------------

echo "=== [2/6] Creating/reusing clean venv ==="
mkdir -p "$(dirname "${VENV_DIR}")"

if [ "${RECREATE_VENV}" = "1" ] && [ -d "${VENV_DIR}" ]; then
  backup="${VENV_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
  echo "RECREATE_VENV=1: moving existing venv to ${backup}"
  mv "${VENV_DIR}" "${backup}"
fi

if [ ! -d "${VENV_DIR}" ]; then
  python -m venv "${VENV_DIR}"
  echo "Created venv: ${VENV_DIR}"
else
  echo "Reusing existing venv: ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Active Python:"
which python
python --version
echo

# -----------------------------------------------------------------------------
# Upgrade packaging tools.
# -----------------------------------------------------------------------------

echo "=== [3/6] Upgrading pip/setuptools/wheel ==="
python -m pip install --upgrade pip setuptools wheel
echo

# -----------------------------------------------------------------------------
# Install PyTorch stack from pip wheels.
# -----------------------------------------------------------------------------

echo "=== [4/6] Installing pip PyTorch stack ==="
# shellcheck disable=SC2086
python -m pip install --force-reinstall ${TORCH_SPEC} --index-url "${TORCH_INDEX_URL}"
echo

# -----------------------------------------------------------------------------
# Install project dependencies, filtering out torch/CUDA wheel packages.
# -----------------------------------------------------------------------------

echo "=== [5/6] Installing Gene2Image dependencies ==="
python -m pip install "${NUMPY_SPEC}"

if [ "${SKIP_PROJECT_REQUIREMENTS}" != "1" ] && [ -f "${PROJECT_DIR}/requirements.txt" ]; then
  FILTERED_REQ="$(mktemp)"
  # Exclude only the packages that must remain owned by the pip PyTorch stack.
  # Keep torchmetrics; do not match "torchmetrics" accidentally.
  grep -v -E '^[[:space:]]*(torch|torchvision|torchaudio|nvidia-|triton)([<=>[:space:]]|$)' \
    "${PROJECT_DIR}/requirements.txt" > "${FILTERED_REQ}"

  echo "Installing filtered requirements from ${PROJECT_DIR}/requirements.txt"
  echo "Filtered out lines matching: torch, torchvision, torchaudio, nvidia-*, triton"
  python -m pip install -r "${FILTERED_REQ}"
else
  echo "Skipping project requirements."
fi

# shellcheck disable=SC2086
python -m pip install ${EXTRA_DEPS}
echo

# -----------------------------------------------------------------------------
# Final sanity checks.
# -----------------------------------------------------------------------------

echo "=== [6/6] Sanity checks ==="
python - <<'PY'
import sys
import importlib.metadata as md

print("python executable:", sys.executable)

for pkg in ["torch", "torchvision", "torchmetrics", "numpy", "scanpy", "anndata", "gseapy", "skimage", "h5py", "tifffile"]:
    try:
        print(f"{pkg:12s}:", md.version(pkg))
    except Exception as exc:
        print(f"{pkg:12s}: NOT FOUND ({exc})")

import torch
import torchvision
from torchvision import transforms
import numpy

print()
print("torch file       :", torch.__file__)
print("torch version    :", torch.__version__)
print("torch cuda       :", torch.version.cuda)
print("torchvision file :", torchvision.__file__)
print("torchvision ver  :", torchvision.__version__)
print("numpy version    :", numpy.__version__)
print("transforms OK    :", transforms)

# CUDA may be unavailable on login nodes. This is only a diagnostic.
try:
    print("cuda available   :", torch.cuda.is_available())
    print("device count     :", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("gpu name         :", torch.cuda.get_device_name(0))
except Exception as exc:
    print("cuda check WARN  :", repr(exc))
PY

cat <<EOF

===============================================================================
Environment setup complete.

Activate with:
  source "${VENV_DIR}/bin/activate"

For Slurm scripts using this venv, do NOT load a PyTorch module.
Use only the base modules, for example:
  module purge
  module load "${RELEASE_MODULE}" "${GCCCORE_MODULE}" "${PYTHON_MODULE}"
  source "${VENV_DIR}/bin/activate"

Recommended smoke-test export:
  export PROJECT_DIR="${PROJECT_DIR}"
  export VENV_DIR="${VENV_DIR}"
  export RELEASE_MODULE="${RELEASE_MODULE}"
  export GCCCORE_MODULE="${GCCCORE_MODULE}"
  export PYTHON_MODULE="${PYTHON_MODULE}"

If a specific GPU node reports CUDA Error 802, submit with:
  sbatch --exclude=<bad-node> ...

===============================================================================
EOF
