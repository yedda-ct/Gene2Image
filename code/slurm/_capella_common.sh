#!/bin/bash
# =============================================================================
# Gene2Image — shared Capella env + safety checks.
# `source` this from every capella_*.slurm AFTER its #SBATCH block.
# All hard-coded site paths live HERE (edit once), per the "centralise config" rule.
# =============================================================================
set -euo pipefail

# ---------- Hard-coded site config (EDIT for your Capella workspace) ----------
PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/chwu350f-g2i/Gene2Image/code}"
VENV_DIR="${VENV_DIR:-/data/horse/ws/chwu350f-g2i/venv_piptorch}"
RELEASE_MODULE="${RELEASE_MODULE:-release/24.10}"
GCCCORE_MODULE="${GCCCORE_MODULE:-GCCcore/13.2.0}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5}"

DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/processed_data}"
MASK_DIR="${MASK_DIR:-$PROJECT_DIR/data/pathway_masks}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results}"       # per-run dirs land here
CKPT_DIR="${CKPT_DIR:-$OUTPUT_DIR}"                    # ckpts: $OUTPUT_DIR/<run>/checkpoints
GMT_HALLMARK="${GMT_HALLMARK:-$PROJECT_DIR/../gmt/msigdb_2023.2_Hs/h.all.v2023.2.Hs.symbols.gmt}"

# Model weights / caches / logging
export UNI2H_MODEL_PATH="${UNI2H_MODEL_PATH:-}"        # gated MahmoodLab/UNI2-h dir (pytorch_model.bin); unset -> UNI2-h FID = NaN
export HE2RNA_MODEL_PATH="${HE2RNA_MODEL_PATH:-}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export WANDB_MODE="${WANDB_MODE:-offline}"            # training also passes --no_wandb
export PYTHONHASHSEED=0                                # deterministic 80/20 split across train/eval procs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-14}"        # 14 cores per GPU on Capella
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Shared with scripts/run_experiments.sh
export DATA_ROOT="$DATA_DIR"
export MASK_DIR OUTPUT_DIR
export OUT_ROOT="$OUTPUT_DIR"
export DB="${DB:-hallmark}"

# ---------- Modules + venv ----------
module --force purge >/dev/null 2>&1 || module purge || true
module load "$RELEASE_MODULE" "$GCCCORE_MODULE" "$PYTHON_MODULE"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---------- Safety checks + dirs ----------
[ -d "$PROJECT_DIR" ] || { echo "[FATAL] PROJECT_DIR missing: $PROJECT_DIR"; exit 1; }
[ -d "$DATA_DIR" ]    || { echo "[FATAL] DATA_DIR missing: $DATA_DIR (download Zenodo 17429142)"; exit 1; }
cd "$PROJECT_DIR"
mkdir -p logs outputs "$MASK_DIR" "$OUTPUT_DIR" "$HF_HOME"

echo "============================================================"
echo "host=$(hostname)  SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}  task=${SLURM_ARRAY_TASK_ID:-none}  date=$(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_DIR=$DATA_DIR  OUTPUT_DIR=$OUTPUT_DIR  CKPT_DIR=$CKPT_DIR"
echo "UNI2H_MODEL_PATH=${UNI2H_MODEL_PATH:-<unset -> UNI2-h FID = NaN>}"
echo "============================================================"

# ---------- GPU fail-fast (never silently fall back to CPU) ----------
command -v nvidia-smi >/dev/null 2>&1 || { echo "[FATAL] nvidia-smi missing (not a GPU node)"; exit 101; }
nvidia-smi -L || { echo "[FATAL] no GPU visible"; exit 102; }
python - <<'PY' || { echo "[FATAL] torch CUDA check failed; refusing CPU run"; exit 103; }
import sys, torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("cuda_available=%s count=%s" % (torch.cuda.is_available(), torch.cuda.device_count()), file=sys.stderr); sys.exit(10)
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), file=sys.stderr)
PY

# ---------- helpers ----------
dataset_folder() {   # short id -> processed_data folder (mirrors scripts/run_experiments.sh)
  case "$1" in
    c1) echo "Xenium_V1_hSkin_Melanoma_Base_FFPE" ;;
    c2) echo "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE" ;;
    p1) echo "Xenium_Prime_Human_Skin_FFPE" ;;
    *)  echo "UNKNOWN" ;;
  esac
}
ensure_masks_for() {   # $1 = dataset id; builds real/per-seed rand/none if any are missing
  local ds="$1" need=""
  for v in real rand none; do [ -f "$MASK_DIR/${ds}_${DB}_${v}.npz" ] || need=1; done
  for s in 42 43 44; do [ -f "$MASK_DIR/${ds}_${DB}_rand_s${s}.npz" ] || need=1; done
  [ -z "$need" ] && { echo "[masks] $ds masks present"; return 0; }
  [ -f "$GMT_HALLMARK" ] || { echo "[FATAL] masks missing and Hallmark GMT not at $GMT_HALLMARK"; exit 111; }
  echo "[masks] building $ds masks -> $MASK_DIR"
  python scripts/build_pathway_mask.py \
    --adata "$DATA_DIR/$(dataset_folder "$ds")/adata.h5ad" \
    --prefix "$ds" --db "$DB" --gmt "$GMT_HALLMARK" \
    --out_dir "$MASK_DIR" --seed 42 --rand_seeds 42 43 44
}
