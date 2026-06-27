#!/bin/bash
# =============================================================================
# Gene2Image — Capella GPU fail-fast sbatch template
#
# 用法：
#   sbatch -J g2i_b1 --export=ALL,BATCH=1 slurm/capella.slurm
#   sbatch -J g2i_b2 --export=ALL,BATCH=2 slurm/capella.slurm
#
# 关键特性：
#   1. 自动检测 CUDA 是否可用
#   2. 如果没有 GPU / torch.cuda 不可用，立即退出，不偷偷跑 CPU
#   3. 自动根据实际可见 GPU 数量设置 run_all.sh 并发数
#   4. 避免 “申请 4 卡却 run_all.sh 8” 导致部分任务 CUDA 失效
# =============================================================================

#SBATCH --job-name=g2i_batch
#SBATCH --partition=capella
#SBATCH --account=swtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=752520
#SBATCH --cpu-freq=High
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Paths / environment
# -----------------------------------------------------------------------------

PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/chwu350f-g2i/Gene2Image/code}"
VENV_DIR="${VENV_DIR:-/data/horse/ws/chwu350f-g2i/venv_piptorch}"

RELEASE_MODULE="${RELEASE_MODULE:-release/24.10}"
GCCCORE_MODULE="${GCCCORE_MODULE:-GCCcore/13.2.0}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5}"

GMT_HALLMARK="${GMT_HALLMARK:-}"
GMT_REACTOME="${GMT_REACTOME:-}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/processed_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR}"
INTERPRET_SEED="${INTERPRET_SEED:-42}"

: "${BATCH:?必须设置 BATCH=1..8，例如 sbatch -J g2i_b1 --export=ALL,BATCH=1 slurm/capella.slurm}"

module --force purge >/dev/null 2>&1 || module purge || true
module load "${RELEASE_MODULE}" "${GCCCORE_MODULE}" "${PYTHON_MODULE}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${PROJECT_DIR}"
mkdir -p logs

# -----------------------------------------------------------------------------
# GPU fail-fast self-check
# -----------------------------------------------------------------------------

echo "============================================================"
echo "Gene2Image Capella self-check"
echo "date=$(date)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-<none>}"
echo "job_name=${SLURM_JOB_NAME:-<none>}"
echo "partition=${SLURM_JOB_PARTITION:-<none>}"
echo "node_list=${SLURM_JOB_NODELIST:-<none>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_GPUS=${SLURM_GPUS:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "============================================================"

echo "=== nvidia-smi ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This job is probably not on a GPU node." >&2
  exit 101
fi

if ! nvidia-smi; then
  echo "ERROR: nvidia-smi failed. GPU is not visible inside this Slurm job." >&2
  exit 102
fi

echo "=== PyTorch CUDA self-check ==="
N_GPU="$(
python - <<'PY'
import sys
import torch

print("torch_version=", torch.__version__, file=sys.stderr)
print("torch_cuda_version=", torch.version.cuda, file=sys.stderr)
print("cuda_available=", torch.cuda.is_available(), file=sys.stderr)
print("cuda_device_count=", torch.cuda.device_count(), file=sys.stderr)

if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False. Refusing to run on CPU.", file=sys.stderr)
    sys.exit(10)

n = torch.cuda.device_count()
if n < 1:
    print("ERROR: torch sees zero CUDA devices. Refusing to run.", file=sys.stderr)
    sys.exit(11)

for i in range(n):
    props = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(
        f"gpu[{i}] name={props.name} total={total/1024**3:.2f}GiB free={free/1024**3:.2f}GiB",
        file=sys.stderr,
    )

print(n)
PY
)" || {
  echo "ERROR: CUDA self-check failed. Job will exit instead of silently running on CPU." >&2
  exit 103
}

echo "Detected visible CUDA GPU count: ${N_GPU}"

if [ "${N_GPU}" -lt 1 ]; then
  echo "ERROR: detected N_GPU=${N_GPU}. Refusing to run." >&2
  exit 104
fi

# run_all.sh expects logical GPU ids relative to the current CUDA_VISIBLE_DEVICES.
# If torch sees 4 devices, use GPUS='0 1 2 3' and MAX_PARALLEL=4.
GPUS="$(seq -s ' ' 0 $((N_GPU - 1)))"
export GPUS
MAX_PARALLEL="${MAX_PARALLEL:-$N_GPU}"

echo "Using GPUS='${GPUS}'"
echo "Using MAX_PARALLEL='${MAX_PARALLEL}'"

if [ "${MAX_PARALLEL}" -gt "${N_GPU}" ]; then
  echo "ERROR: MAX_PARALLEL=${MAX_PARALLEL} > visible CUDA GPUs=${N_GPU}." >&2
  echo "This would cause some child jobs to run without CUDA. Refusing to continue." >&2
  exit 105
fi

# Optional: prevent PyTorch allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Reproducibility / common env
export PYTHONHASHSEED=0
export GMT_HALLMARK GMT_REACTOME
export DATA_ROOT="${DATA_DIR}"
export MASK_DIR="${MASK_DIR:-$PROJECT_DIR/data/pathway_masks}"

# -----------------------------------------------------------------------------
# Common experiment hyperparameters
# -----------------------------------------------------------------------------

COMMON="EPOCHS=100 BATCH_SIZE=16 EVAL_BATCH=8 GEN_STEPS=100 WORKERS=5 \
INCLUDE_REACTOME=0 EXTRA=--use_amp OUT_ROOT=${OUTPUT_DIR}"

run_batch() {
  local ds="$1"
  local var="$2"
  local cross="$3"

  echo "============================================================"
  echo "run_all:"
  echo "  DATASETS='${ds}'"
  echo "  VARIANTS='${var}'"
  echo "  INCLUDE_CROSS=${cross}"
  echo "  SEEDS='${SEEDS:-42 43 44}'"
  echo "  MAX_PARALLEL=${MAX_PARALLEL}"
  echo "  GPUS='${GPUS}'"
  echo "============================================================"

  env $COMMON \
    DATASETS="${ds}" \
    VARIANTS="${var}" \
    SEEDS="${SEEDS:-42 43 44}" \
    INCLUDE_CROSS="${cross}" \
    INCLUDE_INTERPRET=0 \
    bash scripts/run_all.sh "${MAX_PARALLEL}"
}

case "${BATCH}" in
  1)
    run_batch "c1 c2" "gene2image geneflow" 0
    ;;

  2)
    run_batch "c1 c2" "randpath pathprior" 0
    ;;

  3)
    run_batch "c1 c2" "notrans nomask" 0
    ;;

  4)
    run_batch "p1" "gene2image geneflow" 0
    ;;

  5)
    run_batch "p1" "randpath pathprior" 0
    ;;

  6)
    run_batch "p1" "notrans nomask" 0
    ;;

  7)
    # VARIANTS="" means only cross-dataset jobs, no retraining main/ablation.
    run_batch "c1 c2 p1" "" 1
    ;;

  8)
    echo "=== Batch 8: RQ4 interpretability + final summary ==="

    # This block runs c1/c2/p1 interpretation in parallel.
    # It needs at least 3 visible GPUs.
    if [ "${N_GPU}" -lt 3 ]; then
      echo "ERROR: BATCH=8 needs at least 3 visible GPUs, but only ${N_GPU} detected." >&2
      exit 106
    fi

    dataset_dir() {
      case "$1" in
        c1) echo "Xenium_V1_hSkin_Melanoma_Base_FFPE" ;;
        c2) echo "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE" ;;
        p1) echo "Xenium_Prime_Human_Skin_FFPE" ;;
        *)  echo "ERROR_UNKNOWN_DATASET" ;;
      esac
    }

    adata_of() {
      echo "${DATA_DIR}/$(dataset_dir "$1")/adata.h5ad"
    }

    imgpaths_of() {
      echo "${DATA_DIR}/$(dataset_dir "$1")/cell_patch_256_aux/input/cell_image_paths_local.json"
    }

    missing=0
    for ds in c1 c2 p1; do
      ck="${OUTPUT_DIR}/gene2image_${ds}_seed${INTERPRET_SEED}/checkpoints/best_checkpoint.pt"
      if [ ! -f "$ck" ]; then
        echo "Missing checkpoint: $ck" >&2
        missing=1
      fi
    done

    if [ "$missing" = "1" ]; then
      echo "ERROR: Required gene2image checkpoints are missing. Run batches 1 and 4 first." >&2
      exit 1
    fi

    pids=()
    gpu=0

    for ds in c1 c2 p1; do
      ck="${OUTPUT_DIR}/gene2image_${ds}_seed${INTERPRET_SEED}/checkpoints/best_checkpoint.pt"
      gf="${OUTPUT_DIR}/geneflow_${ds}_seed${INTERPRET_SEED}/gene_importance_scores.csv"

      gfarg=""
      if [ -f "$gf" ]; then
        gfarg="--geneflow_importance $gf"
      else
        echo "WARNING: $gf not found; GSEA consistency sub-analysis will be skipped for ${ds}."
      fi

      echo "START interpret ${ds} on logical GPU ${gpu}"

      (
        export CUDA_VISIBLE_DEVICES="${gpu}"

        python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA unavailable inside interpret subprocess.", file=sys.stderr)
    sys.exit(20)
print("interpret subprocess CUDA OK:", torch.cuda.get_device_name(0), file=sys.stderr)
PY

        python analysis/pathway_interpret.py \
          --model_path "$ck" \
          --adata "$(adata_of "$ds")" \
          --image_paths "$(imgpaths_of "$ds")" \
          --out_dir "${OUTPUT_DIR}/interpret/${ds}" \
          $gfarg \
          --gen_steps 100
      ) > "logs/interpret_${ds}.log" 2>&1 &

      pids+=($!)
      gpu=$((gpu + 1))
    done

    fail=0
    for p in "${pids[@]}"; do
      wait "$p" || fail=1
    done

    if [ "$fail" = "1" ]; then
      echo "ERROR: at least one interpret task failed. See logs/interpret_*.log" >&2
      exit 1
    fi

    echo "=== Final summary ==="
    python scripts/summarize_results.py \
      --results_root "${OUTPUT_DIR}" \
      --out_dir "${OUTPUT_DIR}"
    ;;

  *)
    echo "ERROR: BATCH must be 1..8, got '${BATCH}'." >&2
    exit 1
    ;;
esac

echo "=== Batch ${BATCH} completed at $(date) ==="
