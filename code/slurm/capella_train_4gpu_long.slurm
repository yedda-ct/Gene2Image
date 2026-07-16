#!/bin/bash
# =============================================================================
# Capella TRAIN 4-GPU LONG (non-default) — same 4-parallel-single-GPU as capella_train_4gpu
# but 2-day walltime, for a BATCH that genuinely needs >12h. Prefer the 12h default +
# --auto_resume/--requeue (split into short jobs) over this. Full node, 4 experiments PARALLEL, via
# run_all.sh's built-in scheduler (MAX_PARALLEL=4). NOT DDP: each experiment is a
# normal single-GPU run pinned with CUDA_VISIBLE_DEVICES, so no unvalidated
# multi-GPU training path. EPOCHS=50, early-stop off (--patience 9999), --auto_resume.
#
# BATCH selects a chunk so a job fits the walltime and queues fast:
#   BATCH=1 -> c1   BATCH=2 -> c2   BATCH=3 -> p1   (each = 6 variants x 3 seeds = 18 runs)
#   BATCH=4 -> cross-dataset + RQ4 interpret + summary (after 1-3 finish)
#   (unset)  -> whole matrix c1 c2 p1 (won't fit 12h; prefer BATCH=1/2/3, resubmit/resume as needed)
# Resources: 4 GPU, 56 CPU, 700G, 12h (default short; capella_train_4gpu_long.slurm for >12h).
# Run:  sbatch --export=ALL,BATCH=1 slurm/capella_train_4gpu.slurm
# =============================================================================
#SBATCH --job-name=g2i_tr4L
#SBATCH --partition=capella
#SBATCH --account=CHANGEME_capella_account     # <- set your Capella project/account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56                      # 4 GPU x 14 cores (<= 56 node cap)
#SBATCH --mem=700G                              # < 752,520 MB node cap (leave headroom)
#SBATCH --time=2-00:00:00                       # 2 days; ONLY when a BATCH truly needs >12h. NOT a default. Capella max 7d.
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/chwu350f-g2i/Gene2Image/code}"
source "$PROJECT_DIR/slurm/_capella_common.sh"

# --- pick the chunk (keeps each job short so it queues fast; resume covers >12h) ---
case "${BATCH:-all}" in
  1) DATASETS="c1"; INC_CROSS=0; INC_INTERP=1 ;;
  2) DATASETS="c2"; INC_CROSS=0; INC_INTERP=1 ;;
  3) DATASETS="p1"; INC_CROSS=0; INC_INTERP=1 ;;
  4) DATASETS="c1 c2 p1"; VARIANTS=""; INC_CROSS=1; INC_INTERP=1 ;;   # cross + interpret + summary only
  all) DATASETS="c1 c2 p1"; INC_CROSS=1; INC_INTERP=1 ;;
  *) echo "[FATAL] BATCH must be 1|2|3|4|all (got '${BATCH}')"; exit 1 ;;
esac
VARIANTS="${VARIANTS-gene2image geneflow randpath pathprior notrans nomask}"

# --- run 4 experiments in parallel (one per GPU) via run_all.sh MAX_PARALLEL=4 ---
export GPUS="0 1 2 3"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
[ "$MAX_PARALLEL" -le 4 ] || { echo "[FATAL] MAX_PARALLEL=$MAX_PARALLEL > 4 GPUs"; exit 1; }
echo ">>> TRAIN(4gpu parallel) BATCH=${BATCH:-all} DATASETS='${DATASETS}' VARIANTS='${VARIANTS}' MAX_PARALLEL=${MAX_PARALLEL}"

EPOCHS="${EPOCHS:-50}" BATCH_SIZE="${BATCH_SIZE:-16}" EVAL_BATCH="${EVAL_BATCH:-8}" \
GEN_STEPS="${GEN_STEPS:-100}" WORKERS="${WORKERS:-14}" DB="${DB:-hallmark}" SEEDS="${SEEDS:-42 43 44}" \
DATASETS="${DATASETS}" VARIANTS="${VARIANTS}" \
INCLUDE_CROSS="${INC_CROSS}" INCLUDE_INTERPRET="${INC_INTERP}" INCLUDE_REACTOME="${INCLUDE_REACTOME:-0}" \
DATA_ROOT="$DATA_DIR" MASK_DIR="$MASK_DIR" OUT_ROOT="$OUTPUT_DIR" GMT_HALLMARK="$GMT_HALLMARK" \
EXTRA="${EXTRA:---use_amp --patience 9999 --auto_resume}" \
  bash scripts/run_all.sh "${MAX_PARALLEL}"

echo "=== [ok] train(4gpu) BATCH=${BATCH:-all} done at $(date) ==="
