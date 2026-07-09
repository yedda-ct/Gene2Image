#!/bin/bash
# =============================================================================
# Capella ARRAY 1-GPU — DEFAULT way to run the 54-run main matrix.
# A Slurm job array of 54 single-GPU tasks (one experiment each); %N throttles
# concurrency. 1-GPU tasks backfill into ANY free GPU, so when GPUs are plentiful
# this is the FASTEST + most queue-friendly option: it scales past 4 (no whole-
# node wait that a 4-GPU job needs). Each task = one experiment (train 50ep+eval);
# 2-day walltime fits the longest (~41h). --auto_resume/--requeue cover preemption.
#
# SUBMIT (build masks once, then the array, up to 8 concurrent):
#   PREP=$(sbatch --parsable --export=ALL,MODE=prep --gres=gpu:1 --time=00:30:00 slurm/capella.slurm)
#   sbatch --array=0-53%8 --dependency=afterok:$PREP slurm/capella_array_1gpu.slurm
#   # raise %8 -> %16/%32 for more speed if your account/QoS allows.
# RE-SUBMIT only failed/unfinished tasks after checking:
#   python scripts/validate_runs.py --results_root results --expected_epochs 50 --patience 9999
#   sbatch --array=<idx list, e.g. 3,7,40> slurm/capella_array_1gpu.slurm
# AFTER the matrix is done — cross-dataset + RQ4 interpret + summary:
#   sbatch --export=ALL,BATCH=4 slurm/capella_train_4gpu.slurm
#
# Task index -> (variant, dataset, seed):  idx = variant*9 + dataset*3 + seed
#   0-8 gene2image, 9-17 geneflow, ... ; within: c1 c2 p1 ; within: seed 42 43 44
# =============================================================================
#SBATCH --job-name=g2i_arr
#SBATCH --partition=capella
#SBATCH --account=CHANGEME_capella_account     # <- set your Capella project/account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14                      # Capella: 14 cores / GPU
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00                       # fits one full experiment (~30-41h); 1-GPU still backfills easily
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/chwu350f-g2i/Gene2Image/code}"
source "$PROJECT_DIR/slurm/_capella_common.sh"

VARIANTS=(gene2image geneflow randpath pathprior notrans nomask)
DATASETS=(c1 c2 p1)
SEEDS=(42 43 44)
ALL=()
for v in "${VARIANTS[@]}"; do for d in "${DATASETS[@]}"; do for s in "${SEEDS[@]}"; do ALL+=("$v $d $s"); done; done; done   # 54

idx="${SLURM_ARRAY_TASK_ID:?MODE array needs a job array, e.g. --array=0-53%8}"
[ "$idx" -lt "${#ALL[@]}" ] || { echo "[FATAL] array idx $idx >= ${#ALL[@]} (0-53)"; exit 1; }
read -r VARIANT DS SEED <<< "${ALL[$idx]}"
[ "$VARIANT" = "geneflow" ] || ensure_masks_for "$DS"   # no-op if MODE=prep already built masks
echo ">>> ARRAY task ${idx}: ${VARIANT} ${DS} seed${SEED}"

EPOCHS="${EPOCHS:-50}" BATCH_SIZE="${BATCH_SIZE:-16}" EVAL_BATCH="${EVAL_BATCH:-8}" \
GEN_STEPS="${GEN_STEPS:-100}" WORKERS=14 DB="${DB:-hallmark}" \
DATA_ROOT="$DATA_DIR" MASK_DIR="$MASK_DIR" OUT_ROOT="$OUTPUT_DIR" \
EXTRA="${EXTRA:---use_amp --patience 9999 --auto_resume}" \
  bash scripts/run_experiments.sh "${VARIANT}" "${DS}" "${SEED}"

echo "=== [ok] array task ${idx} (${VARIANT}_${DS}_seed${SEED}) done at $(date) ==="
