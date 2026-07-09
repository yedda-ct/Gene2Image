#!/bin/bash
# =============================================================================
# Capella TRAIN 1-GPU — one full experiment (train+eval) on a single H100.
# EPOCHS=50, early-stop OFF (--patience 9999), --auto_resume (checkpoint-resumable
# so a long run can be split into short jobs). best_checkpoint selected on val_mse.
# Resources: 1 GPU, 14 CPU, 180G, 1 day.
# Run:  sbatch --export=ALL,VARIANT=gene2image,DS=c1,SEED=42 slurm/capella_train_1gpu.slurm
#       # resubmit the SAME command to resume after preemption.
# =============================================================================
#SBATCH --job-name=g2i_tr1
#SBATCH --partition=capella
#SBATCH --account=CHANGEME_capella_account     # <- set your Capella project/account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=180G
#SBATCH --time=12:00:00                         # 12h default (resume if needed; do NOT change EPOCHS between runs)
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/chwu350f-g2i/Gene2Image/code}"
source "$PROJECT_DIR/slurm/_capella_common.sh"

VARIANT="${VARIANT:-gene2image}"; DS="${DS:-c1}"; SEED="${SEED:-42}"
ensure_masks_for "$DS"

echo ">>> TRAIN(1gpu) ${VARIANT} ${DS} seed${SEED}: EPOCHS=${EPOCHS:-50}"
EPOCHS="${EPOCHS:-50}" BATCH_SIZE="${BATCH_SIZE:-16}" EVAL_BATCH="${EVAL_BATCH:-8}" \
GEN_STEPS="${GEN_STEPS:-100}" WORKERS=14 \
EXTRA="${EXTRA:---use_amp --patience 9999 --auto_resume}" \
  bash scripts/run_experiments.sh "$VARIANT" "$DS" "$SEED"
echo "=== [ok] train(1gpu) ${VARIANT}_${DS}_seed${SEED} done at $(date) ==="
