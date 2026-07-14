#!/bin/bash
# =============================================================================
# Capella DEBUG — env / paths / import / data read / model load / 1 batch.
# Purpose: prove the environment works. NOT a real training run.
# Uses geneflow (encoder_type=rna, needs NO pathway mask) + a tiny 64-cell subset.
# Resources: 1 GPU, 14 CPU, 120G, 1h.
# Run:  sbatch slurm/capella_debug_1gpu.slurm
# =============================================================================
#SBATCH --job-name=g2i_debug
#SBATCH --partition=capella
#SBATCH --account=CHANGEME_capella_account     # <- set your Capella project/account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14                      # Capella: 14 cores / GPU
#SBATCH --mem=120G                              # per-GPU share ~188G; 120G is enough for debug
#SBATCH --time=01:00:00                         # debug: 1 hour
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/chwu350f-g2i/Gene2Image/code}"
# Write to a SEPARATE dir so this FAKE 64-cell debug run never lands in the formal
# results/. The formal array uses --auto_resume, so a debug geneflow_c1_seed42 checkpoint
# left in results/ would be auto-resumed by the formal run and invalidate it.
# DEDICATED var (NOT ${OUTPUT_DIR:-...}): env_g2i.sh exports OUTPUT_DIR=$PROJECT_DIR/results
# and Slurm inherits the submit env, so `${OUTPUT_DIR:-...}` would keep the formal results/
# and let this fake 64-cell debug run pollute it. Only DEBUG_OUTPUT_DIR can repoint this.
export OUTPUT_DIR="${DEBUG_OUTPUT_DIR:-$PROJECT_DIR/results_debug}"
source "$PROJECT_DIR/slurm/_capella_common.sh"

VARIANT="${VARIANT:-geneflow}"; DS="${DS:-c1}"; SEED="${SEED:-42}"
[ "$VARIANT" != "geneflow" ] && ensure_masks_for "$DS"

echo ">>> DEBUG ${VARIANT} ${DS} seed${SEED}: 1 epoch, 64-cell subset, train+eval"
# NOTE: eval now supports --debug/--debug_samples too, so EVAL_EXTRA mirrors training's
# 64-cell subset: eval reconstructs the SAME 80/10/10 split (same held-out test cells) and
# runs on the ~7 test cells of that subset — a few minutes total. (Without this, eval would
# split the FULL dataset and run min(test_split, 5000) samples ~ 625 batches of UNI2-h +
# nuclei morphology, ~3h, on cells that don't match the debug training split.)
EPOCHS=1 BATCH_SIZE=8 EVAL_BATCH=8 GEN_STEPS=20 WORKERS=8 \
EXTRA="--use_amp --debug --debug_samples 64" \
EVAL_EXTRA="--debug --debug_samples 64" \
  bash scripts/run_experiments.sh "$VARIANT" "$DS" "$SEED"
echo "=== [ok] debug passed at $(date) ==="
