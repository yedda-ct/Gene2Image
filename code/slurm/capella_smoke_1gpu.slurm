#!/bin/bash
# =============================================================================
# Capella SMOKE — a few epochs on a small subset to confirm the REAL training
# flow: masks -> pathway encoder -> train -> checkpoint save -> eval -> summary.
# Resources: 1 GPU, 14 CPU, 180G, 4h.
# Run:  sbatch slurm/capella_smoke_1gpu.slurm         (defaults gene2image c1 42)
#       sbatch --export=ALL,VARIANT=pathprior,DS=c2 slurm/capella_smoke_1gpu.slurm
# =============================================================================
#SBATCH --job-name=g2i_smoke
#SBATCH --partition=capella
#SBATCH --account=CHANGEME_capella_account     # <- set your Capella project/account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=150G                              # smoke: ~150G is plenty
#SBATCH --time=04:00:00                         # smoke: 4 hours
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/chwu350f-g2i/Gene2Image/code}"
source "$PROJECT_DIR/slurm/_capella_common.sh"

VARIANT="${VARIANT:-gene2image}"; DS="${DS:-c1}"; SEED="${SEED:-42}"
ensure_masks_for "$DS"

echo ">>> SMOKE ${VARIANT} ${DS} seed${SEED}: 3 epochs, 256-cell subset, train+eval+summary"
EPOCHS="${EPOCHS:-3}" BATCH_SIZE=16 EVAL_BATCH=8 GEN_STEPS=50 WORKERS=14 \
EXTRA="--use_amp --patience 9999 --auto_resume --debug --debug_samples 256" \
  bash scripts/run_experiments.sh "$VARIANT" "$DS" "$SEED"
python scripts/validate_runs.py --results_root "$OUTPUT_DIR" --expected_epochs "${EPOCHS:-3}" --patience 9999 || true
echo "=== [ok] smoke done at $(date) ==="
