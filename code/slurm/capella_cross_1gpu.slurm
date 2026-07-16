#!/bin/bash
# =============================================================================
# Capella ARRAY 1-GPU — cross-panel transfer (v10 「编码效率与跨 panel 探索」).
#
# WHY THIS EXISTS: the documented finisher `sbatch --export=ALL,BATCH=4 capella_train_4gpu.slurm`
# runs cross-dataset + RQ4 interpret + summary inside ONE 12h job. But the cross step is not a
# post-processing pass -- each of its 9 settings TRAINS a model for 50 epochs
# (run_cross_dataset.sh:68), i.e. ~24-44h each, ~300 GPU-hours total. It cannot fit 12h, has no
# self-requeue, and would be killed part-way with nothing to show.
#
# Note these 9 are NOT duplicates of the main matrix: run_cross_dataset.sh trains on the ALIGNED
# mask <src>_to_<tgt>_src.npz, which build_cross_masks.py restricts to the pathways SHARED with the
# target (its header: "intersects their pathway-name sets"). c1_to_c2_src has |c1 ∩ c2| pathways,
# while the main run's c1_hallmark_real has all 33 of c1's. Different mask -> different model.
#
# So: give the cross step the same treatment as the main matrix -- one task per setting, walltime as
# a scheduling knob, self-requeue before the kill.
#
# SUBMIT (after the main 54-run finishes -- gap_vs_native needs the main matrix's target-panel runs):
#   sbatch --array=0-8 slurm/capella_cross_1gpu.slurm
#   # then RQ4 interpret + summary only (they ARE cheap post-processing):
#   sbatch --export=ALL,MODE=post --gres=gpu:1 --time=04:00:00 slurm/capella.slurm
#
# Task index -> (src, tgt, seed):  3 pairs x 3 seeds = 9
# =============================================================================
#SBATCH --job-name=g2i_cross
#SBATCH --partition=capella
#SBATCH --account=CHANGEME_capella_account     # <- set your Capella project/account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=180G
# Same reasoning as capella_array_1gpu.slurm: the USR1 trap requeues before the kill, so this is a
# scheduling knob, not a bet. Cross trains on the source panel, so c1-source ~33h, c2-source ~24h.
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/chwu350f-g2i/Gene2Image/code}"
source "$PROJECT_DIR/slurm/_capella_common.sh"

# Must match run_all.sh's CROSS_PAIRS (:180) and SEEDS.
PAIRS=("c1 c2" "c2 c1" "c1 p1")
SEEDS=(42 43 44)
ALL=()
for p in "${PAIRS[@]}"; do for s in "${SEEDS[@]}"; do ALL+=("$p $s"); done; done   # 9

idx="${SLURM_ARRAY_TASK_ID:?needs a job array, e.g. --array=0-8}"
[ "$idx" -lt "${#ALL[@]}" ] || { echo "[FATAL] array idx $idx >= ${#ALL[@]} (0-8)"; exit 1; }
read -r SRC TGT SEED <<< "${ALL[$idx]}"
ensure_masks_for "$SRC"
ensure_masks_for "$TGT"
echo ">>> CROSS task ${idx}: ${SRC} -> ${TGT} seed${SEED}"

RUN_LOG="$OUTPUT_DIR/cross_dataset/logs/cross_${SRC}_to_${TGT}_s${SEED}.log"
mkdir -p "$OUTPUT_DIR/cross_dataset/logs"
echo "=== attempt at $(date) (job ${SLURM_JOB_ID:-?}, task ${idx}) ===" >> "$RUN_LOG"

# --- trap block: keep identical to capella_array_1gpu.slurm -------------------
_requeued=0
_on_walltime() {
  local _target="${SLURM_JOB_ID}"
  [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && _target="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  echo "[signal] walltime approaching -- requeueing ${_target} (${SRC}->${TGT} seed${SEED});" \
       "next attempt resumes from latest_checkpoint.pt" | tee -a "$RUN_LOG"
  # Set _requeued ONLY on success: a rejected requeue must NOT take the "exit 0, cleanly requeued"
  # path, or SLURM records success and the task is silently dropped with nothing to re-submit.
  if scontrol requeue "${_target}"; then
    _requeued=1
  else
    echo "[signal] scontrol requeue FAILED -- failing loudly so this task can be resubmitted." \
         | tee -a "$RUN_LOG"
  fi
}
trap _on_walltime USR1

set -o pipefail
(
  EPOCHS="${EPOCHS:-50}" BATCH_SIZE="${BATCH_SIZE:-16}" GEN_STEPS="${GEN_STEPS:-100}" \
  WORKERS=14 DB="${DB:-hallmark}" \
  DATA_ROOT="$DATA_DIR" MASK_DIR="$MASK_DIR" OUT_ROOT="$OUTPUT_DIR/cross_dataset" \
  EXTRA="${EXTRA:---use_amp --patience 9999 --auto_resume}" \
    bash scripts/run_cross_dataset.sh "${SRC}" "${TGT}" "${SEED}" 2>&1 | tee -a "$RUN_LOG"
) &
_child=$!
# `|| rc=$?` is load-bearing: _capella_common.sh sets `set -euo pipefail`, so a bare wait returning
# 138 (128+SIGUSR1) would make -e kill this script before the requeue bookkeeping runs.
rc=0; wait "$_child" || rc=$?
while [ "$rc" -gt 128 ] && kill -0 "$_child" 2>/dev/null; do
  rc=0; wait "$_child" || rc=$?
done
[ "$_requeued" = "1" ] && { echo "=== task ${idx} requeued at walltime; exiting cleanly ==="; exit 0; }
# -----------------------------------------------------------------------------

echo "=== [$([ "$rc" -eq 0 ] && echo ok || echo FAIL rc=$rc)] cross ${SRC}->${TGT} seed${SEED} done at $(date) ==="
exit "$rc"
