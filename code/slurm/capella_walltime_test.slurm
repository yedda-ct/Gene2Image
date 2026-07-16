#!/bin/bash
# =============================================================================
# Walltime / requeue / auto-resume PROOF — the one path a normal smoke test cannot reach.
#
# WHY THIS EXISTS
#   #SBATCH --requeue does NOT cover TIMEOUT: it only makes a job eligible for requeue after
#   preemption or node failure. A job that runs out of walltime is killed and stays dead. The
#   formal array therefore relies on --signal=B:USR1@900 + a trap that calls `scontrol requeue`
#   itself. That path is what lets you request a SHORT walltime (good backfill) without betting
#   the run on a runtime estimate — but the smoke test runs 3 epochs in minutes and never gets
#   anywhere near a walltime, so it proves nothing about it.
#
#   This job is sized to be KILLED by walltime on purpose: 100 epochs of a 256-cell subset cannot
#   finish in 8 minutes, so the trap MUST fire, the task MUST requeue, and the second attempt MUST
#   resume from latest_checkpoint.pt instead of restarting at epoch 1.
#
#   The trap block below is copied VERBATIM from capella_array_1gpu.slurm. If you change one,
#   change both — a test of different code proves nothing about the code you will actually run.
#
# RUN
#   sbatch slurm/capella_walltime_test.slurm
#   # then, once BOTH attempts have finished (~15 min total):
#   bash scripts/check_walltime_test.sh
#
# Writes to results_walltime_test/ — never touches the formal results/.
# =============================================================================
#SBATCH --job-name=g2i_wt
#SBATCH --partition=capella
#SBATCH --account=CHANGEME_capella_account     # <- set your Capella project/account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=00:08:00                        # deliberately too short: forces the trap to fire
#SBATCH --signal=B:USR1@120                    # SIGUSR1 to the batch shell 2 min before the kill
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/chwu350f-g2i/Gene2Image/code}"
export OUTPUT_DIR="${WALLTIME_TEST_OUTPUT_DIR:-$PROJECT_DIR/results_walltime_test}"
source "$PROJECT_DIR/slurm/_capella_common.sh"

VARIANT=gene2image; DS=c1; SEED=42
idx=0
ensure_masks_for "$DS"

RUN_LOG="$OUTPUT_DIR/logs/exp_${VARIANT}_${DS}_s${SEED}.log"
mkdir -p "$OUTPUT_DIR/logs"
echo "=== attempt at $(date) (job ${SLURM_JOB_ID:-?}, restarts=${SLURM_RESTART_COUNT:-0}) ===" >> "$RUN_LOG"
echo ">>> WALLTIME TEST: 100 epochs on a 256-cell subset in an 8-min window."
echo ">>> Attempt #$(( ${SLURM_RESTART_COUNT:-0} + 1 )). Expect: attempt 1 requeues, attempt 2 resumes."

# --- trap block: keep identical to capella_array_1gpu.slurm ------------------
_requeued=0
_on_walltime() {
  _requeued=1
  # Requeue THIS TASK ONLY. In an array, SLURM_JOB_ID is already the individual task's job id
  # (SLURM_ARRAY_JOB_ID is the parent), but spell the target as <arrayjob>_<taskid> anyway: this
  # code is one typo away from requeueing all 54 tasks, and the explicit form cannot be misread.
  local _target="${SLURM_JOB_ID}"
  [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && _target="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  echo "[signal] walltime approaching -- requeueing ${_target} (${VARIANT}_${DS}_seed${SEED});" \
       "next attempt resumes from latest_checkpoint.pt" | tee -a "$RUN_LOG"
  scontrol requeue "${_target}" || echo "[signal] scontrol requeue FAILED; job will be lost"
}
trap _on_walltime USR1

set -o pipefail
(
  EPOCHS=100 BATCH_SIZE=16 EVAL_BATCH=8 GEN_STEPS=50 WORKERS=14 DB="${DB:-hallmark}" \
  DATA_ROOT="$DATA_DIR" MASK_DIR="$MASK_DIR" OUT_ROOT="$OUTPUT_DIR" \
  EXTRA="--use_amp --patience 9999 --auto_resume --debug --debug_samples 256" \
  EVAL_EXTRA="--debug --debug_samples 256" \
    bash scripts/run_experiments.sh "${VARIANT}" "${DS}" "${SEED}" 2>&1 | tee -a "$RUN_LOG"
) &
_child=$!
rc=0; wait "$_child" || rc=$?
while [ "$rc" -gt 128 ] && kill -0 "$_child" 2>/dev/null; do
  rc=0; wait "$_child" || rc=$?
done
[ "$_requeued" = "1" ] && { echo "=== requeued at walltime; exiting cleanly ==="; exit 0; }
# ---------------------------------------------------------------------------

echo "=== [$([ "$rc" -eq 0 ] && echo ok || echo FAIL rc=$rc)] walltime test finished at $(date) ==="
echo ">>> Now run:  bash scripts/check_walltime_test.sh"
exit "$rc"
