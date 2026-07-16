#!/bin/bash
# =============================================================================
# Capella ARRAY 1-GPU — DEFAULT way to run the 54-run main matrix.
# A Slurm job array of 54 single-GPU tasks (one experiment each); %N throttles
# concurrency. 1-GPU tasks backfill into ANY free GPU, so when GPUs are plentiful
# this is the FASTEST + most queue-friendly option: it scales past 4 (no whole-
# node wait that a 4-GPU job needs). Each task = one experiment (train 50ep+eval).
# Measured on H100: c2 ~24h, c1 ~33h, p1 ~44h. Since the 54 runs are independent, wall-clock =
# the SLOWEST single run (~44h) once you have 54 GPUs -- concurrency, not total GPU-hours, sets it.
#
# SUBMIT (build masks once, then the array):
#   PREP=$(sbatch --parsable --export=ALL,MODE=prep --gres=gpu:1 --time=00:30:00 slurm/capella.slurm)
#   sbatch --array=0-53 --dependency=afterok:$PREP slurm/capella_array_1gpu.slurm      # all 54 at once
#   sbatch --array=0-53%8 --dependency=afterok:$PREP slurm/capella_array_1gpu.slurm    # throttle to 8
#   # %N caps concurrency: 54 -> ~44h, 27 -> ~4d, 8 -> ~11d.
#   # --time is a scheduling knob (see below), NOT a bet: the USR1 trap requeues before the kill.
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
# Walltime is a SCHEDULING knob, not a bet on the runtime estimate: the USR1 trap below requeues
# the task before the kill and --auto_resume continues it, so a short request never loses work.
# Measured on H100 (2.60 it/s @ batch 16, from the smoke run): c2 ~24h, c1 ~33h, p1 ~44h per run.
#   24h  (default) : backfills into common gaps; c2 ~1 cycle, c1/p1 ~2 cycles
#   12h            : best backfill on a busy queue; ~2-4 cycles
#   3-00:00:00     : one shot, finishes in ~44h wall -- use when the queue is empty
# Do NOT go below ~4h: evaluation (~2.6h) is not resumable and would restart every cycle.
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900                     # SIGUSR1 to the batch shell 15 min before the kill
#SBATCH --requeue                               # covers preemption/node failure; USR1 covers TIMEOUT
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

# Mirror this run's output to the per-run log validate_runs.py reads:
#   validate_runs.py:124 parses <results_root>/logs/exp_<variant>_<ds>_s<seed>.log
# The SBATCH --output above writes logs/g2i_arr-<jobid>_<taskid>.out instead — a different name
# AND a different directory — so without this tee the gate would find no log and could not verify
# the DOPRI5 solver gates at all (it reports them as "unverified").
RUN_LOG="$OUTPUT_DIR/logs/exp_${VARIANT}_${DS}_s${SEED}.log"
mkdir -p "$OUTPUT_DIR/logs"

# tee -a, NOT tee: a requeued task re-enters this script, and a truncating tee would wipe the
# earlier attempt's log -- including the DOPRI5_DIAGNOSTICS lines and epoch history the gate reads.
echo "=== attempt at $(date) (job ${SLURM_JOB_ID:-?}, task ${idx}) ===" >> "$RUN_LOG"

# ---------------------------------------------------------------------------
# Self-requeue before the walltime kill.
#
# #SBATCH --requeue does NOT cover TIMEOUT -- it only makes the job eligible for requeue after
# preemption or node failure. A job that simply runs out of walltime is killed and stays dead, and
# --auto_resume never gets a chance to continue it. That makes walltime a hard bet on the runtime
# estimate. Instead: --signal=B:USR1@900 delivers SIGUSR1 to this batch shell 15 min before the
# limit, and we requeue ourselves; the next attempt resumes from latest_checkpoint.pt.
#
# With this, --time is purely a SCHEDULING knob -- shorter requests backfill into smaller gaps --
# and completion no longer depends on the estimate being right. Override per submission:
#   sbatch --time=12:00:00 --array=0-53 slurm/capella_array_1gpu.slurm
# Floor: evaluation (~2.6h, single-cell 5000 samples) is NOT resumable, so a window under ~4h risks
# re-doing eval every cycle. 12-24h is the sweet spot; use 3-00:00:00 to finish in one shot when
# the queue is empty.
# ---------------------------------------------------------------------------
_requeued=0
_on_walltime() {
  # Requeue THIS TASK ONLY. In an array, SLURM_JOB_ID is already the individual task's job id
  # (SLURM_ARRAY_JOB_ID is the parent), but spell the target as <arrayjob>_<taskid> anyway: this
  # code is one typo away from requeueing all 54 tasks, and the explicit form cannot be misread.
  local _target="${SLURM_JOB_ID}"
  [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && _target="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  echo "[signal] walltime approaching -- requeueing ${_target} (${VARIANT}_${DS}_seed${SEED});" \
       "next attempt resumes from latest_checkpoint.pt" | tee -a "$RUN_LOG"
  # Set _requeued ONLY if scontrol actually succeeded. Setting it first would make a REJECTED
  # requeue (account/QoS forbids it, job not in a requeueable state) still take the "exit 0, cleanly
  # requeued" path below -- SLURM records success, and the task is silently dropped from the matrix
  # with nothing to re-submit.
  if scontrol requeue "${_target}"; then
    _requeued=1
  else
    echo "[signal] scontrol requeue FAILED -- NOT exiting 0; this task will fail loudly so it can" \
         "be resubmitted. Consider a walltime that fits the run in one shot (--time=3-00:00:00)." \
         | tee -a "$RUN_LOG"
  fi
}
trap _on_walltime USR1

set -o pipefail   # tee must not mask run_experiments.sh's exit status
# Run in the background so the USR1 trap can fire while training is in flight; a foreground
# command would delay the trap until it returned, i.e. until after the kill.
(
  EPOCHS="${EPOCHS:-50}" BATCH_SIZE="${BATCH_SIZE:-16}" EVAL_BATCH="${EVAL_BATCH:-8}" \
  GEN_STEPS="${GEN_STEPS:-100}" WORKERS=14 DB="${DB:-hallmark}" \
  DATA_ROOT="$DATA_DIR" MASK_DIR="$MASK_DIR" OUT_ROOT="$OUTPUT_DIR" \
  EXTRA="${EXTRA:---use_amp --patience 9999 --auto_resume}" \
    bash scripts/run_experiments.sh "${VARIANT}" "${DS}" "${SEED}" 2>&1 | tee -a "$RUN_LOG"
) &
_child=$!
# `wait` returns 128+signum when the USR1 trap interrupts it while the child is still running, so
# re-wait until the child actually exits.
#
# The `|| rc=$?` is load-bearing: _capella_common.sh sets `set -euo pipefail`, so a bare
# `wait "$_child"` returning 138 (128+SIGUSR1) makes -e kill this script on the spot -- the requeue
# bookkeeping and the clean `exit 0` below never run, and SLURM sees a task that died with 138.
# Do NOT write it as `while ! wait "$_child"` either: `!` inverts the status, so $? inside the loop
# is the inverted value rather than the child's exit code, and a genuinely failed run (rc=7) would
# be reported as success.
rc=0; wait "$_child" || rc=$?
while [ "$rc" -gt 128 ] && kill -0 "$_child" 2>/dev/null; do
  rc=0; wait "$_child" || rc=$?
done
[ "$_requeued" = "1" ] && { echo "=== task ${idx} requeued at walltime; exiting cleanly ==="; exit 0; }

echo "=== [$([ "$rc" -eq 0 ] && echo ok || echo FAIL rc=$rc)] array task ${idx} (${VARIANT}_${DS}_seed${SEED}) done at $(date) ==="
exit "$rc"
