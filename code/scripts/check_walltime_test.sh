#!/bin/bash
# Verdict for slurm/capella_walltime_test.slurm: did the walltime trap requeue the task, and did
# the second attempt RESUME rather than restart?
#
#   bash scripts/check_walltime_test.sh
#
# Exit 0 only if every check passes. Run it after both attempts have finished (~15 min).
set -uo pipefail

OUT="${WALLTIME_TEST_OUTPUT_DIR:-${PROJECT_DIR:-.}/results_walltime_test}"
LOG="$OUT/logs/exp_gene2image_c1_s42.log"
fail=0
ok()   { printf '  [ OK ] %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; fail=1; }

echo "=== walltime / requeue / auto-resume proof ==="
echo "log: $LOG"
echo

[ -f "$LOG" ] || { bad "per-run log missing -- did the job run? (checked $LOG)"; exit 1; }

# 1. The trap fired at all.
if grep -q "walltime approaching -- requeueing" "$LOG"; then
  ok "trap fired: SIGUSR1 arrived before the kill and the handler ran"
else
  bad "trap never fired -- --signal=B:USR1@120 did not reach the batch shell, or training finished
         early. Without this the formal array's short walltime would silently lose runs."
fi

# 2. scontrol requeue actually succeeded.
if grep -q "scontrol requeue FAILED" "$LOG"; then
  bad "scontrol requeue FAILED -- the account/QoS may forbid self-requeue. Fall back to a walltime
         that fits the longest run in one shot (--time=3-00:00:00)."
else
  ok "scontrol requeue reported no error"
fi

# 3. There really was a second attempt.
n_attempt=$(grep -c "^=== attempt at " "$LOG")
if [ "$n_attempt" -ge 2 ]; then
  ok "$n_attempt attempts recorded -- SLURM re-ran the task after the requeue"
else
  bad "only $n_attempt attempt in the log -- the task was NOT re-run. A requeue that never comes
         back is worse than a long walltime."
fi

# 4. The heart of it: attempt 2 RESUMED instead of restarting at epoch 1.
if grep -q "Will resume training from epoch" "$LOG"; then
  ep=$(grep -o "Will resume training from epoch [0-9]*" "$LOG" | tail -1 | grep -o '[0-9]*')
  if [ "${ep:-0}" -gt 1 ]; then
    ok "resumed from epoch $ep (not 1) -- latest_checkpoint.pt was picked up"
  else
    bad "resumed from epoch $ep -- that is a restart, not a resume; the checkpoint was not used"
  fi
else
  bad "no 'Will resume training from epoch' line -- attempt 2 started from scratch, so every
         requeue would throw away the previous window's work"
fi

# 5. training_losses.csv survived. A zero-epoch invocation used to overwrite it with a
#    header-only file, which validate_runs.py then reads as a truncated run.
CSV="$OUT/gene2image_c1_seed42/training_losses.csv"
if [ -f "$CSV" ]; then
  rows=$(( $(wc -l < "$CSV") - 1 ))
  if [ "$rows" -gt 0 ]; then
    ok "training_losses.csv has $rows epoch row(s) -- not clobbered by the resume"
  else
    bad "training_losses.csv is header-only -- the resume wiped the training history"
  fi
else
  bad "training_losses.csv missing at $CSV"
fi

# 6. It finished, and the earlier segment's log survived the append.
grep -q "walltime test finished" "$LOG" 2>/dev/null || \
  grep -q "\[ok\] walltime test finished" "$OUT/../logs/g2i_wt-"*.out 2>/dev/null && \
  ok "run reached completion" || echo "  [ -- ] completion marker not found (still running?)"

if grep -q "DOPRI5_DIAGNOSTICS" "$LOG"; then
  ok "DOPRI5 diagnostics present in the appended log (tee -a preserved the segments)"
else
  echo "  [ -- ] no DOPRI5 lines yet (eval may not have run)"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "VERDICT: PASS — short walltime is safe. The formal array can use --time=12:00:00/24:00:00"
  echo "         for better backfill; work is never lost, only paused."
  exit 0
fi
echo "VERDICT: FAIL — do NOT rely on a short walltime. Submit the array with a window that fits the"
echo "         longest run in one shot:  sbatch --time=3-00:00:00 --array=0-53 slurm/capella_array_1gpu.slurm"
exit 1
