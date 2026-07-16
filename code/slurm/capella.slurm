#!/bin/bash
# =============================================================================
# Gene2Image — Capella (TU Dresden ZIH) SPLIT launcher: short 4-GPU "wave" jobs
#
# The matrix is split so each job is SHORT (≈ one experiment's walltime, not the
# whole matrix) — far easier to schedule/backfill than one multi-day job, and a
# preempted job only loses one wave. Each job still uses a full 4-GPU node.
#
# 3 modes (pick with MODE=...):
#   MODE=prep : build image-path remap + pathway masks ONCE (small, ~30 min).
#   MODE=run  : (default) ONE array task = 4 experiments in parallel on 4 GPUs.
#               54 experiments -> 14 waves (array 0-13). Walltime ≈ one experiment.
#   MODE=post : RQ4 interpret (from trained checkpoints) + final summary CSVs.
#
# CAPELLA RESOURCE TABLE (ZIH) -> SBATCH (full node share):
#   4 GPU/node, 56 cores (14/GPU), 13,437 MB/core, 7-day max.
#
# SUBMISSION (with dependencies so it's one command chain):
#   PREP=$(sbatch --parsable -J g2i_prep --export=ALL,MODE=prep --gres=gpu:1 \
#                 --cpus-per-task=14 --time=00:30:00 slurm/capella.slurm)
#   RUN=$(sbatch  --parsable -J g2i --export=ALL,MODE=run --array=0-13%4 \
#                 --dependency=afterok:$PREP slurm/capella.slurm)
#   sbatch -J g2i_post --export=ALL,MODE=post --gres=gpu:1 --cpus-per-task=14 \
#          --time=04:00:00 --dependency=afterok:$RUN slurm/capella.slurm
#   # %4 in --array = up to 4 waves (=16 GPUs) at once; raise/lower to taste.
#   # Calibrate --time from the smoke test (§preflight); default 1 day per wave.
#   # Re-queue only failed waves:  sbatch --array=<idx list> ... slurm/capella.slurm
# =============================================================================

#SBATCH --job-name=g2i
#SBATCH --partition=capella
#SBATCH --account=CHANGEME_capella_account     # <- set your Capella project/account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4                            # full node: 4 GPUs (one experiment per GPU)
#SBATCH --cpus-per-task=56                      # 56 cores (14 per GPU)
#SBATCH --mem-per-cpu=13000M                    # Capella core=13,437MB (per resource table)
#SBATCH --time=1-00:00:00                       # per wave ≈ one experiment; Capella max 7 days
#SBATCH --requeue                               # auto re-queue on preemption (pairs with --auto_resume)
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Site paths / modules / venv  (hard-coded Capella defaults, override via env)
# -----------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/chwu350f-g2i/Gene2Image/code}"
VENV_DIR="${VENV_DIR:-/data/horse/ws/chwu350f-g2i/venv_piptorch}"
RELEASE_MODULE="${RELEASE_MODULE:-release/24.10}"
GCCCORE_MODULE="${GCCCORE_MODULE:-GCCcore/13.2.0}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5}"

DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/processed_data}"
MASK_DIR="${MASK_DIR:-$PROJECT_DIR/data/pathway_masks}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/results}"
# This script does NOT source slurm/_capella_common.sh -- it carries its own site setup -- so the
# UNI2-h pin and fail-fast must be repeated here. An empty default is how the previous 54-run batch
# was invalidated: utils_uni2h caught the missing weights, logged a WARNING, and the evaluator wrote
# overall_uni2h_fid = NaN while the job still exited 0.
export UNI2H_MODEL_PATH="${UNI2H_MODEL_PATH:-$PROJECT_DIR/../../models/UNI2-h}"
export HE2RNA_MODEL_PATH="${HE2RNA_MODEL_PATH:-}"
if [ "${ALLOW_NO_UNI2H:-0}" != "1" ] && [ ! -f "$UNI2H_MODEL_PATH/pytorch_model.bin" ]; then
  echo "[FATAL] UNI2-h weights not found: $UNI2H_MODEL_PATH/pytorch_model.bin"
  echo "        Biological FID would be NaN on every run and the job would still exit 0."
  echo "        Fix: export UNI2H_MODEL_PATH=/path/to/UNI2-h  (dir containing pytorch_model.bin)"
  echo "        Or:  ALLOW_NO_UNI2H=1 to run anyway WITHOUT biological FID."
  exit 104
fi
GMT_HALLMARK="${GMT_HALLMARK:-$PROJECT_DIR/../gmt/msigdb_2023.2_Hs/h.all.v2023.2.Hs.symbols.gmt}"

module --force purge >/dev/null 2>&1 || module purge || true
module load "${RELEASE_MODULE}" "${GCCCORE_MODULE}" "${PYTHON_MODULE}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
mkdir -p logs "${MASK_DIR}" "${OUT_ROOT}"
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# GPU fail-fast self-check
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-?} mode=${MODE:-run} task=${SLURM_ARRAY_TASK_ID:-none} ==="
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi missing"; exit 101; }
nvidia-smi -L || { echo "ERROR: no GPU visible"; exit 102; }
N_GPU="$(python - <<'PY'
import sys, torch
sys.exit(10) if (not torch.cuda.is_available() or torch.cuda.device_count()<1) else print(torch.cuda.device_count())
PY
)" || { echo "ERROR: torch CUDA self-check failed; refusing CPU run"; exit 103; }
echo "Visible GPUs: ${N_GPU}"

# -----------------------------------------------------------------------------
# MODE=prep : build masks once (image-path remap + real/per-seed rand/none).
# -----------------------------------------------------------------------------
if [ "${MODE:-run}" = "prep" ]; then
  [ -f "${GMT_HALLMARK}" ] || { echo "ERROR: Hallmark GMT not at ${GMT_HALLMARK}"; exit 111; }
  # MASKS_ONLY=1 is what makes this a real prep job: run_all.sh stops after PHASE 0/0.5.
  # It replaces the old VARIANTS="" trick, which did NOT stop training — build_train_jobs
  # gates cross-dataset on INCLUDE_CROSS (not VARIANTS), so VARIANTS="" still queued
  # 3 pairs x 3 seeds = 9 full 50-epoch run_cross_dataset.sh trainings. Those blew past
  # this job's 30-min walltime, prep exited non-zero, and `sbatch --dependency=afterok:$PREP`
  # left the whole 54-task array stuck in DependencyNeverSatisfied.
  # VARIANTS is now the real list so PHASE 0.5 asserts every mask the 54-run will load.
  DATA_ROOT="$DATA_ROOT" MASK_DIR="$MASK_DIR" OUT_ROOT="$OUT_ROOT" GMT_HALLMARK="$GMT_HALLMARK" \
  SEEDS="42 43 44" DATASETS="c1 c2 p1" INCLUDE_CROSS=1 INCLUDE_INTERPRET=0 \
  MASKS_ONLY=1 DRY_RUN="${DRY_RUN:-0}" \
    bash scripts/run_all.sh 1
  echo "[ok] prep done"; exit 0
fi

# -----------------------------------------------------------------------------
# MODE=post : RQ4 interpret (needs trained checkpoints) + final summary.
# -----------------------------------------------------------------------------
if [ "${MODE:-run}" = "post" ]; then
  DATA_ROOT="$DATA_ROOT" MASK_DIR="$MASK_DIR" OUT_ROOT="$OUT_ROOT" GMT_HALLMARK="$GMT_HALLMARK" \
  DATASETS="c1 c2 p1" VARIANTS="" INCLUDE_CROSS=0 INCLUDE_INTERPRET=1 \
    bash scripts/run_all.sh 1
  echo "[ok] interpret + summary done"; exit 0
fi

# -----------------------------------------------------------------------------
# MODE=run : one array task = up to 4 experiments (one per GPU), run in parallel.
#   The 54 (variant,dataset,seed) triples are a fixed list; task t = triples[4t..4t+3].
#   Masks must already exist (MODE=prep). Fixed EPOCHS=50, early-stop off (--patience 9999),
#   --auto_resume; best checkpoint still selected on val_mse.
# -----------------------------------------------------------------------------
VARIANTS=(gene2image geneflow randpath pathprior notrans nomask)
DATASETS=(c1 c2 p1)
SEEDS=(42 43 44)
ALL=()
for v in "${VARIANTS[@]}"; do for d in "${DATASETS[@]}"; do for s in "${SEEDS[@]}"; do ALL+=("$v $d $s"); done; done; done   # 54

task="${SLURM_ARRAY_TASK_ID:?MODE=run needs a job array, e.g. --array=0-13}"
per=4                                            # experiments per wave (one per GPU)
start=$(( task * per ))
[ "$N_GPU" -ge 1 ] || { echo "ERROR: N_GPU<1"; exit 104; }

pids=(); gpu=0
for j in $(seq 0 $((per-1))); do
  idx=$(( start + j ))
  [ "$idx" -ge "${#ALL[@]}" ] && break
  g=$(( gpu % N_GPU ))
  read -r v d s <<< "${ALL[$idx]}"
  echo ">>> wave task ${task}: exp ${idx} = ${v} ${d} seed${s} on GPU ${g}"
  ( CUDA_VISIBLE_DEVICES="$g" \
    EPOCHS="${EPOCHS:-50}" BATCH_SIZE="${BATCH_SIZE:-16}" EVAL_BATCH="${EVAL_BATCH:-8}" \
    GEN_STEPS="${GEN_STEPS:-100}" WORKERS="${WORKERS:-14}" DB="${DB:-hallmark}" \
    DATA_ROOT="$DATA_ROOT" MASK_DIR="$MASK_DIR" OUT_ROOT="$OUT_ROOT" \
    EXTRA="${EXTRA:---use_amp --patience 9999 --auto_resume}" \
      bash scripts/run_experiments.sh "$v" "$d" "$s" ) \
    > "logs/${v}_${d}_s${s}.log" 2>&1 &
  pids+=($!); gpu=$(( gpu + 1 ))
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] && echo "=== [ok] wave ${task} done at $(date) ===" || { echo "=== wave ${task} had failures (see logs/) ==="; exit 1; }
