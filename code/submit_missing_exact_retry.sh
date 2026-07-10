#!/usr/bin/env bash
# =============================================================================
# Gene2Image — retry exactly missing Capella tasks, one task per Slurm job
#
# Missing tasks from your check:
#   exp_nomask_c1_s43
#   exp_nomask_c1_s44
#   exp_nomask_c2_s44
#   cross_c1_to_c2_s44
#   cross_c1_to_p1_s42
#   cross_c1_to_p1_s43
#   cross_c1_to_p1_s44
#
# Usage:
#   cd /data/horse/ws/chwu350f-g2i/Gene2Image/code
#   mkdir -p logs results/logs
#   bash submit_missing_exact_retry.sh
#
# Design:
#   - Each missing run is submitted as a separate 1-GPU job.
#   - Each job checks evaluation_summary.json first; if already complete, it exits 0.
#   - B8 is submitted only after all seven precise retry jobs finish successfully.
#   - Uses --auto_resume to continue interrupted checkpoints when possible.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# User/environment defaults. Override from shell if needed:
#   PROJECT_DIR=/... VENV_DIR=/... bash submit_missing_exact_retry.sh
# -----------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-/data/horse/ws/chwu350f-g2i/venv_piptorch}"

RELEASE_MODULE="${RELEASE_MODULE:-release/24.10}"
GCCCORE_MODULE="${GCCCORE_MODULE:-GCCcore/13.2.0}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5}"

DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/processed_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results}"
MASK_DIR="${MASK_DIR:-$PROJECT_DIR/data/pathway_masks}"

# Keep training setting consistent with previous full experiments.
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# Lower eval batch and workers to reduce memory pressure after previous ExitCode=0:9.
EVAL_BATCH="${EVAL_BATCH:-2}"
GEN_STEPS="${GEN_STEPS:-100}"
WORKERS="${WORKERS:-2}"

# Keep AMP and enable checkpoint resume for interrupted training.
EXTRA="${EXTRA:---use_amp --auto_resume}"

PARTITION="${PARTITION:-capella}"
ACCOUNT="${ACCOUNT:-swtest}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"

# Request one GPU but full node memory to reduce SIGKILL/OOM risk.
CPUS_PER_TASK="${CPUS_PER_TASK:-14}"
MEMORY="${MEMORY:-188118M}"

cd "$PROJECT_DIR"
mkdir -p logs results/logs

COMMON_EXPORT="ALL,PROJECT_DIR=$PROJECT_DIR,VENV_DIR=$VENV_DIR,RELEASE_MODULE=$RELEASE_MODULE,GCCCORE_MODULE=$GCCCORE_MODULE,PYTHON_MODULE=$PYTHON_MODULE,DATA_DIR=$DATA_DIR,OUTPUT_DIR=$OUTPUT_DIR,MASK_DIR=$MASK_DIR,EPOCHS=$EPOCHS,BATCH_SIZE=$BATCH_SIZE,EVAL_BATCH=$EVAL_BATCH,GEN_STEPS=$GEN_STEPS,WORKERS=$WORKERS,EXTRA=$EXTRA"

echo "============================================================"
echo "Submitting exact missing Gene2Image retry jobs"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "VENV_DIR=$VENV_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MASK_DIR=$MASK_DIR"
echo "BATCH_SIZE=$BATCH_SIZE EVAL_BATCH=$EVAL_BATCH WORKERS=$WORKERS"
echo "EXTRA=$EXTRA"
echo "============================================================"

submit_exp() {
  local variant="$1"
  local ds="$2"
  local seed="$3"
  local name="g2i_${variant}_${ds}_s${seed}"
  local result="$OUTPUT_DIR/${variant}_${ds}_seed${seed}/evaluation_summary.json"

  if [ -s "$result" ]; then
    echo "SKIP submit: $name already complete: $result"
    return 0
  fi

  sbatch --parsable \
    -J "$name" \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="logs/%x-%j.out" \
    --error="logs/%x-%j.err" \
    --export="$COMMON_EXPORT,VARIANT=$variant,DS=$ds,SEED=$seed" \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

module --force purge >/dev/null 2>&1 || module purge || true
module load "$RELEASE_MODULE" "$GCCCORE_MODULE" "$PYTHON_MODULE"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR/logs"

TASK_NAME="exp_${VARIANT}_${DS}_s${SEED}"
TASK_LOG="$OUTPUT_DIR/logs/${TASK_NAME}.log"
RESULT="$OUTPUT_DIR/${VARIANT}_${DS}_seed${SEED}/evaluation_summary.json"

exec > >(tee -a "$TASK_LOG") 2>&1

echo "============================================================"
echo "START $TASK_NAME"
echo "date=$(date)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RESULT=$RESULT"
echo "============================================================"

if [ -s "$RESULT" ]; then
  echo "SKIP completed: $RESULT"
  exit 0
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python - <<'PY'
import sys, torch
print("torch_version=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("ERROR: no CUDA device visible", file=sys.stderr)
    sys.exit(10)
print("gpu0=", torch.cuda.get_device_name(0))
PY

export PYTHONHASHSEED=0
export DATA_ROOT="$DATA_DIR"
export MASK_DIR="$MASK_DIR"
export OUT_ROOT="$OUTPUT_DIR"
export EPOCHS="$EPOCHS"
export BATCH_SIZE="$BATCH_SIZE"
export EVAL_BATCH="$EVAL_BATCH"
export GEN_STEPS="$GEN_STEPS"
export WORKERS="$WORKERS"
export EXTRA="$EXTRA"
export EVAL=1

bash scripts/run_experiments.sh "$VARIANT" "$DS" "$SEED"

if [ ! -s "$RESULT" ]; then
  echo "ERROR: task ended but result summary is missing or empty: $RESULT" >&2
  exit 20
fi

echo "DONE $TASK_NAME"
SBATCH
}

submit_cross() {
  local src="$1"
  local tgt="$2"
  local seed="$3"
  local name="g2i_cross_${src}_to_${tgt}_s${seed}"
  local base="$OUTPUT_DIR/cross_dataset/${src}_to_${tgt}_seed${seed}"
  local result_tgt="$base/eval_on_${tgt}/evaluation_summary.json"
  local result_src="$base/eval_on_${src}/evaluation_summary.json"

  if [ -s "$result_tgt" ] && [ -s "$result_src" ]; then
    echo "SKIP submit: $name already complete:"
    echo "  $result_tgt"
    echo "  $result_src"
    return 0
  fi

  sbatch --parsable \
    -J "$name" \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="logs/%x-%j.out" \
    --error="logs/%x-%j.err" \
    --export="$COMMON_EXPORT,SRC=$src,TGT=$tgt,SEED=$seed" \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

module --force purge >/dev/null 2>&1 || module purge || true
module load "$RELEASE_MODULE" "$GCCCORE_MODULE" "$PYTHON_MODULE"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR/logs"

TASK_NAME="cross_${SRC}_to_${TGT}_s${SEED}"
TASK_LOG="$OUTPUT_DIR/logs/${TASK_NAME}.log"
BASE="$OUTPUT_DIR/cross_dataset/${SRC}_to_${TGT}_seed${SEED}"
RESULT_TGT="$BASE/eval_on_${TGT}/evaluation_summary.json"
RESULT_SRC="$BASE/eval_on_${SRC}/evaluation_summary.json"

exec > >(tee -a "$TASK_LOG") 2>&1

echo "============================================================"
echo "START $TASK_NAME"
echo "date=$(date)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RESULT_TGT=$RESULT_TGT"
echo "RESULT_SRC=$RESULT_SRC"
echo "============================================================"

if [ -s "$RESULT_TGT" ] && [ -s "$RESULT_SRC" ]; then
  echo "SKIP completed:"
  echo "  $RESULT_TGT"
  echo "  $RESULT_SRC"
  exit 0
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python - <<'PY'
import sys, torch
print("torch_version=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("ERROR: no CUDA device visible", file=sys.stderr)
    sys.exit(10)
print("gpu0=", torch.cuda.get_device_name(0))
PY

export PYTHONHASHSEED=0
export DATA_ROOT="$DATA_DIR"
export MASK_DIR="$MASK_DIR"
export OUT_ROOT="$OUTPUT_DIR/cross_dataset"
export EPOCHS="$EPOCHS"
export BATCH_SIZE="$BATCH_SIZE"
export EVAL_BATCH="$EVAL_BATCH"
export GEN_STEPS="$GEN_STEPS"
export WORKERS="$WORKERS"
export EXTRA="$EXTRA"
export EVAL=1

bash scripts/run_cross_dataset.sh "$SRC" "$TGT" "$SEED"

if [ ! -s "$RESULT_TGT" ] || [ ! -s "$RESULT_SRC" ]; then
  echo "ERROR: cross task ended but one or both result summaries are missing:" >&2
  echo "  $RESULT_TGT" >&2
  echo "  $RESULT_SRC" >&2
  exit 20
fi

echo "DONE $TASK_NAME"
SBATCH
}

jobids=()

maybe_add_jobid() {
  local jid="$1"
  if [ -n "$jid" ]; then
    jobids+=("$jid")
    echo "submitted jobid=$jid"
  fi
}

# -----------------------------------------------------------------------------
# Exact missing tasks
# -----------------------------------------------------------------------------
maybe_add_jobid "$(submit_exp nomask c1 43)"
maybe_add_jobid "$(submit_exp nomask c1 44)"
maybe_add_jobid "$(submit_exp nomask c2 44)"

maybe_add_jobid "$(submit_cross c1 c2 44)"
maybe_add_jobid "$(submit_cross c1 p1 42)"
maybe_add_jobid "$(submit_cross c1 p1 43)"
maybe_add_jobid "$(submit_cross c1 p1 44)"

echo "============================================================"
echo "Retry jobs submitted:"
printf '  %s\n' "${jobids[@]:-<none>}"
echo "============================================================"

if [ "${#jobids[@]}" -eq 0 ]; then
  echo "No retry jobs were needed. Submitting B8 immediately."
  sbatch -J g2i_b8_manual \
    --export="$COMMON_EXPORT,BATCH=8" \
    slurm/capella.slurm
else
  dep="$(IFS=:; echo "${jobids[*]}")"
  echo "Submitting B8 with dependency afterok:$dep"
  sbatch -J g2i_b8_after_exact_retry \
    --dependency="afterok:$dep" \
    --export="$COMMON_EXPORT,BATCH=8" \
    slurm/capella.slurm
fi

echo "Use:"
echo "  squeue -u \$USER"
echo "  sacct -j $(IFS=,; echo "${jobids[*]:-}") --format=JobID,JobName%40,State,ExitCode,Elapsed,MaxRSS -X"
