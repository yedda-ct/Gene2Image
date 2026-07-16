#!/bin/bash
# =============================================================================
# Gene2Image — single master runner for EVERY experiment in the paper.
#
# Covers all four experiment types (docs/idea_report.md Part 3):
#   2.1 Main          : Gene2Image vs GeneFlow, 3 datasets x 3 seeds
#   2.2 Ablation      : randPath / PathPrior / noTrans / noMask (+ optional Reactome)
#   2.3 Cross-dataset : c1->c2, c2->c1, c1->p1 transfer (degradation rate)
#   2.4 RQ4 interpret : CLS->pathway attention, GSEA consistency, intervention
# plus prerequisites (image-path remap, pathway masks, cross masks) and the final
# results aggregation. Each "run" = train + evaluate, so every run drops an
# evaluation_summary.json into its own results dir.
#
# GPU SCHEDULING
#   Pass the max number of concurrent jobs as the first argument. One job runs on
#   one GPU (pinned via CUDA_VISIBLE_DEVICES); when a job finishes the next in the
#   queue starts on the freed GPU.
#       bash scripts/run_all.sh <MAX_PARALLEL>
#   e.g. a 4-GPU box:   bash scripts/run_all.sh 4
#        a single GPU:  bash scripts/run_all.sh 1
#   GPU ids default to 0..MAX_PARALLEL-1; override with GPUS="0 2 3" (count wins).
#
# DRY RUN (print the full plan + write the catalog, run nothing):
#       DRY_RUN=1 bash scripts/run_all.sh 4
#
# COMMON KNOBS (env vars, with defaults):
#   DATASETS="c1 c2 p1"  VARIANTS="gene2image geneflow randpath pathprior notrans nomask"
#   SEEDS="42 43 44"     EPOCHS=50   BATCH_SIZE=16  EVAL_BATCH=8  GEN_STEPS=100  WORKERS=4
#   INCLUDE_CROSS=1  INCLUDE_INTERPRET=1  INCLUDE_REACTOME=0  INTERPRET_SEED=42
#   DATA_ROOT=data/processed_data  MASK_DIR=data/pathway_masks  OUT_ROOT=results
#   EXTRA="--use_amp --patience 9999"   PY=python      # 9999 = early stopping off (equal budget)
#   OFFLINE pathway masks (no Enrichr/network): point these at local .gmt files
#     GMT_HALLMARK=/path/h.all.v2023.2.Hs.symbols.gmt
#     GMT_REACTOME=/path/c2.cp.reactome.v2023.2.Hs.symbols.gmt   (only for INCLUDE_REACTOME=1)
#   If masks can't be built (no network and no GMT_*), PHASE 0.5 aborts before
#   any training, instead of letting the gene2image arm crash silently.
#
# OUTPUTS: everything lands under $OUT_ROOT (default results/). See the generated
#   $OUT_ROOT/EXPERIMENTS_CATALOG.md for the experiment -> file mapping. The summary
#   CSVs (summary_main.csv, ablation/summary.csv, cross_dataset/summary.csv) are the
#   headline deliverables; per-run dirs keep checkpoints, training_losses.csv,
#   evaluation_summary.json, gene_importance_scores.csv.
# =============================================================================

# Run from the code/ root regardless of where this is invoked.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.." || exit 1

# Pin Python's hash seed so set/dict iteration order is identical across the
# separate train / evaluate / generate processes. The dataset derives its cell
# ordering from a set intersection; an unpinned hash seed would make the 80/20
# random_split land on different cells per process -> evaluation-set leakage.
# (dataset.py also sorts cell_ids; this is defence in depth.)
export PYTHONHASHSEED=0

MAX_PARALLEL=${1:-10}   # default: up to 10 concurrent jobs (one per GPU)

PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-data/processed_data}
MASK_DIR=${MASK_DIR:-data/pathway_masks}
OUT_ROOT=${OUT_ROOT:-results}
DB=${DB:-hallmark}
# Offline pathway library: point these at local .gmt files to build masks without
# network access (Enrichr). Leave empty to fetch from Enrichr via gseapy.
#   GMT_HALLMARK=/path/h.all.*.gmt  GMT_REACTOME=/path/c2.cp.reactome.*.gmt
GMT_HALLMARK=${GMT_HALLMARK:-}
GMT_REACTOME=${GMT_REACTOME:-}
EPOCHS=${EPOCHS:-50}          # 对齐 GeneFlow 源代码 train.sh(EPOCHS=50)
BATCH_SIZE=${BATCH_SIZE:-16}
EVAL_BATCH=${EVAL_BATCH:-8}
GEN_STEPS=${GEN_STEPS:-100}
WORKERS=${WORKERS:-4}
# --patience 9999 = early stopping OFF, which the cross-variant comparison REQUIRES. GeneFlow's
# train.sh sets PATIENCE=5 and we follow it on the other four hyper-parameters (batch 16 /
# 50 epochs / lr 1e-4 / wd 0.01), but with patience=5 each arm stops at a different epoch
# (previous batch: gene2image ~26.6 vs geneflow ~14.9), confounding the encoder comparison with
# training budget -- one of the four reasons that batch was binned. Best checkpoint is still the
# val_mse minimum, so a full budget only widens the search for every arm, including the baseline.
EXTRA=${EXTRA:-"--use_amp --patience 9999"}
EVAL_EXTRA=${EVAL_EXTRA:-}   # eval has no --use_amp; keep eval extras separate from train

DATASETS=${DATASETS:-"c1 c2 p1"}
# Note: ${VARIANTS-default} (no colon) so an explicit empty VARIANTS="" is honoured
# (used to run ONLY cross-dataset / interpret without retraining main+ablation).
# Unset -> default 6 variants; empty string -> no main/ablation train jobs.
VARIANTS=${VARIANTS-"gene2image geneflow randpath pathprior notrans nomask"}
SEEDS=${SEEDS:-"42 43 44"}
INCLUDE_CROSS=${INCLUDE_CROSS:-1}
INCLUDE_INTERPRET=${INCLUDE_INTERPRET:-1}
INCLUDE_REACTOME=${INCLUDE_REACTOME:-0}
INTERPRET_SEED=${INTERPRET_SEED:-42}
DRY_RUN=${DRY_RUN:-0}
# MASKS_ONLY=1 -> build + verify pathway masks, then exit BEFORE any train/eval job is
# queued. This is what a "prep" job wants. Do NOT emulate it with VARIANTS="": the
# cross-dataset block in build_train_jobs is gated on INCLUDE_CROSS, NOT on VARIANTS, so
# an empty VARIANTS still queues 3 CROSS_PAIRS x 3 SEEDS = 9 full 50-epoch trainings
# (run_cross_dataset.sh trains the source panel). On a 30-min prep job those get SIGKILLed,
# prep exits non-zero, and an array submitted with --dependency=afterok never launches.
MASKS_ONLY=${MASKS_ONLY:-0}

# RQ4 interpret runs on a SINGLE seed's checkpoint (INTERPRET_SEED). If that seed
# is not in SEEDS, its checkpoint is never trained and interpret would silently
# produce nothing; fall back to the first trained seed instead.
if [ -n "${SEEDS// /}" ]; then
  case " $SEEDS " in
    *" $INTERPRET_SEED "*) ;;
    *) read -r _first_seed _ <<< "$SEEDS"   # whitespace-robust first token (leading spaces safe)
       echo "  NOTE: INTERPRET_SEED=$INTERPRET_SEED not in SEEDS='$SEEDS'; using seed $_first_seed for RQ4 interpret."
       INTERPRET_SEED=$_first_seed ;;
  esac
fi

# GPU pool: explicit GPUS env wins; else 0..MAX_PARALLEL-1.
if [ -n "${GPUS// /}" ]; then            # non-empty after stripping whitespace
  read -r -a GPU_IDS <<< "$GPUS"
else
  case "$MAX_PARALLEL" in
    ''|*[!0-9]*) echo "ERROR: MAX_PARALLEL must be a positive integer (got '$MAX_PARALLEL')." >&2; exit 1 ;;
  esac
  GPU_IDS=(); for i in $(seq 0 $((MAX_PARALLEL-1))); do GPU_IDS+=("$i"); done
fi
if [ ${#GPU_IDS[@]} -eq 0 ]; then
  echo "ERROR: no GPU slots resolved (MAX_PARALLEL='$MAX_PARALLEL', GPUS='$GPUS'). Pass a positive integer, e.g. 'bash scripts/run_all.sh 10'." >&2
  exit 1
fi

LOGDIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$MASK_DIR" "$LOGDIR"

# Export so the child scripts (run_experiments.sh / run_cross_dataset.sh) inherit.
export PY DATA_ROOT MASK_DIR OUT_ROOT DB EPOCHS BATCH_SIZE EVAL_BATCH GEN_STEPS WORKERS EXTRA EVAL_EXTRA EVAL=1

dataset_dir() {
  case "$1" in
    c1) echo "Xenium_V1_hSkin_Melanoma_Base_FFPE" ;;
    c2) echo "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE" ;;
    p1) echo "Xenium_Prime_Human_Skin_FFPE" ;;
    *)  echo "UNKNOWN" ;;
  esac
}
adata_of()   { echo "$DATA_ROOT/$(dataset_dir "$1")/adata.h5ad"; }
imgpaths_of(){ echo "$DATA_ROOT/$(dataset_dir "$1")/cell_patch_256_aux/input/cell_image_paths_local.json"; }
rawpaths_of(){ echo "$DATA_ROOT/$(dataset_dir "$1")/cell_patch_256_aux/input/cell_image_paths.json"; }

runcmd() {  # run (or echo in dry-run) a prerequisite command
  if [ "$DRY_RUN" = "1" ]; then echo "  [prep] $*"; else echo "  [prep] $*"; eval "$@"; fi
}

# ---------------------------------------------------------------------------
# Prerequisites (CPU, run ONCE up front and sequentially so parallel GPU jobs
# never race to build the same mask). The child scripts also self-heal, but doing
# it here avoids concurrent writes to the same .npz.
# ---------------------------------------------------------------------------
# Build the "--gmt <file...>" fragment for a given db, or empty for online mode.
gmt_args() {  # $1 = db (hallmark | hallmark_reactome)
  case "$1" in
    hallmark)
      [ -n "$GMT_HALLMARK" ] && echo "--gmt \"$GMT_HALLMARK\"" ;;
    hallmark_reactome)
      if [ -n "$GMT_HALLMARK" ] && [ -n "$GMT_REACTOME" ]; then
        echo "--gmt \"$GMT_HALLMARK\" \"$GMT_REACTOME\""
      elif [ -n "$GMT_HALLMARK" ] || [ -n "$GMT_REACTOME" ]; then
        echo "ERROR: hallmark_reactome offline needs BOTH GMT_HALLMARK and GMT_REACTOME" >&2
        return 1
      fi ;;
  esac
  return 0
}

prep_dataset() {
  local ds=$1 adata img raw gmt
  adata=$(adata_of "$ds"); img=$(imgpaths_of "$ds"); raw=$(rawpaths_of "$ds")
  [ ! -f "$img" ] && runcmd "$PY scripts/fix_image_paths.py --json \"$raw\" --local_root \"$DATA_ROOT\""
  # Rebuild unless ALL three variants (real/rand/none) are present — a partial build
  # (interrupted, or rand/none deleted) would otherwise crash randpath/nomask jobs.
  local need=""
  for v in real rand none; do [ ! -f "$MASK_DIR/${ds}_hallmark_${v}.npz" ] && need=1; done
  # randPath uses a per-seed random mask (so its multi-seed std reflects mask-draw
  # variance, not just optimization noise); rebuild if any per-seed rand mask is missing.
  for s in $SEEDS; do [ ! -f "$MASK_DIR/${ds}_hallmark_rand_s${s}.npz" ] && need=1; done
  gmt=$(gmt_args hallmark) || return 1
  [ -n "$need" ] && \
    runcmd "$PY scripts/build_pathway_mask.py --adata \"$adata\" --prefix $ds --db hallmark $gmt --out_dir \"$MASK_DIR\" --seed 42 --rand_seeds $SEEDS"
  return 0
}
build_cross_masks() {
  local src=$1 tgt=$2
  [ ! -f "$MASK_DIR/${src}_to_${tgt}_src.npz" ] && \
    runcmd "$PY scripts/build_cross_masks.py --src \"$MASK_DIR/${src}_hallmark_real.npz\" --tgt \"$MASK_DIR/${tgt}_hallmark_real.npz\" --src_name $src --tgt_name $tgt --out_dir \"$MASK_DIR\""
  return 0
}

CROSS_PAIRS=("c1 c2" "c2 c1" "c1 p1")

run_prereqs() {
  echo "=== PHASE 0: prerequisites (image paths + pathway masks) ==="
  # Datasets needed = requested datasets + cross endpoints.
  local needed="$DATASETS"
  [ "$INCLUDE_CROSS" = "1" ] && needed="$needed c1 c2 p1"
  needed=$(echo "$needed" | tr ' ' '\n' | sort -u | tr '\n' ' ')
  for ds in $needed; do prep_dataset "$ds"; done
  if [ "$INCLUDE_REACTOME" = "1" ] && [ ! -f "$MASK_DIR/p1_hallmark_reactome_real.npz" ]; then
    local gmt_hr; gmt_hr=$(gmt_args hallmark_reactome) || return 1
    runcmd "$PY scripts/build_pathway_mask.py --adata \"$(adata_of p1)\" --prefix p1 --db hallmark_reactome $gmt_hr --out_dir \"$MASK_DIR\" --seed 42"
  fi
  if [ "$INCLUDE_CROSS" = "1" ]; then
    for pair in "${CROSS_PAIRS[@]}"; do build_cross_masks $pair; done
  fi
}

# ---------------------------------------------------------------------------
# Fail-fast mask verification (Blocker #2). PHASE 0 builds masks with `runcmd`,
# which does NOT check exit codes — a failed build (e.g. no network for Enrichr
# and no --gmt) would otherwise let PHASE 1 start, the geneflow arm (no mask)
# run fine, and every gene2image/ablation arm crash at np.load -> half a
# comparison with no up-front signal. This asserts every mask a planned job will
# load actually exists, and aborts the whole run with a clear message if not.
# ---------------------------------------------------------------------------
mask_suffix_for_variant() {  # echo the mask variant (real|rand|none) a variant needs, or empty
  case "$1" in
    gene2image|pathprior|notrans) echo real ;;
    randpath) echo rand ;;
    nomask)   echo none ;;
    *) echo "" ;;   # geneflow and unknowns need no Hallmark mask
  esac
}
verify_masks() {
  [ "$DRY_RUN" = "1" ] && { echo "=== (dry-run) skipping mask verification ==="; return 0; }
  echo "=== PHASE 0.5: verifying pathway masks exist (fail-fast) ==="
  local missing=()
  # Main + ablation variants x datasets.
  for v in $VARIANTS; do
    local suf; suf=$(mask_suffix_for_variant "$v")
    [ -z "$suf" ] && continue
    for ds in $DATASETS; do
      # Resolve via $DB (default hallmark) to match run_experiments.sh's mask path
      # ${ds}_${DB}_${suf}.npz; hardcoding 'hallmark' would pass fail-fast then crash at
      # load under a global DB override.
      local f="$MASK_DIR/${ds}_${DB}_${suf}.npz"
      [ ! -f "$f" ] && missing+=("$f  (needed by variant '$v' on '$ds')")
      # randpath consumes a PER-SEED rand mask (${ds}_${DB}_rand_s<seed>.npz); assert each so
      # it doesn't silently fall back to the shared mask and collapse RQ2's 3-seed variance.
      if [ "$suf" = "rand" ]; then
        for s in $SEEDS; do
          local fp="$MASK_DIR/${ds}_${DB}_rand_s${s}.npz"
          [ ! -f "$fp" ] && missing+=("$fp  (per-seed rand mask for 'randpath' on '$ds' seed $s)")
        done
      fi
    done
  done
  # Optional Reactome granularity ablation on P1 (real mask only).
  if [ "$INCLUDE_REACTOME" = "1" ]; then
    local f="$MASK_DIR/p1_hallmark_reactome_real.npz"
    [ ! -f "$f" ] && missing+=("$f  (needed by Reactome ablation)")
  fi
  # Cross-dataset transfer masks.
  if [ "$INCLUDE_CROSS" = "1" ]; then
    for pair in "${CROSS_PAIRS[@]}"; do
      set -- $pair; local src=$1 tgt=$2
      for end in src tgt; do
        local f="$MASK_DIR/${src}_to_${tgt}_${end}.npz"
        [ ! -f "$f" ] && missing+=("$f  (needed by cross ${src}->${tgt})")
      done
    done
  fi
  if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: ${#missing[@]} required pathway mask file(s) are missing:" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    echo "" >&2
    echo "PHASE 0 mask building failed (likely no network for Enrichr). Build masks" >&2
    echo "offline by setting GMT_HALLMARK (and GMT_REACTOME for Reactome) to local" >&2
    echo ".gmt files, or pre-generate the .npz on a networked machine and copy them" >&2
    echo "into $MASK_DIR. Aborting before PHASE 1 to avoid a half-finished comparison." >&2
    exit 1
  fi
  echo "  OK: all required pathway masks present."
}

# ---------------------------------------------------------------------------
# GPU job scheduler: run specs ("<name>\t<command>") with at most one per GPU,
# starting the next as soon as a GPU frees up.
# ---------------------------------------------------------------------------
run_queue() {
  local -a specs=("$@")
  [ ${#specs[@]} -eq 0 ] && { echo ">>> queue empty, nothing to do"; return 0; }
  local -a free=("${GPU_IDS[@]}")
  [ ${#free[@]} -eq 0 ] && { echo ">>> ERROR: no GPU slots; aborting queue." >&2; return 1; }
  declare -A pidgpu pidname
  local idx=0 total=${#specs[@]} n_fail=0
  echo ">>> queue: $total job(s) over ${#free[@]} GPU slot(s): ${GPU_IDS[*]}"
  while [ $idx -lt $total ] || [ ${#pidgpu[@]} -gt 0 ]; do
    while [ ${#free[@]} -gt 0 ] && [ $idx -lt $total ]; do
      local gpu=${free[0]}; free=("${free[@]:1}")
      local name=${specs[$idx]%%$'\t'*}; local cmd=${specs[$idx]#*$'\t'}; idx=$((idx+1))
      if [ "$DRY_RUN" = "1" ]; then
        echo "  [plan] gpu=$gpu  $name"
        echo "         $cmd"
        free+=("$gpu"); continue
      fi
      echo "[$(date +%H:%M:%S)] START gpu=$gpu  $name  (log: $LOGDIR/$name.log)"
      ( CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" ) >"$LOGDIR/$name.log" 2>&1 &
      pidgpu[$!]=$gpu; pidname[$!]=$name
    done
    [ ${#pidgpu[@]} -eq 0 ] && continue
    wait -n 2>/dev/null || true
    for pid in "${!pidgpu[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid"; local rc=$?
        echo "[$(date +%H:%M:%S)] DONE  gpu=${pidgpu[$pid]} rc=$rc  ${pidname[$pid]}"
        [ "$rc" -ne 0 ] && { n_fail=$((n_fail+1)); echo "  >>> FAILED (rc=$rc): ${pidname[$pid]} (log: $LOGDIR/${pidname[$pid]}.log)" >&2; }
        free+=("${pidgpu[$pid]}"); unset 'pidgpu[$pid]' 'pidname[$pid]'
      fi
    done
  done
  if [ "$n_fail" -gt 0 ]; then
    echo ">>> queue: $n_fail/$total job(s) FAILED (rc!=0). See per-job logs in $LOGDIR." >&2
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Build job lists
# ---------------------------------------------------------------------------
build_train_jobs() {
  TRAIN_JOBS=()
  # 2.1 main + 2.2 ablation : variant x dataset x seed (train + eval each).
  for v in $VARIANTS; do
    for ds in $DATASETS; do
      for s in $SEEDS; do
        TRAIN_JOBS+=("exp_${v}_${ds}_s${s}"$'\t'"bash scripts/run_experiments.sh $v $ds $s")
      done
    done
  done
  # 2.2 optional additional ablation: Hallmark+Reactome granularity on P1.
  if [ "$INCLUDE_REACTOME" = "1" ]; then
    for s in $SEEDS; do
      TRAIN_JOBS+=("exp_gene2imageReactome_p1_s${s}"$'\t'"bash scripts/run_experiments.sh gene2imageReactome p1 $s")
    done
  fi
  # 2.3 cross-dataset transfer (train source + eval target & source).
  if [ "$INCLUDE_CROSS" = "1" ]; then
    for pair in "${CROSS_PAIRS[@]}"; do
      set -- $pair; local src=$1 tgt=$2
      for s in $SEEDS; do
        # Put cross products under results/cross_dataset/ to match EXPERIMENTS_CATALOG.md
        # (run_cross_dataset.sh would otherwise inherit the exported OUT_ROOT=results).
        TRAIN_JOBS+=("cross_${src}_to_${tgt}_s${s}"$'\t'"OUT_ROOT=$OUT_ROOT/cross_dataset bash scripts/run_cross_dataset.sh $src $tgt $s")
      done
    done
  fi
}

build_interpret_jobs() {
  INTERPRET_JOBS=()
  [ "$INCLUDE_INTERPRET" != "1" ] && return 0
  # Emit RQ4 interpret jobs when gene2image WILL be trained this run (it is in VARIANTS,
  # so its checkpoint exists by PHASE 2) OR its checkpoint already exists on disk (an
  # interpret-only re-pass with VARIANTS=""). NOTE: build_interpret_jobs runs at PLAN time,
  # BEFORE PHASE 1 training — so gating on checkpoint existence ALONE wrongly skips the
  # from-scratch full run (checkpoints don't exist yet at planning); gating on VARIANTS
  # ALONE wrongly skips the VARIANTS="" interpret-only pass. This covers both (and DRY_RUN).
  for ds in $DATASETS; do
    local ckpt="$OUT_ROOT/gene2image_${ds}_seed${INTERPRET_SEED}/checkpoints/best_checkpoint.pt"
    case " $VARIANTS " in
      *" gene2image "*) : ;;  # trained this run -> checkpoint will exist by PHASE 2
      *)
        if [ ! -f "$ckpt" ]; then
          echo "  (interpret skipped for ${ds}: gene2image not in VARIANTS and no checkpoint at $ckpt)"
          continue
        fi ;;
    esac
    local gfimp="$OUT_ROOT/geneflow_${ds}_seed${INTERPRET_SEED}/gene_importance_scores.csv"
    local cmd="$PY analysis/pathway_interpret.py --model_path \"$ckpt\" --adata \"$(adata_of "$ds")\" --image_paths \"$(imgpaths_of "$ds")\" --out_dir \"$OUT_ROOT/interpret/${ds}\" --geneflow_importance \"$gfimp\" --gen_steps $GEN_STEPS"
    # Optional: name the adata.obs cell-type column for the per-cell-type RQ4-A
    # analysis (else auto-detected). e.g. CELL_TYPE_KEY=cell_type bash scripts/run_all.sh
    [ -n "${CELL_TYPE_KEY:-}" ] && cmd="$cmd --cell_type_key \"$CELL_TYPE_KEY\""
    INTERPRET_JOBS+=("interpret_${ds}"$'\t'"$cmd")
  done
}

# ---------------------------------------------------------------------------
# Results catalog (so the returned output folder is self-documenting)
# ---------------------------------------------------------------------------
write_catalog() {
  cat > "$OUT_ROOT/EXPERIMENTS_CATALOG.md" <<EOF
# Gene2Image — results catalog

Generated by scripts/run_all.sh. All paths are relative to this ($OUT_ROOT/) folder.

## Headline summary tables (read these first)
| Experiment | File | Contents |
|---|---|---|
| 2.1 Main + 2.2 Ablation | \`summary_main.csv\` | per (variant, dataset): FID / SSIM / PSNR / UNI2-h-FID mean±std over seeds |
| 2.2 Ablation (tagged) | \`ablation/summary.csv\` | same table + flipped_switch + target_rq per variant |
| 2.3 Cross-dataset | \`cross_dataset/summary.csv\` | per (model, setting): fid_cross, fid_same, **degradation_rate** mean±std |

## Per-run outputs
| Pattern | Experiment | Key files |
|---|---|---|
| \`<variant>_<ds>_seed<seed>/\` | 2.1 / 2.2 (variant in gene2image, geneflow, randpath, pathprior, notrans, nomask[, gene2imageReactome]) | \`checkpoints/best_checkpoint.pt\`, \`training_losses.csv\`, \`evaluation_summary.json\`, \`gene_importance_scores.csv\` |
| \`cross_dataset/<src>_to_<tgt>_seed<seed>/eval_on_<tgt>/\` | 2.3 cross-panel eval | \`evaluation_summary.json\` (fid_cross) |
| \`cross_dataset/<src>_to_<tgt>_seed<seed>/eval_on_<src>/\` | 2.3 same-panel reference | \`evaluation_summary.json\` (fid_same) |
| \`interpret/<ds>/\` | 2.4 RQ4 interpretability | \`attention.csv\`, \`A_endogeneity.json\`, \`gsea_consistency.json\` (top-k overlap + Spearman), \`intervention.csv\`, \`C_intervention_sensitivity.json\` |
| \`logs/<job>.log\` | all | stdout/stderr per job |

## RQ -> evidence
- RQ1 (main quality): \`summary_main.csv\` — Gene2Image vs GeneFlow.
- RQ2 (mechanism vs semantics): \`ablation/summary.csv\` — randpath vs gene2image vs geneflow.
- RQ3 (learnable vs fixed): \`ablation/summary.csv\` — pathprior vs gene2image.
- Components: notrans / nomask rows in \`ablation/summary.csv\`.
- RQ4 (interpretability): \`interpret/<ds>/gsea_consistency.json\` (+ A/C json/csv).

## Run configuration
DATASETS=$DATASETS | VARIANTS=$VARIANTS | SEEDS=$SEEDS
INCLUDE_CROSS=$INCLUDE_CROSS INCLUDE_INTERPRET=$INCLUDE_INTERPRET INCLUDE_REACTOME=$INCLUDE_REACTOME INTERPRET_SEED=$INTERPRET_SEED
EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE GEN_STEPS=$GEN_STEPS DB=$DB
EOF
  echo "wrote $OUT_ROOT/EXPERIMENTS_CATALOG.md"
}

# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------
echo "############################################################"
echo "# Gene2Image run_all.sh  | MAX_PARALLEL=$MAX_PARALLEL  GPUs=[${GPU_IDS[*]}]  DRY_RUN=$DRY_RUN"
echo "############################################################"

write_catalog
run_prereqs

# Prep path: masks are built (PHASE 0) and asserted present (PHASE 0.5), then stop.
# Nothing below this point may run, or a "30-minute prep" silently becomes a multi-day
# training queue. verify_masks here covers every mask the full 54-run will load, because
# a prep job is expected to pass the real VARIANTS/DATASETS/SEEDS.
if [ "$MASKS_ONLY" = "1" ]; then
  verify_masks
  echo "=== MASKS_ONLY=1: masks built + verified; skipping ALL train/eval/interpret jobs. ==="
  echo "[ok] masks-only prep done"
  exit 0
fi

build_train_jobs
build_interpret_jobs
echo "=== PLAN: ${#TRAIN_JOBS[@]} train/eval job(s); ${#INTERPRET_JOBS[@]} interpret job(s) ==="

verify_masks

OVERALL_RC=0
echo "=== PHASE 1: training + evaluation (main / ablation / cross) ==="
run_queue "${TRAIN_JOBS[@]}" || OVERALL_RC=1

echo "=== PHASE 2: RQ4 interpretability (after training) ==="
run_queue "${INTERPRET_JOBS[@]}" || OVERALL_RC=1

echo "=== PHASE 3: aggregate results into summary CSVs ==="
NSEEDS=$(echo $SEEDS | wc -w)   # so a seed that fails uniformly across ALL arms is flagged, not silently "complete"
if [ "$DRY_RUN" = "1" ]; then
  echo "  [plan] $PY scripts/summarize_results.py --results_root \"$OUT_ROOT\" --out_dir \"$OUT_ROOT\" --expected_seeds $NSEEDS"
else
  $PY scripts/summarize_results.py --results_root "$OUT_ROOT" --out_dir "$OUT_ROOT" --expected_seeds "$NSEEDS" || OVERALL_RC=1
fi

echo "############################################################"
echo "# DONE. Deliverables under: $OUT_ROOT/"
echo "#   summary_main.csv | ablation/summary.csv | cross_dataset/summary.csv"
echo "#   interpret/<ds>/gsea_consistency.json | EXPERIMENTS_CATALOG.md"
echo "############################################################"
if [ "${OVERALL_RC:-0}" -ne 0 ]; then
  echo ">>> WARNING: one or more train/eval/interpret job(s) FAILED (rc!=0); the summary above" >&2
  echo ">>>          may be INCOMPLETE. Inspect the per-job logs, re-run the failed job(s), and" >&2
  echo ">>>          re-run summarize before treating the results as final. Exiting non-zero." >&2
  exit 1
fi
