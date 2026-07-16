#!/bin/bash
# Gene2Image main experiment + ablation runner.
# 6 variants x 3 datasets x 3 seeds (single-cell, img_channels=4), per implementation.md 2.1/2.2.
#
# Variants (three orthogonal switches vs Gene2Image):
#   gene2image : real mask  + learnable + transformer   (main method)
#   geneflow   : encoder_type=rna                         (SOTA baseline / lower bound)
#   randpath   : rand mask  + learnable + transformer    (RQ2 mechanism)
#   pathprior  : real mask  + frozen ssGSEA + transformer (RQ3, --no_learnable_pathway)
#   notrans    : real mask  + learnable + NO transformer (component)
#   nomask     : none mask  + learnable + transformer    (component)
#
# Usage:
#   bash scripts/run_experiments.sh <variant> <dataset> <seed> [extra args...]
#   e.g. bash scripts/run_experiments.sh gene2image c1 42
# Or loop everything (sequential; each full run ~hours on a single GPU):
#   bash scripts/run_experiments.sh all
#
# Adjust BATCH_SIZE/EPOCHS for your GPU.
set -e

# Deterministic set/dict iteration across the separate train/eval processes so
# the 80/20 split is identical for a given seed (see dataset.py cell_ids).
export PYTHONHASHSEED=0

PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-data/processed_data}
MASK_DIR=${MASK_DIR:-data/pathway_masks}
OUT_ROOT=${OUT_ROOT:-results}
DB=${DB:-hallmark}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-50}          # 对齐 GeneFlow 源代码 train.sh(EPOCHS=50)
GEN_STEPS=${GEN_STEPS:-100}
WORKERS=${WORKERS:-4}
# --patience 9999 = early stopping OFF, so every arm gets the same 50-epoch budget. This is the
# ONE deliberate divergence from GeneFlow's train.sh (which sets PATIENCE=5); batch 16 / 50 epochs
# / lr 1e-4 / wd 0.01 all follow it verbatim. Rationale: with patience=5 arms stop at different
# epochs (the previous batch measured gene2image ~26.6 vs geneflow ~14.9), which entangles the
# encoder effect -- the paper's whole claim -- with how long each arm happened to train, and that
# was one of the four reasons the previous 54-run batch was thrown away. Since the reported
# checkpoint is the val_mse minimum rather than the last epoch, training the full budget only
# WIDENS the search for every arm (including the GeneFlow baseline) and cannot overfit what is
# reported. Costs compute, not correctness.
EXTRA=${EXTRA:-"--use_amp --patience 9999"}
# Eval does not accept training-only flags like --use_amp; keep its extras separate.
EVAL_EXTRA=${EVAL_EXTRA:-}

# Map short dataset id -> processed_data folder.
dataset_dir() {
  case "$1" in
    c1) echo "Xenium_V1_hSkin_Melanoma_Base_FFPE" ;;
    c2) echo "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE" ;;
    p1) echo "Xenium_Prime_Human_Skin_FFPE" ;;
    *)  echo "UNKNOWN" ;;
  esac
}

run_one() {
  local variant=$1 ds=$2 seed=$3; shift 3
  local folder; folder=$(dataset_dir "$ds")
  if [ "$folder" = "UNKNOWN" ]; then echo "Unknown dataset: $ds"; exit 1; fi

  local adata="$DATA_ROOT/$folder/adata.h5ad"
  local imgpaths="$DATA_ROOT/$folder/cell_patch_256_aux/input/cell_image_paths_local.json"
  local out="$OUT_ROOT/${variant}_${ds}_seed${seed}"

  # Ensure local image paths exist (remap once if missing).
  if [ ! -f "$imgpaths" ]; then
    echo "Remapping image paths for $ds ..."
    $PY scripts/fix_image_paths.py \
      --json "$DATA_ROOT/$folder/cell_patch_256_aux/input/cell_image_paths.json" \
      --local_root "$DATA_ROOT"
  fi

  # Per-variant encoder flags.
  local enc_args=""
  case "$variant" in
    gene2image) enc_args="--encoder_type pathway --pathway_mask $MASK_DIR/${ds}_${DB}_real.npz" ;;
    geneflow)   enc_args="--encoder_type rna" ;;
    randpath)
      # Per-seed random mask: RQ2 claims randPath's 3-seed std reflects random-mask DRAW
      # variance, not just optimization noise. Falling back to the shared rand mask would
      # give all 3 seeds the SAME mask and quietly collapse that claim into "one random
      # draw, trained 3x" -- a weaker result reported as the stronger one. Fail instead:
      # ensure_masks_for / run_all.sh PHASE 0 build these, so a miss means the masks are
      # wrong, and a loud stop costs minutes where a silent downgrade costs the claim.
      local rmask="$MASK_DIR/${ds}_${DB}_rand_s${seed}.npz"
      if [ ! -f "$rmask" ]; then
        echo "[FATAL] randpath needs the per-seed rand mask: $rmask" >&2
        echo "        Refusing to fall back to the shared ${ds}_${DB}_rand.npz -- that would" >&2
        echo "        give every seed the same mask and invalidate RQ2's mask-draw variance." >&2
        echo "        Build it: python scripts/build_pathway_mask.py --adata <adata> --prefix $ds \\" >&2
        echo "                    --db $DB --gmt <hallmark.gmt> --out_dir $MASK_DIR --seed 42 --rand_seeds 42 43 44" >&2
        exit 105
      fi
      enc_args="--encoder_type pathway --pathway_mask $rmask" ;;
    pathprior)  enc_args="--encoder_type pathway --pathway_mask $MASK_DIR/${ds}_${DB}_real.npz --no_learnable_pathway" ;;
    notrans)    enc_args="--encoder_type pathway --pathway_mask $MASK_DIR/${ds}_${DB}_real.npz --no_pathway_transformer" ;;
    nomask)     enc_args="--encoder_type pathway --pathway_mask $MASK_DIR/${ds}_${DB}_none.npz" ;;
    # Optional additional ablation (2.2 note): Hallmark+Reactome pathway granularity.
    gene2imageReactome) DB=hallmark_reactome; enc_args="--encoder_type pathway --pathway_mask $MASK_DIR/${ds}_${DB}_real.npz" ;;
    *) echo "Unknown variant: $variant"; exit 1 ;;
  esac

  # TRAIN=0 skips training (eval-only job: reads the existing checkpoint). Default trains.
  if [ "${TRAIN:-1}" = "1" ]; then
    echo "=== TRAIN $variant | $ds | seed=$seed -> $out ==="
    $PY rectified/rectified_main.py \
      --model_type single --img_size 256 --img_channels 4 \
      --adata "$adata" --image_paths "$imgpaths" \
      --output_dir "$out" \
      --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --gen_steps "$GEN_STEPS" \
      --num_dataloader_workers "$WORKERS" --seed "$seed" \
      --pathway_db "$DB" $enc_args $EXTRA "$@"
  fi

  # A "run" = train + evaluate, so each invocation produces evaluation_summary.json
  # in its own run dir (picked up by summarize_results.py). Set EVAL=0 to skip eval.
  # The eval split uses --seed, so it must match the training seed exactly.
  if [ "${EVAL:-1}" = "1" ]; then
    echo "=== EVAL $variant | $ds | seed=$seed ==="
    # Biological UNI2-h FID needs the UNI2-h weights: export UNI2H_MODEL_PATH=/path/to/UNI2-h
    # (a dir containing pytorch_model.bin) before running, otherwise that metric is NaN while
    # basic FID/SSIM/PSNR are still computed. Same idea for HE2RNA_MODEL_PATH / HEST_METADATA_CSV.
    # rectified_evaluate.py reads these env vars as argparse defaults (no --flag needed).
    $PY rectified/rectified_evaluate.py \
      --model_path "$out/checkpoints/best_checkpoint.pt" \
      --model_type single --img_size 256 --img_channels 4 \
      --adata "$adata" --image_paths "$imgpaths" \
      --output_dir "$out" \
      --seed "$seed" --batch_size "${EVAL_BATCH:-8}" --gen_steps "$GEN_STEPS" $EVAL_EXTRA
  fi
}

if [ "$1" = "all" ]; then
  for variant in gene2image geneflow randpath pathprior notrans nomask; do
    for ds in c1 c2 p1; do
      for seed in 42 43 44; do
        run_one "$variant" "$ds" "$seed"
      done
    done
  done
else
  run_one "$@"
fi
