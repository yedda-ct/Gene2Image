#!/bin/bash
# Cross-dataset generalization (implementation.md 2.3): train on source panel,
# evaluate on target panel. Gene2Image transfers through the shared pathway space;
# GeneFlow's gene-indexed encoder cannot (input dim mismatches across panels).
#
# Settings: c1->c2, c2->c1, c1->p1 (hardest cross-panel).
# Requires pathway-aligned mask pairs from scripts/build_cross_masks.py:
#   {src}_to_{tgt}_src.npz  (train), {src}_to_{tgt}_tgt.npz (eval)
#
# Usage:
#   bash scripts/run_cross_dataset.sh <src> <tgt> <seed>
#   bash scripts/run_cross_dataset.sh all
#
# This trains Gene2Image on the source, then evaluates the checkpoint on the target
# using the target mask (same pathway rows, target gene columns). GeneFlow is run
# as the source-only baseline; its cross-panel eval is reported as N/A unless gene
# panels match (documented limitation that motivates the pathway approach).
set -e

# Deterministic set/dict iteration across the separate train/eval processes so
# the 80/20 split is identical for a given seed (see dataset.py cell_ids).
export PYTHONHASHSEED=0

PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-data/processed_data}
MASK_DIR=${MASK_DIR:-data/pathway_masks}
OUT_ROOT=${OUT_ROOT:-results/cross_dataset}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-50}          # 对齐 GeneFlow 源代码 train.sh(EPOCHS=50)
GEN_STEPS=${GEN_STEPS:-100}
WORKERS=${WORKERS:-4}
EXTRA=${EXTRA:-"--use_amp"}
# Eval does not accept training-only flags like --use_amp; keep its extras separate.
EVAL_EXTRA=${EVAL_EXTRA:-}

dataset_dir() {
  case "$1" in
    c1) echo "Xenium_V1_hSkin_Melanoma_Base_FFPE" ;;
    c2) echo "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE" ;;
    p1) echo "Xenium_Prime_Human_Skin_FFPE" ;;
    *)  echo "UNKNOWN" ;;
  esac
}

paths_for() {  # echo "adata imgpaths"
  local ds=$1 folder; folder=$(dataset_dir "$ds")
  echo "$DATA_ROOT/$folder/adata.h5ad $DATA_ROOT/$folder/cell_patch_256_aux/input/cell_image_paths_local.json"
}

run_pair() {
  local src=$1 tgt=$2 seed=$3
  read -r src_adata src_img <<< "$(paths_for "$src")"
  read -r tgt_adata tgt_img <<< "$(paths_for "$tgt")"
  local src_mask="$MASK_DIR/${src}_to_${tgt}_src.npz"
  local tgt_mask="$MASK_DIR/${src}_to_${tgt}_tgt.npz"
  local out="$OUT_ROOT/${src}_to_${tgt}_seed${seed}"

  if [ ! -f "$src_mask" ] || [ ! -f "$tgt_mask" ]; then
    echo "Building aligned cross masks for $src -> $tgt ..."
    $PY scripts/build_cross_masks.py \
      --src "$MASK_DIR/${src}_hallmark_real.npz" --tgt "$MASK_DIR/${tgt}_hallmark_real.npz" \
      --src_name "$src" --tgt_name "$tgt" --out_dir "$MASK_DIR"
  fi

  # TRAIN=0 → eval-only job (reads existing checkpoint); EVAL=0 → train-only job.
  if [ "${TRAIN:-1}" = "1" ]; then
    echo "=== TRAIN Gene2Image on $src (seed=$seed) ==="
    $PY rectified/rectified_main.py \
      --model_type single --img_size 256 --img_channels 4 \
      --adata "$src_adata" --image_paths "$src_img" \
      --output_dir "$out" \
      --encoder_type pathway --pathway_mask "$src_mask" \
      --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --gen_steps "$GEN_STEPS" \
      --num_dataloader_workers "$WORKERS" --seed "$seed" $EXTRA
  fi

  if [ "${EVAL:-1}" = "1" ]; then
    echo "=== EVAL on TARGET $tgt (cross-panel) ==="
    $PY rectified/rectified_evaluate.py \
      --model_path "$out/checkpoints/best_checkpoint.pt" \
      --model_type single --img_size 256 --img_channels 4 \
      --adata "$tgt_adata" --image_paths "$tgt_img" \
      --output_dir "$out/eval_on_${tgt}" \
      --encoder_type pathway --pathway_mask "$tgt_mask" --cross_dataset_eval \
      --batch_size 8 --seed "$seed" --gen_steps "$GEN_STEPS" $EVAL_EXTRA

    echo "=== EVAL on SOURCE $src (same-panel reference for degradation rate) ==="
    $PY rectified/rectified_evaluate.py \
      --model_path "$out/checkpoints/best_checkpoint.pt" \
      --model_type single --img_size 256 --img_channels 4 \
      --adata "$src_adata" --image_paths "$src_img" \
      --output_dir "$out/eval_on_${src}" \
      --encoder_type pathway --pathway_mask "$src_mask" \
      --batch_size 8 --seed "$seed" --gen_steps "$GEN_STEPS" $EVAL_EXTRA
  fi
}

if [ "$1" = "all" ]; then
  for seed in 42 43 44; do
    run_pair c1 c2 "$seed"
    run_pair c2 c1 "$seed"
    run_pair c1 p1 "$seed"
  done
else
  run_pair "$@"
fi
