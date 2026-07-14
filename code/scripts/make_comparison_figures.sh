#!/bin/bash
# =============================================================================
# Produce TWO qualitative comparison figures from trained checkpoints:
#   main_comparison.png      : Real | GeneFlow | Gene2Image
#   ablation_comparison.png  : Real | Gene2Image | randPath | PathPrior | noTrans | noMask
#
# How it stays a FAIR, ALIGNED comparison: every model is generated on the SAME
# dataset + the SAME --seed, so rectified_generate.py selects the SAME held-out test
# cells (seed-fixed) AND gives each cell the SAME initial noise (deterministic per
# cell_id) across all models. Columns therefore align cell-by-cell and any visible
# difference is attributable to the encoder, not the cell choice or the noise draw.
#
# Run this on the server AFTER the relevant checkpoints exist. Edit DS/SEED/NCELLS below.
#   bash scripts/make_comparison_figures.sh
#   DS=p1 SEED=42 NCELLS=6 bash scripts/make_comparison_figures.sh
# =============================================================================
set -euo pipefail

DS="${DS:-c1}"; SEED="${SEED:-42}"; NCELLS="${NCELLS:-8}"; GEN_STEPS="${GEN_STEPS:-100}"
RESULTS="${RESULTS:-results}"
QUAL="${QUAL:-$RESULTS/qualitative_${DS}_seed${SEED}}"

# Dataset folder per panel (edit if your layout differs).
case "$DS" in
  c1) DIR=Xenium_V1_hSkin_Melanoma_Base_FFPE ;;
  c2) DIR=Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE ;;
  p1) DIR=Xenium_Prime_Human_Skin_FFPE ;;
  *)  echo "Unknown DS=$DS (expected c1/c2/p1)"; exit 1 ;;
esac
ADATA="data/processed_data/$DIR/adata.h5ad"
IMGS="data/processed_data/$DIR/cell_patch_256_aux/input/cell_image_paths_local.json"

MAIN="geneflow gene2image"
ABL="gene2image randpath pathprior notrans nomask"
ALL=$(printf '%s\n' $MAIN $ABL | sort -u)

echo ">>> Comparison figures for DS=$DS SEED=$SEED, $NCELLS cells -> $QUAL/"
for v in $ALL; do
  ckpt="$RESULTS/${v}_${DS}_seed${SEED}/checkpoints/best_checkpoint.pt"
  if [ ! -f "$ckpt" ]; then
    echo "!! MISSING $ckpt -- train '${v} ${DS} seed${SEED}' first (or drop it from MAIN/ABL)."; exit 1
  fi
  echo ">>> generating $v (same DS+seed -> same cells + paired noise) ..."
  # --model_type/--img_channels/--img_size MUST be passed: rectified_generate.py builds the
  # dataset from these args BEFORE it reads the checkpoint config, so the config override is
  # too late. Omitting them defaults to multi-cell / 3-channel -> crash (open(None) TypeError,
  # or a [:,:,3] IndexError when saving the 4-channel model's aux channel). Match train/eval.
  python rectified/rectified_generate.py \
    --model_type single --img_size 256 --img_channels 4 \
    --model_path "$ckpt" --adata "$ADATA" --image_paths "$IMGS" \
    --num_samples "$NCELLS" --gen_steps "$GEN_STEPS" --seed "$SEED" \
    --output_dir "$QUAL/$v"
done

echo ">>> assembling grids ..."
python scripts/assemble_comparison.py --qual_root "$QUAL" \
  --main $MAIN --ablation $ABL --out_dir "$QUAL" --n_show "$NCELLS"

echo "=== DONE ==="
echo "  export these two (small):  $QUAL/main_comparison.{png,pdf}  and  $QUAL/ablation_comparison.{png,pdf}"
