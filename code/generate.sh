#!/bin/bash

# GeneFlow Image Generation Script
# Generate synthetic histopathological images from gene expression

# Model path. Edit the variant/dataset/seed to pick which trained checkpoint to visualize
# (default: the main method on c1, seed 42). The encoder type + pathway mask are auto-rebuilt
# from the checkpoint config, so this works for ANY variant (gene2image / geneflow / ...).
MODEL_PATH="results/gene2image_c1_seed42/checkpoints/best_checkpoint.pt"

# Data paths (canonical layout under code/; cell_image_paths_local.json is produced by
# run_all.sh PHASE 0 via fix_image_paths.py). Keep adata/image_paths matching MODEL_PATH's dataset.
ADATA="data/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad"
IMAGE_PATHS="data/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/cell_patch_256_aux/input/cell_image_paths_local.json"
OUTPUT_DIR="results/qualitative_gene2image_c1"

# Model configuration
MODEL_TYPE="single"
IMG_SIZE=256
IMG_CHANNELS=4

# Generation parameters
BATCH_SIZE=8
NUM_SAMPLES=100
GEN_STEPS=50

# Stain normalization (optional)
ENABLE_STAIN_NORM=""          # Add --enable_stain_normalization to enable
STAIN_NORM_METHOD="skimage_hist_match"  # Method for stain normalization

# Run generation
python rectified/rectified_generate.py \
    --model_path ${MODEL_PATH} \
    --model_type ${MODEL_TYPE} \
    --adata ${ADATA} \
    --image_paths ${IMAGE_PATHS} \
    --img_size ${IMG_SIZE} \
    --img_channels ${IMG_CHANNELS} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --gen_steps ${GEN_STEPS} \
    ${ENABLE_STAIN_NORM}