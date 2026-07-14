#!/bin/bash
# =============================================================================
# Gene2Image — 单个实验作业（1 张卡跑 1 个 train+eval；给“补跑剩下的”用）
#
# 这是最细的粒度：1 卡 1 实验，占位最小、最容易排队。一般不手动提，由
# slurm/submit_remaining.sh 自动按“还没跑的”逐个 sbatch 进来（见 README §A4“补跑剩下的”）。
#
# 三种工作单元（用环境变量选一种）：
#   主/消融：  KIND=exp     VARIANT=<...> DS=<c1|c2|p1> SEED=<n>
#   跨数据集： KIND=cross   SRC=<..> TGT=<..> SEED=<n>
#   可解释性： KIND=interpret DS=<c1|c2|p1>            （读已训好的 gene2image ckpt）
#
# 手动提的例子（--time 按数据集给：c1/c2=12h，p1/cross=36h，interpret=4h）：
#   sbatch -J g2i_one_notrans_p1_s44 --time=36:00:00 \
#     --export=ALL,KIND=exp,VARIANT=notrans,DS=p1,SEED=44 slurm/one_exp.slurm
#
# ⚠️ 不幂等：会【覆盖】同名 run 目录（<变体>_<数据集>_seed<种子>）。所以只对【还没跑过】的
#    triple 用——submit_remaining.sh 会自动跳过“目录已存在”的（含正在跑的），不碰在飞的作业。
# =============================================================================
#SBATCH --job-name=g2i_one
#SBATCH --partition=capella
#SBATCH --account=CHANGEME_capella_account     # <- set your Capella project/account
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 1 张卡：Alpha 每 GPU≈6 物理核。给 6 核 + 120G 足够单实验（WORKERS=5 ≤ 6）。
#SBATCH --cpus-per-task=14                      # Capella: 1-GPU share = 56/4 cores
#SBATCH --mem-per-cpu=13000M                    # Capella: 13,437 MB per core (per resource table)
#SBATCH --cpu-freq=High
#SBATCH --gres=gpu:1
# ↓ 安全默认；实际按数据集在 sbatch CLI 上给（c1/c2=12h，p1/cross=36h，interpret=4h）。
#SBATCH --time=2-00:00:00                       # single run; Capella max 7 days (7-00:00:00)
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ★ 与 alpha_batch.slurm 顶部一致（同一组站点变量，OUTPUT_DIR 必须一致）     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/CHANGEME-gene2image/Gene2Image/code}"
VENV_DIR="${VENV_DIR:-/data/horse/ws/CHANGEME-gene2image/venv}"
RELEASE_MODULE="${RELEASE_MODULE:-release/CHANGEME}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/CHANGEME}"
GMT_HALLMARK="${GMT_HALLMARK:-}"
GMT_REACTOME="${GMT_REACTOME:-}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/processed_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results}"        # ★ 与其它作业一致
INTERPRET_SEED="${INTERPRET_SEED:-42}"
# ───────────────────────────────────────────────────────────────────────────

: "${KIND:?必须设置 KIND=exp|cross|interpret（见文件头）}"

module purge
module load "${RELEASE_MODULE}" "${PYTORCH_MODULE}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"; mkdir -p logs

echo "=== one_exp KIND=${KIND} host=$(hostname) date=$(date) CVD=${CUDA_VISIBLE_DEVICES:-<unset>} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python -c "import torch;print('torch',torch.__version__,'cuda?',torch.cuda.is_available(),'n_gpu',torch.cuda.device_count())"

export PYTHONHASHSEED=0
export GMT_HALLMARK GMT_REACTOME
export DATA_ROOT="${DATA_DIR}"
export MASK_DIR="${MASK_DIR:-$PROJECT_DIR/data/pathway_masks}"
# 公共训练超参（与 alpha_batch.slurm 一致）。1 卡：WORKERS=5。
export EPOCHS="${EPOCHS:-50}" BATCH_SIZE="${BATCH_SIZE:-16}" EVAL_BATCH="${EVAL_BATCH:-8}" \
       GEN_STEPS="${GEN_STEPS:-100}" WORKERS="${WORKERS:-5}" \
       EXTRA="${EXTRA:---use_amp}" OUT_ROOT="${OUTPUT_DIR}" DB="${DB:-hallmark}"

dataset_dir() { case "$1" in
  c1) echo "Xenium_V1_hSkin_Melanoma_Base_FFPE";;
  c2) echo "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE";;
  p1) echo "Xenium_Prime_Human_Skin_FFPE";; esac; }

case "${KIND}" in
  exp)
    : "${VARIANT:?KIND=exp 需要 VARIANT}"; : "${DS:?需要 DS}"; : "${SEED:?需要 SEED}"
    echo "=== 单实验 exp：${VARIANT} ${DS} seed=${SEED} ==="
    bash scripts/run_experiments.sh "${VARIANT}" "${DS}" "${SEED}"
    ;;
  cross)
    : "${SRC:?KIND=cross 需要 SRC}"; : "${TGT:?需要 TGT}"; : "${SEED:?需要 SEED}"
    echo "=== 单实验 cross：${SRC}->${TGT} seed=${SEED} ==="
    OUT_ROOT="${OUTPUT_DIR}/cross_dataset" bash scripts/run_cross_dataset.sh "${SRC}" "${TGT}" "${SEED}"
    ;;
  interpret)
    : "${DS:?KIND=interpret 需要 DS}"
    folder=$(dataset_dir "${DS}")
    ck="${OUTPUT_DIR}/gene2image_${DS}_seed${INTERPRET_SEED}/checkpoints/best_checkpoint.pt"
    if [ ! -f "$ck" ]; then
      echo "ERROR: 找不到 ${ck} —— 先把 gene2image_${DS}_seed${INTERPRET_SEED} 跑完再 interpret。" >&2
      exit 1
    fi
    gf="${OUTPUT_DIR}/geneflow_${DS}_seed${INTERPRET_SEED}/gene_importance_scores.csv"
    gfarg=""; [ -f "$gf" ] && gfarg="--geneflow_importance $gf" || \
      echo "  注意：$gf 不存在 → 子分析 B(GSEA一致性) 跳过（不影响 A/C）。"
    echo "=== 单实验 interpret：${DS} ==="
    python analysis/pathway_interpret.py \
      --model_path "$ck" \
      --adata "${DATA_DIR}/${folder}/adata.h5ad" \
      --image_paths "${DATA_DIR}/${folder}/cell_patch_256_aux/input/cell_image_paths_local.json" \
      --out_dir "${OUTPUT_DIR}/interpret/${DS}" $gfarg --gen_steps "${GEN_STEPS:-100}"
    ;;
  *) echo "ERROR: KIND 必须是 exp|cross|interpret（当前=${KIND}）" >&2; exit 1 ;;
esac

echo "=== one_exp 完成 date=$(date) ==="
