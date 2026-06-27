#!/bin/bash
# =============================================================================
# Gene2Image — 参数化分批训练脚本 (8 卡 / 每批 ≤72h)
#
# 把全部实验拆成 8 个批次，【每批在 8 卡上最多 2 轮】，所以即使单个实验最坏跑满
# 100 epoch（约 36h 上界），每批也不会超过 72h。c1/c2（基因少、快）合并；
# p1（G=5006，慢）每批只占 1 轮单独跑。
#
# 用法：用环境变量 BATCH=1..8 选批次，并用 -J 设作业名（决定日志文件名）：
#     sbatch -J g2i_b1 --export=ALL,BATCH=1 slurm/alpha_batch.slurm
#     sbatch -J g2i_b2 --export=ALL,BATCH=2 slurm/alpha_batch.slurm
#     ... 完整 8 条命令 + 环境搭建/冒烟/取回结果，见【仓库根目录 README.md 的 §A】。
#
# 批次表（job 数 = 实验数；轮数 = ceil(job/8)）：
#   1  c1+c2 主对照 gene2image+geneflow ×3种子      12 job  2 轮
#   2  c1+c2 消融 randpath+pathprior  ×3            12 job  2 轮
#   3  c1+c2 消融 notrans+nomask      ×3            12 job  2 轮
#   4  p1   主对照 gene2image+geneflow ×3            6 job  1 轮
#   5  p1   消融 randpath+pathprior   ×3            6 job  1 轮
#   6  p1   消融 notrans+nomask       ×3            6 job  1 轮
#   7  跨数据集 c1↔c2 / c1→p1          ×3            9 job  2 轮
#   8  RQ4 可解释性 + 最终汇总（依赖批 1 和批 4）       3 job  1 轮
#
# 依赖：批 8 需要批 1（c1/c2 的 gene2image+geneflow）和批 4（p1 的）都跑完。
# 4 个 ★ 路径变量在所有批次间共享 —— 尤其 OUTPUT_DIR 必须始终一致。
# =============================================================================
#SBATCH --job-name=g2i_batch
#SBATCH --partition=alpha
#SBATCH --exclude=i8009
#SBATCH --account=swtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --mem=990000
#SBATCH --cpu-freq=High
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ★ TODO：Alpha 站点变量（所有批次共享，OUTPUT_DIR 必须始终一致）           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/chwu350f-g2i/Gene2Image/code}"
VENV_DIR="${VENV_DIR:-/data/horse/ws/chwu350f-g2i/venv_piptorch}"
RELEASE_MODULE="${RELEASE_MODULE:-release/24.10}"     # module spider release


GCCCORE_MODULE="${GCCCORE_MODULE:-GCCcore/13.2.0}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5}"


GMT_HALLMARK="${GMT_HALLMARK:-}"
GMT_REACTOME="${GMT_REACTOME:-}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/processed_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results}"        # ★ 所有批次必须相同
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR}"
INTERPRET_SEED="${INTERPRET_SEED:-42}"
# ───────────────────────────────────────────────────────────────────────────

: "${BATCH:?必须设置 BATCH=1..8，例如 sbatch -J g2i_b1 --export=ALL,BATCH=1 slurm/alpha_batch.slurm}"

module --force purge >/dev/null 2>&1 || module purge || true
module load "${RELEASE_MODULE}" "${GCCCORE_MODULE}" "${PYTHON_MODULE}"


# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"; mkdir -p logs

echo "=== 自检  BATCH=${BATCH} ==="
echo "host=$(hostname) date=$(date) CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python -c "import torch;print('torch',torch.__version__,'cuda?',torch.cuda.is_available(),'n_gpu',torch.cuda.device_count())"

export PYTHONHASHSEED=0
export GMT_HALLMARK GMT_REACTOME
export DATA_ROOT="${DATA_DIR}"
export MASK_DIR="${MASK_DIR:-$PROJECT_DIR/data/pathway_masks}"

# 公共训练超参（保留 early stopping：实际多数实验会提前停，比 100ep 快）
COMMON="EPOCHS=100 BATCH_SIZE=16 EVAL_BATCH=8 GEN_STEPS=100 WORKERS=5 \
INCLUDE_REACTOME=0 EXTRA=--use_amp OUT_ROOT=${OUTPUT_DIR}"

run_batch() {  # $1=DATASETS $2=VARIANTS $3=INCLUDE_CROSS
  local ds="$1" var="$2" cross="$3"
  echo "=== run_all：DATASETS='${ds}' VARIANTS='${var}' INCLUDE_CROSS=${cross} ==="
  env $COMMON \
    DATASETS="${ds}" VARIANTS="${var}" SEEDS="${SEEDS:-42 43 44}" \
    INCLUDE_CROSS="${cross}" INCLUDE_INTERPRET=0 \
    bash scripts/run_all.sh 8
}

case "${BATCH}" in
  1) run_batch "c1 c2" "gene2image geneflow"   0 ;;
  2) run_batch "c1 c2" "randpath pathprior"    0 ;;
  3) run_batch "c1 c2" "notrans nomask"        0 ;;
  4) run_batch "p1"    "gene2image geneflow"   0 ;;
  5) run_batch "p1"    "randpath pathprior"    0 ;;
  6) run_batch "p1"    "notrans nomask"        0 ;;
  7) run_batch "c1 c2 p1" "" 1 ;;   # VARIANTS="" => 只跑跨数据集，不重训
  8)
     # ---- RQ4 可解释性：直调 pathway_interpret.py（不走 run_all，避免重训）----
     echo "=== 批 8：RQ4 三子分析 + 最终汇总 ==="
     dataset_dir() { case "$1" in
       c1) echo "Xenium_V1_hSkin_Melanoma_Base_FFPE";;
       c2) echo "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE";;
       p1) echo "Xenium_Prime_Human_Skin_FFPE";; esac; }
     adata_of()    { echo "${DATA_DIR}/$(dataset_dir "$1")/adata.h5ad"; }
     imgpaths_of() { echo "${DATA_DIR}/$(dataset_dir "$1")/cell_patch_256_aux/input/cell_image_paths_local.json"; }

     # 依赖检查：批 1（c1/c2）和批 4（p1）的 gene2image checkpoint 必须存在
     missing=0
     for ds in c1 c2 p1; do
       ck="${OUTPUT_DIR}/gene2image_${ds}_seed${INTERPRET_SEED}/checkpoints/best_checkpoint.pt"
       [ ! -f "$ck" ] && { echo "  缺：$ck"; missing=1; }
     done
     if [ "$missing" = "1" ]; then
       echo "ERROR: 批 1（c1/c2）和/或批 4（p1）还没跑完，找不到 gene2image checkpoint。" >&2
       echo "       请先把批 1 和批 4 跑完，再提交批 8。" >&2
       exit 1
     fi

     pids=(); gpu=0
     for ds in c1 c2 p1; do
       ck="${OUTPUT_DIR}/gene2image_${ds}_seed${INTERPRET_SEED}/checkpoints/best_checkpoint.pt"
       gf="${OUTPUT_DIR}/geneflow_${ds}_seed${INTERPRET_SEED}/gene_importance_scores.csv"
       gfarg=""; [ -f "$gf" ] && gfarg="--geneflow_importance $gf" || \
         echo "  注意：$gf 不存在 → 子分析 B(GSEA一致性) 跳过（不影响 A/C）。"
       echo "  [gpu=$gpu] interpret ${ds}"
       CUDA_VISIBLE_DEVICES=$gpu python analysis/pathway_interpret.py \
         --model_path "$ck" --adata "$(adata_of "$ds")" --image_paths "$(imgpaths_of "$ds")" \
         --out_dir "${OUTPUT_DIR}/interpret/${ds}" $gfarg --gen_steps 100 \
         > "logs/interpret_${ds}.log" 2>&1 &
       pids+=($!); gpu=$((gpu+1))
     done
     fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
     [ "$fail" = "1" ] && echo "  警告：有 interpret 任务非零退出，见 logs/interpret_*.log"

     echo "=== 最终汇总（全部批次结果 -> CSV）==="
     python scripts/summarize_results.py --results_root "${OUTPUT_DIR}" --out_dir "${OUTPUT_DIR}" || true
     ;;
  *) echo "ERROR: BATCH 必须是 1..8（当前=${BATCH}）" >&2; exit 1 ;;
esac

echo "=== 批 ${BATCH} 完成 date=$(date) ==="
