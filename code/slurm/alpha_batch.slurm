#!/bin/bash
# =============================================================================
# Gene2Image — 参数化分批训练脚本（默认 4 卡 A100 / 每个作业 ≤4 实验，时间短好排队）
#
# 【为什么 4 卡 + 短时间】4 卡的坑位比 8 卡好排队得多；再把每个作业封顶在【1 轮】内
# （4 张卡各跑 1 个独立实验，做完即止），并按数据集给 --time：c1/c2 快给 12h、p1 慢给 36h。
# 占位小 + 时间短 = 最容易被 Slurm 排进去。要点：每个作业的实验数 ≤ 卡数（默认 4）= 1 轮。
#
# 【--time 按数据集在提交命令上给】本文件顶部 #SBATCH --time 只是个安全默认；真正的值在
#   README §A4 的 sbatch 命令行上按批给（c1/c2 批 -> 12:00:00；p1 / 跨数据集批 -> 36:00:00；
#   批 8 interpret -> 04:00:00）。CLI 上的 --time/--gres/... 会覆盖这里的 #SBATCH。
#
# 【默认集群 = Alpha（A100×8/节点，40GB，partition=alpha）】这里只用其中 4 张卡。
#   batch 8 的 afterok 依赖能原生工作（同一个 Slurm）。要把【重活】分到 H100 集群 Capella
#   并行加速，见文件末尾“▼ 分到 H100/Capella”与 README §A4（afterok 不能跨集群，见 README）。
#
# 【怎么切】沿用 BATCH=1..8 的分组（数据集/变体/跨数据集/可解释性的范围划分都已验证）。
# 4 卡一轮装 4 个实验，故按下面切，每个作业 ≤4 实验 = 1 轮：
#   - 批 1/2/3：c1+c2 × 2 变体 × 3 种子 = 12 实验 → 按种子拆 42/43/44，每种子=2变体×2数据集=4实验，每批 3 作业
#   - 批 4/5/6：p1 × 2 变体 × 3 种子 = 6 实验 → 拆 [42 43]=4 + [44]=2，每批 2 作业
#   - 批 7    ：3 个跨数据集对 × 3 种子 = 9 实验 → 按种子拆 42/43/44，每种子=3实验，3 作业
#   - 批 8    ：RQ4 可解释性(3 数据集)+汇总，本就 ≤1h，1 个作业即可
# 共 (3+3+3) + (2+2+2) + 3 + 1 = 19 个短作业（互不重叠）。
#
# ⚠️ 不同 SEEDS 落到不同输出目录(<变体>_<数据集>_seed<种子>)，互不覆盖，可放心并行/分提。
#    （编排不幂等：同一 SEEDS 重提会重训该种子，所以按上面方式【不重叠】地切。）
#
# 用法：用环境变量 BATCH=1..8 + SEEDS 选范围，-J 设作业名，并在 CLI 上给 --time（见 README）：
#     # 批 1/2/3（c1+c2，12h），按种子拆 3 个作业：
#     sbatch -J g2i_b1_s42 --time=12:00:00 --export=ALL,BATCH=1,SEEDS="42" slurm/alpha_batch.slurm
#     sbatch -J g2i_b1_s43 --time=12:00:00 --export=ALL,BATCH=1,SEEDS="43" slurm/alpha_batch.slurm
#     sbatch -J g2i_b1_s44 --time=12:00:00 --export=ALL,BATCH=1,SEEDS="44" slurm/alpha_batch.slurm
#     # 批 4/5/6（p1，36h），拆 [42 43] + [44]：
#     sbatch -J g2i_b4_s4243 --time=36:00:00 --export=ALL,BATCH=4,SEEDS="42 43" slurm/alpha_batch.slurm
#     sbatch -J g2i_b4_s44   --time=36:00:00 --export=ALL,BATCH=4,SEEDS="44"    slurm/alpha_batch.slurm
#     # 批 8（interpret，4h；等批 1+批 4 全部种子完成，见 README §A4 afterok）：
#     sbatch -J g2i_b8 --time=04:00:00 --export=ALL,BATCH=8 slurm/alpha_batch.slurm
#     ... 完整 19 条命令 + 环境搭建/冒烟/取回结果，见【仓库根目录 README.md 的 §A4】。
#
# 批次表（每批实验数；4 卡上的切法见上，每个作业 ≤4 实验 = 1 轮）：
#   1  c1+c2 主对照 gene2image+geneflow ×3种子   12 实验 → 3 作业(每种子4实验)  12h
#   2  c1+c2 消融 randpath+pathprior  ×3         12 实验 → 3 作业              12h
#   3  c1+c2 消融 notrans+nomask      ×3         12 实验 → 3 作业              12h
#   4  p1   主对照 gene2image+geneflow ×3          6 实验 → 2 作业([42 43]+[44]) 36h
#   5  p1   消融 randpath+pathprior   ×3          6 实验 → 2 作业              36h
#   6  p1   消融 notrans+nomask       ×3          6 实验 → 2 作业              36h
#   7  跨数据集 c1↔c2 / c1→p1          ×3          9 实验 → 3 作业(每种子3实验)  36h
#   8  RQ4 可解释性 + 最终汇总（依赖批 1 和批 4）    3 实验 → 1 作业              04h
#
# 依赖：批 8 需要批 1（c1/c2 的 gene2image+geneflow）和批 4（p1 的）的【全部种子】跑完。
# 4 个 ★ 路径变量在所有批次/种子间共享 —— 尤其 OUTPUT_DIR 必须始终一致。
#
# 调度并行度：脚本把 RUN_PARALLEL（默认 4）传给 run_all.sh 作为并发实验数（= 卡数）。
#   保持默认 4（与 --gres=gpu:4 一致）；想用 8 卡时设 RUN_PARALLEL=8 且 --gres=gpu:8。
# =============================================================================
#SBATCH --job-name=g2i_batch
# ★ 默认 Alpha（用 4 张 A100）。account 按你有权限的改。换集群见文件末尾“▼ 分到 H100/Capella”。
#SBATCH --partition=alpha
#SBATCH --account=swtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 4 张卡：Alpha 每 GPU≈6 物理核。24 核 + 480G 足够（4 并行 × WORKERS=5 = 20 worker ≤ 24）。
#SBATCH --cpus-per-task=24
#SBATCH --mem=480000
#SBATCH --cpu-freq=High
#SBATCH --gres=gpu:4
# ↓ 安全默认；实际按批在 sbatch CLI 上给（c1/c2=12h，p1/cross=36h，interpret=4h，见 README §A4）。
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ★ TODO：Alpha 站点变量（所有批次共享，OUTPUT_DIR 必须始终一致）           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/CHANGEME-gene2image/Gene2Image/code}"
VENV_DIR="${VENV_DIR:-/data/horse/ws/CHANGEME-gene2image/venv}"
RELEASE_MODULE="${RELEASE_MODULE:-release/CHANGEME}"     # module spider release
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/CHANGEME}"     # module spider PyTorch
GMT_HALLMARK="${GMT_HALLMARK:-}"
GMT_REACTOME="${GMT_REACTOME:-}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/processed_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results}"        # ★ 所有批次必须相同
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR}"
INTERPRET_SEED="${INTERPRET_SEED:-42}"
# ───────────────────────────────────────────────────────────────────────────

: "${BATCH:?必须设置 BATCH=1..8，例如 sbatch -J g2i_b1 --export=ALL,BATCH=1 slurm/alpha_batch.slurm}"

module purge
module load "${RELEASE_MODULE}" "${PYTORCH_MODULE}"
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

# 调度并行度（= 并发实验数 = 卡数）。默认 4（与 --gres=gpu:4 一致）；想用 8 卡设 RUN_PARALLEL=8。
RUN_PARALLEL="${RUN_PARALLEL:-4}"

# 公共训练超参（保留 early stopping：实际多数实验会提前停，比 100ep 快）
# WORKERS=5：4 个并行实验共享 24 核，每个约 5 个 DataLoader worker（4*5=20 ≤ 24）。
COMMON="EPOCHS=100 BATCH_SIZE=16 EVAL_BATCH=8 GEN_STEPS=100 WORKERS=5 \
INCLUDE_REACTOME=0 EXTRA=--use_amp OUT_ROOT=${OUTPUT_DIR}"

# 提醒：在 4 卡上每个作业最多 4 实验（1 轮）。批 1～7 全部 3 个种子都 >4 实验，必须按种子拆：
#   批 1/2/3（12 实验）和批 7（9 实验）-> 按单种子拆（42 / 43 / 44）；批 4/5/6（6 实验）-> [42 43]+[44]。
# 若批 1～7 忘了拆 SEEDS（默认 3 种子），会排到 2 轮、可能超 --time 被砍。
case "${BATCH}" in
  1|2|3|4|5|6|7)
    if [ -z "${SEEDS:-}" ]; then
      echo "WARNING: 批 ${BATCH} 用全部 3 个种子在 4 卡上 >1 轮（>4 实验），可能超 --time 被砍。" >&2
      echo "         请按种子拆（见 README §A4）：批1/2/3/7 用单种子 SEEDS=\"42\"…；批4/5/6 用 SEEDS=\"42 43\" + \"44\"。" >&2
    fi ;;
esac

run_batch() {  # $1=DATASETS $2=VARIANTS $3=INCLUDE_CROSS
  local ds="$1" var="$2" cross="$3"
  echo "=== run_all：DATASETS='${ds}' VARIANTS='${var}' SEEDS='${SEEDS:-42 43 44}' INCLUDE_CROSS=${cross} RUN_PARALLEL=${RUN_PARALLEL} ==="
  env $COMMON \
    DATASETS="${ds}" VARIANTS="${var}" SEEDS="${SEEDS:-42 43 44}" \
    INCLUDE_CROSS="${cross}" INCLUDE_INTERPRET=0 \
    bash scripts/run_all.sh "${RUN_PARALLEL}"
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

# =============================================================================
# ▼ 分到 H100/Capella 并行（可选，给重活提速）
# -----------------------------------------------------------------------------
# 默认全部用 Alpha 的 4 张 A100。Capella 单节点也是 4×H100，所以挪过去【只需改 --partition】，
# --gres=gpu:4 / RUN_PARALLEL=4 都不用动。把【慢的 p1 / 跨数据集】放 Capella 与 Alpha 并行：
# 在【capella 登录节点】用 sbatch CLI 覆盖分区即可（--time 仍按 README：p1/cross=36h）：
#
#   sbatch -J g2i_b4_s4243 --partition=capella --time=36:00:00 \
#     --export=ALL,BATCH=4,SEEDS="42 43" slurm/alpha_batch.slurm
#
# （CLI 上的 --partition/--time/... 会覆盖本文件顶部的 #SBATCH。H100≈3.2–3.5×A100，p1 快很多；
#   core/mem 上限以 Capella 文档 / `scontrol show partition capella` 为准，不同就再覆盖 --cpus-per-task/--mem。）
#
# ⚠️ 跨集群依赖：Alpha 与 Capella 是【两个独立 Slurm】，job id 互不可见——
#    `--dependency=afterok:<Alpha的jobid>` 在 Capella 上【无效】。所以若批 1（在 Alpha）
#    和批 8（想在 Capella）分属两集群，afterok 串不起来：改成【手动】等两边 squeue 都空了
#    再提批 8（本脚本批 8 自带 checkpoint 文件检查，缺文件会报错退出，是保底）。
#    两集群共享 /data/horse 文件系统，所以同一个 OUTPUT_DIR 在两边都能读写、结果自动合并。
#    详见仓库根目录 README.md §A4“可选：把重活分到 H100/Capella”。
# =============================================================================
