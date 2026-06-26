#!/bin/bash
# =============================================================================
# Gene2Image — 冒烟测试 (smoke test) on Alpha Centauri
#
# 目的：用最小资源/最短时间验证【环境 + 编排 + 训练 + 评估能跑通】，不是出结果。
# 资源 = 测试档：1 GPU / 6 核 / 100G / 30 分钟。
#
# 提交（先 cd 到 PROJECT_DIR 且已 mkdir -p logs）：
#     sbatch slurm/smoke_test.slurm
#
# Alpha 站点规则：GPU 集群禁纯 CPU 作业（必须 --gres=gpu:N）；每 GPU 限 6 核
# （ntasks*cpus-per-task ≤ 6*GPU数）。这里 1*6=6 ✓
# =============================================================================
#SBATCH --job-name=g2i_smoke
#SBATCH --partition=alpha
#SBATCH --account=swtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ★ TODO：与 setup_env.sh 顶部保持一致                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/CHANGEME-gene2image/Gene2Image/code}"
VENV_DIR="${VENV_DIR:-/data/horse/ws/CHANGEME-gene2image/venv}"
RELEASE_MODULE="${RELEASE_MODULE:-release/CHANGEME}"     # module spider release
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/CHANGEME}"     # module spider PyTorch
# ★ 离线通路库 .gmt（计算节点无外网时填；有外网留空走 gseapy）：
GMT_HALLMARK="${GMT_HALLMARK:-}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/processed_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results_smoke}"   # 冒烟单独输出，不污染正式 results/
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR}"          # 检查点落在 $OUTPUT_DIR/<run>/checkpoints/
# ───────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------
module purge
module load "${RELEASE_MODULE}" "${PYTORCH_MODULE}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${PROJECT_DIR}"
mkdir -p logs

# ---------------------------------------------------------------------------
# 自检：节点 / 时间 / GPU
# ---------------------------------------------------------------------------
echo "=== 自检 ==="
echo "hostname          : $(hostname)"
echo "date              : $(date)"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda?', torch.cuda.is_available(), '| n_gpu', torch.cuda.device_count())"

# 跨进程划分确定性（blocker #1 的纵深防御；run_all.sh 内部也会设）
export PYTHONHASHSEED=0
# 把 ★ 变量透传给 run_all.sh
export GMT_HALLMARK
export DATA_ROOT="${DATA_DIR}"
export MASK_DIR="${MASK_DIR:-$PROJECT_DIR/data/pathway_masks}"

# ---------------------------------------------------------------------------
# 关卡 1：DRY_RUN —— 秒级验证编排计划 / 路径 / 掩码逻辑（必须先过）
# ---------------------------------------------------------------------------
echo "=== [关卡1] DRY_RUN 预览全计划 ==="
DRY_RUN=1 \
DATASETS="c1" VARIANTS="gene2image geneflow" SEEDS="42" \
INCLUDE_CROSS=0 INCLUDE_INTERPRET=0 INCLUDE_REACTOME=0 \
OUT_ROOT="${OUTPUT_DIR}" \
bash scripts/run_all.sh 1

# ---------------------------------------------------------------------------
# 关卡 2：最小真跑 —— c1 上 gene2image + geneflow，1 种子，1 epoch
#   训练：--debug --debug_samples 200（限训练样本，很快）
#   评估：--max_samples 64（已有参数，确定性取前 64 个验证样本；这是 30min 内的关键）
#   GEN_STEPS=20（DOPRI5 上限步数调小，加速生成）
# ---------------------------------------------------------------------------
echo "=== [关卡2] 最小真跑 (c1, gene2image+geneflow, seed42, 1 epoch) ==="
DATASETS="c1" VARIANTS="gene2image geneflow" SEEDS="42" \
EPOCHS=1 BATCH_SIZE=8 EVAL_BATCH=4 GEN_STEPS=20 WORKERS=6 \
INCLUDE_CROSS=0 INCLUDE_INTERPRET=0 INCLUDE_REACTOME=0 \
EXTRA="--use_amp --debug --debug_samples 200" \
EVAL_EXTRA="--max_samples 64" \
OUT_ROOT="${OUTPUT_DIR}" \
bash scripts/run_all.sh 1

echo "=== 冒烟完成。预期在 ${OUTPUT_DIR}/ 下生成 evaluation_summary.json 与 summary_main.csv ==="
find "${OUTPUT_DIR}" -name "evaluation_summary.json" -o -name "summary_main.csv" 2>/dev/null | head
echo "date(end)         : $(date)"
