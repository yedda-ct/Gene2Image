#!/bin/bash
# =============================================================================
# Gene2Image — 正式实验 (8 GPU) on Alpha Centauri
#
# 提交（先 cd 到 PROJECT_DIR 且已 mkdir -p logs，且冒烟已跑通）：
#     sbatch slurm/alpha_8gpu_train.slurm
#
# ── 这个项目如何用 8 张卡（重要，请读）──────────────────────────────────────
# 本项目的实验是【6 变体 × 3 数据集 × 3 种子 = 54 个独立单卡训练+评估】，外加
# 跨数据集与 RQ4 可解释性。scripts/run_all.sh 自带一个 GPU 任务队列：一张卡跑
# 一个独立实验（CUDA_VISIBLE_DEVICES 固定到该卡），做完一个补一个。传给它的第一个
# 参数 = 并行任务数 = GPU 数。所以下面用 `bash scripts/run_all.sh 8` —— 8 张卡
# 【同时各跑一个独立实验】，这是吞吐并行，也是代码真正验证过的路径。
#
# ── 关于 PyTorch DDP（如实说明，未假装支持）──────────────────────────────────
# 代码里【存在】DDP 分支（--use_ddp + torchrun 风格的 setup_ddp 读 RANK/LOCAL_RANK/
# WORLD_SIZE，DistributedDataParallel 包装模型）。但：
#   (1) 项目文档（dev_log）明确写“DDP 分支保留不主动删除但不保证维护”；
#   (2) run_all.sh / run_experiments.sh 默认编排【不使用】DDP，用的是上面的单卡并行；
#   (3) DDP 路径未经真实多卡验证，且代码审查发现过 DDP 下的潜在隐患。
# 因此本脚本【不】用 DDP，而用 8 单卡并行（对 54 个独立 run 来说反而更高效）。
# 若你确实想用 DDP 把【单个】实验跑在 8 卡上（未验证、风险自负），可参考文件末尾
# 注释里的 torchrun 写法——但不建议用于本实验矩阵。
# =============================================================================
#SBATCH --job-name=g2i_train8
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
# ║  ★ TODO：与 setup_env.sh / smoke_test.slurm 顶部保持一致                    ║
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


# ★ 离线通路库 .gmt（计算节点无外网时填；有外网留空走 gseapy）。
#   无外网且这里为空 → run_all.sh 会在建掩码失败后 fail-fast 报错退出（这是有意的，
#   避免 gene2image 臂静默崩溃）。请务必在无外网时填好 GMT_HALLMARK（和 Reactome 若用）。
# 仅当跑 Hallmark+Reactome 扩展消融时需要          # 检查点落在 $OUTPUT_DIR/<run>/checkpoints/
# ───────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------

module --force purge >/dev/null 2>&1 || module purge || true
module load "${RELEASE_MODULE}" "${GCCCORE_MODULE}" "${PYTHON_MODULE}"


# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${PROJECT_DIR}"
mkdir -p logs

echo "=== 自检 ==="
echo "hostname          : $(hostname)"
echo "date(start)       : $(date)"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda?', torch.cuda.is_available(), '| n_gpu', torch.cuda.device_count())"

# 跨进程划分确定性（blocker #1 的纵深防御；run_all.sh 内部也会设）
export PYTHONHASHSEED=0
# 透传 ★ 变量给 run_all.sh
export GMT_HALLMARK GMT_REACTOME
export DATA_ROOT="${DATA_DIR}"
export MASK_DIR="${MASK_DIR:-$PROJECT_DIR/data/pathway_masks}"

# ---------------------------------------------------------------------------
# 正式：完整实验矩阵，8 张卡并行调度
#   - 6 变体 × 3 数据集 × 3 种子 + 跨数据集 + RQ4 可解释性
#   - WORKERS=5：8 个并行实验共享 44 核，每个约 5 个 DataLoader worker（8*5=40 ≤ 44）
#   - 若 72h 不够全跑完，可先缩小（见下方注释），分批提交
# ---------------------------------------------------------------------------
echo "=== 启动完整实验（run_all.sh 8）==="
EPOCHS=100 BATCH_SIZE=16 EVAL_BATCH=8 GEN_STEPS=100 WORKERS=5 \
DATASETS="c1 c2 p1" \
VARIANTS="gene2image geneflow randpath pathprior notrans nomask" \
SEEDS="42 43 44" \
INCLUDE_CROSS=1 INCLUDE_INTERPRET=1 INCLUDE_REACTOME=0 \
EXTRA="--use_amp" \
OUT_ROOT="${OUTPUT_DIR}" \
bash scripts/run_all.sh 8

echo "=== 全部完成。汇总表在 ${OUTPUT_DIR}/ ：summary_main.csv / ablation/summary.csv / cross_dataset/summary.csv ==="
echo "date(end)         : $(date)"

# 想先小规模验证再全量，可把上面那段换成（例：只主对照 c1，2 变体，1 种子）：
#   EPOCHS=100 DATASETS="c1" VARIANTS="gene2image geneflow" SEEDS="42" \
#   INCLUDE_CROSS=0 INCLUDE_INTERPRET=0 OUT_ROOT="${OUTPUT_DIR}" bash scripts/run_all.sh 8
#
# =============================================================================
# 【可选，未验证】把单个实验用 DDP 跑在 8 卡上（不建议用于本实验矩阵）：
#   不要用 run_all.sh；直接 torchrun 单个 train。注意 --use_ddp 路径未经多卡验证。
#
#   torchrun --standalone --nproc_per_node=8 rectified/rectified_main.py \
#       --use_ddp --model_type single --img_size 256 --img_channels 4 \
#       --adata "${DATA_DIR}/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad" \
#       --image_paths ".../cell_image_paths_local.json" \
#       --encoder_type pathway --pathway_mask "${MASK_DIR}/c1_hallmark_real.npz" \
#       --output_dir "${OUTPUT_DIR}/gene2image_c1_seed42" \
#       --epochs 100 --batch_size 16 --seed 42 --use_amp
#   （evaluate 同理加 --use_ddp。若报 model.rna_dim AttributeError，即审查指出的 DDP 隐患。）
# =============================================================================
