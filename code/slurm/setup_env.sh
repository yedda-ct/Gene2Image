#!/bin/bash
# =============================================================================
# Gene2Image — 在 TU Dresden ZIH "Alpha Centauri" 上创建运行环境
#
# 在【登录节点 login.alpha】上运行（登录节点有外网可装包；计算节点可能没有）：
#     cd <PROJECT_DIR>          # 见下方 ★ 变量
#     bash slurm/setup_env.sh
#
# 本脚本幂等：重复运行只补装缺失的包，不会删除已有环境。
# 不含任何密码 / token / API key；不使用 rm -rf。
#
# 站点约定（ZIH HPC Compendium）：用 module + python venv（不要 conda，二者不可混用）；
# venv 建在 workspace（/data/horse/ws/...），不要放 $HOME。
# =============================================================================
set -euo pipefail

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ★ TODO：Alpha 站点相关变量，全部在这里改（三个脚本顶部保持一致）          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# ★ 仓库 code/ 目录在 Alpha 上的绝对路径（含 scripts/ rectified/ src/ slurm/）
PROJECT_DIR="${PROJECT_DIR:-/data/horse/ws/CHANGEME-gene2image/Gene2Image/code}"

# ★ venv 路径（本脚本会在此创建/复用 venv；建议放在 workspace 内）
VENV_DIR="${VENV_DIR:-/data/horse/ws/CHANGEME-gene2image/venv}"

# ★ module 版本：先在 Alpha 上跑下面两条命令查可用版本，再填实际值（别猜死）：
#       module spider release
#       module spider PyTorch
#   项目期望 torch 2.2.x + CUDA 12.x；找不到精确版本就选最接近的 2.x(cu12x)。
RELEASE_MODULE="${RELEASE_MODULE:-release/CHANGEME}"     # 例 release/24.04
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/CHANGEME}"     # 例 PyTorch/2.1.2-CUDA-12.1.1

# ★ 离线通路库 .gmt（计算节点无外网时用；建掩码会用我加的离线分支）。
#   有外网则留空，run_all.sh 会自动用 gseapy 联网。setup 阶段用不到，仅运行时用。
GMT_HALLMARK="${GMT_HALLMARK:-}"   # 例 /data/horse/ws/CHANGEME/gmt/h.all.v2023.2.Hs.symbols.gmt

# ★ 数据 / 输出 / 检查点目录（setup 阶段用不到，仅运行脚本用；此处占位以便统一对照）。
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/processed_data}"   # 三个数据集目录的父目录
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results}"           # run_all.sh 的 OUT_ROOT
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR}"            # 检查点落在 $OUTPUT_DIR/<run>/checkpoints/
# ───────────────────────────────────────────────────────────────────────────

echo "=== [1/5] module ==="
module purge
module load "${RELEASE_MODULE}"
module load "${PYTORCH_MODULE}"
echo "已加载模块："; module list 2>&1 | sed 's/^/    /'

echo "=== [2/5] 用 module 的 Python 创建/复用 venv ==="
mkdir -p "$(dirname "${VENV_DIR}")"
if [ ! -d "${VENV_DIR}" ]; then
  # --system-site-packages 继承 module 提供的 torch/torchvision，
  # 避免在节点重装 nvidia-*-cu12 大包。
  python -m venv --system-site-packages "${VENV_DIR}"
  echo "已创建 venv: ${VENV_DIR}"
else
  echo "复用已存在的 venv: ${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "=== [3/5] 升级 pip ==="
python -m pip install --upgrade pip

echo "=== [4/5] 安装项目依赖 ==="
# torch/torchvision 来自 module，不在这里装。
# numpy 必须锁 1.26.4（numba/scanpy 兼容；scikit-image 会想拉 numpy>=2 破坏链路）。
# opencv 需 <4.11（numpy<2 兼容）。
# 若 import torch 报 iJIT_NotifyEvent（mkl 冲突），取消下面 mkl 那行注释。
# pip install "mkl==2024.0.0"
pip install "numpy==1.26.4"
if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
  # requirements.txt 已把 torch/torchvision 注释掉（由 module 提供）。
  pip install -r "${PROJECT_DIR}/requirements.txt"
fi
# 训练/评估/掩码必需，且保证版本（requirements 里部分是无版本号的软依赖）：
pip install \
  gseapy "torchmetrics==1.7.1" scikit-image timm einops safetensors \
  "opencv-python-headless==4.10.0.84"
# 可选（缺失时评估自动降级为 N/A，不影响主流程）：cellpose、UNI2-h、sequoia(HE2RNA) 权重。

echo "=== [5/5] 检查信息 ==="
echo "which python : $(which python)"
echo "which pip    : $(which pip)"
python --version
pip --version
python - <<'PY'
import torch, numpy
print("torch        :", torch.__version__)
print("torch.cuda   :", torch.version.cuda)
print("cuda avail   :", torch.cuda.is_available())
print("device count :", torch.cuda.device_count())
print("numpy        :", numpy.__version__, "(应为 1.26.4)")
try:
    import gseapy, torchmetrics, skimage, anndata, scanpy, h5py, tifffile
    print("deps OK      : gseapy/torchmetrics/skimage/anndata/scanpy/h5py/tifffile 均可导入")
except Exception as e:
    print("deps WARN    :", e)
PY

cat <<EOF

============================================================
✅ 环境就绪。把下面两行记下来，确认 smoke_test.slurm /
   alpha_8gpu_train.slurm 顶部的 ★ 变量与此一致：
   VENV_DIR       = ${VENV_DIR}
   RELEASE_MODULE = ${RELEASE_MODULE}
   PYTORCH_MODULE = ${PYTORCH_MODULE}
注意：cuda avail 必须为 True（在登录节点可能为 False，属正常——
真正判定要在 GPU 作业里看，见 smoke_test.slurm 的自检输出）。
============================================================
EOF
