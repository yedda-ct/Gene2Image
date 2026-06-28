#!/bin/bash
# =============================================================================
# Gene2Image — 只补提“还没跑的”实验（不碰已在跑/已跑完的）
#
# 背景：之前已用 8 卡版提了一批，部分作业【正在跑】（如 gene2image_*、notrans_c2_seed42）。
# 重提整批会【重训覆盖】已完成/在飞的 run（编排不幂等）。本脚本只对【输出目录还不存在】的
# triple 逐个提交一个【单卡单实验】作业（slurm/one_exp.slurm），从而：
#   - 跳过【已跑完】的（目录在）；
#   - 跳过【正在跑】的（目录也在）→ 不打断在飞作业；
#   - 只补【从没开始】的（目录不存在）。
#
# ⚠️ 判据 = “输出目录是否存在”。所以【正在跑】的会被跳过（对，别动它）；但若有 run【早早崩了】
#    只留半个空目录，也会被当成“在跑”而跳过——这种少数情况请你手动看 logs 单独处理。
#
# 用法（在【登录节点】，先 export 好站点变量，与 README §A2 一致）：
#   export PROJECT_DIR=$PWD VENV_DIR=... RELEASE_MODULE=... PYTORCH_MODULE=... GMT_HALLMARK=...
#   export OUTPUT_DIR=$PROJECT_DIR/results          # ★ 必须和已在跑的那批用的是同一个！
#   COMMON="ALL,PROJECT_DIR,VENV_DIR,RELEASE_MODULE,PYTORCH_MODULE,GMT_HALLMARK,OUTPUT_DIR"
#
#   # 1) 先预览（默认 DRY_RUN=1，只打印要提哪些、跳过哪些，什么都不提）：
#   COMMON=$COMMON bash slurm/submit_remaining.sh
#
#   # 2) 确认无误后真正提交：
#   COMMON=$COMMON DRY_RUN=0 bash slurm/submit_remaining.sh
#
# 可选缩小范围（默认跑全矩阵）：
#   VARIANTS / DATASETS / SEEDS / WITH_CROSS(1) / WITH_INTERPRET(1) 同 run_all.sh 语义。
#   PARTITION=alpha（默认）或 capella（把补的活提到 Capella，见 README §A4“▼ 可选”）。
# =============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CODE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$CODE_DIR"

DRY_RUN="${DRY_RUN:-1}"
: "${COMMON:?必须设置 COMMON（透传给 sbatch 的 --export 列表，见文件头/README §A2）}"
: "${OUTPUT_DIR:?必须设置 OUTPUT_DIR（且与已在跑的那批一致）}"

PARTITION="${PARTITION:-alpha}"
VARIANTS="${VARIANTS:-gene2image geneflow randpath pathprior notrans nomask}"
DATASETS="${DATASETS:-c1 c2 p1}"
SEEDS="${SEEDS:-42 43 44}"
WITH_CROSS="${WITH_CROSS:-1}"
WITH_INTERPRET="${WITH_INTERPRET:-1}"
INTERPRET_SEED="${INTERPRET_SEED:-42}"
CROSS_PAIRS=("c1 c2" "c2 c1" "c1 p1")

# 数据集 -> 该作业的 --time（c1/c2 快 12h；p1/cross 慢 36h；interpret 4h）。
time_for_ds() { case "$1" in c1|c2) echo "12:00:00";; *) echo "36:00:00";; esac; }

n_submit=0; n_skip=0
emit() {  # $1=jobname $2=time $3=export_extra $4=human
  local jn="$1" tm="$2" ex="$3" human="$4"
  if [ "$DRY_RUN" = "1" ]; then
    echo "  [提] $human   (-J $jn --time=$tm)"
  else
    echo "  [提] $human"
    sbatch -J "$jn" --partition="$PARTITION" --time="$tm" \
      --export="${COMMON},${ex}" slurm/one_exp.slurm
  fi
  n_submit=$((n_submit+1))
}
skip() { echo "  [跳] $1（目录已存在：$2）"; n_skip=$((n_skip+1)); }

echo "############################################################"
echo "# submit_remaining  | OUTPUT_DIR=$OUTPUT_DIR  PARTITION=$PARTITION  DRY_RUN=$DRY_RUN"
echo "#  判据：run 目录已存在 => 跳过（含正在跑的）；不存在 => 补提单卡单实验作业"
echo "############################################################"

# ---- 主 / 消融：每个 (variant, ds, seed) 一个 run 目录 ----
echo "=== 主/消融 ==="
for v in $VARIANTS; do
  for ds in $DATASETS; do
    for s in $SEEDS; do
      d="$OUTPUT_DIR/${v}_${ds}_seed${s}"
      if [ -d "$d" ]; then skip "exp ${v} ${ds} s${s}" "$d"; continue; fi
      emit "g2i_one_${v}_${ds}_s${s}" "$(time_for_ds "$ds")" \
           "KIND=exp,VARIANT=${v},DS=${ds},SEED=${s}" "exp ${v} ${ds} seed=${s}"
    done
  done
done

# ---- 跨数据集：每个 (src->tgt, seed) 一个 run 目录（在 cross_dataset/ 下）----
if [ "$WITH_CROSS" = "1" ]; then
  echo "=== 跨数据集 ==="
  for pair in "${CROSS_PAIRS[@]}"; do
    set -- $pair; src=$1; tgt=$2
    for s in $SEEDS; do
      d="$OUTPUT_DIR/cross_dataset/${src}_to_${tgt}_seed${s}"
      if [ -d "$d" ]; then skip "cross ${src}->${tgt} s${s}" "$d"; continue; fi
      emit "g2i_one_cross_${src}_${tgt}_s${s}" "36:00:00" \
           "KIND=cross,SRC=${src},TGT=${tgt},SEED=${s}" "cross ${src}->${tgt} seed=${s}"
    done
  done
fi

# ---- 可解释性：每个 ds 一个 interpret 目录（依赖 gene2image_<ds>_seed<INTERPRET_SEED>）----
if [ "$WITH_INTERPRET" = "1" ]; then
  echo "=== 可解释性（依赖对应 gene2image ckpt 已存在）==="
  for ds in $DATASETS; do
    d="$OUTPUT_DIR/interpret/${ds}"
    if [ -d "$d" ]; then skip "interpret ${ds}" "$d"; continue; fi
    ck="$OUTPUT_DIR/gene2image_${ds}_seed${INTERPRET_SEED}/checkpoints/best_checkpoint.pt"
    if [ ! -f "$ck" ]; then
      echo "  [等] interpret ${ds}：gene2image_${ds}_seed${INTERPRET_SEED} 还没跑完（缺 best_checkpoint.pt）→ 等它完成后再跑本脚本"
      continue
    fi
    emit "g2i_one_interpret_${ds}" "04:00:00" "KIND=interpret,DS=${ds}" "interpret ${ds}"
  done
fi

echo "############################################################"
if [ "$DRY_RUN" = "1" ]; then
  echo "# 预览：将补提 ${n_submit} 个作业，跳过 ${n_skip} 个已存在。确认后用 DRY_RUN=0 真正提交。"
else
  echo "# 已补提 ${n_submit} 个作业，跳过 ${n_skip} 个已存在。用 squeue -u \$USER 看排队。"
fi
echo "#  最后别忘了：等全部 gene2image(c1/c2/p1, seed=${INTERPRET_SEED}) 跑完后，"
echo "#  RQ4 interpret + 汇总可用 alpha_batch.slurm 的 BATCH=8（见 README §A4）一次出 CSV。"
echo "############################################################"
