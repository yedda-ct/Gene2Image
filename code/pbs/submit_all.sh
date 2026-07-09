#!/bin/bash
# =============================================================================
# Gene2Image — 一键提交全部实验到 PBS（1×H100/作业）。含依赖链 + 幂等跳过 + 预览。
#
# 顺序与依赖：prereqs(建掩码) → 主/消融 + 跨数据集(依赖 prereqs) → interpret(依赖对应
# gene2image[+geneflow] ckpt) → summarize(依赖全部)。判据=run 目录存在则跳过（可续提）。
#
# 用法（在【登录节点】,先 export 站点变量,与 setup_env.sh 一致）：
#   export PROJECT_DIR=$PWD              # 仓库 code/ 目录
#   export VENV_DIR=/data/.../venv RELEASE_MODULE=release/24.04 PYTORCH_MODULE=PyTorch/2.1.2-CUDA-12.1.1
#   export OUTPUT_DIR=$PROJECT_DIR/results DATA_DIR=$PROJECT_DIR/data/processed_data
#   export MASK_DIR=$PROJECT_DIR/data/pathway_masks
#   # 离线建掩码时: export GMT_HALLMARK=/path/h.all.*.gmt  (Reactome 另加 GMT_REACTOME)
#   # 自动探测不到细胞类型列名时(RQ4-A): export CELL_TYPE_KEY=cell_type
#
#   DRY_RUN=1 bash pbs/submit_all.sh      # 预览:打印将 qsub 什么、跳过什么
#   DRY_RUN=0 bash pbs/submit_all.sh      # 真正提交
#
# 缩小范围(同 run_all.sh 语义): VARIANTS / DATASETS / SEEDS / INCLUDE_CROSS / INCLUDE_INTERPRET
#   / INCLUDE_REACTOME / INTERPRET_SEED。H100 提速: export BATCH_SIZE=32(先冒烟测显存)。
#   walltime 可能被打断: export AUTO_RESUME=1(作业续训,配合“跳过已存在”重复提交直到跑完)。
# =============================================================================
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd); CODE_DIR=$(cd "$SCRIPT_DIR/.." && pwd); cd "$CODE_DIR"
export PROJECT_DIR="${PROJECT_DIR:-$CODE_DIR}"
mkdir -p "$CODE_DIR/logs"   # PBS -o logs/ needs it to exist at submit time

DRY_RUN="${DRY_RUN:-1}"
: "${VENV_DIR:?先 export VENV_DIR}"; : "${RELEASE_MODULE:?}"; : "${PYTORCH_MODULE:?}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results}"
export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/processed_data}"
export MASK_DIR="${MASK_DIR:-$PROJECT_DIR/data/pathway_masks}"
export GMT_HALLMARK="${GMT_HALLMARK:-}" GMT_REACTOME="${GMT_REACTOME:-}"
export INTERPRET_SEED="${INTERPRET_SEED:-42}" BATCH_SIZE="${BATCH_SIZE:-16}" EPOCHS="${EPOCHS:-50}" \
       WORKERS="${WORKERS:-8}" AUTO_RESUME="${AUTO_RESUME:-0}" CELL_TYPE_KEY="${CELL_TYPE_KEY:-}"
VARIANTS="${VARIANTS:-gene2image geneflow randpath pathprior notrans nomask}"
DATASETS="${DATASETS:-c1 c2 p1}"; SEEDS="${SEEDS:-42 43 44}"
INCLUDE_CROSS="${INCLUDE_CROSS:-1}"; INCLUDE_INTERPRET="${INCLUDE_INTERPRET:-1}"
INCLUDE_REACTOME="${INCLUDE_REACTOME:-0}"; QUEUE_ARG="${QUEUE:+-q $QUEUE}"
CROSS_PAIRS=("c1 c2" "c2 c1" "c1 p1")

# 透传给作业的站点变量（名字继承自当前 export 的环境）+ 每作业变量（VAR=val）。
SITE="PROJECT_DIR,VENV_DIR,RELEASE_MODULE,PYTORCH_MODULE,OUTPUT_DIR,DATA_DIR,MASK_DIR,GMT_HALLMARK,GMT_REACTOME,INTERPRET_SEED,BATCH_SIZE,EPOCHS,WORKERS,AUTO_RESUME,CELL_TYPE_KEY"
wall_of(){ case "$1" in c1|c2) echo 12:00:00;; interpret) echo 04:00:00;; *) echo 36:00:00;; esac; }

n=0; skip=0
# qsub 包装：$1=jobname $2=walltime $3=extra -v(VAR=val,...) $4=depend(空则无) ; echo jobid
submit(){
  local jn="$1" tm="$2" ev="$3" dep="$4"
  local deparg=""; [ -n "$dep" ] && deparg="-W depend=afterok:$dep"
  if [ "$DRY_RUN" = "1" ]; then
    echo "  [提] $jn  (-l walltime=$tm ${deparg:+$deparg} -v $ev)" >&2; echo "DRYRUN.$jn"
  else
    qsub -N "$jn" -l walltime="$tm" $QUEUE_ARG $deparg -v "${SITE},${ev}" pbs/one_exp.pbs
  fi
}

echo "############ submit_all | OUTPUT_DIR=$OUTPUT_DIR DRY_RUN=$DRY_RUN ############" >&2

# ---- 0) 前置(掩码/路径)：一个作业,后续全部依赖它 ----
echo "=== [0] prereqs ===" >&2
if [ "$DRY_RUN" = "1" ]; then PRE="DRYRUN.prereqs"; echo "  [提] g2i_prereqs (pbs/prereqs.pbs)" >&2
else PRE=$(qsub -N g2i_prereqs $QUEUE_ARG \
      -v "${SITE},DATASETS=${DATASETS// /_},SEEDS=${SEEDS// /_},INCLUDE_CROSS=${INCLUDE_CROSS},INCLUDE_REACTOME=${INCLUDE_REACTOME}" \
      pbs/prereqs.pbs); fi
# 注：DATASETS/SEEDS 里的空格在 -v 里会断行,故用下划线占位,prereqs.pbs 会还原（见其头部注释调整）。
echo "  prereqs job = $PRE" >&2

declare -A G2I GF   # ds -> jobid(用于 interpret 依赖)
alldeps="$PRE"

# ---- 1) 主/消融：每 (variant,ds,seed) 拆成【train 作业】+【eval 作业】(eval 依赖 train) ----
# 细粒度好处：eval 短(4h,进短队列);eval 失败/需重跑不必重训;train 被杀只重训不重评。
echo "=== [1] 主/消融 (train + eval 拆分) ===" >&2
for v in $VARIANTS; do for ds in $DATASETS; do for s in $SEEDS; do
  d="$OUTPUT_DIR/${v}_${ds}_seed${s}"; ck="$d/checkpoints/best_checkpoint.pt"
  # 判据=最终产物存在则跳过(可续提)。⚠️ 补提前先 qstat 确认无同名作业在跑。
  if [ -f "$d/evaluation_summary.json" ]; then echo "  [跳] $v $ds s$s (已完成)" >&2; skip=$((skip+1)); continue; fi
  tid=""
  if [ ! -f "$ck" ]; then
    tid=$(submit "g2i_train_${v}_${ds}_s${s}" "$(wall_of "$ds")" "KIND=train,VARIANT=${v},DS=${ds},SEED=${s}" "$PRE"); n=$((n+1))
  else echo "  [部分] $v $ds s$s: 已有 checkpoint,只补 eval" >&2; fi
  eid=$(submit "g2i_eval_${v}_${ds}_s${s}" 04:00:00 "KIND=eval,VARIANT=${v},DS=${ds},SEED=${s}" "${tid:-$PRE}"); n=$((n+1)); alldeps="$alldeps:$eid"
  [ "$v" = gene2image ] && [ "$s" = "$INTERPRET_SEED" ] && G2I[$ds]="$eid"
  [ "$v" = geneflow ]   && [ "$s" = "$INTERPRET_SEED" ] && GF[$ds]="$eid"
done; done; done

# ---- 2) 跨数据集：每 (src->tgt,seed) 拆成 cross_train + cross_eval(eval 依赖 train) ----
if [ "$INCLUDE_CROSS" = "1" ]; then echo "=== [2] 跨数据集 (train + eval 拆分) ===" >&2
  for pair in "${CROSS_PAIRS[@]}"; do set -- $pair; src=$1; tgt=$2
    for s in $SEEDS; do
      d="$OUTPUT_DIR/cross_dataset/${src}_to_${tgt}_seed${s}"; ck="$d/checkpoints/best_checkpoint.pt"
      if [ -f "$d/eval_on_${tgt}/evaluation_summary.json" ]; then echo "  [跳] cross $src->$tgt s$s (已完成)" >&2; skip=$((skip+1)); continue; fi
      tid=""
      if [ ! -f "$ck" ]; then
        tid=$(submit "g2i_ctrain_${src}_${tgt}_s${s}" 36:00:00 "KIND=cross_train,SRC=${src},TGT=${tgt},SEED=${s}" "$PRE"); n=$((n+1))
      else echo "  [部分] cross $src->$tgt s$s: 已有 checkpoint,只补 eval" >&2; fi
      eid=$(submit "g2i_ceval_${src}_${tgt}_s${s}" 04:00:00 "KIND=cross_eval,SRC=${src},TGT=${tgt},SEED=${s}" "${tid:-$PRE}"); n=$((n+1)); alldeps="$alldeps:$eid"
    done
  done
fi

# ---- 3) interpret：每 ds 一个,依赖对应 gene2image(+geneflow) 作业(若在本批提交) ----
if [ "$INCLUDE_INTERPRET" = "1" ]; then echo "=== [3] RQ4 interpret ===" >&2
  for ds in $DATASETS; do
    d="$OUTPUT_DIR/interpret/${ds}"
    if [ -f "$d/A_endogeneity.json" ]; then echo "  [跳] interpret $ds (已完成)" >&2; skip=$((skip+1)); continue; fi
    dep="${G2I[$ds]:-}"; [ -n "${GF[$ds]:-}" ] && dep="${dep:+$dep:}${GF[$ds]}"
    # dep 为空 = 对应 ckpt 已存在(本批未提)→ 只依赖 prereqs(其实已就绪),直接依赖 prereqs。
    jid=$(submit "g2i_interpret_${ds}" 04:00:00 "KIND=interpret,DS=${ds}" "${dep:-$PRE}")
    n=$((n+1)); alldeps="$alldeps:$jid"
  done
fi

echo "############" >&2
if [ "$DRY_RUN" = "1" ]; then
  echo "# 预览：将提交 $n 个作业,跳过 $skip 个已存在。确认后 DRY_RUN=0 真跑。" >&2
  echo "# 汇总(全部完成后手动跑一次): python scripts/summarize_results.py --results_root \$OUTPUT_DIR --out_dir \$OUTPUT_DIR --expected_seeds 3" >&2
else
  echo "# 已提交 $n 个作业,跳过 $skip 个。qstat -u \$USER 看排队；全部完成后跑 summarize(见 README)。" >&2
fi
