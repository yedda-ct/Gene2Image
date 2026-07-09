#!/bin/bash
# =============================================================================
# 任务F(1) 预飞冒烟 — 重跑前用极小设置验证"最终代码"的三条命脉:
#   (A) val_mse 确实被记录(用于选最优 checkpoint / 早停)
#   (B) resume(--auto_resume)能从上次 epoch 续跑
#   (C) summarize_results.py 能把 evaluation_summary.json 聚合成 summary_main.csv
#
# 用极小规模跑完 1 变体 x 1 数据集 x 1 seed x 极少步,几分钟内在单卡上完成。
# 需要:GPU(eval 的 FID/UNI2-h 必须在 CUDA 上)、已备好的 adata.h5ad + 该数据集
# 的 hallmark real 掩码(先跑 run_all.sh 的 PHASE 0,或手动 build_pathway_mask.py)。
#
# 用法(在 code/ 目录下,或任意目录——脚本会自行 cd 到 code/):
#   bash scripts/preflight_smoke.sh
#   # 可覆盖: V=gene2image DS=c1 SEED=42 SMOKE_ROOT=results_smoke
#
# 退出码:0 = 三项全过;非 0 = 有项失败(打印 FAIL 原因)。
# =============================================================================
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.." || { echo "cannot cd to code/"; exit 2; }

export PYTHONHASHSEED=0                 # 与正式跑一致:保证 80/20 split 可复现
PY=${PY:-python}
V=${V:-gene2image}
DS=${DS:-c1}
SEED=${SEED:-42}
SMOKE_ROOT=${SMOKE_ROOT:-results_smoke}
RUN="$SMOKE_ROOT/${V}_${DS}_seed${SEED}"
LOG1="$SMOKE_ROOT/_smoke_stage1_train_eval.log"
LOG2="$SMOKE_ROOT/_smoke_stage2_resume.log"

mkdir -p "$SMOKE_ROOT"
FAIL=0
say()  { printf '\n=== %s ===\n' "$*"; }
pass() { printf '  PASS: %s\n' "$*"; }
bad()  { printf '  FAIL: %s\n' "$*"; FAIL=1; }

# 极小规模旋钮(不改任何源码,全部走 run_experiments.sh 的 env 覆盖)
export OUT_ROOT="$SMOKE_ROOT" BATCH_SIZE=4 EVAL_BATCH=4 GEN_STEPS=5 WORKERS=2 PY

# -----------------------------------------------------------------------------
say "STAGE 1  short train(2 ep) + eval  ->  $RUN"
# --debug + --debug_samples 64 让数据集缩到 64 个 cell(train 51 / val 13)。
EPOCHS=2 EXTRA="--use_amp --debug --debug_samples 64" \
  bash scripts/run_experiments.sh "$V" "$DS" "$SEED" 2>&1 | tee "$LOG1"
S1=${PIPESTATUS[0]}
[ "$S1" -eq 0 ] || bad "stage1 run_experiments.sh exit=$S1 (see $LOG1)"

# (A) training_losses.csv 存在且含 val_loss 列
if [ -f "$RUN/training_losses.csv" ]; then
  head -1 "$RUN/training_losses.csv" | grep -q 'val_loss' \
    && pass "training_losses.csv written with val_loss column" \
    || bad "training_losses.csv missing val_loss header"
else
  bad "training_losses.csv not written (training did not finish cleanly)"
fi

# (A') 最优 checkpoint 用 val_mse 选择 —— 证据在 checkpoint 字典的 'val_mse' 键,
#      以及日志里的 'Saved checkpoint ... (val_mse: ...)'。CSV 里【没有】val_mse 列,
#      这是最终代码的设计(见 rectified_train.py:705/732/745)。
CKPT="$RUN/checkpoints/best_checkpoint.pt"
if [ -f "$CKPT" ]; then
  "$PY" - "$CKPT" <<'PYEOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert isinstance(ck, dict), "checkpoint is not a dict"
assert 'val_mse' in ck, f"checkpoint has NO 'val_mse' key -> selection not on val_mse. keys={list(ck)[:12]}"
assert 'best_val_loss' in ck, "checkpoint missing best_val_loss"
print(f"  PASS: best_checkpoint carries val_mse={ck['val_mse']:.6f} "
      f"(best epoch idx={ck.get('epoch')}, best_val_loss==best_val_mse={ck['best_val_loss']:.6f})")
PYEOF
  [ $? -eq 0 ] || bad "best_checkpoint.pt lacks val_mse selection metadata"
else
  bad "best_checkpoint.pt not created"
fi
grep -q 'val_mse:' "$LOG1" && pass "log records per-improvement 'val_mse:' lines" \
  || bad "log has no 'val_mse:' line (best-ckpt selection not on val_mse?)"

# (C-part) evaluation_summary.json 存在且含 overall_fid
if [ -f "$RUN/evaluation_summary.json" ]; then
  "$PY" - "$RUN/evaluation_summary.json" <<'PYEOF'
import sys, json, math
d = json.load(open(sys.argv[1]))
need = ['overall_fid','mean_ssim','mean_psnr','overall_uni2h_fid']
miss = [k for k in need if k not in d]
assert not miss, f"evaluation_summary.json missing keys: {miss}"
fid = d['overall_fid']
assert isinstance(fid,(int,float)) and math.isfinite(fid), f"overall_fid not finite: {fid}"
print(f"  PASS: evaluation_summary.json ok (overall_fid={fid:.3f}, ssim={d['mean_ssim']:.3f})")
PYEOF
  [ $? -eq 0 ] || bad "evaluation_summary.json malformed"
else
  bad "evaluation_summary.json not written"
fi

# -----------------------------------------------------------------------------
say "STAGE 2  resume test  (auto_resume, extend 2 -> 4 epochs, train only)"
# 关键:EXTRA 里加 --auto_resume。它读取 $RUN/checkpoints/latest_checkpoint.pt,
# start_epoch = ckpt.epoch + 1。这正是正式跑对抗 SLURM 抢占/walltime 必须补上的开关。
EPOCHS=4 TRAIN=1 EVAL=0 EXTRA="--use_amp --debug --debug_samples 64 --auto_resume" \
  bash scripts/run_experiments.sh "$V" "$DS" "$SEED" 2>&1 | tee "$LOG2"
S2=${PIPESTATUS[0]}
[ "$S2" -eq 0 ] || bad "stage2 resume run exit=$S2 (see $LOG2)"

if grep -Eq 'Auto-resuming from latest checkpoint' "$LOG2"; then
  pass "auto_resume picked up latest_checkpoint.pt"
else
  bad "auto_resume did NOT find a checkpoint (resume broken)"
fi
# 续跑必须从 epoch >= 3 起(stage1 训到 epoch 2)。日志: 'resume training from epoch N'
RESUME_EP=$(grep -Eo 'resume training from epoch [0-9]+' "$LOG2" | grep -Eo '[0-9]+' | tail -1)
if [ -n "${RESUME_EP:-}" ] && [ "$RESUME_EP" -ge 3 ]; then
  pass "resumed from epoch $RESUME_EP (>=3, i.e. continued not restarted)"
else
  bad "did not resume past stage1 (resume epoch='${RESUME_EP:-none}', expected >=3)"
fi

# -----------------------------------------------------------------------------
say "STAGE 3  summarize -> summary_main.csv"
"$PY" scripts/summarize_results.py --results_root "$SMOKE_ROOT" --out_dir "$SMOKE_ROOT" 2>&1 | tee -a "$LOG2"
if [ -f "$SMOKE_ROOT/summary_main.csv" ]; then
  "$PY" - "$SMOKE_ROOT/summary_main.csv" "$V" "$DS" <<'PYEOF'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1]); v, ds = sys.argv[2], sys.argv[3]
row = df[(df.variant==v) & (df.dataset==ds)]
assert len(row), f"summary_main.csv has no {v}/{ds} row. got:\n{df}"
assert 'fid_mean' in df.columns, f"summary_main.csv missing fid_mean col: {list(df.columns)}"
print(f"  PASS: summary_main.csv aggregated {v}/{ds} (fid_mean={row.iloc[0]['fid_mean']:.3f})")
PYEOF
  [ $? -eq 0 ] || bad "summary_main.csv did not contain the smoke run"
else
  bad "summary_main.csv not produced by summarize_results.py"
fi

# -----------------------------------------------------------------------------
say "SMOKE RESULT"
if [ "$FAIL" -eq 0 ]; then
  echo "  ALL GREEN — val_mse recorded, resume works, summarize emits table."
  echo "  (清理: rm -rf $SMOKE_ROOT)"
  exit 0
else
  echo "  SMOKE FAILED — 修好后再启动 54-run 正式重跑。见上面的 FAIL 行与 $LOG1/$LOG2。"
  exit 1
fi
