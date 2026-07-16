#!/usr/bin/env python
# =============================================================================
# 任务F(2) 跑后有效性门禁 —— 在 54 个主 run 全部跑完后运行,核对结果是否"有效",
# 专门针对上一轮把实验做废的三个病根:
#   (1) 选择判据:确认最优 checkpoint 是按 val_mse(纯速度 MSE,跨变体可比)选的,
#       而不是复合 val_loss(含量级随变体差 ~100x 的 L1)。
#   (2) 训练量:确认每个 run 训到了目标 epoch(或合法早停),没有被截断在远低于预算处。
#   (3) 抢占/walltime:确认每个 run 干净收尾(有 evaluation_summary.json),没有停在
#       半成品 checkpoint 上做 eval。
#
# 依据最终代码实际输出结构(均给出 file:line 证据):
#   run 目录         results/<variant>_<ds>_seed<seed>/            (run_experiments.sh:56)
#   最优权重         checkpoints/best_checkpoint.pt (symlink)      (rectified_train.py:749)
#   选择元数据       该 ckpt 字典的 'val_mse' / 'best_val_loss' 键 (rectified_train.py:732,706)
#   训练曲线         training_losses.csv [epoch,train_loss,val_loss] (rectified_main.py:738-743)
#                    注意: 【没有】val_mse 列 —— 这是设计如此,val_mse 存在 ckpt 里。
#   评测指标         evaluation_summary.json                        (rectified_evaluate.py:1329)
#   每-run 日志      results/logs/exp_<variant>_<ds>_s<seed>.log    (run_all.sh:114,277)
#
# 用法:
#   python scripts/validate_runs.py --results_root results \
#          --expected_epochs 50 --patience 9999
#   # 变体/数据集/seed/期望种子数可覆盖;默认对齐 run_all.sh 的 6x3x3=54。
#
# 退出码:0 = 全部 run 通过 (a)-(e) 且主对比合理;非 0 = 有 run 不合格。
# =============================================================================
import os
import re
import csv
import glob
import json
import math
import argparse

VARIANTS = ['gene2image', 'geneflow', 'randpath', 'pathprior', 'notrans', 'nomask']
DATASETS = ['c1', 'c2', 'p1']
SEEDS = [42, 43, 44]
METRIC_KEYS = ['overall_fid', 'mean_ssim', 'mean_psnr', 'overall_uni2h_fid']

EP_RE = re.compile(r'Epoch (\d+)/(\d+) - Train Loss')       # rectified_train.py:619
EARLY_RE = re.compile(r'Early stopping triggered after (\d+) epochs')  # :786
VALMSE_RE = re.compile(r'val_mse:\s*[0-9.]+')               # :745


def load_ckpt_val_mse(ckpt_path):
    """Return (has_val_mse, val_mse, epoch) from best_checkpoint.pt, or (None,..) if torch absent.

    Proof that selection was on val_mse lives in the checkpoint dict, NOT in the CSV
    (rectified_train.py:732 writes 'val_mse'; :705 gates the save on val_mse<best).
    """
    try:
        import torch
    except Exception:
        return (None, None, None)  # torch not importable on this node -> fall back to log grep
    try:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        return (False, None, f'load-error: {e}')
    if not isinstance(ck, dict):
        return (False, None, None)
    return ('val_mse' in ck, ck.get('val_mse'), ck.get('epoch'))


DOPRI5_RE = re.compile(
    r'DOPRI5_DIAGNOSTICS rejected=(\d+) dt_floor=(\d+) '
    r'under_integration_fallback=(\d+) final_t=([0-9.]+)')


def parse_log(log_path):
    """Return dict(last_epoch, budget, early_stop_epoch, has_valmse_line, dopri5_*) from a run log."""
    info = {'last_epoch': None, 'budget': None, 'early_stop_epoch': None,
            'has_valmse_line': False, 'exists': False,
            'dopri5_lines': 0, 'dopri5_dt_floor': 0, 'dopri5_fallback': 0, 'dopri5_bad_t': 0}
    if not os.path.isfile(log_path):
        return info
    info['exists'] = True
    with open(log_path, errors='ignore') as f:
        for line in f:
            m = EP_RE.search(line)
            if m:
                info['last_epoch'] = int(m.group(1))
                info['budget'] = int(m.group(2))
            me = EARLY_RE.search(line)
            if me:
                info['early_stop_epoch'] = int(me.group(1))
            if VALMSE_RE.search(line):
                info['has_valmse_line'] = True
            # Solver gates 1-2 (see check_run): rejected steps are NORMAL adaptive behaviour and
            # are not a failure; dt-floor force-accepts and under-integration top-ups are not.
            md = DOPRI5_RE.search(line)
            if md:
                info['dopri5_lines'] += 1
                info['dopri5_dt_floor'] += int(md.group(2))
                info['dopri5_fallback'] += int(md.group(3))
                if abs(float(md.group(4)) - 1.0) > 1e-6:
                    info['dopri5_bad_t'] += 1
    return info


def csv_epochs(csv_path):
    """Rows in training_losses.csv and whether val_loss column exists.

    NOTE: after a --auto_resume, the CSV holds only the FINAL segment's epochs
    (rectified_main.py rebuilds train_losses each process), so its row count is a
    LOWER BOUND on total epochs. Cross-check with the log's 'Epoch N/M'.
    """
    if not os.path.isfile(csv_path):
        return (None, False)
    with open(csv_path) as f:
        r = csv.reader(f)
        header = next(r, [])
        n = sum(1 for _ in r)
    return (n, 'val_loss' in header)


def check_run(root, variant, ds, seed, expected_epochs, patience, logs_dir):
    run = os.path.join(root, f'{variant}_{ds}_seed{seed}')
    name = f'{variant}/{ds}/seed{seed}'
    problems, notes = [], []

    # (a) run 目录 + 关键文件齐全
    if not os.path.isdir(run):
        return {'name': name, 'ok': False, 'problems': ['run directory missing (never launched)'],
                'notes': [], 'final_epoch': None, 'fid': None, 'log_exists': False}
    ckpt = os.path.join(run, 'checkpoints', 'best_checkpoint.pt')
    ckpt_sp = os.path.join(run, 'checkpoints', 'best_checkpoint_spatial.pt')
    csv_path = os.path.join(run, 'training_losses.csv')
    eval_path = os.path.join(run, 'evaluation_summary.json')
    have_ckpt = os.path.exists(ckpt) or os.path.exists(ckpt_sp)
    if not have_ckpt:
        problems.append('checkpoints/best_checkpoint.pt missing')
    if not os.path.exists(csv_path):
        problems.append('training_losses.csv missing (training did not finish -> likely preempted)')
    # (e) evaluation_summary.json 存在
    if not os.path.exists(eval_path):
        problems.append('evaluation_summary.json missing (eval never ran -> job truncated)')

    # 日志(best-effort;只有经 run_all.sh 跑才有 per-job 日志)
    log = parse_log(os.path.join(logs_dir, f'exp_{variant}_{ds}_s{seed}.log'))

    # (c) val_mse 选择判据
    valmse_ok = False
    if have_ckpt:
        target = ckpt if os.path.exists(ckpt) else ckpt_sp
        has_vm, vm, _ = load_ckpt_val_mse(target)
        if has_vm is True:
            valmse_ok = True
            if isinstance(vm, (int, float)):
                notes.append(f'best val_mse={vm:.5f}')
        elif has_vm is False:
            problems.append("best_checkpoint has no 'val_mse' key -> NOT selected on val_mse")
        else:  # torch unavailable -> fall back to log
            if log['has_valmse_line']:
                valmse_ok = True
                notes.append("val_mse via log grep (torch unavailable to read ckpt)")
            else:
                problems.append("cannot confirm val_mse selection (no torch, no 'val_mse:' in log)")
    if not valmse_ok and log['has_valmse_line']:
        valmse_ok = True

    # training_losses.csv 结构
    n_csv, has_valloss_col = csv_epochs(csv_path)
    if n_csv is not None and not has_valloss_col:
        problems.append('training_losses.csv has no val_loss column (unexpected schema)')

    # (b)+(3) 训练量 / 截断:final_epoch = max(csv行数, 日志末 epoch)
    final_epoch = max([x for x in (n_csv, log['last_epoch']) if x], default=None)
    budget = log['budget'] or expected_epochs
    early = log['early_stop_epoch']

    if final_epoch is None:
        problems.append('cannot determine epochs reached (no CSV, no log)')
    else:
        reached_budget = final_epoch >= budget
        legit_early = early is not None
        if legit_early:
            # (d) 早停必须发生在 >= patience+1 epoch(第一个 epoch 必改善,故最早在 patience+1 触发)
            if early < patience + 1:
                problems.append(f'anomalous early stop at epoch {early} (< patience+1={patience+1}) '
                                f'-> looks like a crash mislabelled, not a real plateau')
            else:
                notes.append(f'legit early stop @ep{early}/{budget}')
        elif not reached_budget:
            # 没到预算、也没有早停行 == 被 walltime/抢占截断(正是上一轮的病根)
            problems.append(f'TRUNCATED: reached epoch {final_epoch}/{budget}, no early-stop -> '
                            f'preempted/walltime; resume with --auto_resume and finish')
        else:
            notes.append(f'reached full budget {final_epoch}/{budget}')
        # 软警告:即便"早停",也不该停在极低 epoch(上一轮灾难是 ep2~5)
        if final_epoch < patience + 1 and not reached_budget:
            notes.append('WARN: suspiciously few epochs')

    # Run gates 1-2 (paper 「评价指标与统计方法」): dt_floor == 0 and under_integration_fallback == 0.
    # dt_floor>0 = steps force-accepted over tolerance; fallback>0 = the solver never reached t=1 and
    # the image is under-integrated. Rejected steps are normal and deliberately NOT a failure here.
    # These live only in the per-run log, so they can only be checked when that log exists.
    if log['exists'] and log['dopri5_lines'] > 0:
        if log['dopri5_dt_floor'] > 0:
            problems.append(f"DOPRI5 forced {log['dopri5_dt_floor']} step(s) at the dt-floor "
                            f"(over tolerance) across {log['dopri5_lines']} generation(s)")
        if log['dopri5_fallback'] > 0:
            problems.append(f"DOPRI5 hit the under-integration fallback {log['dopri5_fallback']} "
                            f"time(s) -> images generated without reaching t=1")
        if log['dopri5_bad_t'] > 0:
            problems.append(f"{log['dopri5_bad_t']}/{log['dopri5_lines']} DOPRI5 generation(s) "
                            f"did not reach final_t=1.0")
        if not (log['dopri5_dt_floor'] or log['dopri5_fallback'] or log['dopri5_bad_t']):
            notes.append(f"dopri5 clean ({log['dopri5_lines']} gen)")
    elif log['exists']:
        notes.append('WARN: no DOPRI5_DIAGNOSTICS in log -> solver gates unverified')
    else:
        notes.append('WARN: per-run log absent -> solver gates unverified '
                     '(array must tee to results/logs/exp_<v>_<ds>_s<seed>.log)')

    # (e) eval 指标可用
    fid = None
    if os.path.exists(eval_path):
        try:
            d = json.load(open(eval_path))
            miss = [k for k in METRIC_KEYS if k not in d]
            if miss:
                problems.append(f'evaluation_summary.json missing metrics {miss}')
            fid = d.get('overall_fid')
            if fid is None or (isinstance(fid, float) and not math.isfinite(fid)):
                problems.append('overall_fid missing/NaN in evaluation_summary.json')
            # UNI2-h biological FID must be finite too. The `miss` check above only tests key
            # PRESENCE, and rectified_evaluate always writes the key -- so a NaN parses back
            # as float('nan') and the run passes. That blind spot is exactly how the previous
            # 54-run batch shipped 54 NaN biological FIDs unnoticed.
            ufid = d.get('overall_uni2h_fid')
            if ufid is None or (isinstance(ufid, float) and not math.isfinite(ufid)):
                problems.append('overall_uni2h_fid missing/NaN — UNI2-h weights were not '
                                'loaded (check UNI2H_MODEL_PATH); biological FID is unusable')
            # Run gate 3 (paper 「评价指标与统计方法」): n_ssim_used == n_psnr_used == total_samples.
            # A shortfall means rows were dropped as non-finite and the reported means silently
            # cover fewer samples than the table claims.
            tot, ns, npsnr = d.get('total_samples'), d.get('n_ssim_used'), d.get('n_psnr_used')
            if None in (tot, ns, npsnr):
                problems.append('evaluation_summary.json lacks total_samples/n_ssim_used/'
                                'n_psnr_used -> cannot verify no samples were silently dropped')
            elif ns != tot or npsnr != tot:
                problems.append(f'non-finite samples dropped: n_ssim_used={ns}, n_psnr_used={npsnr}, '
                                f'total_samples={tot} -> reported SSIM/PSNR means cover fewer '
                                f'samples than the table implies')
            # A tile that failed to decode is replaced by a BLACK image (src/dataset.py). That image
            # becomes the "real" one the model is scored against and enters the FID reference stats,
            # while staying finite -- so the check above cannot see it.
            _zsub = d.get('zero_image_substitutions')
            if _zsub is None:
                # The current evaluator writes this key unconditionally, so its absence means the
                # run came from an older build. Say so rather than pass silently: a black tile that
                # was scored as ground truth is finite, so no other check here can see it, and a
                # quiet skip would read as "verified clean".
                notes.append('WARN: no zero_image_substitutions in evaluation_summary.json -> '
                             'black-tile contamination UNVERIFIED (run predates the counter)')
            elif isinstance(_zsub, int) and _zsub > 0:
                problems.append(f'{_zsub} unreadable tile(s) were replaced by BLACK images and '
                                f'scored as ground truth -> FID/SSIM/PSNR are contaminated; fix the '
                                f'source .tif files and re-evaluate')
        except Exception as e:
            problems.append(f'evaluation_summary.json unreadable: {e}')

    return {'name': name, 'ok': not problems, 'problems': problems, 'notes': notes,
            'final_epoch': final_epoch, 'fid': fid, 'log_exists': log['exists']}


def main():
    ap = argparse.ArgumentParser(description='Gene2Image 54-run post-hoc validity gate.')
    ap.add_argument('--results_root', default='results')
    ap.add_argument('--variants', nargs='*', default=VARIANTS)
    ap.add_argument('--datasets', nargs='*', default=DATASETS)
    ap.add_argument('--seeds', nargs='*', type=int, default=SEEDS)
    ap.add_argument('--expected_epochs', type=int, default=50,
                    help='Target epoch budget (run_all.sh default EPOCHS=50).')
    ap.add_argument('--patience', type=int, default=9999,
                    help='Early-stopping patience USED IN TRAINING. Must match it: this decides '
                         'whether an early stop happened at a legitimate epoch. The formal protocol '
                         'is 9999 (early stopping off), not main.py argparse default of 5.')
    ap.add_argument('--logs_dir', default=None, help='Per-job logs dir (default: <results_root>/logs).')
    ap.add_argument('--rerun_summarize', action='store_true',
                    help='Re-run summarize_results.py and print the main comparison.')
    args = ap.parse_args()

    logs_dir = args.logs_dir or os.path.join(args.results_root, 'logs')
    expected_n = len(args.variants) * len(args.datasets) * len(args.seeds)
    print(f'== Gene2Image validity gate ==  root={args.results_root}  '
          f'expect {expected_n} runs  budget={args.expected_epochs}ep  patience={args.patience}')

    results = []
    for v in args.variants:
        for ds in args.datasets:
            for s in args.seeds:
                results.append(check_run(args.results_root, v, ds, s,
                                         args.expected_epochs, args.patience, logs_dir))

    n_ok = sum(r['ok'] for r in results)
    print(f'\n-- per-run --  ({n_ok}/{len(results)} PASS)')
    for r in results:
        tag = 'PASS' if r['ok'] else 'FAIL'
        extra = '; '.join(r['notes'])
        print(f'  [{tag}] {r["name"]:<28} ep={r["final_epoch"]} fid={r["fid"]}  {extra}')
        for p in r['problems']:
            print(f'         - {p}')

    if not any(r['log_exists'] for r in results):
        print('\n  NOTE: no per-job logs found under', logs_dir,
              '- epoch/early-stop checks fell back to CSV row counts only',
              '(a lower bound after --auto_resume). Run via run_all.sh to get logs.')

    # (f) 重新聚合 + 主对比是否合理(gene2image 应 <= geneflow 的 FID)
    if args.rerun_summarize:
        print('\n-- re-aggregating summary_main.csv --')
        import subprocess, sys
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'summarize_results.py'),
                        '--results_root', args.results_root, '--out_dir', args.results_root,
                        '--expected_seeds', str(len(args.seeds))], check=False)
    smain = os.path.join(args.results_root, 'summary_main.csv')
    unreasonable = []
    try:
        import pandas as pd
    except Exception:
        pd = None
    if os.path.exists(smain) and pd is None:
        print('\n  summary_main.csv present but pandas unavailable here; skip main-comparison check '
              '(run this gate where pandas is installed, e.g. the training env).')
    elif os.path.exists(smain):
        df = pd.read_csv(smain)
        print('\n-- main comparison: gene2image vs geneflow FID (lower better) --')
        for ds in args.datasets:
            g2 = df[(df.variant == 'gene2image') & (df.dataset == ds)]
            gf = df[(df.variant == 'geneflow') & (df.dataset == ds)]
            if len(g2) and len(gf) and 'fid_mean' in df.columns:
                a, b = g2.iloc[0]['fid_mean'], gf.iloc[0]['fid_mean']
                verdict = 'OK (method wins/ties)' if a <= b else '*** method LOSES to baseline ***'
                if a > b:
                    unreasonable.append(ds)
                print(f'  {ds}: gene2image FID={a:.3f}  vs  geneflow FID={b:.3f}   {verdict}')
            else:
                print(f'  {ds}: missing gene2image or geneflow row in summary_main.csv')
    else:
        print('\n  summary_main.csv not found; pass --rerun_summarize or run summarize_results.py first.')

    n_fail = len(results) - n_ok
    print('\n== GATE RESULT ==')
    if n_fail == 0 and not unreasonable:
        print('  ALL PASS — 54 runs complete, val_mse-selected, trained to budget/legit-early-stop, '
              'metrics present, main comparison reasonable.')
        raise SystemExit(0)
    if n_fail:
        print(f'  {n_fail} run(s) FAILED the gate (see above). Fix/resume those before writing the paper.')
    if unreasonable:
        print(f'  WARNING: gene2image still loses on FID for: {unreasonable} — investigate before claiming.')
    raise SystemExit(1)


if __name__ == '__main__':
    main()
