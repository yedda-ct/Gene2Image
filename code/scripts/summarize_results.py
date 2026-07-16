"""Aggregate per-run evaluation_summary.json into the summary CSVs (implementation.md 5.4-5.6).

Scans an experiments root for ``evaluation_summary.json`` files (written by
rectified_evaluate.py), parses variant/dataset/seed from the run directory name
(``<variant>_<ds>_seed<seed>``, e.g. ``gene2image_c1_seed42``), and produces
mean +/- std tables across seeds.

Metric keys map to the evaluator's actual schema:
    overall_fid, mean_ssim, mean_psnr, overall_uni2h_fid.

Outputs:
    summary_main.csv      : per (variant, dataset) mean+/-std of each metric
    ablation/summary.csv   : same table tagged with the flipped switch / target RQ

Usage:
    python scripts/summarize_results.py --results_root results --out_dir results
"""
import os
import re
import csv
import json
import glob
import argparse
import numpy as np
import pandas as pd


RUN_RE = re.compile(r'(?P<variant>[a-zA-Z0-9]+)_(?P<ds>c1|c2|p1)_seed(?P<seed>\d+)')
# Cross-dataset run dirs (run_cross_dataset.sh): e.g. c1_to_p1_seed42, with
# eval_on_<src>/ and eval_on_<tgt>/ subdirs holding the evaluation_summary.json.
CROSS_RE = re.compile(r'(?P<src>c1|c2|p1)_to_(?P<tgt>c1|c2|p1)_seed(?P<seed>\d+)')

SWITCH = {
    'gene2image': ('full', 'RQ1 main'),
    'geneflow':   ('no pathway encoder', 'lower bound'),
    'randpath':   ('real->random mask', 'RQ2 mechanism'),
    'pathprior':  ('learnable->frozen', 'RQ3 fixed scoring'),
    'notrans':    ('remove transformer', 'pathway co-regulation'),
    'nomask':     ('sparse->dense', 'structured sparsity'),
}

# eval json key -> friendly metric name (direction: fid/uni2h lower better, ssim/psnr higher)
# NOTE: per-batch FID (mean_batch_fid) is deliberately NOT aggregated: with small
# eval batches it is statistically invalid (rank-deficient covariance). Only the
# full-set overall_fid / overall_uni2h_fid are reported.
METRIC_KEYS = {
    'overall_fid': 'fid',
    'mean_ssim': 'ssim',
    'mean_psnr': 'psnr',
    'overall_uni2h_fid': 'uni2h_fid',
}


EPOCH_RE = re.compile(r'Epoch (\d+)/(\d+) - Train Loss')          # rectified_train.py
LASTEP_RE = re.compile(r'TRAIN_GATES [^\n]*last_epoch=(\d+)')     # rectified_main.py


def _stop_epoch(run_dir):
    """Epochs actually completed by this run, or None.

    training_losses.csv alone is NOT the answer. rectified_train.py re-initialises train_losses per
    process and rectified_main.py rewrites the CSV with rows renumbered from 1, so after an
    --auto_resume it holds ONLY the final segment. The array's walltime (24h) is shorter than c1
    (~33h) and p1 (~44h), so resume is the DESIGNED path: a CSV-only reading would report ~15-25
    for most runs that in fact trained the full 50. The paper reads this column as proof the
    budgets were equal and treats "< 50" as truncated-and-excludable, so that would discard most of
    the matrix -- and would contradict validate_runs.py:167, which already takes the max.

    So take max(csv rows, the log's last epoch), exactly as validate_runs.py does.
    """
    n_csv = None
    csv_path = os.path.join(run_dir, 'training_losses.csv')
    if os.path.isfile(csv_path):
        try:
            with open(csv_path) as f:
                r = csv.reader(f)
                next(r, None)                       # header
                n_csv = sum(1 for _ in r) or None
        except Exception:
            n_csv = None

    # Per-run log: <results_root>/logs/exp_<variant>_<ds>_s<seed>.log -- written by run_all.sh and
    # tee -a'd by capella_array_1gpu.slurm, so it survives resumes and carries the true last epoch.
    n_log = None
    m = RUN_RE.search(os.path.basename(run_dir.rstrip('/')))
    if m:
        log_path = os.path.join(os.path.dirname(run_dir.rstrip('/')), 'logs',
                                f"exp_{m.group('variant')}_{m.group('ds')}_s{m.group('seed')}.log")
        if os.path.isfile(log_path):
            try:
                with open(log_path, errors='ignore') as f:
                    for line in f:
                        mg = LASTEP_RE.search(line) or EPOCH_RE.search(line)
                        if mg:
                            n_log = max(n_log or 0, int(mg.group(1)))
            except Exception:
                n_log = None

    vals = [v for v in (n_csv, n_log) if v]
    return max(vals) if vals else None


def collect(results_root):
    """Return a long DataFrame of all evaluation_summary.json found under results_root."""
    rows = []
    pattern = os.path.join(results_root, '**', 'evaluation_summary.json')
    for jpath in glob.glob(pattern, recursive=True):
        # Cross-dataset eval results live under eval_on_* subdirs and are summarized
        # separately in the cross-dataset table (5.6); skip them here so they do not
        # pollute the main (variant x dataset) table.
        if os.path.basename(os.path.dirname(jpath)).startswith('eval_on_'):
            continue
        # Identity comes from the run directory name.
        ident = None
        d = os.path.dirname(jpath)
        for _ in range(4):
            m = RUN_RE.search(os.path.basename(d))
            if m:
                ident = m
                break
            d = os.path.dirname(d)
        with open(jpath) as f:
            data = json.load(f)
        rec = {}
        if ident:
            rec['variant'] = ident.group('variant')
            rec['dataset'] = ident.group('ds')
            rec['seed'] = int(ident.group('seed'))
        else:
            rec['variant'] = data.get('encoder_type', 'unknown')
            rec['dataset'] = 'unknown'
            rec['seed'] = data.get('seed')
        rec['eval_path'] = jpath
        # Realized training length. Early stopping is OFF (--patience 9999), so this should be 50
        # for every arm, and the paper prints it beside the metrics as EVIDENCE that all arms got the
        # same budget. A value below 50 therefore means the run was TRUNCATED (preemption/walltime),
        # not that it converged early -- that run must be resumed, not reported.
        # Row count of training_losses.csv == epochs completed by the last invocation.
        rec['stop_epoch'] = _stop_epoch(os.path.dirname(jpath))
        for src_key, name in METRIC_KEYS.items():
            v = data.get(src_key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                rec[name] = v
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate(df, expected_seeds=None):
    """Mean +/- std across seeds for each (variant, dataset).

    Also emits an ``n_seeds`` column (runs contributing to each group) plus a
    per-metric ``<metric>_n`` count, and prints a LOUD warning for any group with
    fewer runs than expected (default: the max observed across groups) or any
    metric with missing/NaN values. Without this, a crashed or NaN seed is silently
    dropped and the group's mean/std is formatted identically to a full multi-seed
    result -- so a 2-seed arm could be compared against a 3-seed arm unnoticed.
    """
    if df.empty:
        return df
    present = [m for m in METRIC_KEYS.values() if m in df.columns]
    # stop_epoch aggregates like a metric: the paper prints it next to FID/SSIM/PSNR as evidence
    # that every arm got the same budget. With early stopping off it should read 50.00 +/- 0.00 for
    # all of them; anything less is a truncated (preempted) run to resume, not a converged one.
    if 'stop_epoch' in df.columns and df['stop_epoch'].notna().any():
        present = present + ['stop_epoch']
    grp = df.groupby(['variant', 'dataset'])
    g = grp[present].agg(['mean', 'std'])
    g.columns = [f'{m}_{stat}' for m, stat in g.columns]
    g = g.reset_index()
    # runs per group, and non-NaN contributing count per metric
    g = g.merge(grp.size().reset_index(name='n_seeds'), on=['variant', 'dataset'])
    g = g.merge(grp[present].count().reset_index().rename(
        columns={m: f'{m}_n' for m in present}), on=['variant', 'dataset'])

    exp = int(expected_seeds) if expected_seeds is not None else int(g['n_seeds'].max())
    problems = []
    for _, row in g.iterrows():
        vd = f"{row['variant']}/{row['dataset']}"
        if int(row['n_seeds']) < exp:
            problems.append(f"  {vd}: only {int(row['n_seeds'])}/{exp} seed run(s)")
        for m in present:
            if int(row[f'{m}_n']) < int(row['n_seeds']):
                problems.append(f"  {vd}: metric {m} present for "
                                f"{int(row[f'{m}_n'])}/{int(row['n_seeds'])} run(s)")
    if problems:
        print("=" * 72)
        print(f"WARNING: INCOMPLETE AGGREGATES (expected {exp} seed(s) per group).")
        print("These rows are means over FEWER runs than expected -- do NOT compare")
        print("them as if they were complete multi-seed results:")
        for p in problems:
            print(p)
        print("=" * 72)
    return g


def collect_cross(results_root):
    """Collect cross-dataset transfer eval results into per-run rows (5.6).

    Each cross run dir (``<src>_to_<tgt>_seed<seed>``) has two eval subdirs:
    eval_on_<tgt> (cross-panel) and eval_on_<src> (same-panel reference). Pairs them
    and computes degradation_rate = (fid_cross - fid_same) / fid_same.
    """
    model_map = {'pathway': 'gene2image', 'rna': 'geneflow'}
    runs = {}
    pattern = os.path.join(results_root, '**', 'evaluation_summary.json')
    for jpath in glob.glob(pattern, recursive=True):
        parent = os.path.basename(os.path.dirname(jpath))
        m_eval = re.match(r'eval_on_(c1|c2|p1)$', parent)
        if not m_eval:
            continue
        run_dir = os.path.basename(os.path.dirname(os.path.dirname(jpath)))
        mc = CROSS_RE.search(run_dir)
        if not mc:
            continue
        with open(jpath) as f:
            data = json.load(f)
        fid = data.get('overall_fid')
        if fid is None or (isinstance(fid, float) and np.isnan(fid)):
            continue
        src, tgt, seed = mc.group('src'), mc.group('tgt'), int(mc.group('seed'))
        key = (src, tgt, seed)
        rec = runs.setdefault(key, {
            'src': src, 'tgt': tgt, 'seed': seed,
            'model': model_map.get(data.get('encoder_type'), 'gene2image')})
        if m_eval.group(1) == tgt:
            rec['fid_cross'] = fid
        elif m_eval.group(1) == src:
            rec['fid_same'] = fid
    rows = [r for r in runs.values() if 'fid_cross' in r and 'fid_same' in r]

    # The paper (v10 「编码效率与跨 panel 探索」) asks for the transplanted model to be reported
    # "并与目标数据集内训练的 Gene2Image 参考结果并列" -- side by side with a Gene2Image TRAINED ON
    # THE TARGET. That reference is NOT fid_same: fid_same is the source model on its own source
    # panel, which answers "how much worse is it away from home", a different question from "how far
    # short of a native model does transplanting land". The native number already exists -- it is the
    # main matrix's gene2image_<tgt>_seed<seed> -- so pull it in rather than leave the column empty.
    native = {}
    for jpath in glob.glob(os.path.join(results_root, '**', 'evaluation_summary.json'), recursive=True):
        if os.path.basename(os.path.dirname(jpath)).startswith('eval_on_'):
            continue
        mm = RUN_RE.search(os.path.basename(os.path.dirname(jpath)))
        if not mm:
            continue
        try:
            with open(jpath) as f:
                v = json.load(f).get('overall_fid')
        except Exception:
            continue
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            native[(mm.group('variant'), mm.group('ds'), int(mm.group('seed')))] = v

    for r in rows:
        r['degradation_rate'] = ((r['fid_cross'] - r['fid_same']) / r['fid_same']
                                 if r['fid_same'] else float('nan'))
        # Same variant, same seed, but trained on the TARGET panel.
        r['fid_target_native'] = native.get((r['model'], r['tgt'], r['seed']))
        # How far the transplant lands from a model that actually trained on this panel. Positive =
        # worse than native. Left NaN when the native run is missing rather than silently reusing
        # fid_same, which would answer a different question under the same column name.
        r['gap_vs_native'] = ((r['fid_cross'] - r['fid_target_native']) / r['fid_target_native']
                              if r.get('fid_target_native') else float('nan'))
    return pd.DataFrame(rows)


def aggregate_cross(df, expected_seeds=None):
    """Mean +/- std of the cross-panel columns per (model, setting).

    Columns, and what each answers:
      fid_cross          transplanted source model, evaluated on the TARGET panel
      fid_same           the same model on its own SOURCE panel  -> degradation_rate's denominator
      degradation_rate   (cross - same)/same: how much worse away from home
      fid_target_native  a model of the same variant/seed TRAINED on the target (from the main
                         matrix) -- the reference the paper asks to print alongside
      gap_vs_native      (cross - native)/native: how far short of a native model the transplant lands

    Also emits n_seeds and warns on incomplete groups (mirrors aggregate()), so a dropped
    cross run does not silently shrink the sample without notice.
    """
    if df.empty:
        return df
    df = df.copy()
    df['setting'] = df['src'].str.upper() + '->' + df['tgt'].str.upper()
    grp = df.groupby(['model', 'setting'])
    _cols = ['fid_cross', 'fid_same', 'degradation_rate']
    for _c in ('fid_target_native', 'gap_vs_native'):
        if _c in df.columns and df[_c].notna().any():
            _cols.append(_c)
    g = grp[_cols].agg(['mean', 'std'])
    g.columns = [f'{m}_{stat}' for m, stat in g.columns]
    g = g.reset_index()
    g = g.merge(grp.size().reset_index(name='n_seeds'), on=['model', 'setting'])
    exp = int(expected_seeds) if expected_seeds is not None else int(g['n_seeds'].max())
    problems = [f"  {row['model']} {row['setting']}: only {int(row['n_seeds'])}/{exp} seed run(s)"
                for _, row in g.iterrows() if int(row['n_seeds']) < exp]
    if problems:
        print("=" * 72)
        print(f"WARNING: INCOMPLETE CROSS-DATASET AGGREGATES (expected {exp} seed(s) per group).")
        print("These degradation_rate means are over FEWER runs than expected:")
        for p in problems:
            print(p)
        print("=" * 72)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', default='results')
    ap.add_argument('--out_dir', default='results')
    ap.add_argument('--expected_seeds', type=int, default=None,
                    help='Expected #seeds per (variant,dataset); warn if any group has '
                         'fewer. Defaults to the max observed across groups.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = collect(args.results_root)
    if df.empty:
        print(f"No main evaluation_summary.json found under {args.results_root}.")
    else:
        print(f"Collected {len(df)} main eval runs.")
        summary = aggregate(df, expected_seeds=args.expected_seeds)
        main_path = os.path.join(args.out_dir, 'summary_main.csv')
        summary.to_csv(main_path, index=False)
        print(f"wrote {main_path} ({len(summary)} variant x dataset rows)")

        abl = summary.copy()
        abl['flipped_switch'] = abl['variant'].map(lambda v: SWITCH.get(v, ('', ''))[0])
        abl['target_rq'] = abl['variant'].map(lambda v: SWITCH.get(v, ('', ''))[1])
        abl_dir = os.path.join(args.out_dir, 'ablation')
        os.makedirs(abl_dir, exist_ok=True)
        abl_path = os.path.join(abl_dir, 'summary.csv')
        abl.to_csv(abl_path, index=False)
        print(f"wrote {abl_path}")

    # Cross-dataset transfer table (5.6), if any cross runs are present.
    cross = collect_cross(args.results_root)
    if not cross.empty:
        cagg = aggregate_cross(cross, expected_seeds=args.expected_seeds)
        cdir = os.path.join(args.out_dir, 'cross_dataset')
        os.makedirs(cdir, exist_ok=True)
        cpath = os.path.join(cdir, 'summary.csv')
        cagg.to_csv(cpath, index=False)
        print(f"wrote {cpath} ({len(cagg)} model x setting rows)")
    elif df.empty:
        print("No cross-dataset eval results found either. Run experiments + evaluation first.")


if __name__ == "__main__":
    main()
