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
        for src_key, name in METRIC_KEYS.items():
            v = data.get(src_key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                rec[name] = v
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate(df):
    """Mean +/- std across seeds for each (variant, dataset)."""
    if df.empty:
        return df
    present = [m for m in METRIC_KEYS.values() if m in df.columns]
    g = df.groupby(['variant', 'dataset'])[present].agg(['mean', 'std'])
    g.columns = [f'{m}_{stat}' for m, stat in g.columns]
    return g.reset_index()


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
    for r in rows:
        r['degradation_rate'] = ((r['fid_cross'] - r['fid_same']) / r['fid_same']
                                 if r['fid_same'] else float('nan'))
    return pd.DataFrame(rows)


def aggregate_cross(df):
    """Mean +/- std of fid_cross/fid_same/degradation_rate per (model, setting)."""
    if df.empty:
        return df
    df = df.copy()
    df['setting'] = df['src'].str.upper() + '->' + df['tgt'].str.upper()
    g = df.groupby(['model', 'setting'])[['fid_cross', 'fid_same', 'degradation_rate']].agg(['mean', 'std'])
    g.columns = [f'{m}_{stat}' for m, stat in g.columns]
    return g.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', default='results')
    ap.add_argument('--out_dir', default='results')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = collect(args.results_root)
    if df.empty:
        print(f"No main evaluation_summary.json found under {args.results_root}.")
    else:
        print(f"Collected {len(df)} main eval runs.")
        summary = aggregate(df)
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
        cagg = aggregate_cross(cross)
        cdir = os.path.join(args.out_dir, 'cross_dataset')
        os.makedirs(cdir, exist_ok=True)
        cpath = os.path.join(cdir, 'summary.csv')
        cagg.to_csv(cpath, index=False)
        print(f"wrote {cpath} ({len(cagg)} model x setting rows)")
    elif df.empty:
        print("No cross-dataset eval results found either. Run experiments + evaluation first.")


if __name__ == "__main__":
    main()
