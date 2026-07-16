#!/usr/bin/env python3
"""Aggregate the per-run logs into the paper's 编码效率表 (encoding-efficiency table).

    python3 scripts/collect_efficiency.py --results_root results --out results/efficiency_table.csv

Why this exists: params / effective edges / per-epoch time / peak VRAM are emitted only as
MODEL_STATS and EFFICIENCY_STATS lines in each run's log (rectified_train.py). They cannot be
recovered from a checkpoint afterwards -- peak VRAM and epoch wall-time are gone the moment the job
ends -- and hand-grepping 54 logs is not a workflow. This turns them into one CSV the paper table
can be read off directly.

Columns follow v10 「编码效率表」: 有效边数 / 编码器可训练参数 / 总可训练参数 / 单epoch时间 /
峰值显存 / 单样本推理时间 -- the full table. PathPrior's frozen parameters are reported separately
from the trainable ones, as the paper requires ("避免只比较总参数造成误解"). sec_per_sample comes
from the evaluator's EVAL_GATES line and is reported next to gen_steps, since DOPRI5 is adaptive
and seconds/sample is uninterpretable without the step budget it ran under.
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

RUN_RE = re.compile(r'exp_(?P<variant>[a-zA-Z0-9]+)_(?P<ds>c1|c2|p1)_s(?P<seed>\d+)\.log$')
MODEL_RE = re.compile(r'MODEL_STATS ([^\n]+)')
EFF_RE = re.compile(r'EFFICIENCY_STATS ([^\n]+)')
EVAL_RE = re.compile(r'EVAL_GATES ([^\n]+)')   # carries sec_per_sample (单样本推理时间)
EXPECTED_EPOCHS = 50   # overridden by --expected_epochs


def kv(blob):
    """Parse 'k=v k=v' into a dict, mapping the literal string 'None' back to None.

    The emitters interpolate Python values straight into the log line, so a metric that could not be
    computed arrives as the four characters 'None'. Left as a string it is TRUTHY, which would make
    the "this run has no timing" check below silently pass and put a bogus 'None' in the table --
    exactly the class of silent gap this collector exists to surface.
    """
    out = {}
    for tok in blob.split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            out[k] = None if v in ('None', 'nan', 'NaN', '') else v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results_root', default='results')
    ap.add_argument('--out', default=None, help='default: <results_root>/efficiency_table.csv')
    ap.add_argument('--expected_epochs', type=int, default=50,
                    help='Flag runs whose timed epochs do not sum to this (default 50).')
    args = ap.parse_args()

    global EXPECTED_EPOCHS
    EXPECTED_EPOCHS = args.expected_epochs
    logs_dir = os.path.join(args.results_root, 'logs')
    out_path = args.out or os.path.join(args.results_root, 'efficiency_table.csv')
    if not os.path.isdir(logs_dir):
        print(f'no logs dir at {logs_dir} -- the array appends per-run logs there; '
              f'nothing to aggregate', file=sys.stderr)
        return 2

    rows = []
    for fn in sorted(os.listdir(logs_dir)):
        m = RUN_RE.search(fn)
        if not m:
            continue
        with open(os.path.join(logs_dir, fn), errors='ignore') as f:
            text = f.read()
        rec = {'variant': m.group('variant'), 'dataset': m.group('ds'), 'seed': int(m.group('seed'))}
        mm = MODEL_RE.search(text)
        if mm:
            g = kv(mm.group(1))
            rec['encoder_trainable_params'] = g.get('encoder_trainable_params')
            rec['total_trainable_params'] = g.get('total_trainable_params')
            rec['frozen_params'] = g.get('frozen_params')
            # -1 means "no pathway mask" (the geneflow arm); keep it distinguishable from 0.
            rec['effective_edges'] = g.get('effective_edges')
        # AGGREGATE across attempts -- do NOT take the last one. rectified_train.py resets
        # _epoch_times=[] and calls reset_peak_memory_stats() before its epoch loop, so each
        # EFFICIENCY_STATS line covers only THAT attempt's epochs. Requeue is the design here
        # (--time=24h vs c1 ~33h / p1 ~44h), so ~36 of 54 runs carry two or more lines, and taking
        # the last would report: epochs_timed=20 instead of 50; a peak VRAM that is the tail
        # segment's max rather than the run's (a maximum, so this systematically UNDER-states it and
        # prints a peak that never happened); and a mean epoch time skewed high by the short tail
        # segment's checkpoint reload and cold caches.
        me = list(EFF_RE.finditer(text))
        gs = [kv(m.group(1)) for m in me]
        gs = [g for g in gs if g.get('epochs_timed') and g.get('mean_epoch_sec')]
        if gs:
            ns = [int(g['epochs_timed']) for g in gs]
            # Mean weighted by each segment's epoch count, so segments of unequal length combine
            # into the run's true per-epoch mean.
            rec['mean_epoch_sec'] = f"{sum(float(g['mean_epoch_sec']) * n for g, n in zip(gs, ns)) / sum(ns):.1f}"
            vrams = [float(g['peak_vram_gb']) for g in gs if g.get('peak_vram_gb')]
            rec['peak_vram_gb'] = f"{max(vrams):.2f}" if vrams else None
            rec['epochs_timed'] = str(sum(ns))
            rec['n_attempts'] = str(len(gs))
        # Last EVAL_GATES wins, and that IS correct here -- unlike training, evaluation is not
        # resumable: a requeued run re-runs it whole, so the final line is the complete one. Only
        # the training-side accumulators are segmented (see the aggregation above).
        mv = list(EVAL_RE.finditer(text))
        if mv:
            g = kv(mv[-1].group(1))
            rec['sec_per_sample'] = g.get('sec_per_sample')
            rec['gen_samples_timed'] = g.get('gen_samples_timed')
            rec['gen_steps'] = g.get('gen_steps')   # DOPRI5 is adaptive: sec/sample needs its budget
        rows.append(rec)

    if not rows:
        print(f'no exp_*.log with MODEL_STATS under {logs_dir} -- is the server running a build '
              f'that predates the instrumentation?', file=sys.stderr)
        return 2

    cols = ['variant', 'dataset', 'seed', 'effective_edges', 'encoder_trainable_params',
            'total_trainable_params', 'frozen_params', 'mean_epoch_sec', 'peak_vram_gb',
            'epochs_timed', 'n_attempts', 'sec_per_sample', 'gen_steps', 'gen_samples_timed']
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})
    print(f'wrote {out_path} ({len(rows)} run(s))')

    # A row whose timed epochs do not add up to the budget covers only part of the run: its
    # mean/peak describe a fragment, not the run the paper's table names. Flag loudly.
    partial = [f"{r['variant']}/{r['dataset']}/s{r['seed']} ({r.get('epochs_timed')}ep)" for r in rows
               if r.get('epochs_timed') and int(r['epochs_timed']) != EXPECTED_EPOCHS]
    if partial:
        print(f'\nWARNING: {len(partial)} run(s) have EFFICIENCY_STATS covering != {EXPECTED_EPOCHS} '
              f'epochs -- their mean/peak describe only part of the run and must NOT go in the '
              f'table as-is: {", ".join(partial[:8])}' + (' ...' if len(partial) > 8 else ''))

    missing = [f"{r['variant']}/{r['dataset']}/s{r['seed']}" for r in rows
               if not r.get('mean_epoch_sec')]
    if missing:
        print(f'\nWARNING: {len(missing)} run(s) have no EFFICIENCY_STATS (training may not have '
              f'finished, or the run predates the instrumentation): {", ".join(missing[:8])}'
              + (' ...' if len(missing) > 8 else ''))

    # Per (variant, dataset) means, which is the shape the paper table wants.
    agg = defaultdict(list)
    for r in rows:
        if r.get('mean_epoch_sec'):
            agg[(r['variant'], r['dataset'])].append(float(r['mean_epoch_sec']))
    if agg:
        print('\nmean epoch seconds by (variant, dataset):')
        for (v, d), xs in sorted(agg.items()):
            print(f'  {v:<12} {d:<3} {sum(xs)/len(xs):8.1f}s  (n={len(xs)})')

    no_infer = [f"{r['variant']}/{r['dataset']}/s{r['seed']}" for r in rows
                if not r.get('sec_per_sample')]
    if no_infer:
        print(f'\nNOTE: {len(no_infer)} run(s) have no sec_per_sample -- evaluation has not run yet, '
              f'or the run predates the timer: {", ".join(no_infer[:8])}'
              + (' ...' if len(no_infer) > 8 else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
