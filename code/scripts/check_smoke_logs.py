#!/usr/bin/env python3
"""Audit smoke/debug SLURM logs and say GO / NO-GO, one line per job.

Works on the logs ALONE — no server access, no results_smoke/ needed. Hand it whatever
`logs/` tarball came back from the cluster:

    python3 scripts/check_smoke_logs.py /path/to/logs
    python3 scripts/check_smoke_logs.py /path/to/logs --expect gene2image/c2 randpath/c1

What each check is for
----------------------
uni2h_loaded / uni2h_fid
    The previous 54-run batch was thrown away because UNI2-h biological FID was NaN on
    all 54 runs while every job still exited 0. This check catches it from the log alone, which
    is also useful when the server still runs an older validate_runs.py that only NaN-checked
    overall_fid (the current one checks the UNI2-h FID too).
dopri5
    Every DOPRI5_DIAGNOSTICS line must read dt_floor=0, under_integration_fallback=0,
    final_t=1.0000. dt_floor>0 means steps were force-accepted over tolerance;
    fallback>0 means the solver never reached t=1 and the image is under-integrated.
gate3 (n_ssim_used == n_psnr_used == total_samples)
    Read from the EVAL_GATES line the evaluator emits. A shortfall means rows were dropped as
    non-finite and the reported SSIM/PSNR means silently cover fewer samples than claimed.
    ("Dropped ... non-finite" warnings are also flagged, as a belt-and-braces cross-check.)
epochs
    Read from the TRAIN_GATES line: realized epochs vs budget, and the resume point.
traceback / pass_line
    The run actually finished rather than dying or being silently skipped.

Exit code is 0 only if every job passes, so this is usable as a gate in a script.
"""
import argparse
import os
import re
import sys

RE_HEADER = re.compile(r'>>>\s+(SMOKE|DEBUG)\s+(\S+)\s+(\S+)\s+seed(\d+)')
RE_UNI2H_FID = re.compile(r'UNI2-H FID \(overall, full-set\):\s*([0-9.eE+-]+|nan|NaN)')
RE_INCEPT_FID = re.compile(r'Inception FID \(overall, full-set\):\s*([0-9.eE+-]+|nan|NaN)')
RE_DOPRI5 = re.compile(
    r'DOPRI5_DIAGNOSTICS rejected=(\d+) dt_floor=(\d+) '
    r'under_integration_fallback=(\d+) final_t=([0-9.]+)')
RE_PASS = re.compile(r'\[PASS\]\s+(\S+)/(\S+)/seed(\d+)')
RE_SPLIT = re.compile(r'Split \(seed=\d+\): train=(\d+), val=(\d+).*?test=(\d+)')
# Machine-readable gate lines (rectified_evaluate.py / rectified_main.py). These carry the values
# that otherwise live only in evaluation_summary.json / training_losses.csv, so a run is auditable
# from the log alone.
RE_EVAL_GATES = re.compile(r'EVAL_GATES ([^\n]+)')
RE_TRAIN_GATES = re.compile(r'TRAIN_GATES ([^\n]+)')
# Efficiency table inputs (paper 编码效率表). Params / effective edges / per-epoch time / peak VRAM
# exist ONLY during the run, so surface them here too -- discovering after 1800 GPU-hours that a
# table has no source is the expensive failure this checker exists to prevent.
RE_MODEL_STATS = re.compile(r'MODEL_STATS ([^\n]+)')
RE_EFF_STATS = re.compile(r'EFFICIENCY_STATS ([^\n]+)')


def _kv(blob):
    """Parse 'k=v k=v' into a dict, mapping the literal 'None'/'nan' back to None.

    The emitters interpolate Python values straight into the log line, so a metric that could not be
    computed arrives as the characters 'None'. Left as a string it is TRUTHY, so presence checks
    would pass on a value that does not exist.
    """
    out = {}
    for tok in blob.split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            out[k] = None if v in ('None', 'nan', 'NaN', '') else v
    return out


def read(path):
    if not os.path.exists(path):
        return ''
    with open(path, errors='replace') as fh:
        return fh.read()


def audit(out_text, err_text):
    """Return (label, list_of_problems, list_of_facts) for one job."""
    both = out_text + err_text
    problems, facts = [], []

    m = RE_HEADER.search(out_text)
    label = f'{m.group(2)}/{m.group(3)}/seed{m.group(4)}' if m else '<unknown>'
    kind = m.group(1) if m else '?'

    # --- the run finished at all -------------------------------------------------
    n_tb = both.count('Traceback (most recent call last)')
    if n_tb:
        problems.append(f'{n_tb} Traceback(s)')
    # Only SMOKE runs validate_runs.py and therefore emits [PASS]; capella_debug_1gpu.slurm
    # does not, so requiring the line there would fail every debug job for no reason.
    if kind == 'SMOKE' and m:
        hits = [p for l in out_text.splitlines() for p in [RE_PASS.match(l.strip())] if p]
        if any(f'{p.group(1)}/{p.group(2)}/seed{p.group(3)}' == label for p in hits):
            facts.append('gate=PASS')
        else:
            problems.append('no [PASS] line for this run (never finished, or gate not reached)')

    # --- UNI2-h: the failure that invalidated the last batch ----------------------
    if 'UNI2-h model loaded successfully' not in err_text:
        problems.append('UNI2-h model NOT loaded (check UNI2H_MODEL_PATH)')
    mu = RE_UNI2H_FID.search(err_text)
    if not mu:
        problems.append('no UNI2-H FID line — biological validation never ran')
    else:
        raw = mu.group(1)
        if raw.lower() == 'nan':
            problems.append('UNI2-H FID = NaN — biological FID unusable (this is what '
                            'invalidated the previous 54-run batch)')
        else:
            facts.append(f'uni2h_fid={float(raw):.2f}')
    mi = RE_INCEPT_FID.search(err_text)
    if mi and mi.group(1).lower() != 'nan':
        facts.append(f'incept_fid={float(mi.group(1)):.2f}')

    # --- DOPRI5 solver gates ------------------------------------------------------
    ds = RE_DOPRI5.findall(err_text)
    if not ds:
        problems.append('no DOPRI5_DIAGNOSTICS lines — generation never ran')
    else:
        bad_floor = sum(1 for d in ds if int(d[1]) > 0)
        bad_fb = sum(1 for d in ds if int(d[2]) > 0)
        bad_t = sum(1 for d in ds if abs(float(d[3]) - 1.0) > 1e-6)
        if bad_floor:
            problems.append(f'{bad_floor}/{len(ds)} DOPRI5 lines forced steps at dt-floor')
        if bad_fb:
            problems.append(f'{bad_fb}/{len(ds)} DOPRI5 lines hit under-integration fallback')
        if bad_t:
            problems.append(f'{bad_t}/{len(ds)} DOPRI5 lines did not reach final_t=1.0')
        if not (bad_floor or bad_fb or bad_t):
            facts.append(f'dopri5={len(ds)}/{len(ds)} clean')

    # --- gate 3: every evaluated sample counted toward the reported means ----------
    me = RE_EVAL_GATES.search(err_text)
    if me:
        g = _kv(me.group(1))
        try:
            tot, ns, npr = int(g['total_samples']), int(g['n_ssim_used']), int(g['n_psnr_used'])
            if ns != tot or npr != tot:
                problems.append(f'non-finite samples dropped: n_ssim_used={ns}, n_psnr_used={npr}, '
                                f'total_samples={tot} -> reported means cover fewer samples '
                                f'than the run claims')
            else:
                facts.append(f'gate3 {ns}/{tot} used')
            # A tile that failed to decode is replaced by a BLACK image and scored as ground truth,
            # entering the FID reference stats -- while staying finite, so the count above cannot
            # see it. The evaluator reports it on this same line; check it here too.
            _zs = g.get('zero_image_substitutions')
            if _zs is None:
                # Absent = the run predates the counter. Report it as unverified rather than let a
                # quiet skip read as clean: a black tile scored as ground truth is finite, so
                # nothing else here can catch it.
                facts.append('zero-tile UNVERIFIED (no counter in this build)')
            elif _zs not in ('0', '-1'):
                problems.append(f'{_zs} unreadable tile(s) replaced by BLACK images and scored as '
                                f'ground truth -> FID/SSIM/PSNR contaminated')
        except (KeyError, ValueError, TypeError):
            # TypeError matters: _kv maps the literal 'None' to None, so int(None) lands here.
            # A gate line whose counts are None means the evaluator could not compute them --
            # report it rather than crash the whole audit on one malformed run.
            problems.append(f'EVAL_GATES counts missing/unparseable ({me.group(1)[:70]}) -> cannot '
                            f'verify that every sample was counted')
    else:
        # Degrade gracefully instead of failing: logs produced before EVAL_GATES existed (i.e. by a
        # server that has not synced this package yet) are still worth auditing. The weaker but
        # still-valid signal is the ABSENCE of the evaluator's drop warning — nothing dropped means
        # the means used every sample. Flag it as a note so a stale server does not read as a
        # broken run, but say plainly that the check is weaker.
        facts.append('gate3 via drop-warning only (no EVAL_GATES; server predates it)')

    # The evaluator warns whenever it drops rows. This is the fallback signal above, and a
    # cross-check when EVAL_GATES is present.
    if re.search(r'Dropped .*non-finite', err_text):
        for line in err_text.splitlines():
            if re.search(r'Dropped .*non-finite', line):
                problems.append('dropped non-finite samples: ' + line.split(' - ')[-1].strip())

    # --- training reach (from the log, not training_losses.csv) --------------------
    mt = RE_TRAIN_GATES.search(err_text)
    if mt:
        g = _kv(mt.group(1))
        facts.append(f"ep={g.get('last_epoch','?')}/{g.get('budget','?')}")
        if g.get('resumed_from_epoch', '0') != '0':
            facts.append(f"resumed@{g['resumed_from_epoch']}")

    # --- efficiency table inputs (not a gate; report so gaps surface before the 54-run) ------
    mm = RE_MODEL_STATS.search(err_text)
    if mm:
        g = _kv(mm.group(1))
        facts.append(f"params={g.get('total_trainable_params','?')}")
        if g.get('effective_edges', '-1') != '-1':
            facts.append(f"edges={g['effective_edges']}")
    else:
        facts.append('no MODEL_STATS (efficiency table inputs unavailable)')
    mf = RE_EFF_STATS.search(err_text)
    if mf:
        g = _kv(mf.group(1))
        facts.append(f"epoch={g.get('mean_epoch_sec','?')}s")
        facts.append(f"vram={g.get('peak_vram_gb','?')}G")

    ms = RE_SPLIT.search(err_text)
    if ms:
        facts.append(f'split={ms.group(1)}/{ms.group(2)}/{ms.group(3)}')

    return label, kind, problems, facts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logs_dir', help='directory holding g2i_*-<jobid>.{out,err}')
    ap.add_argument('--expect', nargs='*', default=[], metavar='VARIANT/DS',
                    help='runs that MUST appear, e.g. gene2image/c2 randpath/c1')
    args = ap.parse_args()

    errs = []
    for root, _, files in os.walk(args.logs_dir):
        errs += [os.path.join(root, f) for f in files if f.endswith('.err')]
    if not errs:
        print(f'no *.err under {args.logs_dir}', file=sys.stderr)
        return 2

    results, seen = [], set()
    for err_path in sorted(errs):
        out_path = err_path[:-4] + '.out'
        label, kind, problems, facts = audit(read(out_path), read(err_path))
        job = os.path.basename(err_path)[:-4]
        results.append((job, label, kind, problems, facts))
        if label != '<unknown>':
            seen.add('/'.join(label.split('/')[:2]))

    width = max(len(r[1]) for r in results) + 2
    print(f'\n{"JOB":<22}{"RUN":<{width}}{"VERDICT":<8}FACTS / PROBLEMS')
    print('-' * 100)
    n_fail = 0
    for job, label, kind, problems, facts in results:
        ok = not problems
        n_fail += 0 if ok else 1
        print(f'{job:<22}{label:<{width}}{"PASS" if ok else "FAIL":<8}{", ".join(facts)}')
        for p in problems:
            print(f'{"":<22}{"":<{width}}{"":<8}!! {p}')

    missing = [e for e in args.expect if e not in seen]
    print('-' * 100)
    if missing:
        print(f'MISSING: {len(missing)} expected run(s) have no log at all: {", ".join(missing)}')
    print(f'{len(results)} job(s): {len(results) - n_fail} PASS, {n_fail} FAIL'
          + (f', {len(missing)} MISSING' if missing else ''))
    print(f'\nCOVERAGE (variant/dataset pairs with a log): {", ".join(sorted(seen)) or "none"}')

    if n_fail or missing:
        print('\nVERDICT: NO-GO — fix the above before launching the 54-run.')
        return 1
    print('\nVERDICT: GO — every audited job is clean.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
