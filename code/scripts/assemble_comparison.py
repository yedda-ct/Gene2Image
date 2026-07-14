#!/usr/bin/env python
"""Assemble qualitative comparison grids from per-cell PNGs produced by rectified_generate.py.

rectified_generate.py writes, per run, <out>/<variant>/generated_images/<cell_id>_real_rgb.png
and <cell_id>_gen_rgb.png. Because every variant was generated with the SAME dataset + --seed,
they share the same held-out test cells (same cell_ids) and the SAME per-cell initial noise, so
the columns align cell-by-cell and differences reflect the encoder, not the cell or the noise.

Produces two figures (rows = cells, columns = Real + each model):
  main_comparison.png/pdf      : Real | GeneFlow | Gene2Image
  ablation_comparison.png/pdf  : Real | Gene2Image | randPath | PathPrior | noTrans | noMask
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

PRETTY = {'geneflow': 'GeneFlow', 'gene2image': 'Gene2Image', 'randpath': 'randPath',
          'pathprior': 'PathPrior', 'notrans': 'noTrans', 'nomask': 'noMask'}
_SUF = '_gen_rgb.png'


def _ids(qual_root, variant):
    d = os.path.join(qual_root, variant, 'generated_images')
    return sorted(os.path.basename(f)[:-len(_SUF)]
                  for f in glob.glob(os.path.join(d, '*' + _SUF)))


def _img(qual_root, variant, cid, kind):
    return mpimg.imread(os.path.join(qual_root, variant, 'generated_images', f'{cid}_{kind}_rgb.png'))


def build_grid(qual_root, variants, cell_ids, out_path, title):
    ncol, nrow = 1 + len(variants), len(cell_ids)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 1.9 * nrow), squeeze=False)
    col_labels = ['Real'] + [PRETTY.get(v, v) for v in variants]
    for r, cid in enumerate(cell_ids):
        # Real GT is identical across variants -> read it from the first variant's dir.
        panels = [_img(qual_root, variants[0], cid, 'real')] + \
                 [_img(qual_root, v, cid, 'gen') for v in variants]
        for c, im in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(np.clip(im[:, :, :3], 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_labels[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(str(cid), fontsize=6, rotation=0, ha='right', va='center')
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    fig.savefig(out_path[:-4] + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path} (+ .pdf)  [{nrow} cells x {ncol} cols]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qual_root', required=True, help='dir holding <variant>/generated_images/ subdirs')
    ap.add_argument('--main', nargs='+', default=['geneflow', 'gene2image'])
    ap.add_argument('--ablation', nargs='+', default=['gene2image', 'randpath', 'pathprior', 'notrans', 'nomask'])
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--n_show', type=int, default=8, help='number of cells (rows) to show')
    args = ap.parse_args()
    out_dir = args.out_dir or args.qual_root
    os.makedirs(out_dir, exist_ok=True)

    all_variants = list(dict.fromkeys(args.main + args.ablation))
    id_sets = {v: set(_ids(args.qual_root, v)) for v in all_variants}
    for v in all_variants:
        if not id_sets[v]:
            raise SystemExit(f"No *_gen_rgb.png under {args.qual_root}/{v}/generated_images -- "
                             f"run rectified_generate.py for '{v}' first (same DS+seed).")
    # Cells present in EVERY variant (guaranteed aligned), deterministic order.
    common = sorted(set.intersection(*id_sets.values()))
    if not common:
        raise SystemExit("No cell_ids common to all variants -- were they generated with the SAME --seed?")
    cells = common[:args.n_show]
    print(f"[assemble] {len(common)} common cells; showing {len(cells)}.")
    build_grid(args.qual_root, args.main, cells, os.path.join(out_dir, 'main_comparison.png'),
               'Main: Real vs generated (same held-out test cells, paired noise)')
    build_grid(args.qual_root, args.ablation, cells, os.path.join(out_dir, 'ablation_comparison.png'),
               'Ablation: Real vs generated (same held-out test cells, paired noise)')


if __name__ == '__main__':
    main()
