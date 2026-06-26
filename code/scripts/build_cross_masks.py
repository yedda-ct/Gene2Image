"""Build pathway-name-aligned mask pairs for cross-dataset generalization (2.3).

Gene2Image transfers across panels through the shared *pathway* space, not the
gene space. For a (source, target) pair the two masks must therefore expose the
SAME pathway rows in the SAME order (so the P x D_token token sequence is
comparable), while each keeps its own gene columns (panels differ).

This script takes two single-dataset masks built by build_pathway_mask.py,
intersects their pathway-name sets, and rewrites both to that shared, identically
ordered pathway list. Gene columns are untouched (already aligned to each
dataset's own gene order). Output: {src}_to_{tgt}_src.npz / _tgt.npz.

Why intersection (not union): a pathway present in only one panel would be an
all-zero row on the other side, contributing nothing but noise. The 50 Hallmark
pathways are near-universal, so the intersection is large in practice.

Example:
    python scripts/build_cross_masks.py \
        --src data/pathway_masks/c1_hallmark_real.npz \
        --tgt data/pathway_masks/p1_hallmark_real.npz \
        --src_name c1 --tgt_name p1 --out_dir data/pathway_masks
"""
import os
import argparse
import numpy as np


def realign(npz, keep_pathways):
    """Return A restricted/reordered to keep_pathways (list of names)."""
    A = npz['A']
    names = list(npz['pathway_names'])
    name2row = {n: i for i, n in enumerate(names)}
    rows = [A[name2row[p]] for p in keep_pathways]
    return np.stack(rows, axis=0)


def main():
    ap = argparse.ArgumentParser(description="Build pathway-aligned cross-dataset mask pair.")
    ap.add_argument("--src", required=True, help="Source dataset mask .npz (real).")
    ap.add_argument("--tgt", required=True, help="Target dataset mask .npz (real).")
    ap.add_argument("--src_name", required=True, help="Short id for source, e.g. c1.")
    ap.add_argument("--tgt_name", required=True, help="Short id for target, e.g. p1.")
    ap.add_argument("--out_dir", default="data/pathway_masks")
    args = ap.parse_args()

    src = np.load(args.src, allow_pickle=True)
    tgt = np.load(args.tgt, allow_pickle=True)

    src_paths = list(src['pathway_names'])
    tgt_paths = list(tgt['pathway_names'])
    # Intersection, ordered by the source's pathway order for determinism.
    shared = [p for p in src_paths if p in set(tgt_paths)]
    if not shared:
        raise RuntimeError("No shared pathways between source and target masks.")
    print(f"source P={len(src_paths)}, target P={len(tgt_paths)}, shared P={len(shared)}")

    A_src = realign(src, shared)
    A_tgt = realign(tgt, shared)
    shared_arr = np.array(shared, dtype=object)

    os.makedirs(args.out_dir, exist_ok=True)
    src_out = os.path.join(args.out_dir, f"{args.src_name}_to_{args.tgt_name}_src.npz")
    tgt_out = os.path.join(args.out_dir, f"{args.src_name}_to_{args.tgt_name}_tgt.npz")

    # Carry frozen ssGSEA weights for the source only (training uses source mask).
    src_kwargs = dict(A=A_src, pathway_names=shared_arr, gene_names=src['gene_names'])
    if 'W_ssgsea' in src.files:
        # Realign W_ssgsea rows to the shared pathway order too.
        names = list(src['pathway_names'])
        name2row = {n: i for i, n in enumerate(names)}
        W = src['W_ssgsea']
        src_kwargs['W_ssgsea'] = np.stack([W[name2row[p]] for p in shared], axis=0)

    np.savez_compressed(src_out, **src_kwargs)
    np.savez_compressed(tgt_out, A=A_tgt, pathway_names=shared_arr, gene_names=tgt['gene_names'])

    print(f"wrote:\n  {src_out}  A{A_src.shape}\n  {tgt_out}  A{A_tgt.shape}")
    assert A_src.shape[0] == A_tgt.shape[0] == len(shared), "pathway row mismatch"
    print(f"OK: both masks expose {len(shared)} aligned pathway rows")


if __name__ == "__main__":
    main()
