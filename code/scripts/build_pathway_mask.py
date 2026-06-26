"""Build the fixed pathway-gene mask A and its ablation variants for Gene2Image.

For a given dataset's gene panel (read from an AnnData ``var_names`` or an explicit
gene-name JSON), this constructs:

    A_real  [P, G]  : Hallmark (or Hallmark+Reactome) pathway membership, columns
                      aligned to the dataset gene order, pathways with < min_genes
                      hits removed.  -> Gene2Image / PathPrior / noTrans
    A_rand  [P, G]  : per-row random mask preserving each pathway's gene count
                      (same structured sparsity, biology shuffled).  -> randPath (RQ2)
    A_none  [P, G]  : all-ones matrix (no sparsity).                  -> noMask
    W_ssgsea[P, G]  : fixed (pathway, gene) weights for PathPrior (RQ3), frozen at
                      train time. --ssgsea_mode expression (default) weights genes
                      by train-set mean expression within each pathway; 'equal'
                      uses within-pathway 1/k.

The mask column order is *identical* to the dataset gene order, which is the order
the model receives at runtime (``adata.var_names`` for single / unified gene cache
for multi). This alignment is a hard correctness requirement: the model multiplies
``x[:, edge_g]`` by the learnable weights, so a misaligned column silently corrupts
every pathway token.

Outputs one ``.npz`` per variant under ``--out_dir``:
    {prefix}_{db}_real.npz   (keys: A, pathway_names, gene_names, W_ssgsea)
    {prefix}_{db}_rand.npz   (keys: A, pathway_names, gene_names)
    {prefix}_{db}_none.npz   (keys: A, pathway_names, gene_names)

Example:
    python scripts/build_pathway_mask.py \
        --adata data/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad \
        --prefix c1 --db hallmark --out_dir data/pathway_masks --seed 42
"""
import os
import json
import argparse
import numpy as np


HALLMARK_LIB = "MSigDB_Hallmark_2020"
REACTOME_LIB = "Reactome_2022"


def get_gene_names(adata_path=None, gene_json=None):
    """Return the dataset gene-name list in its canonical order.

    Prefers an explicit JSON (e.g. multi's ``unified_genes_cache.json``); otherwise
    reads ``var_names`` from the AnnData in backed mode (no expression matrix load).
    """
    if gene_json is not None:
        with open(gene_json, "r") as f:
            genes = json.load(f)
        return [str(g) for g in genes]
    if adata_path is not None:
        import anndata as ad
        a = ad.read_h5ad(adata_path, backed="r")
        genes = [str(g) for g in a.var_names]
        a.file.close()
        return genes
    raise ValueError("Provide either --adata or --gene_json.")


def parse_gmt(gmt_path):
    """Parse a local MSigDB ``.gmt`` file -> {pathway_name: [gene, ...]}.

    GMT format is one pathway per line, tab-separated:
        <pathway_name>\t<description>\t<gene1>\t<gene2>\t...
    This is the offline equivalent of ``gseapy.get_library`` and produces an
    identically-shaped dict, so no networked Enrichr call is needed. Download the
    Hallmark / Reactome ``.gmt`` once on a networked machine (e.g. from MSigDB or
    Enrichr) and ship it with the data for offline GPU servers.
    """
    library = {}
    with open(gmt_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue  # need name + description + >=1 gene
            pname = parts[0]
            # parts[1] is the description/URL field (ignored); genes follow.
            genes = [g for g in parts[2:] if g]
            if genes:
                library[pname] = genes
    if not library:
        raise RuntimeError(f"No pathways parsed from GMT file: {gmt_path}")
    return library


def load_pathway_library(db, gmt_paths=None):
    """Return pathway -> gene-set dict, offline (GMT) or online (Enrichr).

    db='hallmark' -> MSigDB Hallmark 2020 (50 pathways)
    db='hallmark_reactome' -> Hallmark + Reactome 2022 (~hundreds)

    If ``gmt_paths`` is given (one or more local ``.gmt`` files), the library is
    built offline by merging them and ``db`` is used only for output naming. This
    is the required path on a network-isolated GPU server. Otherwise the library
    is fetched from Enrichr via gseapy (needs network access).
    """
    if gmt_paths:
        lib = {}
        for p in gmt_paths:
            lib.update(parse_gmt(p))
        return lib
    import gseapy as gp
    if db == "hallmark":
        return gp.get_library(HALLMARK_LIB)
    elif db == "hallmark_reactome":
        lib = dict(gp.get_library(HALLMARK_LIB))
        lib.update(gp.get_library(REACTOME_LIB))
        return lib
    else:
        raise ValueError(f"Unknown --db: {db}")


def build_real_mask(gene_names, library, min_genes=3):
    """Build A_real [P, G] and the kept pathway-name list.

    A pathway is kept only if at least ``min_genes`` of its genes appear in the
    dataset panel. Returns (A_real int8 [P, G], pathway_names list).
    """
    gene2col = {g: i for i, g in enumerate(gene_names)}
    G = len(gene_names)

    rows = []
    pathway_names = []
    for pname, geneset in library.items():
        hits = [gene2col[g] for g in geneset if g in gene2col]
        if len(hits) >= min_genes:
            row = np.zeros(G, dtype=np.int8)
            row[hits] = 1
            rows.append(row)
            pathway_names.append(pname)

    if not rows:
        raise RuntimeError(
            f"No pathway retained (min_genes={min_genes}). "
            f"Panel size G={G}; check gene-name format matches the library symbols."
        )
    A_real = np.stack(rows, axis=0)
    return A_real, pathway_names


def make_random_mask(A_real, seed):
    """Per-row random mask with the same per-pathway gene count as A_real.

    Reproduces the 'randomized but structure-preserving' setup of
    Sparsity is All You Need: identical structured sparsity, biology shuffled.
    """
    rng = np.random.default_rng(seed)
    P, G = A_real.shape
    A_rand = np.zeros_like(A_real)
    for p in range(P):
        k = int(A_real[p].sum())
        if k > 0:
            cols = rng.choice(G, size=k, replace=False)
            A_rand[p, cols] = 1
    return A_rand


def make_none_mask(P, G):
    """All-ones mask (no structural sparsity) for the noMask ablation."""
    return np.ones((P, G), dtype=np.int8)


def get_mean_expression(adata_path, gene_names):
    """Mean expression per gene (in ``gene_names`` order) for expression-weighted ssGSEA.

    One-time offline read of the training AnnData. Returns a [G] float32 vector
    aligned to ``gene_names`` (missing genes -> 0). Used only when --ssgsea_mode
    expression is requested.
    """
    import anndata as ad
    a = ad.read_h5ad(adata_path)
    X = a.X
    mean = np.asarray(X.mean(axis=0)).ravel()  # [G_adata]; works for sparse or dense
    var2i = {str(g): i for i, g in enumerate(a.var_names)}
    out = np.array([mean[var2i[g]] if g in var2i else 0.0 for g in gene_names],
                   dtype=np.float32)
    try:
        a.file.close()
    except Exception:
        pass
    return out


def build_ssgsea_weights(A_real, mean_expr=None):
    """Fixed (pathway, gene) weights for PathPrior (RQ3); rows sum to 1.

    Two derivations (both frozen at train time, both keep tokenisation intact so
    only the 'learnable vs fixed' switch is flipped vs Gene2Image):

    - equal (mean_expr=None): within-pathway equal weighting 1/k_p. The minimal,
      data-free proxy for MUPAD's fixed scoring.
    - expression (mean_expr given): within each pathway, weight a gene by its
      train-set mean expression, normalised to sum to 1 over the pathway's genes.
      This is the more faithful ssGSEA-style derivation flagged in
      implementation.md 3.2 / dev_log, capturing which genes actually drive each
      pathway's score rather than treating all members equally.
    """
    A = A_real.astype(np.float32)
    if mean_expr is None:
        k = A.sum(axis=1, keepdims=True)  # [P, 1]
        k[k == 0] = 1.0
        W = A / k
    else:
        W = A * np.asarray(mean_expr, dtype=np.float32)[None, :]  # zero outside mask
        row = W.sum(axis=1, keepdims=True)
        zero = (row.ravel() == 0)
        if zero.any():
            # Pathways whose members all have zero mean expression fall back to equal
            # 1/k weighting so every kept pathway's weights still sum to 1.
            k = A[zero].sum(axis=1, keepdims=True)
            k[k == 0] = 1.0
            W[zero] = A[zero] / k
            row[zero] = 1.0
        W = W / row
    return W.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build pathway mask + variants.")
    parser.add_argument("--adata", default=None, help="AnnData path (reads var_names).")
    parser.add_argument("--gene_json", default=None, help="JSON list of gene names (overrides --adata).")
    parser.add_argument("--prefix", required=True, help="Output filename prefix, e.g. c1/c2/p1.")
    parser.add_argument("--db", default="hallmark", choices=["hallmark", "hallmark_reactome"])
    parser.add_argument("--gmt", nargs="+", default=None,
                        help="Local MSigDB .gmt file(s) for OFFLINE mask building "
                             "(no network). Pass one for --db hallmark, two "
                             "(hallmark + reactome) for --db hallmark_reactome. "
                             "If omitted, the library is fetched from Enrichr via "
                             "gseapy (needs network).")
    parser.add_argument("--min_genes", type=int, default=3, help="Drop pathways with fewer hits.")
    parser.add_argument("--out_dir", default="data/pathway_masks")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random mask variant.")
    parser.add_argument("--ssgsea_mode", default="expression", choices=["equal", "expression"],
                        help="PathPrior W_ssgsea derivation: 'equal' = within-pathway 1/k; "
                             "'expression' (default) = within-pathway weight proportional to "
                             "train-set mean expression (closer to real ssGSEA). 'expression' "
                             "needs --adata; falls back to 'equal' otherwise.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gene_names = get_gene_names(args.adata, args.gene_json)
    G = len(gene_names)
    print(f"[{args.prefix}] dataset panel: G={G} genes")

    library = load_pathway_library(args.db, gmt_paths=args.gmt)
    src = "GMT " + ",".join(os.path.basename(p) for p in args.gmt) if args.gmt else "Enrichr"
    print(f"[{args.prefix}] library '{args.db}' ({src}): {len(library)} pathways before filtering")

    A_real, pathway_names = build_real_mask(gene_names, library, args.min_genes)
    P = A_real.shape[0]
    per_path = A_real.sum(axis=1)
    print(f"[{args.prefix}] retained P={P} pathways (>= {args.min_genes} hits)")
    print(f"[{args.prefix}] genes per pathway: min={per_path.min()}, "
          f"median={int(np.median(per_path))}, max={per_path.max()}")
    covered = (A_real.sum(axis=0) > 0).sum()
    print(f"[{args.prefix}] genes covered by >=1 pathway: {covered}/{G} "
          f"({100*covered/G:.1f}%)")

    A_rand = make_random_mask(A_real, args.seed)
    A_none = make_none_mask(P, G)

    mean_expr = None
    if args.ssgsea_mode == "expression":
        if args.adata is not None:
            mean_expr = get_mean_expression(args.adata, gene_names)
            print(f"[{args.prefix}] W_ssgsea: expression-weighted (train-set mean expression)")
        else:
            print(f"[{args.prefix}] W_ssgsea: --ssgsea_mode expression needs --adata; "
                  f"falling back to equal (1/k) weighting")
    else:
        print(f"[{args.prefix}] W_ssgsea: equal (1/k) within-pathway weighting")
    W_ssgsea = build_ssgsea_weights(A_real, mean_expr)

    gene_names_arr = np.array(gene_names, dtype=object)
    pathway_names_arr = np.array(pathway_names, dtype=object)

    real_path = os.path.join(args.out_dir, f"{args.prefix}_{args.db}_real.npz")
    rand_path = os.path.join(args.out_dir, f"{args.prefix}_{args.db}_rand.npz")
    none_path = os.path.join(args.out_dir, f"{args.prefix}_{args.db}_none.npz")

    np.savez_compressed(real_path, A=A_real, pathway_names=pathway_names_arr,
                        gene_names=gene_names_arr, W_ssgsea=W_ssgsea)
    np.savez_compressed(rand_path, A=A_rand, pathway_names=pathway_names_arr,
                        gene_names=gene_names_arr)
    np.savez_compressed(none_path, A=A_none, pathway_names=pathway_names_arr,
                        gene_names=gene_names_arr)

    print(f"[{args.prefix}] wrote:\n  {real_path}\n  {rand_path}\n  {none_path}")
    # Sanity: column count must equal panel size (alignment guarantee).
    assert A_real.shape[1] == G == A_rand.shape[1] == A_none.shape[1], "mask G mismatch"
    print(f"[{args.prefix}] OK: mask columns aligned to G={G}")


if __name__ == "__main__":
    main()
