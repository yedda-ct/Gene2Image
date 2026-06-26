"""RQ4 pathway interpretability: three sub-analyses for a trained Gene2Image model.

A  Endogeneity   : extract CLS->pathway attention per cell, measure attention
                   entropy (focus) and cell-type / pathway specificity.
B  Biology       : overlap of the model's top pathways with an external reference
                   (GeneFlow gene-importance GSEA, or a known marker pathway list).
C  Causality     : intervene on pathway tokens (ablate / amplify) at inference and
                   measure the morphological shift; dominant vs random specificity.

Loads a single-cell Gene2Image checkpoint (encoder_type='pathway') and its dataset.
Outputs CSV/JSON under --out_dir (see implementation.md 5.7). UNI2-h embedding
distance in sub-analysis C is optional; if UNI2-h is unavailable it falls back to a
pixel-space L2 on generated RGB so the script still runs end to end.

Usage:
    python analysis/pathway_interpret.py \
        --model_path results/gene2image_c1_seed42/checkpoints/best_checkpoint.pt \
        --adata data/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad \
        --image_paths .../cell_image_paths_local.json \
        --out_dir results/interpret/c1 [--analysis A B C]
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.single_model import RNAtoHnEModel
from rectified.rectified_flow import RectifiedFlow
from rectified.utils import generate_images_with_rectified_flow


# ---------------------------------------------------------------------------
# model / data loading
# ---------------------------------------------------------------------------
def load_model(model_path, gene_dim, device):
    """Rebuild a single-cell Gene2Image model from a checkpoint's saved config."""
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = ck.get('config', {})
    if cfg.get('encoder_type') != 'pathway':
        raise ValueError("pathway_interpret expects a pathway (Gene2Image) checkpoint.")
    if cfg.get('model_type', 'single') != 'single':
        raise ValueError(
            "pathway_interpret only supports single-cell pathway checkpoints "
            f"(model_type='single'); got model_type='{cfg.get('model_type')}'.")
    mask = cfg['pathway_mask_array']
    mask = mask if torch.is_tensor(mask) else torch.tensor(np.asarray(mask))
    mask = mask.to(torch.float32)
    model = RNAtoHnEModel(
        rna_dim=gene_dim, img_channels=cfg.get('img_channels', 4),
        img_size=cfg.get('img_size', 256), model_channels=128, num_res_blocks=2,
        attention_resolutions=(16,), channel_mult=(1, 2, 2, 2),
        num_heads=2, num_head_channels=16,
        encoder_type='pathway', pathway_mask=mask,
        d_token=cfg.get('d_token', 48), pt_layers=cfg.get('pt_layers', 2),
        pt_heads=cfg.get('pt_heads', 8),
        learnable_pathway=cfg.get('learnable_pathway', True),
        use_pathway_transformer=cfg.get('use_pathway_transformer', True),
    )
    state = ck.get('model_state_dict', ck.get('model'))
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    pathway_names = list(cfg.get('pathway_names') or []) or None
    return model, pathway_names, mask  # mask: [P, G] tensor, columns = gene order


# ---------------------------------------------------------------------------
# A: endogeneity
# ---------------------------------------------------------------------------
@torch.no_grad()
def analysis_A(model, loader, pathway_names, device, out_dir, max_batches=50):
    """CLS->pathway attention, entropy and per-cell-type dominant pathways."""
    rows = []
    P = model.rna_encoder.embed.P
    pnames = pathway_names or [f"pathway_{i}" for i in range(P)]
    attn_sum = np.zeros(P, dtype=np.float64)   # accumulate per-pathway attention
    n_cells_total = 0
    n = 0
    for batch in loader:
        gene = batch['gene_expr'].to(device)
        attn = model.rna_encoder.get_pathway_attention(gene)  # [B, P]
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
        ent = -(attn * (attn + 1e-12).log()).sum(dim=1)        # [B]
        cell_ids = batch.get('cell_id', [None] * gene.shape[0])
        a = attn.cpu().numpy()
        e = ent.cpu().numpy()
        attn_sum += a.sum(axis=0)
        n_cells_total += a.shape[0]
        for i in range(a.shape[0]):
            top = int(a[i].argmax())
            rows.append({'cell_id': cell_ids[i], 'entropy': float(e[i]),
                         'top_pathway': pnames[top], 'top_attention': float(a[i, top])})
        n += 1
        if n >= max_batches:
            break

    # Per-pathway mean attention across cells = the model's pathway-importance
    # profile (used by sub-analysis B for top-k overlap and Spearman).
    mean_attn = attn_sum / max(1, n_cells_total)
    pathway_scores = {pnames[p]: float(mean_attn[p]) for p in range(P)}

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'attention.csv'), index=False)
    summary = {
        'n_cells': len(df),
        'P': P,
        'uniform_entropy': float(np.log(P)),  # max entropy reference
        'mean_entropy': float(df['entropy'].mean()),
        'dominant_pathways': df['top_pathway'].value_counts().head(10).to_dict(),
        'mean_attention': pathway_scores,
    }
    with open(os.path.join(out_dir, 'A_endogeneity.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[A] {len(df)} cells | mean entropy {summary['mean_entropy']:.3f} "
          f"(uniform {summary['uniform_entropy']:.3f}); lower => more focused")
    return df, summary, pathway_scores


# ---------------------------------------------------------------------------
# B: biology consistency
# ---------------------------------------------------------------------------
def _pathway_gene_sets(pathway_names, mask, gene_names):
    """Build {pathway_name: [member gene names]} from the model's own fixed mask.

    The mask columns are aligned to ``gene_names`` (build_pathway_mask guarantee),
    so this recovers exactly the pathway memberships the model was trained on.
    """
    A = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.asarray(mask)
    A = (A != 0)
    G = len(gene_names)
    sets = {}
    for p, pname in enumerate(pathway_names):
        cols = np.nonzero(A[p])[0]
        genes = [gene_names[c] for c in cols if c < G]
        if genes:
            sets[str(pname)] = genes
    return sets


def _gsea_pathway_ranking(gene_importance_csv, gene_sets, seed=0):
    """Rank the model's pathways by GSEA enrichment of GeneFlow gene importance.

    Runs a pre-ranked GSEA of GeneFlow's ``gene_importance_scores.csv`` against the
    model's own Hallmark pathway sets (passed in-memory, so this stays offline), and
    returns {pathway_name: enrichment_score (NES)} plus the method used. If gseapy is
    unavailable or prerank fails (e.g. too few ranked genes), falls back to a
    rank-based enrichment = mean importance-rank percentile of each pathway's genes.
    """
    imp = pd.read_csv(gene_importance_csv)
    gcol = 'gene_name' if 'gene_name' in imp.columns else imp.columns[0]
    scol = 'importance_score' if 'importance_score' in imp.columns else imp.columns[1]
    rnk = imp[[gcol, scol]].dropna()
    rnk = rnk.sort_values(scol, ascending=False).reset_index(drop=True)

    try:
        import gseapy as gp
        pre = gp.prerank(rnk=rnk, gene_sets=gene_sets, min_size=3, max_size=100000,
                         permutation_num=100, seed=seed, no_plot=True, outdir=None,
                         verbose=False)
        res = pre.res2d
        term_col = 'Term' if 'Term' in res.columns else res.columns[0]
        nes_col = 'NES' if 'NES' in res.columns else ('nes' if 'nes' in res.columns else None)
        if nes_col is not None:
            scores = {str(t): float(v) for t, v in zip(res[term_col], res[nes_col])}
            scores = {k: v for k, v in scores.items() if not np.isnan(v)}
            if scores:
                return scores, 'gsea_prerank_NES'
    except Exception as e:  # pragma: no cover - depends on gseapy/runtime
        print(f"[B] gseapy.prerank unavailable/failed ({e}); using rank-based fallback.")

    n = len(rnk)
    rank_pct = {g: 1.0 - (i / max(1, n - 1)) for i, g in enumerate(rnk[gcol].astype(str))}
    scores = {}
    for pname, genes in gene_sets.items():
        vals = [rank_pct[g] for g in genes if g in rank_pct]
        if vals:
            scores[pname] = float(np.mean(vals))
    return scores, 'mean_rank_fallback'


def analysis_B(model_scores, gene_importance_csv, gene_sets, out_dir, k=10,
               marker_pathways=None, seed=0):
    """Consistency of model-dominant pathways with GeneFlow-importance GSEA (RQ4 B).

    Compares the model's pathway ranking (mean CLS->pathway attention) against the
    GSEA-enriched ranking of GeneFlow's gene importance, via (1) top-k overlap and
    (2) Spearman rank correlation over the shared pathways. Optionally also reports
    overlap with a curated marker-pathway list (--reference_pathways). Output:
    gsea_consistency.json (implementation.md 5.7).
    """
    from scipy.stats import spearmanr

    gsea_scores, method = _gsea_pathway_ranking(gene_importance_csv, gene_sets, seed=seed)
    model_rank = sorted(model_scores, key=model_scores.get, reverse=True)
    gsea_rank = sorted(gsea_scores, key=gsea_scores.get, reverse=True)
    topk_model, topk_gsea = model_rank[:k], gsea_rank[:k]
    overlap = len(set(topk_model) & set(topk_gsea)) / max(1, k)

    shared = [p for p in model_scores if p in gsea_scores]
    if len(shared) >= 3:
        rho, pval = spearmanr([model_scores[p] for p in shared],
                              [gsea_scores[p] for p in shared])
        rho, pval = (float(rho), float(pval)) if not np.isnan(rho) else (None, None)
    else:
        rho, pval = None, None

    result = {'k': k, 'method': method, 'n_pathways_scored': len(gsea_scores),
              'n_shared': len(shared), 'topk_overlap': overlap,
              'spearman': rho, 'spearman_p': pval,
              'topk_model': topk_model, 'topk_gsea': topk_gsea}
    if marker_pathways:
        result['marker_topk_overlap'] = (
            len(set(topk_model) & set(str(m) for m in marker_pathways[:k])) / max(1, k))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'gsea_consistency.json'), 'w') as f:
        json.dump(result, f, indent=2)
    rho_str = f"{rho:.3f}" if rho is not None else "n/a"
    print(f"[B] GSEA consistency ({method}): top-{k} overlap={overlap:.2f}, "
          f"Spearman={rho_str} over {len(shared)} shared pathways")
    return result


# ---------------------------------------------------------------------------
# C: causal intervention
# ---------------------------------------------------------------------------
@torch.no_grad()
def analysis_C(model, loader, pathway_names, device, out_dir, gen_steps=50,
               n_cells=8, mode='ablate'):
    """Intervene on pathway tokens and measure morphological shift.

    For each of the top dominant pathways and a set of random pathways, zero
    (ablate) or amplify that pathway's token before generation, and measure the
    pixel-space L2 between baseline and intervened generations. The specificity
    ratio = mean(dominant shift) / mean(random shift); >1 means dominant pathways
    causally drive morphology more than irrelevant ones.
    """
    rf = RectifiedFlow()
    P = model.rna_encoder.embed.P
    pnames = pathway_names or [f"pathway_{i}" for i in range(P)]

    batch = next(iter(loader))
    gene = batch['gene_expr'][:n_cells].to(device)

    # Baseline and every intervention must start from the SAME initial noise; otherwise
    # (out - base) is dominated by random-noise variance rather than the pathway
    # intervention, making the specificity ratio meaningless. The DOPRI5 sampler draws
    # its initial noise via torch.randn with no seed, so reset the RNG to a fixed seed
    # immediately before each generation to share one noise draw across all of them.
    _NOISE_SEED = 0

    def _gen():
        torch.manual_seed(_NOISE_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_NOISE_SEED)
        return generate_images_with_rectified_flow(
            model=model, rectified_flow=rf, gene_expr=gene, device=device,
            num_steps=gen_steps, is_multi_cell=False)

    # Baseline generation (shared initial noise).
    base = _gen()

    # Identify dominant pathways from attention, and sample random ones.
    attn = model.rna_encoder.get_pathway_attention(gene).mean(dim=0)  # [P]
    order = attn.argsort(descending=True).cpu().numpy()
    dominant = order[:3].tolist()
    rng = np.random.default_rng(0)
    random_p = rng.choice(order[len(order) // 2:], size=min(3, P // 2), replace=False).tolist()

    embed = model.rna_encoder.embed
    rows = []

    def gen_with_intervention(pidx):
        # Patch the embedding forward so pathway pidx's token is zeroed/amplified.
        orig_forward = embed.forward

        def patched(x):
            T = orig_forward(x)            # [N, P, d]
            if mode == 'ablate':
                T[:, pidx, :] = 0.0
            else:                          # amplify
                T[:, pidx, :] = T[:, pidx, :] * 3.0
            return T

        embed.forward = patched
        try:
            out = _gen()   # same shared initial noise as base
        finally:
            embed.forward = orig_forward
        return out

    def shift(pidx, group):
        out = gen_with_intervention(pidx)
        d = torch.norm((out - base).flatten(1), dim=1).mean().item()
        rows.append({'pathway': pnames[pidx], 'group': group,
                     'intervention': mode, 'morph_shift': d})
        return d

    dom_shifts = [shift(p, 'dominant') for p in dominant]
    rnd_shifts = [shift(p, 'random') for p in random_p]
    spec = (np.mean(dom_shifts) / max(1e-8, np.mean(rnd_shifts)))

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'intervention.csv'), index=False)
    with open(os.path.join(out_dir, 'C_causal.json'), 'w') as f:
        json.dump({'mode': mode, 'specificity_ratio': float(spec),
                   'mean_dominant_shift': float(np.mean(dom_shifts)),
                   'mean_random_shift': float(np.mean(rnd_shifts))}, f, indent=2)
    print(f"[C] specificity ratio (dominant/random {mode}): {spec:.2f} "
          f"(>1 => dominant pathways drive morphology)")
    return df


def build_loader(args, device):
    """Minimal single-cell loader mirroring rectified_main's single path."""
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from src.dataset import CellImageGeneDataset
    from src.utils import parse_adata

    expr_df, missing = parse_adata(adata=args.adata, missing_gene_symbols=None)
    with open(args.image_paths) as f:
        image_paths = json.load(f)
    image_paths = {k: v for k, v in image_paths.items() if os.path.exists(v)}
    ds = CellImageGeneDataset(
        expr_df, image_paths, img_size=args.img_size, img_channels=args.img_channels,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((args.img_size, args.img_size), antialias=True)]),
        missing_gene_symbols=missing)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_dataloader_workers)
    return loader, expr_df.shape[1], list(expr_df.columns)


def main():
    ap = argparse.ArgumentParser(description="Gene2Image pathway interpretability (RQ4).")
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--adata', required=True)
    ap.add_argument('--image_paths', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--analysis', nargs='+', default=['A', 'B', 'C'], choices=['A', 'B', 'C'])
    ap.add_argument('--reference_pathways', default=None,
                    help='Optional JSON list of curated marker pathway names; reported as '
                         'an extra marker overlap in sub-analysis B.')
    ap.add_argument('--geneflow_importance', default=None,
                    help='GeneFlow gene_importance_scores.csv (gene_name,importance_score) '
                         'for the GSEA consistency test in sub-analysis B.')
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--img_channels', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_dataloader_workers', type=int, default=2)
    ap.add_argument('--gen_steps', type=int, default=50)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader, gene_dim, gene_names = build_loader(args, device)
    model, pathway_names, mask = load_model(args.model_path, gene_dim, device)
    if pathway_names is None:
        pathway_names = [f"pathway_{i}" for i in range(model.rna_encoder.embed.P)]

    attn_df = None
    pathway_scores = None
    if 'A' in args.analysis:
        attn_df, _, pathway_scores = analysis_A(model, loader, pathway_names, device, args.out_dir)
    if 'B' in args.analysis:
        if pathway_scores is None:
            attn_df, _, pathway_scores = analysis_A(model, loader, pathway_names, device, args.out_dir)
        marker = None
        if args.reference_pathways and os.path.exists(args.reference_pathways):
            with open(args.reference_pathways) as f:
                marker = json.load(f)
        if args.geneflow_importance and os.path.exists(args.geneflow_importance):
            gene_sets = _pathway_gene_sets(pathway_names, mask, gene_names)
            analysis_B(pathway_scores, args.geneflow_importance, gene_sets,
                       args.out_dir, marker_pathways=marker)
        else:
            print("[B] skipped: pass --geneflow_importance <gene_importance_scores.csv> "
                  "(from a GeneFlow/single run) to run the GSEA consistency test; "
                  "the model's pathways and gene importances are compared via GSEA "
                  "enrichment (top-k overlap + Spearman), not against itself.")
    if 'C' in args.analysis:
        analysis_C(model, loader, pathway_names, device, args.out_dir, gen_steps=args.gen_steps)


if __name__ == "__main__":
    main()
