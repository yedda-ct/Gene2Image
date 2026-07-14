"""RQ4 pathway interpretability: three sub-analyses for a trained Gene2Image model.

A  Endogeneity   : extract CLS->pathway attention per cell, measure attention
                   entropy (focus) and cell-type / pathway specificity.
B  Biology       : overlap of the model's top pathways with an external reference
                   (GeneFlow gene-importance GSEA, or a known marker pathway list).
C  Sensitivity   : intervene on pathway tokens (ablation by default) at inference and
                   measure the morphological shift; dominant vs random specificity.
                   A conditioning-sensitivity readout, NOT biological causal evidence.

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
def load_model(model_path, gene_dim, device, gene_names=None):
    """Rebuild a single-cell Gene2Image model from a checkpoint's saved config."""
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = ck.get('config', {})
    if cfg.get('encoder_type') != 'pathway':
        raise ValueError("pathway_interpret expects a pathway (Gene2Image) checkpoint.")
    if cfg.get('model_type', 'single') != 'single':
        raise ValueError(
            "pathway_interpret only supports single-cell pathway checkpoints "
            f"(model_type='single'); got model_type='{cfg.get('model_type')}'.")
    # Hard gene-order guard (mirrors the train-side check in rectified_main.py and
    # rectified_evaluate.py). The mask columns and every x[:, i] index are aligned to
    # the TRAINING gene order stored in the checkpoint. load_state_dict only checks
    # tensor shapes, so a same-length but reordered panel (a re-saved adata, a
    # different layer) would load silently and every attention / intervention result
    # (RQ4) would be computed on misaligned genes.
    ckpt_gene_names = cfg.get('gene_names')
    if ckpt_gene_names and gene_names is not None:
        ck_g = [str(g) for g in ckpt_gene_names]
        cur_g = [str(g) for g in gene_names]
        if ck_g != cur_g:
            n_mis = sum(1 for a, b in zip(ck_g, cur_g) if a != b)
            first = next((i for i, (a, b) in enumerate(zip(ck_g, cur_g)) if a != b), 'NA')
            raise ValueError(
                f"Interpret gene order does not match the checkpoint's training gene "
                f"order: len(ckpt)={len(ck_g)} vs len(eval)={len(cur_g)}, {n_mis} "
                f"positions differ (first at index {first}). RQ4 outputs would be "
                f"computed on misaligned conditioning; align the panel to the training "
                f"adata / layer before interpreting.")
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
def analysis_A(model, loader, pathway_names, device, out_dir, cell_types=None,
               max_batches=None, topk=5, min_cells=10):
    """CLS->pathway attention: entropy (focus), global dominant pathways, and --
    when cell types are available -- per-cell-type dominant pathways plus the
    cross-cell-type Jaccard specificity the plan (Part 3 §2.4-A) asks for.

    Args:
        cell_types: optional {cell_id -> cell_type} map; enables the per-cell-type block.
        max_batches: cap on batches (None = all cells; attention is a cheap forward pass).
        topk: number of dominant pathways per cell type for the Jaccard specificity.
        min_cells: cell types with fewer cells are excluded from the specificity metric.
    """
    rows = []
    P = model.rna_encoder.embed.P
    pnames = pathway_names or [f"pathway_{i}" for i in range(P)]
    attn_sum = np.zeros(P, dtype=np.float64)   # accumulate per-pathway attention
    ct_sum = {}    # cell_type -> np.array[P] summed attention
    ct_count = {}  # cell_type -> #cells
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
            cid = cell_ids[i]
            ct = cell_types.get(str(cid)) if cell_types else None
            rows.append({'cell_id': cid, 'cell_type': ct, 'entropy': float(e[i]),
                         'top_pathway': pnames[top], 'top_attention': float(a[i, top])})
            if ct is not None:
                if ct not in ct_sum:
                    ct_sum[ct] = np.zeros(P, dtype=np.float64)
                    ct_count[ct] = 0
                ct_sum[ct] += a[i]
                ct_count[ct] += 1
        n += 1
        if max_batches and n >= max_batches:
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

    # --- Per-cell-type dominant pathways + cross-type Jaccard specificity (RQ4-A) ---
    # For each cell type, the mean attention profile -> its top-k dominant pathways.
    # Specificity = mean pairwise Jaccard DISTANCE between cell types' top-k sets
    # (1 - |A n B|/|A u B|); higher => cell types focus on more distinct pathways,
    # evidence that the model learned cell-type-specific pathway->morphology mapping.
    per_ct, topk_sets = {}, {}
    for ct, cnt in ct_count.items():
        if cnt < min_cells:
            continue
        m = ct_sum[ct] / cnt
        order = list(np.argsort(m)[::-1][:topk])
        top_names = [pnames[p] for p in order]
        topk_sets[ct] = set(top_names)
        per_ct[ct] = {
            'n_cells': int(cnt),
            'mean_entropy': float(df.loc[df['cell_type'] == ct, 'entropy'].mean()),
            f'top{topk}_pathways': top_names,
            f'top{topk}_attention': [float(m[p]) for p in order],
        }
    if per_ct:
        # Per-cell-type dominant pathways are emitted whenever >=1 type qualifies (a
        # plan deliverable in its own right). The cross-type Jaccard specificity is
        # gated separately below, since it is only defined for >=2 types.
        cts = sorted(per_ct)
        summary['topk'] = int(topk)
        summary['min_cells_per_celltype'] = int(min_cells)
        summary['n_celltypes_analyzed'] = len(cts)
        summary['per_celltype_dominant'] = per_ct
        ct_rows = []
        for ct in cts:
            d = per_ct[ct]
            for rank, (pw, at) in enumerate(zip(d[f'top{topk}_pathways'],
                                                d[f'top{topk}_attention']), 1):
                ct_rows.append({'cell_type': ct, 'n_cells': d['n_cells'], 'rank': rank,
                                'pathway': pw, 'mean_attention': at,
                                'mean_entropy': d['mean_entropy']})
        pd.DataFrame(ct_rows).to_csv(os.path.join(out_dir, 'attention_by_celltype.csv'),
                                     index=False)
        if len(topk_sets) >= 2:
            dists = []
            for i in range(len(cts)):
                for j in range(i + 1, len(cts)):
                    s1, s2 = topk_sets[cts[i]], topk_sets[cts[j]]
                    jac = len(s1 & s2) / max(1, len(s1 | s2))
                    dists.append(1.0 - jac)
            summary['celltype_specificity_jaccard_distance'] = float(np.mean(dists))
            print(f"[A] per-cell-type: {len(cts)} types (>= {min_cells} cells) | top-{topk} "
                  f"Jaccard distance {summary['celltype_specificity_jaccard_distance']:.3f} "
                  f"(higher => more cell-type-specific pathway focus)")
        else:
            print(f"[A] per-cell-type: 1 type (>= {min_cells} cells); dominant pathways "
                  f"written, but Jaccard specificity needs >= 2 types.")
    elif cell_types:
        print(f"[A] per-cell-type skipped: no cell type reached >= {min_cells} cells.")
    else:
        print("[A] per-cell-type analysis skipped (no cell types; pass --cell_type_key).")

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
# C: intervention sensitivity (conditioning-sensitivity, NOT causal)
# ---------------------------------------------------------------------------
@torch.no_grad()
def analysis_C(model, loader, pathway_names, device, out_dir, gen_steps=50,
               n_cells=128, k_pathways=5, mode='ablate', seed=0, gen_batch=16):
    """Pathway-token INTERVENTION SENSITIVITY of the trained model (RQ4-C).

    Design (reduces circularity; still a sensitivity readout, NOT causal evidence):
      * Gather n_cells cells and split into a DISJOINT selection subset and intervention-
        measurement subset. Dominant pathways are chosen by CLS->pathway attention on the
        SELECTION subset; the intervention is applied and measured ONLY on the disjoint
        subset -- so pathways are not selected and validated on the same cells. NOTE: this
        measurement subset is disjoint from selection but is NOT guaranteed held out from
        model training (the loader has no train/test split).
      * Control = k pathways sampled UNIFORMLY AT RANDOM from all pathways (excluding the
        dominant ones), NOT the attention bottom-half, so the contrast is dominant-vs-random
        rather than high-attention-vs-low-attention.
      * Baseline and every intervention share the SAME per-cell initial noise, so the shift
        reflects the intervention rather than noise variance.
      * Only the token-ZEROING (ablation) intervention runs from the default entry point
        (amplification is available via mode='amplify' but is not run by default).
    Shift = 4-channel (RGB+DAPI) pixel-space L2 per cell, averaged over cells. Specificity
    ratio = mean(dominant shift) / mean(random shift) (with a bootstrap 95% CI); >1 means the
    model's self-identified dominant pathways drive morphology more than random ones.
    """
    rf = RectifiedFlow()
    P = model.rna_encoder.embed.P
    pnames = pathway_names or [f"pathway_{i}" for i in range(P)]

    # Gather up to n_cells cells (+ their ids) across batches, then split disjointly.
    genes, cids, got = [], [], 0
    for batch in loader:
        genes.append(batch['gene_expr'])
        cids.extend(batch.get('cell_id', [None] * batch['gene_expr'].shape[0]))
        got += batch['gene_expr'].shape[0]
        if got >= n_cells:
            break
    if not genes:
        print("[C] skipped: loader yielded no cells.")
        return None
    gene_all = torch.cat(genes, dim=0)[:n_cells].to(device)
    cids = cids[:gene_all.shape[0]]
    n = gene_all.shape[0]
    if n < 4:
        print(f"[C] skipped: only {n} cell(s) available (need >= 4 for a select/measure split).")
        return None
    n_sel = n // 2
    sel_gene = gene_all[:n_sel]              # select dominant pathways on this subset ...
    val_gene = gene_all[n_sel:]             # ... and measure the intervention on this DISJOINT subset
    val_cids = [str(c) for c in cids[n_sel:]]

    # Dominant pathways from attention on the SELECTION subset only (no circularity).
    attn_sel = model.rna_encoder.get_pathway_attention(sel_gene).mean(dim=0)  # [P]
    order = attn_sel.argsort(descending=True).cpu().numpy()
    k = min(k_pathways, max(1, P // 3))
    dominant = order[:k].tolist()
    # Control: uniformly random pathways EXCLUDING the dominant ones (not the attn bottom-half).
    rng = np.random.default_rng(seed)
    pool = [p for p in range(P) if p not in set(dominant)]
    random_p = rng.choice(pool, size=min(k, len(pool)), replace=False).tolist() if pool else []
    if not dominant or not random_p:
        print(f"[C] skipped: need >=2 pathways and interv_k>=1 for a dominant-vs-random control "
              f"(P={P}, k={k}, |pool|={len(pool)}).")
        return None

    # Baseline + interventions on the measurement subset, sharing per-cell initial noise.
    _noise_ids = [f"interv_val_{i}" for i in range(val_gene.shape[0])]

    _gb = max(1, gen_batch)   # guard: --batch_size 0/negative must not empty/reverse the slices

    def _gen():
        # Chunk the measurement cells so intervention generation does not OOM (the data
        # loader batch_size only bounds data READING, not this generation call).
        outs = []
        for s in range(0, val_gene.shape[0], _gb):
            outs.append(generate_images_with_rectified_flow(
                model=model, rectified_flow=rf, gene_expr=val_gene[s:s + _gb],
                device=device, num_steps=gen_steps, is_multi_cell=False,
                sample_ids=_noise_ids[s:s + _gb]))
        return torch.cat(outs, dim=0)

    base = _gen()
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
    # Bootstrap a 95% CI for the specificity ratio by resampling the per-pathway shift lists
    # (post-hoc, no extra generation). A single fixed random-control draw is a known
    # limitation; the CI at least quantifies the spread across the chosen pathways.
    _bs = np.random.default_rng(seed + 1)
    _ratios = [float(np.mean(_bs.choice(dom_shifts, len(dom_shifts), replace=True))
                     / max(1e-8, np.mean(_bs.choice(rnd_shifts, len(rnd_shifts), replace=True))))
               for _ in range(2000)]
    ci_lo, ci_hi = float(np.percentile(_ratios, 2.5)), float(np.percentile(_ratios, 97.5))

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'intervention.csv'), index=False)
    with open(os.path.join(out_dir, 'C_intervention_sensitivity.json'), 'w') as f:
        json.dump({'mode': mode,
                   'specificity_ratio': float(spec),
                   'specificity_ratio_ci95': [ci_lo, ci_hi],
                   'mean_dominant_shift': float(np.mean(dom_shifts)),
                   'mean_random_shift': float(np.mean(rnd_shifts)),
                   'dominant_pathways': [pnames[p] for p in dominant],
                   'random_control_pathways': [pnames[p] for p in random_p],
                   'measurement_cell_ids': val_cids,
                   'n_cells_total': int(n), 'n_selection': int(n_sel),
                   'n_measurement': int(val_gene.shape[0]), 'k_pathways': int(k),
                   'seed': int(seed), 'gen_steps': int(gen_steps),
                   'distance': '4-channel (RGB+DAPI) pixel-space L2 per cell, mean over cells',
                   'selection_measurement': 'disjoint subsets (measurement NOT held out from training)',
                   'control': 'uniform_random_excluding_dominant (single seed-fixed draw)',
                   'note': 'conditional-sensitivity readout of the trained model, '
                           'NOT biological causal evidence'}, f, indent=2)
    print(f"[C] specificity ratio (dominant/random {mode}) = {spec:.2f} "
          f"[95% CI {ci_lo:.2f}-{ci_hi:.2f}] on {val_gene.shape[0]} measurement cells "
          f"(dominant selected on {n_sel} DISJOINT cells; random control excl. dominant). "
          f">1 => dominant pathways drive morphology more. NOTE: measurement cells are "
          f"disjoint from SELECTION, not held out from training; this is sensitivity, not causal.")
    return df


def _read_cell_types(adata_path, index, cell_type_key=None):
    """Return {cell_id -> cell_type(str)} aligned to ``index``, or None.

    Reads only ``adata.obs`` (backed mode, so X is not loaded twice) to support the
    per-cell-type RQ4-A analysis. If ``--cell_type_key`` is given it is used (and a
    missing column is a hard error); otherwise a common set of obs column names is
    auto-detected. Returns None (with a printed note) when no cell-type column is
    available, so the analysis degrades gracefully to global-only.
    """
    try:
        import anndata as ad
        try:
            obs = ad.read_h5ad(adata_path, backed='r').obs.copy()
        except Exception:
            obs = ad.read_h5ad(adata_path).obs
    except Exception as e:
        print(f"[A] could not read adata.obs for cell types ({e}); "
              f"per-cell-type analysis skipped.")
        return None
    if cell_type_key:
        if cell_type_key not in obs.columns:
            raise ValueError(f"--cell_type_key '{cell_type_key}' not in adata.obs "
                             f"(available: {list(obs.columns)})")
        key = cell_type_key
    else:
        candidates = ['cell_type', 'celltype', 'cell_types', 'Cell_Type', 'CellType',
                      'cell_type_label', 'annotation', 'annotations', 'cell_annotation',
                      'leiden', 'louvain', 'cluster', 'clusters', 'predicted_labels']
        key = next((c for c in candidates if c in obs.columns), None)
        if key is None:
            print(f"[A] no cell-type column found in adata.obs (looked for {candidates}; "
                  f"available: {list(obs.columns)}). Pass --cell_type_key <col> to enable "
                  f"per-cell-type analysis; running global-only.")
            return None
    # Drop NaN / unassigned entries so unlabeled cells fall OUT of the per-cell-type
    # block instead of forming a spurious 'nan' pseudo-type that would bias the
    # Jaccard specificity (curated cell-type columns commonly have unassigned cells).
    _IGNORE = {'', 'nan', 'none', 'na', 'n/a', '<na>', 'unassigned', 'unknown',
               'unlabeled', 'unlabelled', 'undetermined'}
    mapping = {}
    for cid, ct in obs[key].dropna().items():
        s = str(ct).strip()
        if s.lower() in _IGNORE:
            continue
        mapping[str(cid)] = s
    out = {str(c): mapping[str(c)] for c in index if str(c) in mapping}
    if out:
        print(f"[A] cell types from obs['{key}']: {len(set(out.values()))} types "
              f"over {len(out)} cells")
    return out or None


def build_loader(args, device):
    """Minimal single-cell loader mirroring rectified_main's single path.

    Also returns a {cell_id -> cell_type} map (or None) for the per-cell-type RQ4-A
    analysis; cell types come from adata.obs (see _read_cell_types).
    """
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
    cell_types = _read_cell_types(args.adata, list(expr_df.index),
                                  cell_type_key=getattr(args, 'cell_type_key', None))
    return loader, expr_df.shape[1], list(expr_df.columns), cell_types


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
    ap.add_argument('--cell_type_key', default=None,
                    help='adata.obs column holding cell type for the per-cell-type '
                         'RQ4-A analysis. Default: auto-detect common names; if none '
                         'found, per-cell-type analysis is skipped (global-only).')
    ap.add_argument('--attn_max_batches', type=int, default=0,
                    help='Cap on batches for sub-analysis A (0 = all cells; attention '
                         'is a cheap forward pass).')
    ap.add_argument('--topk_pathways', type=int, default=5,
                    help='Top-k dominant pathways per cell type for the Jaccard specificity.')
    ap.add_argument('--min_cells_per_celltype', type=int, default=10,
                    help='Cell types with fewer cells are excluded from the specificity metric.')
    ap.add_argument('--interv_cells', type=int, default=128,
                    help='RQ4-C: total cells gathered for the token-intervention test, split '
                         'into DISJOINT selection/validation halves (>= 4; larger = stronger).')
    ap.add_argument('--interv_k', type=int, default=5,
                    help='RQ4-C: number of dominant AND of random-control pathways.')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader, gene_dim, gene_names, cell_types = build_loader(args, device)
    model, pathway_names, mask = load_model(args.model_path, gene_dim, device, gene_names=gene_names)
    if pathway_names is None:
        pathway_names = [f"pathway_{i}" for i in range(model.rna_encoder.embed.P)]

    _A_kw = dict(cell_types=cell_types, max_batches=(args.attn_max_batches or None),
                 topk=args.topk_pathways, min_cells=args.min_cells_per_celltype)
    attn_df = None
    pathway_scores = None
    if 'A' in args.analysis:
        attn_df, _, pathway_scores = analysis_A(model, loader, pathway_names, device, args.out_dir, **_A_kw)
    if 'B' in args.analysis:
        if pathway_scores is None:
            attn_df, _, pathway_scores = analysis_A(model, loader, pathway_names, device, args.out_dir, **_A_kw)
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
        analysis_C(model, loader, pathway_names, device, args.out_dir,
                   gen_steps=args.gen_steps, n_cells=args.interv_cells,
                   k_pathways=args.interv_k, gen_batch=args.batch_size)


if __name__ == "__main__":
    main()
