import os
import sys
import time
import json
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import linalg
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import inception_v3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sequoia_root = os.path.join(project_root, 'sequoia')
# sys.path.insert(0, sequoia_root)
# sys.path.insert(1, project_root)

from rectified.rectified_flow import RectifiedFlow
from src.utils import setup_parser, parse_adata
# Deprecated model classes are no longer shipped; import them only if present so
# the legacy fallback path still works, but a missing module never breaks eval.
try:
    from src.single_model_deprecation import RNAtoHnEModel as RNAtoHnEModel_deprecation
    from src.multi_model_deprecation import MultiCellRNAtoHnEModel as MultiCellRNAtoHnEModel_deprecation
except ImportError:
    RNAtoHnEModel_deprecation = None
    MultiCellRNAtoHnEModel_deprecation = None
from src.single_model import RNAtoHnEModel
from src.multi_model import MultiCellRNAtoHnEModel, prepare_multicell_batch
from src.dataset import (
    CellImageGeneDataset, PatchImageGeneDataset, patch_collate_fn,
    load_preprocessed_hest1k_singlecell_data,
    OnDemandMultiSampleHestXeniumDataset, multi_sample_hest_xenium_collate_fn,
    FastSeparatePatchDataset, fast_separate_patch_collate_fn)
# from rectified.rectified_train_ddp import generate_images_with_rectified_flow
from src.stain_normalization import normalize_staining_rgb_skimage_hist_match
from rectified.utils import generate_images_with_rectified_flow
# UNI2-h is an OPTIONAL pathology-FM metric. Its module pulls heavy deps (cv2, timm,
# sklearn); guard the import so a missing dep degrades UNI2-h FID to N/A instead of
# crashing the whole evaluation (basic FID/SSIM/PSNR are unaffected). Downstream calls
# are already gated on `uni2h_model is not None`.
try:
    from rectified.utils_uni2h import (
        load_uni2_h_model, extract_uni2_h_embeddings, extended_biological_evaluation_uni2h,
        calculate_uni2h_fid
    )
except Exception as _uni2h_import_err:
    load_uni2_h_model = extract_uni2_h_embeddings = extended_biological_evaluation_uni2h = calculate_uni2h_fid = None
    logging.getLogger(__name__).warning(
        f"UNI2-h eval unavailable (import failed: {_uni2h_import_err}); "
        f"UNI2-h metrics -> N/A. Basic FID/SSIM/PSNR unaffected.")
from rectified.utils_he2rna import *
from rectified.utils_plot import save_all_evaluation_plots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def setup_ddp():
    """Initialize DDP environment"""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    return rank, local_rank, world_size, device


def cleanup_ddp():
    """Cleanup DDP environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_logging(rank):
    """Setup logging for DDP - only rank 0 logs to avoid spam"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    return logging.getLogger(__name__)


def gather_list_across_ranks(data_list, world_size):
    """Gather lists from all ranks"""
    if world_size == 1:
        return data_list
    
    # Convert to tensor for gathering
    if len(data_list) == 0:
        return []
    
    # Gather all data
    gathered_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_data, data_list)
    
    # Flatten the gathered data
    all_data = []
    for rank_data in gathered_data:
        all_data.extend(rank_data)
    
    return all_data

class InceptionModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.inception_model = inception_v3(weights='IMAGENET1K_V1', transform_input=False).to(device)
        self.inception_model.eval()
        self.inception_model.fc = torch.nn.Identity()
        for param in self.inception_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3, :, :]

        if x.shape[2] != 299 or x.shape[3] != 299:
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        x = (x * 2) - 1
        x = self.inception_model(x)
        return x


def calculate_fid(real_features, gen_features):
    if real_features.shape[0] < 2 or gen_features.shape[0] < 2:
        return np.nan

    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    
    if np.isnan(sigma1).any() or np.isnan(sigma2).any() or np.isinf(sigma1).any() or np.isinf(sigma2).any():
        return np.nan

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    try:
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    except Exception:
        return np.nan
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid if fid >= 0 and not np.isnan(fid) and not np.isinf(fid) else np.nan


def calculate_image_metrics(real_images_batch, generated_images_batch):
    batch_size = real_images_batch.shape[0]
    ssim_scores = []
    psnr_scores = []
    
    for i in range(batch_size):
        real_img_np = real_images_batch[i].cpu().numpy().transpose(1, 2, 0)
        gen_img_np = generated_images_batch[i].cpu().numpy().transpose(1, 2, 0)
        
        real_img_np = np.clip(real_img_np, 0, 1)
        gen_img_np = np.clip(gen_img_np, 0, 1)

        real_img_rgb = real_img_np[:,:,:3]
        gen_img_rgb = gen_img_np[:,:,:3]

        data_range = 1.0 

        current_ssim = ssim(
            real_img_rgb, 
            gen_img_rgb, 
            channel_axis=2,
            data_range=data_range
        )
        
        current_psnr = psnr(
            real_img_rgb, 
            gen_img_rgb, 
            data_range=data_range
        )
        
        ssim_scores.append(current_ssim)
        psnr_scores.append(current_psnr)
    
    return ssim_scores, psnr_scores


def load_checkpoint_ddp(filename, model, device):
    """Load checkpoint and handle DDP model state dict"""
    checkpoint = torch.load(filename, map_location=device)
    
    model_state_dict = checkpoint.get("model", checkpoint)
    
    # Handle DDP model state dict (remove 'module.' prefix if present)
    if isinstance(model, DDP):
        # If loading into DDP model, ensure state dict matches
        if not any(key.startswith('module.') for key in model_state_dict.keys()):
            model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
    else:
        # If loading into non-DDP model, remove 'module.' prefix
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    
    model.load_state_dict(model_state_dict)
    return checkpoint


def transplant_cross_dataset_weights(model, source_state, ckpt_config, target_mask_path):
    """Transfer a source-trained pathway encoder to a target panel by name (2.3).

    Gene2Image transfers across panels through the shared *pathway* space: the
    Pathway Transformer / CLS head / UNet are panel-agnostic and load 1:1, while the
    embedding's per-(pathway, gene) weight ``W`` is panel-specific (its edges index
    the source gene columns). This rebuilds the embedding on the TARGET mask's edges
    and copies each source-learned weight onto the matching target edge by
    (pathway-name, gene-name); target-only edges keep their default init. Per-pathway
    bias transfers by pathway name. ``model`` must already be built with the target
    mask. Returns a small stats dict.
    """
    tgt = np.load(target_mask_path, allow_pickle=True)
    tgt_pnames = [str(p) for p in tgt['pathway_names']]
    tgt_gnames = [str(g) for g in tgt['gene_names']]

    src_A = ckpt_config.get('pathway_mask_array')
    src_A = src_A.cpu().numpy() if torch.is_tensor(src_A) else np.asarray(src_A)
    src_A = (src_A != 0)
    src_pnames = [str(p) for p in (ckpt_config.get('pathway_names') or [])]
    src_gnames = [str(g) for g in (ckpt_config.get('gene_names') or [])]
    if not src_pnames or not src_gnames:
        raise ValueError(
            "Cross-dataset eval needs source 'pathway_names' and 'gene_names' in the "
            "checkpoint config. Re-train with the current rectified_main.py (it now "
            "stores gene_names) or run a same-panel eval instead.")

    # Move source weights onto the target model's current device before indexed
    # assignment into new_W/new_bias, so everything stays on one device.
    _dev = model.rna_encoder.embed.W.device
    src_W = source_state['rna_encoder.embed.W'].to(_dev)      # [E_src, d_token]
    src_bias = source_state['rna_encoder.embed.bias'].to(_dev)  # [P_src, d_token]
    # Source edge order is row-major nonzero of src_A == mask.nonzero() at build time.
    src_edges = np.argwhere(src_A)
    src_edge_index = {(int(p), int(g)): i for i, (p, g) in enumerate(src_edges)}
    src_p2i = {n: i for i, n in enumerate(src_pnames)}
    src_g2i = {n: i for i, n in enumerate(src_gnames)}

    emb = model.rna_encoder.embed
    tgt_edge_p = emb.edge_p.cpu().numpy()
    tgt_edge_g = emb.edge_g.cpu().numpy()

    new_W = emb.W.detach().clone()        # default init retained for unmatched edges
    n_matched = 0
    for i in range(tgt_edge_p.shape[0]):
        pname = tgt_pnames[int(tgt_edge_p[i])]
        gname = tgt_gnames[int(tgt_edge_g[i])]
        sp, sg = src_p2i.get(pname), src_g2i.get(gname)
        if sp is None or sg is None:
            continue
        j = src_edge_index.get((sp, sg))
        if j is None:
            continue
        new_W[i] = src_W[j]
        n_matched += 1
    emb.W.data.copy_(new_W)

    new_bias = emb.bias.detach().clone()
    for p, pname in enumerate(tgt_pnames):
        sp = src_p2i.get(pname)
        if sp is not None and sp < src_bias.shape[0]:
            new_bias[p] = src_bias[sp]
    emb.bias.data.copy_(new_bias)

    # Load everything except the embedding (target buffers + transplanted W/bias stay).
    rest = {k: v for k, v in source_state.items() if not k.startswith('rna_encoder.embed.')}
    model.load_state_dict(rest, strict=False)
    return {'n_target_edges': int(tgt_edge_p.shape[0]), 'n_matched': int(n_matched),
            'P': len(tgt_pnames)}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RNA to H&E model with Rectified Flow (DDP-enabled)."
    )
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model checkpoint (.pt or .pth).')
    parser.add_argument('--batch_size', type=int, default=20, help='Per-GPU batch size for evaluation.')
    parser.add_argument('--max_samples', type=int, default=5000, help='Maximum number of samples to use from the dataset.')
    parser.add_argument('--use_ddp', action='store_true', help='Use Distributed Data Parallel evaluation.')
    parser.add_argument('--uni2h_model_path', type=str,
                        default=os.environ.get('UNI2H_MODEL_PATH', '/depot/natallah/data/Mengbo/HnE_RNA/GeneFlow/UNI2-h'),
                        help='Directory holding the UNI2-h foundation model pytorch_model.bin. '
                             'Defaults to $UNI2H_MODEL_PATH, else the legacy authors-cluster path. '
                             'Required for the biological UNI2-h FID metric; if the path is missing, '
                             'UNI2-h FID is skipped and reported as NaN (basic FID/SSIM/PSNR unaffected).')
    parser.add_argument('--he2rna_model_path', type=str,
                        default=os.environ.get('HE2RNA_MODEL_PATH', '/depot/natallah/data/Mengbo/HnE_RNA/GeneFlow/sequoia/models/he2rna-skcm-0'),
                        help='Path to pretrained HE2RNA model for RNA prediction validation. Defaults to $HE2RNA_MODEL_PATH.')
    parser.add_argument('--save_embeddings', action='store_true', default=True, help='Save UNI2-h embeddings for later analysis')
    parser.add_argument('--embeddings_output_path', type=str, default=None, help='Path to save embeddings (if None, saves to output_dir/embeddings)')
    # Debug/smoke: mirror rectified_main.py's --debug subset so eval reconstructs the SAME
    # 80/10/10 split as a --debug training run (else eval splits the FULL dataset and its
    # "test" chunk overlaps the debug training cells). Pass the SAME --debug_samples as training.
    parser.add_argument('--debug', action='store_true', help='Subset the dataset to --debug_samples before the split (mirror a --debug training run).')
    parser.add_argument('--debug_samples', type=int, default=1000, help='Number of samples in --debug mode (must match the training run).')

    parser = setup_parser(parser)
    args = parser.parse_args()

    # Initialize DDP if requested
    if args.use_ddp:
        rank, local_rank, world_size, device = setup_ddp()
        # Adjust batch size for DDP
        original_batch_size = args.batch_size
        args.batch_size = args.batch_size // world_size
        if args.batch_size == 0:
            args.batch_size = 1
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original_batch_size = args.batch_size

    # Setup logging
    logger = setup_logging(rank)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Using device: {device}, Rank: {rank}, World size: {world_size}")
        if args.use_ddp:
            logger.info(f"Per-GPU batch size: {args.batch_size}, Effective batch size: {original_batch_size}")
        else:
            logger.info(f"Batch size: {args.batch_size}")

    # Set seeds for reproducibility
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Load UNI2-h model for biological validation and FID calculation
    if load_uni2_h_model is not None:
        uni2h_model, uni2h_processor, uni2h_preprocess = load_uni2_h_model(device, model_path=args.uni2h_model_path)
    else:  # import guarded above (optional dep missing) -> UNI2-h path skipped
        uni2h_model = uni2h_processor = uni2h_preprocess = None
    if rank == 0:
        if uni2h_model is not None:  # ← Check model, not processor
            logger.info("UNI2-h model loaded successfully for biological validation")
        else:
            # There is NO fallback -- the old "using ResNet fallback" wording was fiction and read
            # as "something else is still computing it", which is how a run with no biological FID
            # at all looks fine at a glance. Say what actually happens.
            logger.warning("UNI2-h model unavailable: NO biological validation will be computed and "
                           "overall_uni2h_fid will be NaN. This is exactly what invalidated the "
                           "previous 54-run batch. Set UNI2H_MODEL_PATH to a dir containing "
                           "pytorch_model.bin, or pass ALLOW_NO_UNI2H=1 to accept it deliberately.")

    # Load HE2RNA model for RNA prediction validation
    he2rna_model = None
    if args.he2rna_model_path and rank == 0:
        he2rna_model = load_he2rna_model(args.he2rna_model_path, device)
        if he2rna_model is not None:
            logger.info("HE2RNA model loaded successfully for RNA prediction validation")
        else:
            logger.info("HE2RNA model unavailable, skipping RNA prediction validation")

    # Data loading logic (same as original but with distributed considerations)
    expr_df = None
    gene_names = None
    missing_gene_symbols_list = None

    if args.hest1k_base_dir:
        if args.hest1k_sid is None or len(args.hest1k_sid) == 0:
            hest_metadata = pd.read_csv(os.environ.get("HEST_METADATA_CSV", "/depot/natallah/data/Mengbo/HnE_RNA/data/HEST-1k/data/HEST_v1_1_0.csv"))
            args.hest1k_sid = hest_metadata[(hest_metadata['st_technology']=='Xenium') &
                                            (hest_metadata['species']=='Homo sapiens')]['id'].tolist()
        if rank == 0:
            logger.info(f"Loading pre-processed HEST-1k data for sample {args.hest1k_sid}")
        expr_df, image_paths_dict = load_preprocessed_hest1k_singlecell_data(
            args.hest1k_sid, args.hest1k_base_dir, img_size=args.img_size, img_channels=args.img_channels)
        missing_gene_symbols_list = None
    elif args.hest1k_xenium_dir or args.hest1k_xenium_fast_dir:
        if rank == 0:
            logger.info(f"Preparing to load manually processed HEST-1k Xenium samples")
    elif args.adata is not None:
        if rank == 0:
            logger.info(f"Loading AnnData from {args.adata}")
        expr_df, missing_gene_symbols_list = parse_adata(args)
    else:
        if rank == 0:
            logger.warning(f"(deprecated) Loading gene expression data from {args.gene_expr}")
        expr_df = pd.read_csv(args.gene_expr, index_col=0)
        missing_gene_symbols_list = None

    if expr_df is not None and rank == 0:
        logger.info(f"Loaded gene expression data with shape: {expr_df.shape}")
        gene_names = expr_df.columns.tolist()

    # Dataset creation logic (same as training script)
    if args.model_type == 'single':
        if rank == 0:
            logger.info("Creating single-cell dataset for evaluation")
        if not (args.hest1k_sid and args.hest1k_base_dir):
            image_paths_dict = {}
            if args.image_paths:
                if rank == 0:
                    logger.info(f"Loading image paths from {args.image_paths}")
                with open(args.image_paths, "r") as f:
                    image_paths_data = json.load(f)
                image_paths_dict = {k: v for k, v in image_paths_data.items() if os.path.exists(v)}
                if rank == 0:
                    logger.info(f"Loaded {len(image_paths_dict)} valid cell image paths")

        full_dataset = CellImageGeneDataset(
            expr_df, image_paths_dict, img_size=args.img_size, img_channels=args.img_channels,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.img_size, args.img_size), antialias=True)]),
            missing_gene_symbols=missing_gene_symbols_list,
            normalize_aux=args.normalize_aux,
        )
    else:  # multi-cell
        if rank == 0:
            logger.info("Creating multi-cell dataset for evaluation")
        if args.hest1k_xenium_fast_dir is not None:
            if rank == 0:
                logger.info(f"Loading fast reformatted HEST-1k Xenium dataset from {args.hest1k_xenium_fast_dir}")
            full_dataset = FastSeparatePatchDataset(
                reformatted_dir=args.hest1k_xenium_fast_dir,
                sample_metadata_csv=args.hest1k_xenium_metadata,
                sample_ids=args.hest1k_xenium_samples,
                img_size=args.img_size,
                img_channels=args.img_channels,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((args.img_size, args.img_size), antialias=True),
                ]),
                filter_unassigned=True,
                min_gene_samples=1,
                cache_unified_genes=True
            )
            gene_names = full_dataset.gene_names
            if rank == 0:
                logger.info(f"Fast dataset loaded: {len(full_dataset)} patches, {len(gene_names)} unified genes")
        elif args.hest1k_xenium_dir is not None:
            if rank == 0:
                logger.info(f"Loading HEST-1k Xenium on-demand dataset from {args.hest1k_xenium_dir}")
            full_dataset = OnDemandMultiSampleHestXeniumDataset(
                combined_dir=args.hest1k_xenium_dir,
                sample_metadata_csv=args.hest1k_xenium_metadata,
                sample_ids=args.hest1k_xenium_samples,
                img_size=args.img_size,
                img_channels=args.img_channels,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((args.img_size, args.img_size), antialias=True),
                ]),
                normalize_aux=args.normalize_aux,
                cache_metadata=True
            )
            if rank == 0:
                sample_stats = full_dataset.get_sample_stats()
                logger.info("Sample statistics:")
                for sample_id, stats in sample_stats.items():
                    logger.info(f"  {sample_id}: {stats['n_patches']} patches, "
                              f"{stats['n_cells']} cells, {stats['n_genes']} genes")
        else:
            patch_image_paths_dict = None
            with open(args.patch_cell_mapping, "r") as f:
                patch_to_cells = json.load(f)
            if args.patch_image_paths:
                if rank == 0:
                    logger.info(f"Loading patch image paths from {args.patch_image_paths}")
                with open(args.patch_image_paths, "r") as f:
                    patch_image_paths_data = json.load(f)
                patch_image_paths_dict = {k: v for k, v in patch_image_paths_data.items() if os.path.exists(v)}
                if rank == 0:
                    logger.info(f"Loaded {len(patch_image_paths_dict)} valid patch image paths")

            full_dataset = PatchImageGeneDataset(
                expr_df=expr_df, patch_image_paths=patch_image_paths_dict, patch_to_cells=patch_to_cells,
                img_size=args.img_size, img_channels=args.img_channels,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.img_size, args.img_size), antialias=True)]),
                normalize_aux=args.normalize_aux,
            )

    if len(full_dataset) == 0:
        if rank == 0:
            logger.error("Full dataset is empty. Cannot proceed with evaluation.")
        if args.use_ddp:
            cleanup_ddp()
        return

    # Debug/smoke: mirror rectified_main.py's debug subset (same seed + randperm) BEFORE the
    # split, so the 80/10/10 partition reproduces the SAME test set as the --debug training
    # run (otherwise eval splits the full dataset and its test chunk won't match).
    if getattr(args, 'debug', False):
        _dbg_idx = torch.randperm(len(full_dataset),
                                  generator=torch.Generator().manual_seed(args.seed))[:args.debug_samples]
        full_dataset = torch.utils.data.Subset(full_dataset, _dbg_idx)
        if rank == 0:
            logger.info(f"DEBUG MODE: subset full_dataset to {len(full_dataset)} samples "
                        f"(mirroring training) before the split.")

    # FIX (2026-07): evaluate on the held-out TEST split (80/10/10), NOT the
    # validation set used for checkpoint selection. This split MUST match
    # rectified_main.py exactly (same lengths, same seed) so the test set here is the
    # same 10% that was held out (never trained on, never used to pick the checkpoint).
    n_total = len(full_dataset)
    train_size = int(0.8 * n_total)
    val_size = int(0.1 * n_total)
    test_size = n_total - train_size - val_size
    _, _, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)  # same split as training
    )

    if len(eval_dataset) == 0:
        if rank == 0:
            logger.error("Evaluation dataset is empty after splitting. Exiting.")
        if args.use_ddp:
            cleanup_ddp()
        return

    if len(eval_dataset) > args.max_samples:
        logger.info(f"Limiting evaluation dataset to {args.max_samples} samples")
        eval_dataset = torch.utils.data.Subset(eval_dataset, range(args.max_samples))

    # Create distributed sampler
    if args.use_ddp:
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.seed
        )
    else:
        eval_sampler = None

    # Determine collate function based on model type and dataset type
    collate_fn_to_use = None
    if args.model_type == 'multi':
        if args.hest1k_xenium_fast_dir is not None:
            collate_fn_to_use = fast_separate_patch_collate_fn
            if rank == 0:
                logger.info("Using fast separate patch collate function for evaluation")
        elif args.hest1k_xenium_dir is not None:
            collate_fn_to_use = multi_sample_hest_xenium_collate_fn
            if rank == 0:
                logger.info("Using multi-sample Xenium collate function for evaluation")
        else:
            collate_fn_to_use = patch_collate_fn
            if rank == 0:
                logger.info("Using standard patch collate function for evaluation")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=args.num_dataloader_workers,
        pin_memory=True,
        collate_fn=collate_fn_to_use
    )

    if rank == 0:
        logger.info(f"Evaluation set size: {len(eval_dataset)}")

    # Model initialization - Fixed gene dimension handling
    if expr_df is not None:
        gene_dim = expr_df.shape[1]
    elif hasattr(full_dataset, 'gene_names'):
        gene_dim = len(full_dataset.gene_names)
        gene_names = full_dataset.gene_names
    else:
        gene_dim = len(gene_names)

    if gene_dim is None:
        raise ValueError("Could not determine gene dimension. Ensure dataset has gene_names attribute.")

    if rank == 0:
        logger.info(f"Gene dimension: {gene_dim}")

    # Model Loading with Config Consistency
    if rank == 0:
        logger.info(f"Loading pretrained model from {args.model_path}")
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
    except FileNotFoundError:
        if rank == 0:
            logger.error(f"Model checkpoint not found at {args.model_path}")
        if args.use_ddp:
            cleanup_ddp()
        return

    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        if rank == 0:
            logger.error(f"Checkpoint missing model weights. Available keys: {checkpoint.keys()}")
        if args.use_ddp:
            cleanup_ddp()
        return
    
    model_config_ckpt = checkpoint.get("config", {})

    # Prioritize config from checkpoint for critical architectural params
    ckpt_img_channels = model_config_ckpt.get('img_channels', args.img_channels)
    ckpt_img_size = model_config_ckpt.get('img_size', args.img_size)
    ckpt_model_type = model_config_ckpt.get('model_type', args.model_type)

    # Update args if you want them to reflect checkpoint's settings for model init
    args.img_channels = ckpt_img_channels
    args.img_size = ckpt_img_size

    use_gene_attention = model_config_ckpt.get('use_gene_attention', args.use_gene_attention)
    use_multi_head_attention = model_config_ckpt.get('use_multi_head_attention', args.use_multi_head_attention)
    use_feature_gating = model_config_ckpt.get('use_feature_gating', args.use_feature_gating)
    use_residual_blocks = model_config_ckpt.get('use_residual_blocks', args.use_residual_blocks)
    use_layer_norm = model_config_ckpt.get('use_layer_norm', args.use_layer_norm)
    use_gene_relations = model_config_ckpt.get('use_gene_relations', args.use_gene_relations)

    model_constructor_args = dict(
        rna_dim=gene_dim, img_channels=args.img_channels, img_size=args.img_size,
        model_channels=128, num_res_blocks=2, attention_resolutions=(16,),
        dropout=0.1, channel_mult=(1, 2, 2, 2), use_checkpoint=False,
        num_heads=2, num_head_channels=16, use_scale_shift_norm=True,
        resblock_updown=True, use_new_attention_order=True,
        concat_mask=model_config_ckpt.get('concat_mask', getattr(args, 'concat_mask', False)),
        relation_rank=model_config_ckpt.get('relation_rank', getattr(args, 'relation_rank', 50)),
        # Build the RNA (geneflow) encoder with the checkpoint's ablation flags (read above),
        # not the constructor's hardcoded all-True defaults. No-op on the 54-run (all True),
        # but correct if an RNA-flag ablation checkpoint is ever evaluated. Ignored by the
        # pathway encoder. (main.py persists these in model_config; see rectified_main.py:720.)
        use_gene_attention=use_gene_attention,
        use_multi_head_attention=use_multi_head_attention,
        use_feature_gating=use_feature_gating,
        use_residual_blocks=use_residual_blocks,
        use_layer_norm=use_layer_norm,
        use_gene_relations=use_gene_relations,
    )

    # Pathway encoder support: rebuild the same encoder used at train time.
    # The mask is restored from the checkpoint config (shape [P, G]) so the model
    # graph matches the saved state_dict exactly.
    encoder_type = model_config_ckpt.get('encoder_type', getattr(args, 'encoder_type', 'rna'))
    cross_eval = (encoder_type == 'pathway' and getattr(args, 'cross_dataset_eval', False)
                  and getattr(args, 'pathway_mask', None) is not None)

    # Hard correctness check (mirrors the train-side guard in rectified_main.py):
    # the gene COLUMN ORDER at eval time must match the order the model was trained
    # on (stored in the checkpoint config). load_state_dict only checks tensor
    # shapes, so a same-length-but-reordered panel (e.g. re-saved adata, a
    # different layer, or --gene_symbols) would pass silently while x[:, i] indexes
    # the wrong gene -> every metric is computed on corrupted conditioning. Skipped
    # for cross-dataset eval, where source and target panels are expected to differ
    # (the transplant remaps weights by name instead).
    if not cross_eval:
        ckpt_gene_names = model_config_ckpt.get('gene_names')
        if ckpt_gene_names and gene_names is not None:
            ckpt_genes = [str(g) for g in ckpt_gene_names]
            cur_genes = [str(g) for g in gene_names]
            if ckpt_genes != cur_genes:
                n_mis = sum(1 for a, b in zip(ckpt_genes, cur_genes) if a != b)
                first = next((i for i, (a, b) in enumerate(zip(ckpt_genes, cur_genes)) if a != b), 'NA')
                raise ValueError(
                    f"Eval gene order does not match the checkpoint's training gene "
                    f"order: len(ckpt)={len(ckpt_genes)} vs len(eval)={len(cur_genes)}, "
                    f"{n_mis} positions differ (first at index {first}). The model was "
                    f"trained on a different gene ordering; evaluating like this scores "
                    f"the model on misaligned conditioning. Use --cross_dataset_eval for "
                    f"a deliberate cross-panel eval, or align the eval panel to training.")

    if encoder_type == 'pathway':
        import numpy as _np
        import torch as _torch
        if cross_eval:
            # Cross-dataset transfer eval (2.3): build the encoder on the TARGET
            # mask; source-trained weights are transplanted by name after build.
            pathway_mask_arr = _np.load(args.pathway_mask, allow_pickle=True)['A']
        else:
            pathway_mask_arr = model_config_ckpt.get('pathway_mask_array', None)
            if pathway_mask_arr is None:
                # Fall back to loading from the mask file recorded in config / args.
                mask_path = model_config_ckpt.get('pathway_mask', getattr(args, 'pathway_mask', None))
                if mask_path is None:
                    raise ValueError("encoder_type='pathway' but no pathway mask available in checkpoint or args.")
                pathway_mask_arr = _np.load(mask_path)['A']
        if _torch.is_tensor(pathway_mask_arr):
            pathway_mask_tensor = pathway_mask_arr.to(_torch.float32)
        else:
            pathway_mask_tensor = _torch.tensor(_np.asarray(pathway_mask_arr), dtype=_torch.float32)
        model_constructor_args.update(dict(
            encoder_type='pathway',
            pathway_mask=pathway_mask_tensor,
            d_token=model_config_ckpt.get('d_token', getattr(args, 'd_token', 48)),
            pt_layers=model_config_ckpt.get('pt_layers', getattr(args, 'pt_layers', 2)),
            pt_heads=model_config_ckpt.get('pt_heads', getattr(args, 'pt_heads', 8)),
            learnable_pathway=model_config_ckpt.get('learnable_pathway', getattr(args, 'learnable_pathway', True)),
            use_pathway_transformer=model_config_ckpt.get('use_pathway_transformer', getattr(args, 'use_pathway_transformer', True)),
        ))

    model = None
    try:
        if ckpt_model_type == 'single':
            model = RNAtoHnEModel(**model_constructor_args)
        else:  # multi-cell
            model_constructor_args['num_aggregation_heads'] = model_config_ckpt.get('num_aggregation_heads', getattr(args, 'num_aggregation_heads', 4))
            model = MultiCellRNAtoHnEModel(**model_constructor_args)
        if cross_eval:
            info = transplant_cross_dataset_weights(model, model_state, model_config_ckpt, args.pathway_mask)
            if rank == 0:
                logger.info(f"Cross-dataset transfer: matched {info['n_matched']}/{info['n_target_edges']} "
                            f"target edges to source-trained weights across {info['P']} shared pathways.")
        else:
            model.load_state_dict(model_state)
    except Exception as e:
        if cross_eval:
            # Do not silently fall back to a same-panel load for cross-dataset eval.
            raise
        if encoder_type == 'pathway':
            # Never fall back to the RNA constructors for a pathway (Gene2Image /
            # ablation) checkpoint. The feature-detection path below only builds RNA
            # encoders, so strict=False would drop every pathway weight and silently
            # evaluate a RANDOM-init encoder, reporting garbage metrics under the
            # variant's name (multi-cell) or aborting with no metrics (single-cell).
            # Fail loudly so the real load error surfaces.
            raise
        if rank == 0:
            logger.warning(f"Failed to load model with current constructor: {e}")
            logger.info("Attempting to load with feature detection from checkpoint...")
        
        # Inspect checkpoint to determine which features it has
        has_gene_attention = any('gene_attention' in k for k in model_state.keys())
        has_gene_relation = any('gene_relation_net' in k for k in model_state.keys())
        has_multi_head = any('multi_head' in k for k in model_state.keys())
        has_feature_gating = any('gate' in k.lower() for k in model_state.keys())
        
        if rank == 0:
            logger.info(f"Checkpoint features detected:")
            logger.info(f"  - Gene attention: {has_gene_attention}")
            logger.info(f"  - Gene relations: {has_gene_relation}")
            logger.info(f"  - Multi-head attention: {has_multi_head}")
            logger.info(f"  - Feature gating: {has_feature_gating}")
        
        # Build minimal constructor args based on what checkpoint has
        minimal_constructor_args = {
            'rna_dim': gene_dim,
            'img_channels': args.img_channels,
            'img_size': args.img_size,
            'model_channels': 128,
            'num_res_blocks': 2,
            'attention_resolutions': (16,),
            'dropout': 0.1,
            'channel_mult': (1, 2, 2, 2),
            'use_checkpoint': False,
            'num_heads': 2,
            'num_head_channels': 16,
            'use_scale_shift_norm': True,
            'resblock_updown': True,
            'use_new_attention_order': True,
        }
        
        # Try different model constructors in order of decreasing complexity
        model = None
        constructors_to_try = []
        
        if ckpt_model_type == 'single':
            # Try current model first (if checkpoint has all features)
            if has_gene_attention and has_gene_relation:
                constructors_to_try.append(('current', RNAtoHnEModel, {
                    **minimal_constructor_args,
                    'concat_mask': False,
                    'relation_rank': 50,
                    'use_gene_attention': True,
                    'use_multi_head_attention': has_multi_head,
                    'use_feature_gating': has_feature_gating,
                    'use_residual_blocks': True,
                    'use_layer_norm': True,
                    'use_gene_relations': True,
                }))
            # Try deprecated model (only if the legacy module is available)
            if RNAtoHnEModel_deprecation is not None:
                constructors_to_try.append(('deprecated', RNAtoHnEModel_deprecation, minimal_constructor_args))
        else:  # multi-cell
            # Try current model with detected features
            if has_gene_relation:
                constructors_to_try.append(('current_with_relations', MultiCellRNAtoHnEModel, {
                    **minimal_constructor_args,
                    'num_aggregation_heads': 4,
                    'concat_mask': False,
                    'relation_rank': 50,
                    'use_multi_head_attention': has_multi_head,
                    'use_feature_gating': has_feature_gating,
                    'use_residual_blocks': True,
                    'use_layer_norm': True,
                    'use_gene_relations': True,
                }))
            # Try current model without relations
            constructors_to_try.append(('current_minimal', MultiCellRNAtoHnEModel, {
                **minimal_constructor_args,
                'num_aggregation_heads': 4,
                'concat_mask': False,
                'relation_rank': 50,
                'use_multi_head_attention': False,
                'use_feature_gating': False,
                'use_residual_blocks': True,
                'use_layer_norm': True,
                'use_gene_relations': False,
            }))
            # Try deprecated model (only if the legacy module is available)
            if MultiCellRNAtoHnEModel_deprecation is not None:
                constructors_to_try.append(('deprecated', MultiCellRNAtoHnEModel_deprecation, {
                    **minimal_constructor_args,
                    'num_aggregation_heads': 4,
                }))
        
        # Try each constructor until one works
        for constructor_name, constructor_class, constructor_args in constructors_to_try:
            try:
                if rank == 0:
                    logger.info(f"Trying {constructor_name} constructor...")
                
                test_model = constructor_class(**constructor_args)
                
                # Handle DDP prefix
                test_state = model_state.copy()
                if any(key.startswith('module.') for key in test_state.keys()):
                    test_state = {k.replace('module.', ''): v for k, v in test_state.items()}
                
                # Try loading
                test_model.load_state_dict(test_state, strict=False)
                model = test_model
                
                if rank == 0:
                    logger.info(f"✓ Successfully loaded model with {constructor_name} constructor")
                break
                
            except Exception as load_error:
                if rank == 0:
                    logger.debug(f"✗ {constructor_name} failed: {load_error}")
                continue
        
        if model is None:
            if rank == 0:
                logger.error("Failed to load model with any available constructor")
                logger.error("Checkpoint may be incompatible or corrupted")
            if args.use_ddp:
                cleanup_ddp()
            return

    if rank == 0:
        logger.info(f"Model loaded successfully using {ckpt_model_type} constructor.")

    # Move model to device
    model.to(device)

    # Wrap model with DDP if using distributed evaluation
    if args.use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if rank == 0:
            logger.info("Model wrapped with DistributedDataParallel")

    model.eval()

    # Initialize evaluation components
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    # Inception-v3 (ImageNet) backs FID; torchvision downloads its weights on first use.
    # On an OFFLINE compute node that raises URLError -> guard it so a missing FID never
    # crashes the whole eval (SSIM/PSNR/UNI2-h still computed). Pre-fetch on a login node.
    try:
        inception_model = InceptionModel(device)
    except Exception as _inception_err:
        inception_model = None
        logger.warning(
            f"Inception-v3 unavailable ({_inception_err}); overall_fid will be NaN. Pre-fetch on a "
            f"login node: python -c \"import torchvision.models as m; m.inception_v3(weights='IMAGENET1K_V1')\" "
            f"or set TORCH_HOME to a cached copy.")

    # Storage for metrics
    all_ssim_scores = []
    all_psnr_scores = []
    all_real_features_for_fid = []
    all_gen_features_for_fid = []
    per_sample_metrics_list = []
    all_batch_fids_list = []

    all_real_embeddings_with_metadata = []
    all_gen_embeddings_with_metadata = []

    # RNA prediction validation storage
    all_rna_prediction_metrics = []
    all_original_rna = []
    all_predicted_rna = []

    # UNI2-h FID and biological validation storage
    all_uni2h_fids_list = []
    all_biological_results = []
    all_real_uni2h_features_for_fid = []
    all_gen_uni2h_features_for_fid = []

    if rank == 0:
        logger.info(f"Starting evaluation on {len(eval_loader)} batches")

    # Generation-time accumulators for the paper's 单样本推理时间 column (see the timed block below).
    _gen_seconds = 0.0
    _gen_samples = 0
    # Tiles that failed to decode and were replaced by a BLACK image (src/dataset.py). Summed from
    # the batch, NOT from a class counter: __getitem__ runs in DataLoader worker processes, so a
    # class-level count never reaches this process and a gate on it could never fire.
    _n_zero_sub = 0

    with torch.no_grad():
        # Create progress bar only on rank 0
        if rank == 0:
            eval_pbar = tqdm(eval_loader, desc="Evaluating")
        else:
            eval_pbar = eval_loader

        for batch_idx, batch in enumerate(eval_pbar):
            sample_ids_in_batch = []
            real_images_tensor = batch['image'].to(device)
            current_batch_size = real_images_tensor.shape[0]

            if ckpt_model_type == 'single':
                gene_expr = batch['gene_expr'].to(device)
                sample_ids_in_batch = batch['cell_id']
                _zs = batch.get('is_zero_substituted')
                if _zs is not None:
                    _n_zero_sub += int(_zs.sum()) if hasattr(_zs, 'sum') else int(sum(_zs))
                gene_mask = batch.get('gene_mask', None)
                if gene_mask is not None:
                    gene_mask = gene_mask.to(device)

                # Time the generation itself for the paper's 编码效率表 (单样本推理时间).
                # synchronize on both sides: CUDA is async, so without it this would measure kernel
                # launch time, not the work. Wall-time per sample is unrecoverable once the job
                # ends, which is why it is measured here rather than left to be reconstructed.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _gen_t0 = time.perf_counter()
                generated_images_tensor = generate_images_with_rectified_flow(
                    model, rectified_flow, gene_expr, device, args.gen_steps,
                    gene_mask=gene_mask, is_multi_cell=False,
                    sample_ids=sample_ids_in_batch
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _gen_seconds += time.perf_counter() - _gen_t0
                _gen_samples += int(gene_expr.shape[0])
            else:  # multi-cell
                processed_batch = prepare_multicell_batch(batch, device)
                gene_expr = processed_batch['gene_expr']
                num_cells_info = processed_batch.get('num_cells')
                sample_ids_in_batch = batch['patch_id']
                gene_mask = batch.get('gene_mask', None)
                if gene_mask is not None:
                    gene_mask = gene_mask.to(device)

                generated_images_tensor = generate_images_with_rectified_flow(
                    model, rectified_flow, gene_expr, device, args.gen_steps,
                    num_cells=num_cells_info, gene_mask=gene_mask, is_multi_cell=True,
                    sample_ids=sample_ids_in_batch
                )

            # Apply stain normalization if enabled
            if getattr(args, 'enable_stain_normalization', False) and args.img_channels >= 3:
                normalized_generated_images_list = []
                for j in range(current_batch_size):
                    real_img_np_j = real_images_tensor[j].cpu().numpy().transpose(1, 2, 0)
                    gen_img_np_j = generated_images_tensor[j].cpu().numpy().transpose(1, 2, 0)
                    real_rgb_j = real_img_np_j[:, :, :3]
                    gen_rgb_original_j = gen_img_np_j[:, :, :3]

                    if getattr(args, 'stain_normalization_method', '') == 'skimage_hist_match':
                        gen_rgb_normalized_j = normalize_staining_rgb_skimage_hist_match(
                            gen_rgb_original_j, real_rgb_j
                        )
                    else:
                        gen_rgb_normalized_j = gen_rgb_original_j

                    if args.img_channels > 3:
                        gen_aux_j = gen_img_np_j[:, :, 3:]
                        final_gen_img_np_j = np.concatenate((gen_rgb_normalized_j, gen_aux_j), axis=2)
                    else:
                        final_gen_img_np_j = gen_rgb_normalized_j

                    normalized_generated_images_list.append(
                        torch.from_numpy(final_gen_img_np_j.transpose(2,0,1)).to(device)
                    )
                generated_images_tensor = torch.stack(normalized_generated_images_list)

            if he2rna_model is not None and rank == 0:
                # Add debugging for the first batch only
                if batch_idx == 0:
                    logger.info("DEBUGGING FIRST BATCH RNA PREDICTION...")
                    real_rna_pred, gen_rna_pred, _ = debug_rna_prediction_comparison(
                        real_images_tensor, generated_images_tensor, he2rna_model, device
                    )
                else:
                    # Normal prediction for other batches
                    real_rna_pred, gen_rna_pred, rna_features = predict_and_compare_rna_from_images(
                        real_images_tensor, generated_images_tensor, he2rna_model, device
                    )
                
                if real_rna_pred is not None and gen_rna_pred is not None:
                    # Calculate RNA prediction comparison metrics (real vs generated images)
                    rna_metrics = calculate_rna_image_comparison_metrics(
                        real_rna_pred.cpu().numpy(), gen_rna_pred.cpu().numpy()
                    )
                    
                    # Log detailed metrics for first batch
                    if batch_idx == 0:
                        logger.info(f"First batch RNA metrics: {rna_metrics}")
                    
                    # Store for aggregation
                    all_rna_prediction_metrics.append(rna_metrics)
                    all_original_rna.append(real_rna_pred.cpu().numpy())  # Real image RNA predictions
                    all_predicted_rna.append(gen_rna_pred.cpu().numpy())  # Generated image RNA predictions

            # Calculate image metrics (SSIM, PSNR)
            ssim_scores, psnr_scores = calculate_image_metrics(real_images_tensor, generated_images_tensor)
            all_ssim_scores.extend(ssim_scores)
            all_psnr_scores.extend(psnr_scores)

            # Extract features for traditional FID (skip gracefully if Inception unavailable offline).
            # PERF (opt A): FID is computed ONCE at the end from the accumulated FULL-set
            # features (overall_fid). We no longer compute a per-batch FID here: with
            # batch_size ~8 and 2048-d Inception features the per-batch covariance is
            # rank-deficient (statistically invalid), and the per-batch scipy.linalg.sqrtm
            # was the single biggest eval cost. We keep only the cheap per-sample feature
            # L2 distance (for the progress bar and the per-sample CSV).
            per_sample_feat_dists = []
            if inception_model is not None:
                real_features = inception_model(real_images_tensor).cpu().numpy()
                gen_features = inception_model(generated_images_tensor).cpu().numpy()
                all_real_features_for_fid.append(real_features)
                all_gen_features_for_fid.append(gen_features)
                for i in range(real_features.shape[0]):
                    per_sample_feat_dists.append(float(np.linalg.norm(real_features[i] - gen_features[i])))
            batch_fid = np.nan  # per-batch FID removed; see overall_fid at end of eval
            all_batch_fids_list.append(batch_fid)

            # UNI2-h embeddings (optional: skipped if model unavailable).
            # PERF (opt B): extract the ViT-H embeddings ONCE per batch (real + generated)
            # and accumulate them; the UNI2-h FID is computed ONCE at the end
            # (overall_uni2h_fid) from the full set. This drops the old per-batch
            # calculate_uni2h_fid, which redundantly ran TWO more ViT-H forward passes
            # (it re-extracts internally) plus a per-batch sqrtm on a rank-deficient batch.
            if uni2h_model is None:
                real_uni2h_features = None
                gen_uni2h_features = None
            else:
                real_uni2h_features = extract_uni2_h_embeddings(
                    real_images_tensor, uni2h_model, uni2h_processor, uni2h_preprocess, device
                )
                gen_uni2h_features = extract_uni2_h_embeddings(
                    generated_images_tensor, uni2h_model, uni2h_processor, uni2h_preprocess, device
                )
            uni2h_fid = np.nan  # per-batch UNI2-h FID removed; see overall_uni2h_fid at end
            all_uni2h_fids_list.append(uni2h_fid)

            # Store embeddings with metadata for saving
            if args.save_embeddings and uni2h_model is not None:
                for i in range(current_batch_size):
                    # Real image embeddings
                    real_embedding_entry = {
                        'sample_id': sample_ids_in_batch[i],
                        'embedding': real_uni2h_features[i],
                        'type': 'real',
                        'batch_idx': batch_idx,
                        'rank': rank
                    }
                    all_real_embeddings_with_metadata.append(real_embedding_entry)
                    
                    # Generated image embeddings
                    gen_embedding_entry = {
                        'sample_id': sample_ids_in_batch[i],
                        'embedding': gen_uni2h_features[i],
                        'type': 'generated',
                        'batch_idx': batch_idx,
                        'rank': rank
                    }
                    all_gen_embeddings_with_metadata.append(gen_embedding_entry)

            if uni2h_model is not None:
                all_real_uni2h_features_for_fid.append(real_uni2h_features)
                all_gen_uni2h_features_for_fid.append(gen_uni2h_features)

            # Batch-level biological validation (UNI2-h dependent; skip if absent)
            if uni2h_model is not None:
                biological_results = extended_biological_evaluation_uni2h(
                    real_images_tensor, generated_images_tensor,
                    uni2h_model, uni2h_processor, uni2h_preprocess, device
                )
            else:
                biological_results = {}

            # Store as batch-level metrics
            batch_biological_metrics = {
                'batch_idx': batch_idx,
                'rank': rank,
                'batch_size': current_batch_size,
                'uni2h_fid': uni2h_fid,
                'inception_fid': batch_fid
            }
            batch_biological_metrics.update(biological_results)
            all_biological_results.append(batch_biological_metrics)

            # Store per-sample metrics
            for i in range(current_batch_size):
                sample_metrics = {
                    'sample_id': sample_ids_in_batch[i],
                    'ssim': ssim_scores[i],
                    'psnr': psnr_scores[i],
                    'batch_idx': batch_idx,
                    'inception_feature_distance': per_sample_feat_dists[i] if i < len(per_sample_feat_dists) else np.nan,
                    'rank': rank
                }
                per_sample_metrics_list.append(sample_metrics)

            # Update progress bar (only on rank 0)
            if rank == 0:
                # FID / UNI2-h FID are computed ONCE at the end from the full set, so the
                # per-batch values are no longer available for the bar; show the cheap
                # per-batch Inception feature-distance as a live progress signal instead.
                postfix_dict = {
                    'Avg SSIM': f'{np.mean(all_ssim_scores):.4f}',
                    'Avg PSNR': f'{np.mean(all_psnr_scores):.4f}',
                    'FeatDist(batch)': f'{np.mean(per_sample_feat_dists):.3f}' if per_sample_feat_dists else 'N/A',
                }

                # Add RNA correlation if available
                if all_rna_prediction_metrics:
                    latest_rna_corr = all_rna_prediction_metrics[-1].get('rna_image_gene_correlation_mean', 0)
                    postfix_dict['RNA Corr'] = f'{latest_rna_corr:.4f}'
                
                eval_pbar.set_postfix(postfix_dict)

    # Gather results from all ranks
    if args.use_ddp:
        # Synchronize before gathering results
        dist.barrier()
        
        # Gather metrics from all ranks
        all_ssim_scores = gather_list_across_ranks(all_ssim_scores, world_size)
        all_psnr_scores = gather_list_across_ranks(all_psnr_scores, world_size)
        per_sample_metrics_list = gather_list_across_ranks(per_sample_metrics_list, world_size)
        all_batch_fids_list = gather_list_across_ranks(all_batch_fids_list, world_size)
        
        # Gather UNI2-h results
        all_uni2h_fids_list = gather_list_across_ranks(all_uni2h_fids_list, world_size)
        all_biological_results = gather_list_across_ranks(all_biological_results, world_size)

        # Gather features for global FID calculation
        gathered_real_features = [None for _ in range(world_size)]
        gathered_gen_features = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_real_features, all_real_features_for_fid)
        dist.all_gather_object(gathered_gen_features, all_gen_features_for_fid)

        # Gather UNI2-h features for global FID calculation
        gathered_real_uni2h_features = [None for _ in range(world_size)]
        gathered_gen_uni2h_features = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_real_uni2h_features, all_real_uni2h_features_for_fid)
        dist.all_gather_object(gathered_gen_uni2h_features, all_gen_uni2h_features_for_fid)
        
        # Gather embeddings with metadata if saving
        if args.save_embeddings:
            all_real_embeddings_with_metadata = gather_list_across_ranks(all_real_embeddings_with_metadata, world_size)
            all_gen_embeddings_with_metadata = gather_list_across_ranks(all_gen_embeddings_with_metadata, world_size)
           
        # Handle RNA prediction results (only rank 0 has this data)
        if rank == 0 and all_rna_prediction_metrics:
            # RNA data only exists on rank 0, no need to gather across ranks
            pass  # Keep the data as is on rank 0
        else:
            # Other ranks have no RNA data, initialize empty lists
            all_rna_prediction_metrics = []
            all_original_rna = []
            all_predicted_rna = []

        # Flatten features on rank 0
        if rank == 0:
            all_real_features_for_fid = []
            all_gen_features_for_fid = []
            for rank_real_features in gathered_real_features:
                all_real_features_for_fid.extend(rank_real_features)
            for rank_gen_features in gathered_gen_features:
                all_gen_features_for_fid.extend(rank_gen_features)

            # Flatten UNI2-h features on rank 0
            all_real_uni2h_features_for_fid = []
            all_gen_uni2h_features_for_fid = []
            for rank_real_features in gathered_real_uni2h_features:
                all_real_uni2h_features_for_fid.extend(rank_real_features)
            for rank_gen_features in gathered_gen_uni2h_features:
                all_gen_uni2h_features_for_fid.extend(rank_gen_features)

    # Compute final metrics and save results (only on rank 0)
    if rank == 0:
        logger.info("Computing final evaluation metrics...")
        
        # Calculate overall traditional FID (NaN if Inception was unavailable -> no features)
        if all_real_features_for_fid and all_gen_features_for_fid:
            all_real_features_concat = np.concatenate(all_real_features_for_fid, axis=0)
            all_gen_features_concat = np.concatenate(all_gen_features_for_fid, axis=0)
            # Drop non-finite feature rows (independently per set) so one NaN generated
            # image can't make the covariance -> sqrtm NaN and void overall_fid.
            _rfin = np.isfinite(all_real_features_concat).all(axis=1)
            _gfin = np.isfinite(all_gen_features_concat).all(axis=1)
            if not (_rfin.all() and _gfin.all()):
                logger.warning(f"Dropped non-finite feature rows before FID "
                               f"(real: {int((~_rfin).sum())}, gen: {int((~_gfin).sum())}).")
                all_real_features_concat = all_real_features_concat[_rfin]
                all_gen_features_concat = all_gen_features_concat[_gfin]
            overall_fid = calculate_fid(all_real_features_concat, all_gen_features_concat)
        else:
            overall_fid = float('nan')

        # Calculate overall UNI2-h FID (same non-finite row filter as Inception FID above,
        # so one bad embedding drops a row instead of voiding the whole variant's UNI2h-FID)
        try:
            all_real_uni2h_features_concat = np.concatenate(all_real_uni2h_features_for_fid, axis=0)
            all_gen_uni2h_features_concat = np.concatenate(all_gen_uni2h_features_for_fid, axis=0)
            _ru = np.isfinite(all_real_uni2h_features_concat).all(axis=1)
            _gu = np.isfinite(all_gen_uni2h_features_concat).all(axis=1)
            if not (_ru.all() and _gu.all()):
                logger.warning(f"Dropped non-finite UNI2-h feature rows before FID "
                               f"(real: {int((~_ru).sum())}, gen: {int((~_gu).sum())}).")
                all_real_uni2h_features_concat = all_real_uni2h_features_concat[_ru]
                all_gen_uni2h_features_concat = all_gen_uni2h_features_concat[_gu]
            overall_uni2h_fid = calculate_fid(all_real_uni2h_features_concat, all_gen_uni2h_features_concat)
            logger.info(f"Overall UNI2-h FID calculated from {all_real_uni2h_features_concat.shape[0]} samples")
        except Exception as e:
            logger.warning(f"Could not compute overall UNI2-h FID: {e}")
            overall_uni2h_fid = np.nan
        
        # Calculate mean metrics. Filter non-finite per-sample values first: a single
        # NaN/inf generated image (e.g. a diverged cell) would otherwise make np.mean NaN
        # and void the entire variant's SSIM/PSNR (np.clip/torch.clamp do NOT remove NaN).
        # Matches the NaN filtering already done for FID / feature-distance below.
        _ssim_arr = np.asarray(all_ssim_scores, dtype=float)
        _psnr_arr = np.asarray(all_psnr_scores, dtype=float)
        _ssim_ok = _ssim_arr[np.isfinite(_ssim_arr)]
        _psnr_ok = _psnr_arr[np.isfinite(_psnr_arr)]
        _n_bad = (len(_ssim_arr) - len(_ssim_ok)) + (len(_psnr_arr) - len(_psnr_ok))
        if _n_bad > 0:
            logger.warning(f"Dropped {_n_bad} non-finite SSIM/PSNR value(s) before averaging "
                           f"(SSIM: {len(_ssim_arr) - len(_ssim_ok)}, PSNR: {len(_psnr_arr) - len(_psnr_ok)}).")
        mean_ssim = float(np.mean(_ssim_ok)) if len(_ssim_ok) else float('nan')
        mean_psnr = float(np.mean(_psnr_ok)) if len(_psnr_ok) else float('nan')
        std_ssim = float(np.std(_ssim_ok)) if len(_ssim_ok) else float('nan')
        std_psnr = float(np.std(_psnr_ok)) if len(_psnr_ok) else float('nan')

        all_feat_dists = [item['inception_feature_distance'] for item in per_sample_metrics_list if not np.isnan(item.get('inception_feature_distance', np.nan))]
        feat_dist_mean, feat_dist_std = (np.mean(all_feat_dists), np.std(all_feat_dists)) if all_feat_dists else (np.nan, np.nan)

        # Filter out NaN FID values
        valid_fids = [fid for fid in all_batch_fids_list if not np.isnan(fid)]
        mean_batch_fid = np.mean(valid_fids) if valid_fids else np.nan
        
        # Compute UNI2-h FID summary
        valid_uni2h_fids = [fid for fid in all_uni2h_fids_list if not np.isnan(fid)]
        mean_uni2h_fid = np.mean(valid_uni2h_fids) if valid_uni2h_fids else np.nan

        # Compute biological validation summary from batch-level results
        biological_summary = {}
        if all_biological_results:
            biological_keys = [k for k in all_biological_results[0].keys() 
                            if k not in ['batch_idx', 'rank', 'batch_size', 'uni2h_fid', 'inception_fid']]
            
            for key in biological_keys:
                # Guard `key in r`: nuclear_*/spatial_* keys are only present in batches that
                # actually segmented nuclei, so a later batch (e.g. a blurry ablation image ->
                # 0 regions) can omit a key that batch 0 had. Without this guard r[key] raises
                # KeyError and crashes the whole eval BEFORE evaluation_summary.json is written
                # (losing the run's FID/SSIM/PSNR and, under set -e, aborting the matrix).
                values = [r[key] for r in all_biological_results
                          if key in r and r[key] is not None and not np.isnan(r[key])]
                if values:
                    biological_summary[f'mean_{key}'] = float(np.mean(values))
                    biological_summary[f'std_{key}'] = float(np.std(values))

        # Compute RNA prediction summary
        rna_summary = {}
        if all_rna_prediction_metrics:
            for key in all_rna_prediction_metrics[0].keys():
                values = [m[key] for m in all_rna_prediction_metrics if key in m and not np.isnan(m[key])]
                if values:
                    rna_summary[f'mean_{key}'] = float(np.mean(values))
                    rna_summary[f'std_{key}'] = float(np.std(values))

        # results summary
        results_summary = {
            'total_samples': len(all_ssim_scores),
            'n_ssim_used': int(len(_ssim_ok)),  # finite samples the SSIM mean/std were computed over
            'n_psnr_used': int(len(_psnr_ok)),  # finite samples the PSNR mean/std were computed over
            'mean_ssim': float(mean_ssim),
            'std_ssim': float(std_ssim),
            'mean_psnr': float(mean_psnr),
            'std_psnr': float(std_psnr),
            'overall_fid': float(overall_fid),
            # NOTE: per-batch FID (mean over batches) is intentionally NOT reported
            # here. With EVAL_BATCH=8 and 2048-d Inception features the per-batch
            # covariance is rank-deficient, so per-batch FID is statistically
            # invalid (two identically-distributed batches can score FID~480).
            # Only the full-set overall_fid / overall_uni2h_fid are trustworthy.
            # The per-batch values are still computed for the diagnostic plots.
            'overall_uni2h_fid': float(overall_uni2h_fid),
            'model_path': args.model_path,
            'model_type': ckpt_model_type,
            'img_size': args.img_size,
            'img_channels': args.img_channels,
            'generation_steps': args.gen_steps,
            'batch_size': original_batch_size,
            'world_size': world_size,
            'inception_feature_distance_mean': float(feat_dist_mean) if not np.isnan(feat_dist_mean) else None,
            'inception_feature_distance_std': float(feat_dist_std) if not np.isnan(feat_dist_std) else None,
            'use_gene_attention': use_gene_attention,
            'use_multi_head_attention': use_multi_head_attention,
            'use_feature_gating': use_feature_gating,
            'use_residual_blocks': use_residual_blocks,
            'use_layer_norm': use_layer_norm,
            'use_gene_relations': use_gene_relations,
            # Run identity for downstream aggregation (encoder_type, seed, dataset).
            'encoder_type': model_config_ckpt.get('encoder_type', getattr(args, 'encoder_type', 'rna')),
            'pathway_db': model_config_ckpt.get('pathway_db', getattr(args, 'pathway_db', None)),
            'seed': getattr(args, 'seed', None),
        }

        # One machine-readable line carrying every value the validity gates need, so a run can be
        # audited from the LOG ALONE -- no results/ directory, no evaluation_summary.json. That
        # matters because the logs are what comes back from the cluster first (and are small enough
        # to hand around), while these numbers otherwise live only in the JSON. Paired with the
        # DOPRI5_DIAGNOSTICS line from the sampler, this makes the log self-sufficient:
        #   gate 1-2 (dt_floor / under_integration_fallback) -> DOPRI5_DIAGNOSTICS
        #   gate 3   (n_ssim_used == n_psnr_used == total_samples) -> here
        #   UNI2-h NaN (what invalidated the previous 54-run batch)  -> here
        results_summary['zero_image_substitutions'] = _n_zero_sub
        # 单样本推理时间 for the efficiency table. Report gen_steps alongside it: DOPRI5 is adaptive,
        # so seconds/sample is meaningless without the step budget it ran under.
        results_summary['gen_seconds_total'] = round(_gen_seconds, 3)
        results_summary['gen_samples_timed'] = _gen_samples
        results_summary['sec_per_sample'] = round(_gen_seconds / _gen_samples, 4) if _gen_samples else None
        logger.info(
            "EVAL_GATES "
            f"total_samples={results_summary['total_samples']} "
            f"n_ssim_used={results_summary['n_ssim_used']} "
            f"n_psnr_used={results_summary['n_psnr_used']} "
            f"overall_fid={results_summary['overall_fid']:.6g} "
            f"overall_uni2h_fid={results_summary['overall_uni2h_fid']:.6g} "
            f"mean_ssim={results_summary['mean_ssim']:.6g} "
            f"mean_psnr={results_summary['mean_psnr']:.6g} "
            f"encoder_type={results_summary['encoder_type']} "
            f"seed={results_summary['seed']} "
            f"gen_steps={results_summary['generation_steps']} "
            # Tiles that failed to decode and were replaced by a BLACK image. Nonzero means some
            # "real" image a model was scored against was fabricated, and those pixels also entered
            # the FID reference statistics -- invisible to the finite-value gate, since a black
            # image's SSIM/PSNR are finite.
            f"zero_image_substitutions={_n_zero_sub} "
            f"sec_per_sample={results_summary['sec_per_sample']} "
            f"gen_samples_timed={_gen_samples}"
        )

        # Add biological metrics to results summary
        results_summary.update(biological_summary)
        results_summary.update(rna_summary)

        # Save UNI2-h embeddings for UMAP analysis (only if any were collected;
        # they are empty when UNI2-h is unavailable).
        if args.save_embeddings and (all_real_embeddings_with_metadata or all_gen_embeddings_with_metadata):
            logger.info("Saving UNI2-h embeddings for UMAP analysis...")

            embeddings_dir = args.embeddings_output_path if args.embeddings_output_path else os.path.join(args.output_dir, 'embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)
            
            # Combine real and generated embeddings
            all_embeddings_with_metadata = all_real_embeddings_with_metadata + all_gen_embeddings_with_metadata
            
            # Create arrays for embeddings and metadata
            embeddings = np.array([entry['embedding'] for entry in all_embeddings_with_metadata])
            sample_ids = [entry['sample_id'] for entry in all_embeddings_with_metadata]
            types = [entry['type'] for entry in all_embeddings_with_metadata]
            batch_indices = [entry['batch_idx'] for entry in all_embeddings_with_metadata]
            
            # Save embeddings
            np.save(os.path.join(embeddings_dir, 'uni2h_embeddings.npy'), embeddings)
            
            # Save metadata
            metadata_df = pd.DataFrame({
                'sample_id': sample_ids,
                'type': types,
                'batch_idx': batch_indices,
                'embedding_index': range(len(sample_ids))
            })
            metadata_df.to_csv(os.path.join(embeddings_dir, 'embeddings_metadata.csv'), index=False)
            
            # Also save separate arrays for convenience
            real_embeddings = np.array([entry['embedding'] for entry in all_real_embeddings_with_metadata])
            gen_embeddings = np.array([entry['embedding'] for entry in all_gen_embeddings_with_metadata])
            
            np.save(os.path.join(embeddings_dir, 'uni2h_real_embeddings.npy'), real_embeddings)
            np.save(os.path.join(embeddings_dir, 'uni2h_generated_embeddings.npy'), gen_embeddings)
            
            # Save sample IDs separately
            real_sample_ids = [entry['sample_id'] for entry in all_real_embeddings_with_metadata]
            gen_sample_ids = [entry['sample_id'] for entry in all_gen_embeddings_with_metadata]
            
            with open(os.path.join(embeddings_dir, 'real_sample_ids.json'), 'w') as f:
                json.dump(real_sample_ids, f)
            with open(os.path.join(embeddings_dir, 'generated_sample_ids.json'), 'w') as f:
                json.dump(gen_sample_ids, f)
            
            logger.info(f"Saved {embeddings.shape[0]} UNI2-h embeddings ({embeddings.shape[1]} dimensions) to {embeddings_dir}")
            logger.info(f"Real embeddings: {real_embeddings.shape[0]}, Generated embeddings: {gen_embeddings.shape[0]}")
        
        # Save RNA predictions for detailed analysis
        if all_original_rna and all_predicted_rna:
            logger.info("Saving RNA prediction results...")
            np.save(os.path.join(args.output_dir, 'original_rna_expressions.npy'), 
                   np.concatenate(all_original_rna, axis=0))
            np.save(os.path.join(args.output_dir, 'predicted_rna_expressions.npy'), 
                   np.concatenate(all_predicted_rna, axis=0))
            
            # Save gene names if available
            if gene_names:
                with open(os.path.join(args.output_dir, 'gene_names.json'), 'w') as f:
                    json.dump(gene_names, f)
            
            logger.info(f"Saved RNA expressions: {len(all_original_rna)} batches")

        # logging with UNI2-h and biological validation results
        logger.info("="*80)
        logger.info("ENHANCED EVALUATION RESULTS WITH UNI2-H BIOLOGICAL VALIDATION")
        logger.info("(Biological metrics computed at batch level)")
        logger.info("="*80)
        logger.info(f"Total samples evaluated: {len(all_ssim_scores)}")
        logger.info(f"SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
        logger.info(f"PSNR: {mean_psnr:.4f} ± {std_psnr:.4f}")
        logger.info(f"Inception FID (overall, full-set): {overall_fid:.4f}")
        logger.info(f"UNI2-H FID (overall, full-set): {overall_uni2h_fid:.4f}")
        logger.info("-" * 40)
        logger.info("BIOLOGICAL VALIDATION RESULTS:")
        logger.info("-" * 40)
        if biological_summary:
            logger.info(f"Cell Type Classification Accuracy: {biological_summary.get('mean_cell_type_accuracy', 0):.4f}")
            logger.info(f"UNI2-H Embedding Similarity: {biological_summary.get('mean_uni2h_embedding_similarity', 0):.4f}")
            logger.info(f"Nuclear Feature Similarity (avg): {np.mean([biological_summary.get(f'mean_nuclear_{f}_similarity', 0) for f in ['area', 'circularity', 'eccentricity', 'solidity']]):.4f}")
            logger.info(f"Spatial Pattern Similarity (avg): {np.mean([biological_summary.get(f'mean_spatial_{f}_similarity', 0) for f in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'uni2h_spatial_complexity', 'uni2h_feature_magnitude']]):.4f}")
            logger.info(f"Overall Biological Plausibility: {biological_summary.get('mean_overall_biological_plausibility', 0):.4f}")
        logger.info("="*80)
        logger.info("RNA PREDICTION VALIDATION (Round-trip: RNA→Image→RNA):")
        logger.info("-" * 40)
        if rna_summary:
            logger.info(f"Gene-wise RNA Correlation: {rna_summary.get('mean_rna_image_gene_correlation_mean', 0):.4f} ± {rna_summary.get('std_rna_image_gene_correlation_mean', 0):.4f}")
            logger.info(f"Sample-wise RNA Correlation: {rna_summary.get('mean_rna_image_sample_correlation_mean', 0):.4f} ± {rna_summary.get('std_rna_image_sample_correlation_mean', 0):.4f}")
            logger.info(f"Overall RNA Correlation: {rna_summary.get('mean_rna_image_overall_correlation', 0):.4f} ± {rna_summary.get('std_rna_image_overall_correlation', 0):.4f}")
            logger.info(f"RNA MSE: {rna_summary.get('mean_rna_image_mse_mean', 0):.4f} ± {rna_summary.get('std_rna_image_mse_mean', 0):.4f}")
            logger.info(f"Valid Gene Correlations: {rna_summary.get('mean_num_valid_gene_correlations', 0):.0f}")
            logger.info(f"Valid Sample Correlations: {rna_summary.get('mean_num_valid_sample_correlations', 0):.0f}")
            logger.info(f"Genes Compared: {rna_summary.get('mean_genes_compared', 0):.0f}")
        else:
            logger.info("No RNA prediction validation performed (HE2RNA model not provided)")
        logger.info("="*80)

        # Save summary to JSON
        with open(os.path.join(args.output_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=2)

        # Save per-sample metrics to CSV
        per_sample_df = pd.DataFrame(per_sample_metrics_list)
        per_sample_df.to_csv(os.path.join(args.output_dir, 'per_sample_metrics.csv'), index=False)

        # Save batch-level biological validation metrics separately
        biological_df = pd.DataFrame(all_biological_results)
        biological_df.to_csv(os.path.join(args.output_dir, 'batch_level_biological_validation.csv'), index=False)

        # Create and save all plots using the modular plotting functions
        plot_paths = save_all_evaluation_plots(
            args.output_dir,
            all_ssim_scores,
            all_psnr_scores,
            valid_fids,
            valid_uni2h_fids,
            all_biological_results,
            all_rna_prediction_metrics,
            results_summary,
            mean_ssim,
            mean_psnr,
            mean_batch_fid,
            mean_uni2h_fid,
            gene_names
        )
        
        logger.info(f"evaluation results saved to {args.output_dir}")

    # Cleanup DDP
    if args.use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()