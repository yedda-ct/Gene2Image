import hashlib
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from rectified.rectified_flow import DOPRI5Solver


def _deterministic_noise(sample_ids, img_channels, img_size, noise_seed=0):
    """Build a fixed initial-noise tensor keyed by each sample's stable id.

    The noise for a given sample depends ONLY on (noise_seed, sample_id) via a
    salted SHA-256 hash, never on the global RNG state. This means two different
    model variants (run as separate processes, whose model-init consumes the
    global RNG by different amounts) still generate the SAME cell from the SAME
    x(t=0). Comparisons across variants / conditions then reflect the model, not
    the noise draw.
    """
    noises = []
    for sid in sample_ids:
        h = hashlib.sha256(f"{noise_seed}:{sid}".encode()).hexdigest()
        seed = int(h[:16], 16)
        g = torch.Generator()  # CPU generator -> reproducible regardless of device
        g.manual_seed(seed)
        noises.append(torch.randn(img_channels, img_size, img_size, generator=g))
    return torch.stack(noises, dim=0)


def generate_images_with_rectified_flow(
    model,
    rectified_flow,
    gene_expr,
    device,
    num_steps=100,
    gene_mask=None,
    num_cells=None,
    is_multi_cell=False,
    sample_ids=None,
    noise_seed=0,
):
    """
    Generate cell images from gene expression profiles using rectified flow and DOPRI5 solver
    
    Args:
        model: The RNA to H&E model (can be DDP wrapped)
        rectified_flow: The rectified flow module
        gene_expr: RNA expression tensor
        device: Computation device
        num_steps: Number of steps for the solver
        gene_mask: Optional gene mask tensor
        num_cells: Optional number of cells per patch for multi-cell model
        is_multi_cell: Whether using multi-cell model
        
    Returns:
        Generated images tensor
    """
    # Handle DDP model wrapping for inference
    actual_model = model.module if isinstance(model, DDP) else model
    
    # Create the solver with modified forward method for multi-cell model
    if is_multi_cell:
        class MultiCellModelWrapper:
            def __init__(self, model):
                self.model = model
                self.img_channels = model.img_channels
                self.img_size = model.img_size
                
            def __call__(self, x, t, rna_expr):
                # Forward gene_mask as None if not provided
                return self.model(x, t, rna_expr, num_cells, gene_mask)
                
        model_wrapper = MultiCellModelWrapper(actual_model)
        solver = DOPRI5Solver(model_wrapper, rectified_flow)
    else:
        # For single-cell model, use standard wrapper
        class SingleCellModelWrapper:
            def __init__(self, model):
                self.model = model
                self.img_channels = model.img_channels
                self.img_size = model.img_size
                
            def __call__(self, x, t, rna_expr):
                return self.model(x, t, rna_expr, gene_mask)
                
        model_wrapper = SingleCellModelWrapper(actual_model)
        solver = DOPRI5Solver(model_wrapper, rectified_flow)

    # Paired initial noise: when sample ids are supplied, derive a fixed x(t=0)
    # per sample so the same cell starts from the same noise across every variant
    # / condition. Without ids, fall back to a fresh random draw (unpaired).
    fixed_noise = None
    if sample_ids is not None:
        fixed_noise = _deterministic_noise(
            sample_ids, model_wrapper.img_channels, model_wrapper.img_size, noise_seed
        )

    # Generate images
    generated_images = solver.generate_sample(
        rna_expr=gene_expr,
        num_steps=num_steps,
        device=device,
        noise=fixed_noise,
    )
    
    # Denormalize images
    generated_images = torch.clamp(generated_images, 0, 1)
    
    return generated_images

