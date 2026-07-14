import math
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class RectifiedFlow:
    """
    Implements rectified flow dynamics for generative modeling.
    Rectified flow provides a direct path between noise and data distributions.
    """
    def __init__(self, sigma_min=0.002, sigma_max=80.0):
        """
        Initialize the rectified flow model.
        
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def noise_schedule(self, t):
        """
        Noise schedule that determines the noise level at time t.
        
        Args:
            t: Time variable in [0, 1]
            
        Returns:
            Noise level sigma(t)
        """
        # Cosine noise schedule
        return self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - t)
    
    def drift_coefficient(self, t):
        """
        Calculate the drift coefficient for the rectified flow.
        
        Args:
            t: Time variable in [0, 1]
            
        Returns:
            Drift coefficient
        """
        return 1.0
    
    def sample_path(self, x_1, t, noise=None):
        """
        Sample from the path at time t using non-linear interpolation and stochastic noise.
        
        Args:
            x_1: Target data sample (B, C, H, W)
            t: Time variable in [0, 1] (B,)
            noise: Optional noise to use (B, C, H, W)
            
        Returns:
            Dictionary containing:
                x_t: Sample at time t
                velocity: Ground truth velocity
        """
        batch_size = x_1.shape[0]
        
        # Expand time dimension for broadcasting
        t_expanded = t.view(-1, *([1] * (len(x_1.shape) - 1)))
        
        # Get noise if not provided
        if noise is None:
            noise = torch.randn_like(x_1)
        
        # Non-linear interpolation using sinusoidal schedule for smoother trajectories
        # This creates a more natural path between noise and data
        interp_coef = torch.sin(t_expanded * (math.pi / 2))
        
        # Interpolate between noise and data with non-linear coefficient
        x_t = interp_coef * x_1 + (1 - interp_coef) * noise
        
        # FIX (2026-07): the previous implementation added a stochastic term
        # 0.05*(1-t)*xi to x_t, but its time-derivative (-0.05*xi) was NOT included in
        # the supervised velocity below, so the target was not the exact derivative of
        # the training path (it injected irreducible, zero-mean label noise into the
        # regression target). We remove that unmatched term so that x_t lies on the
        # clean sinusoidal path and `velocity` is exactly d x_t / dt.

        # Velocity of the non-linear path = d/dt [ sin(t*pi/2)*x_1 + (1-sin(t*pi/2))*noise ]
        velocity = (x_1 - noise) * (math.pi / 2) * torch.cos(t_expanded * (math.pi / 2))
        
        return {
            "x_t": x_t,
            "velocity": velocity,
            "noise": noise,
            "sigma_t": self.noise_schedule(t_expanded)
        }
    
    def loss_fn(self, model_output, target_velocity):
        """
        Compute the loss between predicted and target velocities.
        
        Args:
            model_output: Predicted velocity from the model
            target_velocity: Target velocity from the path
            
        Returns:
            Loss value
        """
        return torch.mean((model_output - target_velocity) ** 2)


class EulerSolver:
    """
    Implements Euler method for solving ODEs in the context of generative modeling.
    """
    def __init__(self, model, rectified_flow):
        """
        Initialize the Euler solver.
        
        Args:
            model: The neural network model that predicts velocities
            rectified_flow: The rectified flow object
        """
        self.model = model
        self.rf = rectified_flow
        
    def generate_sample(self, rna_expr, num_steps=100, device="cuda", noise=None):
        """
        Generate a sample by solving the ODE using Euler method.

        Args:
            rna_expr: RNA expression data (B, gene_dim)
            num_steps: Number of steps for the Euler method
            device: The device to run the computation on
            noise: Optional fixed initial noise (B, C, H, W). When provided it is used
                verbatim as x(t=0) instead of a fresh draw, so paired comparisons
                (variants / interventions) can share one noise realisation per sample.

        Returns:
            Generated sample
        """
        batch_size = rna_expr.shape[0]

        # Get shapes from the model
        img_channels = self.model.img_channels
        img_size = self.model.img_size

        # Start from noise (t=0): injected (paired) if given, else a fresh draw.
        if noise is not None:
            x = noise.to(device)
        else:
            x = torch.randn(batch_size, img_channels, img_size, img_size, device=device)
        
        # Create time steps (from t=0 to t=1)
        dt = 1.0 / num_steps
        times = torch.linspace(0, 1 - dt, num_steps, device=device)
        
        # Euler integration
        for i, t in enumerate(times):
            # Expand t for batch
            t_batch = torch.ones(batch_size, device=device) * t
            
            # Get velocity prediction from model
            with torch.no_grad():
                velocity = self.model(x, t_batch, rna_expr)
            
            # Update x using Euler method
            x = x + velocity * dt
            
            # Optional: Add progress logging for long generations
            if i % (num_steps // 10) == 0:
                logger.info(f"Generation progress: {i}/{num_steps} steps")
        
        # Clamp to valid image range
        x = torch.clamp(x, 0.0, 1.0)  # data range is [0,1] (ToTensor, no [-1,1] norm); match train/eval
        
        return x

class DOPRI5Solver:
    """
    Implements the Dormand-Prince (DOPRI5) method for solving ODEs.
    This is a fifth-order Runge-Kutta method with adaptive step size
    control and error estimation.
    """
    def __init__(self, model, rectified_flow, rtol=1e-3, atol=1e-4, safety=0.9):
        """
        Initialize the DOPRI5 solver.
        
        Args:
            model: The neural network model that predicts velocities
            rectified_flow: The rectified flow object
            rtol: Relative tolerance for adaptive step size control
            atol: Absolute tolerance for adaptive step size control
            safety: Safety factor for step size adjustments
        """
        self.model = model
        self.rf = rectified_flow
        self.rtol = rtol
        self.atol = atol
        self.safety = safety
        
        # Butcher tableau coefficients for Dormand-Prince method
        self.a = [
            [],  # a[0] is unused
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
        ]
        
        self.b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]  # 5th order solution
        self.b_star = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]  # 4th order solution for error estimation
        
        self.c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    
    def _dormand_prince_step(self, x, t, dt, rna_expr, batch_size, device):
        """
        Perform one step of the Dormand-Prince method.
        
        Args:
            x: Current state
            t: Current time
            dt: Time step
            rna_expr: RNA expression data
            batch_size: Batch size
            device: Computation device
            
        Returns:
            Tuple of (next_state, error_estimate)
        """
        with torch.no_grad():
            # Six stages of the Dormand-Prince method
            k = []
            
            # First stage
            t_batch = torch.ones(batch_size, device=device) * t
            k.append(self.model(x, t_batch, rna_expr))
            
            # Remaining stages
            for i in range(1, 6):
                # Create intermediate state for this stage
                x_i = x.clone()
                for j in range(i):
                    x_i = x_i + dt * self.a[i][j] * k[j]
                
                # Compute new stage value
                t_i = t + self.c[i] * dt
                t_i_batch = torch.ones(batch_size, device=device) * t_i
                k.append(self.model(x_i, t_i_batch, rna_expr))
            
            # Compute the 5th-order solution from stages k[0..5]. In Dormand-Prince
            # this point is ALSO the node for the 7th (FSAL) stage: the tableau's
            # a[6] row equals the b weights, so stage 7 is evaluated at x_next, t+dt.
            x_next = x.clone()
            for i in range(6):
                x_next = x_next + dt * self.b[i] * k[i]

            # Compute the 7th (FSAL) stage at (x_next, t+dt). The previous code
            # evaluated it at the 6th-stage node (reusing self.a[5] / range(5)), so
            # x_final == the 6th-stage input and t_final == t + c[5]*dt == t + dt,
            # yielding k[6] == k[5]. That corrupts the 4th-order estimate x_next_star
            # (its b_star[6] term) and hence the error driving adaptive step control.
            # The accepted x_next is unaffected (it uses only k[0..5]).
            t_final = t + dt
            t_final_batch = torch.ones(batch_size, device=device) * t_final
            k.append(self.model(x_next, t_final_batch, rna_expr))

            # Compute 4th order solution for error estimation (uses all 7 stages).
            x_next_star = x.clone()
            for i in range(7):
                x_next_star = x_next_star + dt * self.b_star[i] * k[i]

            # Error estimate
            error = x_next - x_next_star

            return x_next, error
    
    def _compute_adaptive_step_size(self, error_norm, dt):
        """New step size from the (per-element weighted RMS) error_norm.

        error_norm <= 1 means the step is within tolerance. dt is a python float.
        """
        if error_norm <= 0.0:
            # Error ~ 0: grow aggressively (clamped below).
            return dt * 5.0
        # Standard adaptive step-size formula, clamped to a bounded per-step change.
        dt_new = self.safety * dt * (1.0 / error_norm) ** (1.0 / 5.0)
        return min(max(dt_new, dt * 0.1), dt * 5.0)
    
    def generate_sample(self, rna_expr, num_steps=100, initial_dt=0.01, device="cuda", noise=None):
        """
        Generate a sample by solving the ODE using the DOPRI5 method
        with adaptive step size control.

        Args:
            rna_expr: RNA expression data (B, gene_dim)
            num_steps: Maximum number of steps for the DOPRI5 method
            initial_dt: Initial time step
            device: The device to run the computation on
            noise: Optional fixed initial noise (B, C, H, W). When provided it is used
                verbatim as x(t=0) instead of a fresh draw, so paired comparisons
                (variants / interventions) can share one noise realisation per sample.

        Returns:
            Generated sample
        """
        batch_size = rna_expr.shape[0]

        # Get shapes from the model
        img_channels = self.model.img_channels
        img_size = self.model.img_size

        # Start from noise (t=0): injected (paired) if given, else a fresh draw.
        if noise is not None:
            x = noise.to(device)
        else:
            x = torch.randn(batch_size, img_channels, img_size, img_size, device=device)
        
        # Initialize variables for adaptive stepping
        t = 0.0
        dt = initial_dt
        step_count = 0
        n_reject = 0     # steps rejected because the error exceeded tolerance
        n_dt_floor = 0   # steps force-accepted at the dt floor (stiff point)
        n_fallback = 0   # 1 if the under-integration fallback had to finish to t=1
        
        # Create history arrays for debugging
        t_history = [t]
        dt_history = [dt]
        
        # Integration with adaptive step size + embedded error control.
        while t < 1.0 and step_count < num_steps:
            # Ensure we don't overshoot t=1.0
            if t + dt > 1.0:
                dt = 1.0 - t

            # Perform one step of DOPRI5
            x_next, error = self._dormand_prince_step(x, t, dt, rna_expr, batch_size, device)

            # Embedded-error step control with the STANDARD weighted RMS error: scale each
            # error component by its own tolerance atol + rtol*max(|x|,|x_next|), take the
            # per-IMAGE RMS (over C,H,W), then the batch MAX (worst image). error_norm<=1
            # means the step is within tolerance. This replaces a global ||err||/||x|| ratio
            # whose meaning drifted with image size / batch. ACCEPT if within tolerance;
            # otherwise REJECT (do NOT advance t) and retry with a smaller step. A dt floor
            # force-accepts at a stiff point to avoid an infinite reject loop. The reject /
            # dt-floor / under-integration counts are logged so a pilot run can be inspected
            # before the formal evaluation is trusted.
            _scale = self.atol + self.rtol * torch.maximum(x.abs(), x_next.abs())
            # Per-IMAGE weighted RMS (mean over C,H,W per sample), then the WORST image in the
            # batch -> a genuine per-image tolerance guarantee: a batch of easy images cannot
            # dilute one hard image the way a single batch-wide mean would.
            _per_sample = torch.sqrt(torch.mean((error / _scale) ** 2,
                                                dim=tuple(range(1, error.dim()))))  # [B]
            error_norm = float(_per_sample.max())
            dt_next = float(self._compute_adaptive_step_size(error_norm, dt))
            _dt_min = initial_dt * 0.01
            if error_norm <= 1.0:
                x = x_next
                t += dt
            elif dt <= _dt_min:
                x = x_next
                t += dt
                n_dt_floor += 1     # force-accept at the dt floor (stiff point)
            else:
                n_reject += 1       # rejected -- x, t unchanged; retry with the smaller dt
            step_count += 1

            # Update step size for the next (retry or new) iteration
            dt = max(dt_next, _dt_min)

            # Store history
            t_history.append(t)
            dt_history.append(dt)
            
            # Progress logging
            if step_count % 10 == 0:
                logger.info(f"Generation progress: t={t:.4f}, dt={dt:.6f}, steps={step_count}")
        
        logger.info(f"DOPRI5 generation: {step_count} steps ({n_reject} rejected, "
                    f"{n_dt_floor} forced at dt-floor), final t={t:.4f}")

        # --- Under-integration guard ---
        # The adaptive step budget (num_steps) can be exhausted before t reaches 1.0 when
        # the velocity field is stiff (e.g. a poorly fit / weaker-ablation model), which
        # would otherwise return a partially-denoised (half-noise) sample and silently
        # corrupt FID/SSIM/PSNR. Finish the leftover interval with bounded fixed DOPRI5
        # steps so every returned sample is integrated to t=1.0.
        _eps = 1e-4
        if t < 1.0 - _eps:
            n_fallback = 1
            logger.warning(
                f"UNDER_INTEGRATION_FALLBACK: DOPRI5 step budget ({num_steps}) exhausted at "
                f"t={t:.4f} ({n_reject} rejected, {n_dt_floor} at dt-floor); completing to t=1.0 "
                f"with bounded fixed steps. FREQUENT fallbacks -> DO NOT trust the sample; raise "
                f"--gen_steps or relax rtol/atol and re-generate.")
            _max_dt = max(initial_dt, (1.0 - t) / 20.0)
            while t < 1.0 - _eps:
                _dt = min(_max_dt, 1.0 - t)
                x, _ = self._dormand_prince_step(x, t, _dt, rna_expr, batch_size, device)
                t += _dt
            logger.info(f"Under-integration completed to final t={t:.4f}.")
        
        # Log adaptive step size statistics
        if len(dt_history) > 1:
            logger.info(f"Step size statistics: min={min(dt_history):.6f}, max={max(dt_history):.6f}, avg={sum(dt_history)/len(dt_history):.6f}")

        # Clamp to valid image range
        x = torch.clamp(x, 0.0, 1.0)  # data range is [0,1] (ToTensor, no [-1,1] norm); match train/eval

        # Hard-gate diagnostics (grep this line): for a TRUSTWORTHY formal sample require
        # dt_floor=0 AND under_integration_fallback=0. rejected>0 is normal/healthy.
        logger.info(f"DOPRI5_DIAGNOSTICS rejected={n_reject} dt_floor={n_dt_floor} "
                    f"under_integration_fallback={n_fallback} final_t={t:.4f}")
        return x