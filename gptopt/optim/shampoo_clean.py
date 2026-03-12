import torch
from torch.optim import Optimizer
import math


class ShampooClean(Optimizer):
    """
    Clean ungrafted Shampoo implementation following the Clarifying Shampoo paper
    (Eschenhagen et al., 2025, arXiv:2602.09314).

    Standard Shampoo update (Eq. 4):
        L_t = β₂ L_{t-1} + (1-β₂) G_t G_tᵀ
        R_t = β₂ R_{t-1} + (1-β₂) G_tᵀ G_t
        update = (L_t + εI)^{-p} M_t (R_t + εI)^{-p}

    EShampoo mode (eshampoo=True): eigenvalue-corrected Shampoo (SOAP).
    Uses eigenvectors from gradient covariance but replaces Kronecker
    eigenvalues with per-element second moments (Adam in eigenbasis):
        Q_L, Q_R = eigenvectors of L_t, R_t
        G̃ = Q_Lᵀ G Q_R  (project to eigenbasis)
        D_t = β₂ D_{t-1} + (1-β₂) G̃²  (per-element second moment)
        update = Q_L (M̃ / (D_t^p + ε)) Q_Rᵀ

    2D non-embedding params get Shampoo; everything else gets AdamW.
    No grafting, no block partitioning, no Newton-Schulz iterations.
    """

    def __init__(
        self,
        named_params,
        lr=0.016,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        beta2=0.8,
        epsilon=1e-15,
        exponent=0.5,
        momentum_after=False,
        use_bias_correction=True,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        eshampoo=False,
        trace_scaling=False,
        kl_shampoo=False,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        if trace_scaling and eshampoo:
            raise ValueError("trace_scaling=True is not meaningful with eshampoo=True")
        if kl_shampoo and eshampoo:
            raise ValueError("kl_shampoo=True is not supported with eshampoo=True")

        self.beta2 = beta2
        self.epsilon = epsilon
        self.exponent = exponent
        self.momentum_after = momentum_after
        self.use_bias_correction = use_bias_correction
        self.eshampoo = eshampoo
        self.trace_scaling = trace_scaling
        self.kl_shampoo = kl_shampoo

        shampoo_params, shampoo_names = [], []
        adamw_params, adamw_names = [], []
        excluded = ["embeddings", "embed_tokens", "wte", "lm_head", "weight_proj", "wpe"]

        for name, p in named_params:
            if p.ndim >= 2 and not any(ex in name for ex in excluded):
                shampoo_params.append(p)
                shampoo_names.append(name)
            else:
                adamw_params.append(p)
                adamw_names.append(name)

        print(f"\n=== ShampooClean Optimizer Parameter Classification ===")
        print(f"Shampoo Parameters ({len(shampoo_params)}):")
        for name in shampoo_names:
            print(f"  - {name}")
        print(f"\nAdamW Parameters ({len(adamw_params)}):")
        for name in adamw_names:
            print(f"  - {name}")
        print(f"=======================================================\n")

        params = list(shampoo_params) + list(adamw_params)
        super().__init__(params, defaults)

        for p in shampoo_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_shampoo"] = True
        for p in adamw_params:
            self.state[p]["use_shampoo"] = False

    @torch.no_grad()
    def _matrix_inv_root(self, C, exponent):
        """Compute C^{-exponent} via eigendecomposition with eigenvalue perturbation.

        Matches the PyTorch Distributed Shampoo reference (PerturbationConfig):
        1. Add ε*I before eigendecomposition for numerical stability
        2. eigh on perturbed matrix
        3. Shift eigenvalues: if lambda_min < ε, shift all by -lambda_min then add ε
        4. Raise to -exponent
        """
        C_reg = C + self.epsilon * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
        try:
            eigvals, eigvecs = torch.linalg.eigh(C_reg)
        except torch.linalg.LinAlgError:
            # Promote to float64 for ill-conditioned matrices (matches reference retry_double_precision)
            eigvals, eigvecs = torch.linalg.eigh(C_reg.to(torch.float64))
            eigvals = eigvals.to(C.dtype)
            eigvecs = eigvecs.to(C.dtype)
        lambda_min = eigvals.min().item()
        if lambda_min < self.epsilon:
            eigvals = eigvals - lambda_min + self.epsilon
        inv_root_eigvals = eigvals.pow(-exponent)
        return (eigvecs * inv_root_eigvals.unsqueeze(0)) @ eigvecs.T

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]

            ############################
            #        Shampoo           #
            ############################

            for p in group["params"]:
                if not self.state[p].get("use_shampoo", False):
                    continue
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                m, n = g.shape

                # Initialize state
                if "step" not in state:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["L"] = torch.zeros(m, m, device=g.device, dtype=torch.float32)
                    state["R"] = torch.zeros(n, n, device=g.device, dtype=torch.float32)
                    if self.kl_shampoo:
                        state["L_inv_root"] = torch.eye(m, device=g.device, dtype=torch.float32)
                        state["R_inv_root"] = torch.eye(n, device=g.device, dtype=torch.float32)

                state["step"] += 1
                step = state["step"]
                buf = state["momentum_buffer"]

                # Weight decay (decoupled)
                p.data.mul_(1 - lr * wd)

                # Update covariance matrices using outer products in original grad dtype
                # (matches DistributedShampoo: torch.tensordot on bfloat16 gradients).
                # The bfloat16 rounding noise in the outer product provides implicit
                # regularization for rank-deficient covariance matrices.
                L = state["L"]
                R = state["R"]
                if self.kl_shampoo:
                    L_inv_root = state["L_inv_root"]
                    R_inv_root = state["R_inv_root"]
                    # Cast preconditioned grad back to original dtype before outer product
                    # (matches Meta's _precondition_grad which returns grad.dtype)
                    G_R = (g.float() @ R_inv_root).to(g.dtype)
                    G_L = (L_inv_root @ g.float()).to(g.dtype)
                    L.mul_(self.beta2).add_((G_R @ G_R.T).float(), alpha=1 - self.beta2)
                    R.mul_(self.beta2).add_((G_L.T @ G_L).float(), alpha=1 - self.beta2)
                else:
                    L.mul_(self.beta2).add_(
                        (g @ g.T).float(), alpha=1 - self.beta2
                    )
                    R.mul_(self.beta2).add_(
                        (g.T @ g).float(), alpha=1 - self.beta2
                    )

                # Compute momentum (EMA-style, matching reference)
                buf.mul_(momentum).add_(g, alpha=1 - momentum)
                if self.use_bias_correction and momentum > 0:
                    bc1 = 1 - momentum ** step
                    g_mom = buf / bc1
                else:
                    g_mom = buf
                if nesterov:
                    g_mom = g + momentum * g_mom

                # Bias-correct covariance before computing preconditioners
                # Use float32 tensor arithmetic to match Meta's DistributedShampoo reference
                # (Python float64 bc2 causes tiny rounding diffs that amplify in KL feedback loop)
                if self.use_bias_correction and self.beta2 < 1.0:
                    bc2 = torch.tensor(1.0) - self.beta2 ** torch.tensor(step)
                    L_bc = L / bc2
                    R_bc = R / bc2
                else:
                    L_bc, R_bc = L, R
                    bc2 = 1.0

                if self.eshampoo:
                    # EShampoo: Adam in the eigenbasis of Shampoo's preconditioner
                    # Eigendecompose covariances for eigenbasis (no inverse root needed)
                    L_reg = L_bc + self.epsilon * torch.eye(m, device=g.device, dtype=L_bc.dtype)
                    R_reg = R_bc + self.epsilon * torch.eye(n, device=g.device, dtype=R_bc.dtype)
                    try:
                        _, Q_L = torch.linalg.eigh(L_reg)
                    except torch.linalg.LinAlgError:
                        _, Q_L = torch.linalg.eigh(L_reg.to(torch.float64))
                        Q_L = Q_L.to(L_bc.dtype)
                    try:
                        _, Q_R = torch.linalg.eigh(R_reg)
                    except torch.linalg.LinAlgError:
                        _, Q_R = torch.linalg.eigh(R_reg.to(torch.float64))
                        Q_R = Q_R.to(R_bc.dtype)

                    # Project gradient into eigenbasis: G̃ = Q_Lᵀ g Q_R
                    G_tilde = Q_L.T @ g.float() @ Q_R

                    # Track per-element second moment D_t (Adam-style in eigenbasis)
                    if "D_eshampoo" not in state:
                        state["D_eshampoo"] = torch.zeros(m, n, device=g.device, dtype=torch.float32)
                    D = state["D_eshampoo"]
                    D.mul_(self.beta2).add_(G_tilde.square(), alpha=1 - self.beta2)

                    # Bias-correct D
                    D_corrected = D / bc2

                    # Project momentum into eigenbasis and scale by corrected eigenvalues
                    M_tilde = Q_L.T @ g_mom.float() @ Q_R
                    update_tilde = M_tilde / (D_corrected.pow(self.exponent).add_(self.epsilon))

                    # Project back to parameter space
                    update = Q_L @ update_tilde @ Q_R.T
                else:
                    # Standard Shampoo: compute inverse root preconditioners
                    L_inv = self._matrix_inv_root(L_bc, self.exponent)
                    R_inv = self._matrix_inv_root(R_bc, self.exponent)

                    # Cache inverse roots for next step's KL-Shampoo covariance update
                    if self.kl_shampoo:
                        state["L_inv_root"] = L_inv
                        state["R_inv_root"] = R_inv

                    # Apply preconditioning
                    if self.momentum_after:
                        # LaProp-style: precondition raw gradient, then momentum on update
                        precond_g = L_inv @ g.float() @ R_inv
                        if self.trace_scaling:
                            precond_g = precond_g / (L_bc.trace().clamp(min=self.epsilon) * R_bc.trace().clamp(min=self.epsilon))

                        if "update_buffer" not in state:
                            state["update_buffer"] = torch.zeros_like(precond_g)
                        ubuf = state["update_buffer"]
                        ubuf.mul_(momentum).add_(precond_g)
                        if nesterov:
                            update = precond_g + momentum * ubuf
                        else:
                            update = ubuf
                    else:
                        # Standard: precondition momentum buffer
                        update = L_inv @ g_mom.float() @ R_inv
                        if self.trace_scaling:
                            update = update / (L_bc.trace().clamp(min=self.epsilon) * R_bc.trace().clamp(min=self.epsilon))

                p.data.add_(update.to(p.dtype), alpha=-lr)

            ############################
            #       AdamW backup       #
            ############################

            for p in group["params"]:
                if self.state[p].get("use_shampoo", False):
                    continue
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                beta1, beta2 = group["adamw_betas"]
                eps = group["adamw_eps"]

                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]

                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                update = buf1 / (buf2.sqrt() + eps)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                scale = bias_correction1 / bias_correction2 ** 0.5

                p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-lr / scale)

        return loss
