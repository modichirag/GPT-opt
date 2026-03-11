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

    where M_t is the momentum buffer and p is the exponent (0.5 for Shampoo^{1/2}).

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
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        self.beta2 = beta2
        self.epsilon = epsilon
        self.exponent = exponent
        self.momentum_after = momentum_after
        self.use_bias_correction = use_bias_correction

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

        Matches the PyTorch Distributed Shampoo reference:
        1. eigh on raw matrix (no ε added before decomposition)
        2. Clamp negative eigenvalues to 0
        3. Add ε to all eigenvalues
        4. Raise to -exponent
        """
        try:
            eigvals, eigvecs = torch.linalg.eigh(C)
        except torch.linalg.LinAlgError:
            # Promote to float64 for ill-conditioned matrices (matches reference retry_double_precision)
            eigvals, eigvecs = torch.linalg.eigh(C.to(torch.float64))
            eigvals = eigvals.to(C.dtype)
            eigvecs = eigvecs.to(C.dtype)
        eigvals = eigvals.clamp(min=0.0) + self.epsilon
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

                state["step"] += 1
                step = state["step"]
                buf = state["momentum_buffer"]

                # Weight decay (decoupled)
                p.data.mul_(1 - lr * wd)

                # Update covariance matrices (using raw gradient, in float32)
                g_f32 = g.float()
                L = state["L"]
                R = state["R"]
                L.mul_(self.beta2).addmm_(g_f32, g_f32.T, alpha=1 - self.beta2)
                R.mul_(self.beta2).addmm_(g_f32.T, g_f32, alpha=1 - self.beta2)

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
                if self.use_bias_correction and self.beta2 < 1.0:
                    bc2 = 1 - self.beta2 ** step
                    L_bc = L / bc2
                    R_bc = R / bc2
                else:
                    L_bc, R_bc = L, R

                # Compute preconditioners
                L_inv = self._matrix_inv_root(L_bc, self.exponent)
                R_inv = self._matrix_inv_root(R_bc, self.exponent)

                # Apply preconditioning
                if self.momentum_after:
                    # LaProp-style: precondition raw gradient, then momentum on update
                    precond_g = L_inv @ g.float() @ R_inv

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
