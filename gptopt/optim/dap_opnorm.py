import torch
from torch import nn
from torch.optim import Optimizer
from gptopt.optim.dap import LinearWithXtX
from gptopt.optim.muon import PolarExpress

from typing import Optional, Dict


class DAPOpNorm(Optimizer):
    """
    DAP with operator-norm steepest descent (Lemma 2.3).

    Combines DAP's covariance tracking (LinearWithXtX, EMA) with Muon's
    Newton-Schulz orthogonalization.  The LMO update is:

        W -= lr * sign(G @ C^{-1/2}) @ C^{-1/2}

    where C = XX^T is the input covariance, sign() is the matrix sign
    computed via PolarExpress, and G is the (momentum-processed) gradient.
    """

    def __init__(
        self,
        model,
        named_params,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        ema_beta=0.0,
        ns_steps=5,
        rcond=1e-3,
        damping=0.0,
        opnorm_target=None,
        per_layer_damping=False,
        output_cov_mode=None,
        abs_damping=None,
        trace_damping=None,
        precond_only_opnorm=False,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        num_microbatches: Optional[int] = None,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        dap_params, dap_params_names = [], []
        adamw_params, adamw_params_names = [], []

        excluded_names = ["lm_head", "weight_proj", "embeddings", "embed_tokens", "wte", "wpe"]

        self.param_to_name = {}

        for name, p in named_params:
            self.param_to_name[p] = name

            if p.ndim >= 2 and not any(excluded in name for excluded in excluded_names):
                dap_params.append(p)
                dap_params_names.append(name)
            else:
                adamw_params.append(p)
                adamw_params_names.append(name)

        print(f"\n=== DAPOpNorm Optimizer Parameter Classification ===")
        print(f"DAPOpNorm Parameters ({len(dap_params)}):")
        for name in dap_params_names:
            print(f"  - {name}")
        print(f"\nAdamW Parameters ({len(adamw_params)}):")
        for name in adamw_params_names:
            print(f"  - {name}")
        print(f"=====================================================\n")

        params = list(dap_params) + list(adamw_params)
        super().__init__(params, defaults)

        for p in dap_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_dap_opnorm"] = True

        for p in adamw_params:
            self.state[p]["use_dap_opnorm"] = False

        self.ema_beta = ema_beta
        self.ns_steps = ns_steps
        self.rcond = float(rcond)
        self.damping = float(damping)
        self.opnorm_target = float(opnorm_target) if opnorm_target is not None else None
        self.per_layer_damping = per_layer_damping

        if output_cov_mode not in (None, "sign_input", "kfac_sign", "full", "shampoo_sign", "kfac", "shampoo", "eshampoo"):
            raise ValueError(f"output_cov_mode must be None, 'sign_input', 'kfac_sign', 'full', 'shampoo_sign', 'kfac', 'shampoo', or 'eshampoo', got {output_cov_mode}")
        self.output_cov_mode = output_cov_mode
        self.abs_damping = float(abs_damping) if abs_damping is not None else None
        self.trace_damping = float(trace_damping) if trace_damping is not None else None
        self.precond_only_opnorm = precond_only_opnorm

        self.dap_modules = []
        self.param_to_module: Dict[nn.Parameter, LinearWithXtX] = {}
        self._step_cov: Dict[nn.Parameter, torch.Tensor] = {}
        self._step_cov_out: Dict[nn.Parameter, torch.Tensor] = {}

        self._wire_up_xtx_sources(model, dap_params)

    def _wire_up_xtx_sources(self, model: nn.Module, dap_params):
        dap_param_ids = {id(p) for p in dap_params}

        for mod in model.modules():
            if isinstance(mod, nn.Linear) and id(mod.weight) in dap_param_ids:
                if not isinstance(mod, LinearWithXtX):
                    raise TypeError(
                        "DAPOpNorm requires LinearWithXtX. Call swap_linears_for_xtx(model) first."
                    )
                mod._dap_accum_enabled = (self.output_cov_mode not in ("shampoo_sign", "shampoo", "eshampoo"))
                mod._accum_output_cov = (self.output_cov_mode in ("kfac_sign", "full", "kfac"))
                self.param_to_module[mod.weight] = mod
                self.dap_modules.append(mod)

        if len(self.param_to_module) != len(dap_params):
            missing = [p for p in dap_params if p not in self.param_to_module]
            raise ValueError(
                f"Some DAPOpNorm params not owned by any LinearWithXtX: {missing}"
            )

    def _compute_C_inv_sqrt(self, C: torch.Tensor, damping=None, opnorm_target=None, rcond=None, abs_damping=None, trace_damping=None):
        """Compute C^{-1/2} via eigendecomposition with optional damping.

        Args:
            C: the covariance matrix
            damping: relative damping coefficient; if None, uses self.damping.
                Ignored when opnorm_target, abs_damping, or trace_damping is set.
                Relative: C_eff = C + damping * lambda_max(C) * I
            opnorm_target: if set, compute per-layer relative damping to achieve
                this target opnorm exactly: delta = (1/opnorm_target^2 - lambda_min) / lambda_max,
                clamped >= 0.
            abs_damping: absolute damping (standard KFAC/Shampoo style).
                C_eff = C + abs_damping * I.  Takes precedence over damping but
                not over opnorm_target.
            trace_damping: trace-scaled damping (Ishikawa & Karakida 2023).
                C_eff = C + trace_damping * (tr(C)/d) * I.  Takes precedence
                over damping but not over opnorm_target or abs_damping.

        Returns:
            (C_inv_sqrt, opnorm, C_eigmax, C_eigmin, delta_used): C_inv_sqrt is the matrix
            square root inverse; opnorm = 1/sqrt(min active eigval of C_eff);
            C_eigmax and C_eigmin are the max and min eigenvalues of C *before*
            damping (among active directions); delta_used is the relative damping
            coefficient actually applied (useful when opnorm_target computes it).
        """
        C_f32 = C.to(torch.float32)

        # For trace_damping, pre-damp C before eigendecomposition to stabilize eigh
        if trace_damping is not None and opnorm_target is None and abs_damping is None:
            d = C_f32.shape[0]
            trace_val = C_f32.trace().item()
            eps = trace_damping * trace_val / d
            C_f32 = C_f32 + eps * torch.eye(d, device=C_f32.device, dtype=C_f32.dtype)

        # Single eigendecomposition of C (after pre-damping for trace, before damping otherwise)
        # Fall back to SVD if eigh fails (ill-conditioned matrices during LR cooldown)
        try:
            eigvals_C, eigvecs = torch.linalg.eigh(C_f32)
        except torch._C._LinAlgError:
            U, S, Vh = torch.linalg.svd(C_f32)
            eigvals_C, eigvecs = S.flip(0), U.flip(1)
        eigmax_C = eigvals_C[-1].item()

        if opnorm_target is not None:
            # Compute per-layer delta to hit target opnorm exactly
            # opnorm = 1/sqrt(lambda_min(C_eff)) = 1/sqrt(lambda_min(C) + delta * lambda_max(C))
            # => delta = (1/opnorm_target^2 - lambda_min(C)) / lambda_max(C)
            eigmin_C_raw = eigvals_C[0].item()
            target_min_eigval = 1.0 / (opnorm_target ** 2)
            if eigmax_C > 0:
                damping = max(0.0, (target_min_eigval - eigmin_C_raw) / eigmax_C)
            else:
                damping = self.damping
        elif abs_damping is not None:
            # Standard KFAC/Shampoo damping: C_eff = C + epsilon * I
            # Convert to "relative" form for the eigenvalue addition below
            damping = abs_damping / eigmax_C if eigmax_C > 0 else 0.0
        elif trace_damping is not None:
            # Already pre-damped above; no additional damping needed
            damping = 0.0
        elif damping is None:
            damping = self.damping

        # Apply relative damping to eigenvalues directly (avoids second decomp)
        eigvals_eff = eigvals_C + damping * eigmax_C if damping > 0 else eigvals_C

        # rcond truncation: skip when opnorm_target is set (damping handles regularization)
        if opnorm_target is not None:
            mask = eigvals_eff > 0
        else:
            threshold = (rcond if rcond is not None else self.rcond) * eigvals_eff.max()
            mask = eigvals_eff > threshold

        inv_sqrt_vals = torch.zeros_like(eigvals_eff)
        inv_sqrt_vals[mask] = 1.0 / eigvals_eff[mask].sqrt()

        C_inv_sqrt = (eigvecs * inv_sqrt_vals.unsqueeze(0)) @ eigvecs.T

        if mask.any():
            opnorm = inv_sqrt_vals.max().item()
            eigmin_C = eigvals_C[mask].min().item()
        else:
            opnorm = float('inf')
            eigmin_C = 0.0

        return C_inv_sqrt, opnorm, eigmax_C, eigmin_C, damping

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.diagnostics = {'C_inv_sqrt_opnorm': [], 'C_eigmax': [], 'C_eigmin': [], 'delta_adaptive': None, 'delta_per_layer': [],
                             'C_out_eigmax': [], 'C_out_eigmin': [], 'delta_out_per_layer': []}

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            ############################
            #       DAPOpNorm          #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_dap_opnorm"]]

            # Finalize accumulated input covariance
            self._step_cov.clear()
            for p, mod in self.param_to_module.items():
                count = mod.C_accum_count.item()
                if not count:
                    continue

                C_mean = mod.C_accum_sum / count
                state = self.state[p]

                if self.ema_beta == 0.0:
                    self._step_cov[p] = C_mean
                else:
                    C_prev = state.get("C_ema", None)
                    if C_prev is None:
                        state["C_ema"] = C_mean.detach().clone()
                    else:
                        C_prev.mul_(self.ema_beta).add_(C_mean, alpha=(1.0 - self.ema_beta))

                mod.C_accum_sum.zero_()
                mod.C_accum_count.zero_()

            # Finalize accumulated output gradient covariance
            self._step_cov_out: Dict[torch.nn.Parameter, torch.Tensor] = {}
            if self.output_cov_mode is not None:
                for p, mod in self.param_to_module.items():
                    if mod.S_accum_count is None:
                        continue
                    count = mod.S_accum_count.item()
                    if not count:
                        continue

                    S_mean = mod.S_accum_sum / count
                    state = self.state[p]

                    if self.ema_beta == 0.0:
                        self._step_cov_out[p] = S_mean
                    else:
                        C_out_prev = state.get("C_out_ema", None)
                        if C_out_prev is None:
                            state["C_out_ema"] = S_mean.detach().clone()
                        else:
                            C_out_prev.mul_(self.ema_beta).add_(S_mean, alpha=(1.0 - self.ema_beta))

                    mod.S_accum_sum.zero_()
                    mod.S_accum_count.zero_()

            # Sign-only modes (sign_input, kfac_sign) produce ~orthogonal updates
            # (opnorm ≈ 1), so opnorm_target doesn't control update scale via damping.
            # Instead: use no damping for whitening (rcond handles stability),
            # then scale the update by opnorm_target.
            # Non-sign modes (None, full) use opnorm_target to control damping directly.
            is_sign_mode = self.output_cov_mode in ("sign_input", "kfac_sign", "shampoo_sign"
                                                       )  # kfac, shampoo are NOT sign modes

            # First pass: compute global adaptive damping (only for non-sign modes)
            if self.opnorm_target is not None and not self.per_layer_damping and not is_sign_mode:
                eigmax_vals = []
                for p in params:
                    state_p = self.state[p]
                    C = self._step_cov.get(p, state_p.get("C_ema", None))
                    if C is not None:
                        lam_max = torch.linalg.eigh(C.to(torch.float32))[0][-1].item()
                        eigmax_vals.append(lam_max)
                if eigmax_vals:
                    mean_inv_sqrt_eigmax = sum(1.0 / v ** 0.5 for v in eigmax_vals) / len(eigmax_vals)
                    delta_adaptive = mean_inv_sqrt_eigmax ** 2 / self.opnorm_target ** 2
                else:
                    delta_adaptive = self.damping
                self.diagnostics['delta_adaptive'] = delta_adaptive
            else:
                delta_adaptive = None

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1

                # Momentum
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g_mom = g.add(buf, alpha=momentum)
                else:
                    g_mom = buf

                # EShampoo: eigenvalue-corrected Shampoo (SOAP-like)
                # Uses eigenvectors from gradient covariance but replaces Kronecker
                # eigenvalues with per-element second moments (like Adam in eigenbasis).
                if self.output_cov_mode == "eshampoo":
                    R = g.t() @ g          # [d_in, d_in]
                    L = g @ g.t()          # [d_out, d_out]

                    # EMA of covariance for eigenbasis
                    beta2 = self.ema_beta
                    if beta2 > 0.0:
                        if "R_ema" not in state:
                            state["R_ema"] = R.detach().clone()
                        else:
                            state["R_ema"].mul_(beta2).add_(R, alpha=1 - beta2)
                        if "L_ema" not in state:
                            state["L_ema"] = L.detach().clone()
                        else:
                            state["L_ema"].mul_(beta2).add_(L, alpha=1 - beta2)
                        R_use, L_use = state["R_ema"], state["L_ema"]
                    else:
                        R_use, L_use = R, L

                    # Eigendecompose for eigenbasis (not for inverse — we use D_t instead)
                    R_f32 = R_use.to(torch.float32)
                    L_f32 = L_use.to(torch.float32)
                    try:
                        _, Q_R = torch.linalg.eigh(R_f32)
                    except torch._C._LinAlgError:
                        _, _, Vh = torch.linalg.svd(R_f32)
                        Q_R = Vh.T.flip(1)
                    try:
                        _, Q_L = torch.linalg.eigh(L_f32)
                    except torch._C._LinAlgError:
                        _, _, Vh = torch.linalg.svd(L_f32)
                        Q_L = Vh.T.flip(1)

                    # Project gradient into eigenbasis
                    G_tilde = Q_L.T.to(g.dtype) @ g @ Q_R.to(g.dtype)

                    # Update per-element second moment D_t (Adam-style)
                    if "D_eshampoo" not in state:
                        state["D_eshampoo"] = G_tilde.square().detach().clone()
                    else:
                        state["D_eshampoo"].mul_(beta2).add_(G_tilde.square(), alpha=1 - beta2)

                    # Bias correction
                    step_t = state["step"]
                    bc = 1.0 - beta2 ** step_t if beta2 < 1.0 else 1.0
                    D_corrected = state["D_eshampoo"] / bc

                    # Project momentum gradient into eigenbasis and scale
                    M_tilde = Q_L.T.to(g_mom.dtype) @ g_mom @ Q_R.to(g_mom.dtype)
                    eps = 1e-8
                    update_tilde = M_tilde / (D_corrected.sqrt() + eps)

                    # Project back to parameter space
                    update = Q_L.to(update_tilde.dtype) @ update_tilde @ Q_R.T.to(update_tilde.dtype)

                    # Diagnostics
                    eigmax_R = R_f32.diagonal().max().item()
                    eigmin_R = R_f32.diagonal().min().item()
                    eigmax_L = L_f32.diagonal().max().item()
                    eigmin_L = L_f32.diagonal().min().item()
                    self.diagnostics['C_inv_sqrt_opnorm'].append(0.0)
                    self.diagnostics['C_eigmax'].append(eigmax_R)
                    self.diagnostics['C_eigmin'].append(eigmin_R)
                    self.diagnostics['C_out_eigmax'].append(eigmax_L)
                    self.diagnostics['C_out_eigmin'].append(eigmin_L)
                    self.diagnostics['delta_per_layer'].append(0.0)
                    self.diagnostics['delta_out_per_layer'].append(0.0)

                    p.data.mul_(1 - lr * wd)
                    p.data.add_(update, alpha=-lr)
                    continue

                # Shampoo_sign / shampoo: gradient covariances instead of activation covariances
                if self.output_cov_mode in ("shampoo_sign", "shampoo"):
                    R = g.t() @ g          # [d_in, d_in]
                    L = g @ g.t()          # [d_out, d_out]

                    # EMA (same pattern as hook-based cov)
                    if self.ema_beta > 0.0:
                        if "R_ema" not in state:
                            state["R_ema"] = R.detach().clone()
                        else:
                            state["R_ema"].mul_(self.ema_beta).add_(R, alpha=1 - self.ema_beta)
                        if "L_ema" not in state:
                            state["L_ema"] = L.detach().clone()
                        else:
                            state["L_ema"].mul_(self.ema_beta).add_(L, alpha=1 - self.ema_beta)
                        R_use, L_use = state["R_ema"], state["L_ema"]
                    else:
                        R_use, L_use = R, L

                    if self.output_cov_mode == "shampoo_sign":
                        # Sign mode: no damping, minimal rcond
                        R_inv_sqrt, opnorm_R, eigmax_R, eigmin_R, _ = self._compute_C_inv_sqrt(R_use, damping=0.0, rcond=1e-12)
                        L_inv_sqrt, opnorm_L, eigmax_L, eigmin_L, _ = self._compute_C_inv_sqrt(L_use, damping=0.0, rcond=1e-12)
                    else:
                        # shampoo: needs damping to control update scale.
                        # opnorm_target targets ||update||_op = opnorm_target by default,
                        # accounting for ||G||_op. With precond_only_opnorm, it only
                        # targets the preconditioner opnorm (ignoring gradient magnitude).
                        if self.opnorm_target is not None:
                            if self.precond_only_opnorm:
                                side_target = self.opnorm_target ** 0.5
                            else:
                                g_opnorm = torch.linalg.svdvals(g_mom.to(torch.float32))[0].item()
                                side_target = (self.opnorm_target / max(g_opnorm, 1e-12)) ** 0.5
                        else:
                            side_target = None
                        if self.abs_damping is not None:
                            R_inv_sqrt, opnorm_R, eigmax_R, eigmin_R, delta_R = self._compute_C_inv_sqrt(R_use, abs_damping=self.abs_damping)
                            L_inv_sqrt, opnorm_L, eigmax_L, eigmin_L, delta_L = self._compute_C_inv_sqrt(L_use, abs_damping=self.abs_damping)
                        elif self.trace_damping is not None:
                            R_inv_sqrt, opnorm_R, eigmax_R, eigmin_R, delta_R = self._compute_C_inv_sqrt(R_use, trace_damping=self.trace_damping)
                            L_inv_sqrt, opnorm_L, eigmax_L, eigmin_L, delta_L = self._compute_C_inv_sqrt(L_use, trace_damping=self.trace_damping)
                        elif self.per_layer_damping and side_target is not None:
                            R_inv_sqrt, opnorm_R, eigmax_R, eigmin_R, delta_R = self._compute_C_inv_sqrt(R_use, opnorm_target=side_target)
                            L_inv_sqrt, opnorm_L, eigmax_L, eigmin_L, delta_L = self._compute_C_inv_sqrt(L_use, opnorm_target=side_target)
                        else:
                            R_inv_sqrt, opnorm_R, eigmax_R, eigmin_R, delta_R = self._compute_C_inv_sqrt(R_use, damping=delta_adaptive)
                            L_inv_sqrt, opnorm_L, eigmax_L, eigmin_L, delta_L = self._compute_C_inv_sqrt(L_use, damping=delta_adaptive)

                    # Diagnostics
                    self.diagnostics['C_inv_sqrt_opnorm'].append(opnorm_R)
                    self.diagnostics['C_eigmax'].append(eigmax_R)
                    self.diagnostics['C_eigmin'].append(eigmin_R)
                    self.diagnostics['C_out_eigmax'].append(eigmax_L)
                    self.diagnostics['C_out_eigmin'].append(eigmin_L)
                    self.diagnostics['delta_per_layer'].append(0.0)
                    self.diagnostics['delta_out_per_layer'].append(0.0)

                    # Whitened gradient
                    G_w = L_inv_sqrt.to(g_mom.dtype) @ g_mom @ R_inv_sqrt.to(g_mom.dtype)
                    if self.output_cov_mode == "shampoo_sign":
                        update = PolarExpress(G_w, steps=self.ns_steps)
                    else:
                        update = G_w  # shampoo: raw whitened gradient

                    p.data.mul_(1 - lr * wd)
                    p.data.add_(update, alpha=-lr)
                    continue

                # Get input covariance
                C = self._step_cov.get(p, state.get("C_ema", None))

                # Get output covariance (only for kfac_sign and full)
                C_out = None
                if self.output_cov_mode in ("kfac_sign", "full", "kfac"):
                    C_out = self._step_cov_out.get(p, state.get("C_out_ema", None))

                if C is not None:
                    if is_sign_mode:
                        # Sign modes: no damping, minimal rcond (sign absorbs scale)
                        C_inv_sqrt, opnorm, eigmax_C, eigmin_C, delta_used = self._compute_C_inv_sqrt(C, damping=0.0, rcond=1e-12)
                        self.diagnostics['delta_per_layer'].append(0.0)
                    else:
                        # null/full/kfac: opnorm_target or abs_damping controls update scale.
                        # For null/full (sign modes), ||update||_op = ||C^{-1/2}||_op since sign
                        # absorbs gradient magnitude — opnorm_target already targets update opnorm.
                        # For kfac (no sign), ||update||_op ≈ ||C_out^{-1/2}||·||G||·||C_in^{-1/2}||,
                        # so we adjust per-side target by ||G||_op to keep unified meaning.
                        if C_out is not None and self.opnorm_target is not None:
                            if self.output_cov_mode == "kfac" and not self.precond_only_opnorm:
                                g_opnorm = torch.linalg.svdvals(g_mom.to(torch.float32))[0].item()
                                side_target = (self.opnorm_target / max(g_opnorm, 1e-12)) ** 0.5
                            else:
                                side_target = self.opnorm_target ** 0.5
                        else:
                            side_target = self.opnorm_target

                        if self.abs_damping is not None:
                            C_inv_sqrt, opnorm, eigmax_C, eigmin_C, delta_used = self._compute_C_inv_sqrt(C, abs_damping=self.abs_damping)
                            self.diagnostics['delta_per_layer'].append(delta_used)
                        elif self.trace_damping is not None:
                            C_inv_sqrt, opnorm, eigmax_C, eigmin_C, delta_used = self._compute_C_inv_sqrt(C, trace_damping=self.trace_damping)
                            self.diagnostics['delta_per_layer'].append(delta_used)
                        elif self.per_layer_damping and self.opnorm_target is not None:
                            C_inv_sqrt, opnorm, eigmax_C, eigmin_C, delta_used = self._compute_C_inv_sqrt(C, opnorm_target=side_target)
                            self.diagnostics['delta_per_layer'].append(delta_used)
                        else:
                            C_inv_sqrt, opnorm, eigmax_C, eigmin_C, delta_used = self._compute_C_inv_sqrt(C, damping=delta_adaptive)
                    self.diagnostics['C_inv_sqrt_opnorm'].append(opnorm)
                    self.diagnostics['C_eigmax'].append(eigmax_C)
                    self.diagnostics['C_eigmin'].append(eigmin_C)

                    if self.output_cov_mode == "sign_input":
                        # sign(G @ C_in^{-1/2}) — unit opnorm, lr controls scale
                        G_w = g_mom @ C_inv_sqrt.to(g_mom.dtype)
                        update = PolarExpress(G_w, steps=self.ns_steps)

                    elif self.output_cov_mode in ("kfac_sign", "full", "kfac") and C_out is not None:
                        # Output-side C_out^{-1/2}
                        if is_sign_mode:
                            C_out_inv_sqrt, _, eigmax_out, eigmin_out, delta_out = self._compute_C_inv_sqrt(C_out, damping=0.0, rcond=1e-12)
                            self.diagnostics['delta_out_per_layer'].append(0.0)
                        elif self.abs_damping is not None:
                            C_out_inv_sqrt, _, eigmax_out, eigmin_out, delta_out = self._compute_C_inv_sqrt(C_out, abs_damping=self.abs_damping)
                            self.diagnostics['delta_out_per_layer'].append(delta_out)
                        elif self.trace_damping is not None:
                            C_out_inv_sqrt, _, eigmax_out, eigmin_out, delta_out = self._compute_C_inv_sqrt(C_out, trace_damping=self.trace_damping)
                            self.diagnostics['delta_out_per_layer'].append(delta_out)
                        else:
                            if self.per_layer_damping and self.opnorm_target is not None:
                                C_out_inv_sqrt, _, eigmax_out, eigmin_out, delta_out = self._compute_C_inv_sqrt(C_out, opnorm_target=side_target)
                                self.diagnostics['delta_out_per_layer'].append(delta_out)
                            else:
                                C_out_inv_sqrt, _, eigmax_out, eigmin_out, delta_out = self._compute_C_inv_sqrt(C_out, damping=delta_adaptive)
                        self.diagnostics['C_out_eigmax'].append(eigmax_out)
                        self.diagnostics['C_out_eigmin'].append(eigmin_out)

                        G_w = C_out_inv_sqrt.to(g_mom.dtype) @ g_mom @ C_inv_sqrt.to(g_mom.dtype)

                        if self.output_cov_mode == "kfac":
                            # Raw doubly-whitened gradient (no sign)
                            update = G_w
                        else:
                            sign_Gw = PolarExpress(G_w, steps=self.ns_steps)

                            if self.output_cov_mode == "full":
                                update = C_out_inv_sqrt.to(sign_Gw.dtype) @ sign_Gw @ C_inv_sqrt.to(sign_Gw.dtype)
                            else:
                                # kfac_sign — unit opnorm, lr controls scale
                                update = sign_Gw

                    else:
                        # None (null): sign(G @ C^{-1/2}) @ C^{-1/2}
                        G_w = g_mom @ C_inv_sqrt.to(g_mom.dtype)
                        sign_Gw = PolarExpress(G_w, steps=self.ns_steps)
                        update = sign_Gw @ C_inv_sqrt.to(sign_Gw.dtype)
                else:
                    # Fallback: pure Muon (no whitening available yet)
                    update = PolarExpress(g_mom, steps=self.ns_steps)

                # Weight decay
                p.data.mul_(1 - lr * wd)
                # Apply update
                p.data.add_(update, alpha=-lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_dap_opnorm"]]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
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

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                scale = bias_correction1 / bias_correction2 ** 0.5
                p.data.mul_(1 - lr * wd)
                p.data.add_(g, alpha=-lr / scale)

        for mod in self.dap_modules:
            mod._xtx_mults_this_step.zero_()

        self._step_cov.clear()
        self._step_cov_out.clear()

        return loss
