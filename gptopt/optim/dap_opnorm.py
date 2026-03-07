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

        self.dap_modules = []
        self.param_to_module: Dict[nn.Parameter, LinearWithXtX] = {}
        self._step_cov: Dict[nn.Parameter, torch.Tensor] = {}

        self._wire_up_xtx_sources(model, dap_params)

    def _wire_up_xtx_sources(self, model: nn.Module, dap_params):
        dap_param_ids = {id(p) for p in dap_params}

        for mod in model.modules():
            if isinstance(mod, nn.Linear) and id(mod.weight) in dap_param_ids:
                if not isinstance(mod, LinearWithXtX):
                    raise TypeError(
                        "DAPOpNorm requires LinearWithXtX. Call swap_linears_for_xtx(model) first."
                    )
                mod._dap_accum_enabled = True
                self.param_to_module[mod.weight] = mod
                self.dap_modules.append(mod)

        if len(self.param_to_module) != len(dap_params):
            missing = [p for p in dap_params if p not in self.param_to_module]
            raise ValueError(
                f"Some DAPOpNorm params not owned by any LinearWithXtX: {missing}"
            )

    def _compute_C_inv_sqrt(self, C: torch.Tensor, damping=None, opnorm_target=None):
        """Compute C^{-1/2} via eigendecomposition with optional damping.

        Args:
            C: the covariance matrix
            damping: relative damping coefficient; if None, uses self.damping.
                Ignored when opnorm_target is set.
            opnorm_target: if set, compute per-layer relative damping to achieve
                this target opnorm exactly: delta = (1/opnorm_target^2 - lambda_min) / lambda_max,
                clamped >= 0.

        Returns:
            (C_inv_sqrt, opnorm, C_eigmax, C_eigmin, delta_used): C_inv_sqrt is the matrix
            square root inverse; opnorm = 1/sqrt(min active eigval of C_eff);
            C_eigmax and C_eigmin are the max and min eigenvalues of C *before*
            damping (among active directions); delta_used is the relative damping
            coefficient actually applied (useful when opnorm_target computes it).
        """
        # Single eigendecomposition of C (before damping)
        eigvals_C, eigvecs = torch.linalg.eigh(C.to(torch.float32))
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
        elif damping is None:
            damping = self.damping

        # Apply relative damping to eigenvalues directly (avoids second decomp)
        eigvals_eff = eigvals_C + damping * eigmax_C if damping > 0 else eigvals_C

        threshold = self.rcond * eigvals_eff.max()
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

        self.diagnostics = {'C_inv_sqrt_opnorm': [], 'C_eigmax': [], 'C_eigmin': [], 'delta_adaptive': None, 'delta_per_layer': []}

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            ############################
            #       DAPOpNorm          #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_dap_opnorm"]]

            # Finalize accumulated covariance
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

            # First pass: compute global adaptive damping (skipped for per-layer mode)
            if self.opnorm_target is not None and not self.per_layer_damping:
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

                # Get covariance
                C = self._step_cov.get(p, state.get("C_ema", None))

                if C is not None:
                    if self.per_layer_damping and self.opnorm_target is not None:
                        C_inv_sqrt, opnorm, eigmax_C, eigmin_C, delta_used = self._compute_C_inv_sqrt(C, opnorm_target=self.opnorm_target)
                        self.diagnostics['delta_per_layer'].append(delta_used)
                    else:
                        C_inv_sqrt, opnorm, eigmax_C, eigmin_C, delta_used = self._compute_C_inv_sqrt(C, damping=delta_adaptive)
                    self.diagnostics['C_inv_sqrt_opnorm'].append(opnorm)
                    self.diagnostics['C_eigmax'].append(eigmax_C)
                    self.diagnostics['C_eigmin'].append(eigmin_C)
                    # Whiten gradient
                    G_w = g_mom @ C_inv_sqrt.to(g_mom.dtype)
                    # Matrix sign via PolarExpress
                    sign_Gw = PolarExpress(G_w, steps=self.ns_steps)
                    # LMO update: W -= lr * sign(G C^{-1/2}) @ C^{-1/2}
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

        return loss
