import torch
from torch import nn
from torch.optim import Optimizer
from gptopt.linalg_utils import ns_pinv, ns_pinv_v2, accelerated_ns_pinv, power_method
from gptopt.optim.timing import SimpleTimer, install_forward_cuda_timers

from typing import Optional, Dict
import torch.nn.functional as F

import os
import sys
import time


class LinearWithXtX(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, *, track_xtx=False):
        super().__init__(in_features, out_features, bias=bias)
        # We default to track_xtx=False; the optimizer will enable it only for DAP layers.
        self.track_xtx = bool(track_xtx)

        # fp32 accumulators for input covariance (same device as module)
        self.register_buffer(
            "C_accum_sum", torch.zeros(in_features, in_features, dtype=torch.float32), persistent=True
        )
        self.register_buffer("C_accum_count", torch.zeros((), dtype=torch.int64), persistent=True)

        # fp32 accumulators for output gradient covariance (lazy — allocated on first use)
        self._out_features = out_features
        self.register_buffer("S_accum_sum", None, persistent=False)
        self.register_buffer("S_accum_count", None, persistent=False)

        self.register_buffer(
            "_xtx_mults_this_step", torch.tensor(0, dtype=torch.int64), persistent=False
        )

        # wired by the optimizer
        self._dap_timing_enabled: bool = False
        self._dap_timing_sink: Optional[list] = None
        self._dap_accum_enabled: bool = False
        self._accum_output_cov: bool = False
        self.xtx_subsample: Optional[float] = None

        self.xtx_mode: bool = True

        # No module-level backward hook — we use tensor hooks in forward() instead
        # to avoid keeping all output gradients alive simultaneously during backward.

    @torch.no_grad()
    def _accum_sts(self, grad_out: torch.Tensor) -> None:
        S = grad_out.detach().reshape(-1, grad_out.shape[-1])  # [N, d_out]
        if self.xtx_subsample is not None:
            k = int(self.xtx_subsample * S.shape[0])
            S = S[:k]
        StS = S.transpose(0, 1) @ S  # [d_out, d_out]
        # Lazy allocation of output covariance buffers
        if self.S_accum_sum is None:
            self.S_accum_sum = torch.zeros(self._out_features, self._out_features, dtype=torch.float32, device=S.device)
            self.S_accum_count = torch.zeros((), dtype=torch.int64, device=S.device)
        self.S_accum_sum.add_(StS.to(self.S_accum_sum.dtype))
        self.S_accum_count.add_(S.shape[0])

    @torch.no_grad()
    def _accum_xtx(self, x: torch.Tensor) -> None:
        X = x.detach().reshape(-1, x.shape[-1])  # [N, d]
        if self.xtx_subsample is not None:
            k = int(self.xtx_subsample * X.shape[0])
            X = X[:k]
        C = X.transpose(0, 1) @ X  # [d, d] in activation dtype
        self.C_accum_sum.add_(C.to(self.C_accum_sum.dtype))
        self.C_accum_count.add_(X.shape[0])
        self._xtx_mults_this_step.add_(X.shape[0] * X.shape[1] * X.shape[1])

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.training and self._dap_accum_enabled and self.xtx_mode:
            self._accum_xtx(x)
        if self.training and self._accum_output_cov and y.requires_grad:
            y.register_hook(lambda grad, mod=self: mod._accum_sts(grad))
        return y

class DAP(Optimizer):
    def __init__(
        self,
        model,
        named_params,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        damping=0.0,
        ema_beta=0.0,
        dap_beta1: float = 0.0,
        dap_beta2: float = 0.0,
        dap_eps: float = 1e-8,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        disable_preconditioning: bool = False,
        scalar: bool = False,
        include_output: bool = False,
        include_embed: bool = False,
        use_ns_pinv: bool = False,
        ns_pinv_steps: int = 30,
        rcond: float = 1e-3,
        debug_timing: bool = True,
        use_bf16: bool = False,
        use_fp64: bool = False,
        refresh_precond_iters=None,
        accelerated: bool = False,
        spectral_norm_estimator: str = "power_method",  # "power_method", "frobenius"
        xtx_subsample: Optional[float] = None,
        mb_subsampling: bool = False,
        num_microbatches: Optional[int] = None,
        update_norm: bool = False,
        update_opnorm: bool = False,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            damping=damping,
            ema_beta=ema_beta,
            dap_beta1=dap_beta1,
            dap_beta2=dap_beta2,
            dap_eps=dap_eps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            update_norm=update_norm,
            update_opnorm=update_opnorm,
        )

        dap_params, dap_params_names = [], []
        adamw_params, adamw_params_names = [], []

        excluded_names = []
        if not include_output:
            excluded_names.extend(["lm_head", "weight_proj"])

        if not include_embed:
            excluded_names.extend(["embeddings", "embed_tokens", "wte", "wpe"])

        self.param_to_name = {}

        for name, p in named_params:
            self.param_to_name[p] = name

            if p.ndim >= 2 and not any(excluded in name for excluded in excluded_names):
                dap_params.append(p)
                dap_params_names.append(name)
            else:
                adamw_params.append(p)
                adamw_params_names.append(name)

        # Print informative summary of parameter classification
        print(f"\n=== DAP Optimizer Parameter Classification ===")
        print(f"DAP Parameters ({len(dap_params)}):")
        for name in dap_params_names:
            print(f"  - {name}")
        print(f"\nAdamW Parameters ({len(adamw_params)}):")
        for name in adamw_params_names:
            print(f"  - {name}")
        print(f"==============================================\n")

        # One-time params debug print at initialization
        self._num_dap_params_total = len(dap_params)
        if debug_timing:
            print(f"[DAP] params={self._num_dap_params_total}")

        params = list(dap_params)
        params.extend(adamw_params)
        super().__init__(params, defaults)

        # Sort parameters into those for which we will use DAP, and those for which we will not
        # Use DAP for every parameter in dap_params which is >= 2D and doesn't look like an embedding or head layer
        for p in dap_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_dap"] = True

        for p in adamw_params:
            # Do not use DAP for parameters in adamw_params
            self.state[p]["use_dap"] = False

        self.ema_beta = ema_beta
        # Whether to bypass the expensive preconditioner and use a plain SGD update.
        self.disable_preconditioning = disable_preconditioning
        self.scalar = scalar
        self.include_output = include_output
        self.include_embed = include_embed
        self.use_ns_pinv = use_ns_pinv
        self.ns_pinv_steps = ns_pinv_steps
        self.rcond = rcond

        # Debug timing controls
        self.debug_timing = debug_timing
        self._debug_step_idx = 0
        self.xtx_subsample = xtx_subsample
        self.mb_subsampling = mb_subsampling
        self.num_microbatches = num_microbatches

        # Accumulator for average relative Frobenius change of C_ema per step
        self._C_ema_rel_change_accum = 0.0

        self._C_ema_update_time_accum = (
            0.0  # total wall-clock seconds spent updating C_ema since last step()
        )
        self._C_ema_update_count = 0  # how many EMA updates occurred since last step()
        self._pending_timing_events = (
            []
        )  # list of (label, start_event, end_event, device)

        # Accumulators for XtX timing across micro-batches between step() calls
        self._XtX_update_time_accum = 0.0
        self._XtX_update_count = 0

        # Precision control
        self.rcond = float(rcond)
        self.bf16 = bool(use_bf16)
        self.fp64 = bool(use_fp64)

        self.accelerated = accelerated
        if spectral_norm_estimator not in ["power_method", "frobenius"]:
            raise ValueError(
                f"Invalid spectral_norm_estimator: {spectral_norm_estimator}"
            )
        self.spectral_norm_estimator = spectral_norm_estimator

        if self.bf16 and self.fp64:
            raise ValueError("use_bf16 and use_fp64 are mutually exclusive")

        # dtype used for power_method / pinv / ns_pinv operations
        if self.bf16:
            self.op_dtype = torch.bfloat16
        elif self.fp64:
            self.op_dtype = torch.float64
        else:
            self.op_dtype = torch.float32

        assert not (
            self.scalar and self.disable_preconditioning
        ), "choose either scalar or disable_preconditioning"

        # Control how often to refresh preconditioned vector; None disables caching
        self.refresh_precond_iters = refresh_precond_iters

        self.dap_modules = []
        self.param_to_module = {}
        self._step_cov: Dict[torch.nn.Parameter, torch.Tensor] = {}

        if not self.disable_preconditioning:
            self._wire_up_xtx_sources(model, dap_params)

    def _wire_up_xtx_sources(self, model: nn.Module, dap_params):
        dap_param_ids = {id(p) for p in dap_params}

        for mod in model.modules():
            if isinstance(mod, nn.Linear) and id(mod.weight) in dap_param_ids:
                if not isinstance(mod, LinearWithXtX):
                    raise TypeError(
                        "Buffer mode requires LinearWithXtX. Call swap_linears_for_xtx(model) first."
                    )
                # timing + accumulation only for DAP layers
                mod._dap_timing_enabled = self.debug_timing
                mod._dap_timing_sink = self._pending_timing_events
                mod._dap_accum_enabled = True  # <- enable XtX accumulation
                mod._xtx_mults_this_step.zero_()
                mod.xtx_subsample = self.xtx_subsample

                self.param_to_module[mod.weight] = mod
                self.dap_modules.append(mod)

        if len(self.param_to_module) != len(dap_params):
            missing = [p for p in dap_params if p not in self.param_to_module]
            raise ValueError(
                f"Some DAP params not owned by any LinearWithXtX: {missing}"
            )

        if self.debug_timing and torch.cuda.is_available() and self.dap_modules:
            install_forward_cuda_timers(
                modules=self.dap_modules,
                pending=self._pending_timing_events,
                label="xtx",  # matches your existing aggregator keys
            )

    def _estimate_spectral_norm(self, C: torch.Tensor, psd=True, return_iters=True):
        if self.spectral_norm_estimator == "power_method":
            return power_method(
                C, max_iters=self.ns_pinv_steps, psd=psd, return_iters=return_iters
            )
        elif self.spectral_norm_estimator == "frobenius":
            if return_iters:
                return torch.linalg.norm(C, ord="fro"), 0
            else:
                return torch.linalg.norm(C, ord="fro")
        else:
            raise ValueError(
                f"Invalid spectral_norm_estimator: {self.spectral_norm_estimator}"
            )

    def _precondition(
        self, tensor: torch.Tensor, C: torch.Tensor, p: torch.Tensor, step: int
    ):
        """Precondition a gradient-like tensor using covariance C.
        Returns (preconditioned_tensor, pm_iters, ns_iters).
        """
        pm_iters = 0
        ns_iters = 0

        C_op = C.to(self.op_dtype)
        v_op = tensor.to(self.op_dtype)

        if self.use_ns_pinv:
            device = p.device if p.is_cuda else None
            with SimpleTimer(
                "power", device, self._pending_timing_events, enabled=self.debug_timing
            ):
                sig_max, pm_iters = self._estimate_spectral_norm(
                    C_op, psd=True, return_iters=True
                )
            sig_max = sig_max.item()
            eps = sig_max * self.rcond

            with SimpleTimer(
                "pinv", device, self._pending_timing_events, enabled=self.debug_timing
            ):
                # Determine if accelerated should use an explicit add_eps value (floats only)
                add_eps_value = None
                if isinstance(self.accelerated, (int, float)) and not isinstance(self.accelerated, bool):
                    add_eps_value = float(self.accelerated)

                if add_eps_value is not None:
                    Cinv, info = accelerated_ns_pinv(
                        C_op,
                        l=eps,
                        u=1.2 * sig_max,
                        max_steps=self.ns_pinv_steps,
                        psd=True,
                        add_eps=add_eps_value,  # absolute, not relative to sig_max
                        early_stop_eps=eps,
                        dtype=torch.float32,
                        diagnostics=False,
                        return_iters=True,
                    )
                elif self.accelerated:
                    Cinv, info = accelerated_ns_pinv(
                        C_op,
                        l=eps,
                        u=1.2 * sig_max,
                        max_steps=self.ns_pinv_steps,
                        psd=False,  # for now
                        add_eps=0,
                        early_stop_eps=eps,
                        dtype=torch.float32,
                        diagnostics=False,
                        return_iters=True,
                    )
                else:
                    Cinv, info = ns_pinv_v2(
                        C_op,
                        eps=eps,
                        max_steps=self.ns_pinv_steps,
                        return_iters=True,
                        diagnostics=False,
                    )

            ns_iters = int(info.get("iterations", 0))
            if torch.isnan(Cinv).any() or torch.isinf(Cinv).any():
                debug_dir = "debug_matrices"
                os.makedirs(debug_dir, exist_ok=True)
                matrix_path = os.path.join(
                    debug_dir, f"C_{self.param_to_name[p]}_step_{step}.pt"
                )
                grad_path = os.path.join(
                    debug_dir, f"grad_{self.param_to_name[p]}_step_{step}.pt"
                )
                torch.save(C_op, matrix_path)
                torch.save(v_op, grad_path)
                print(
                    f"ERROR: NaN/inf detected in pseudo-inverse calculation for parameter {self.param_to_name[p]} at step {step}"
                )
                print(
                    f"Matrix saved to debug_matrices/C_{self.param_to_name[p]}_step_{step}.pt"
                )
                sys.exit(1)

            u_local = v_op @ Cinv
        else:
            if self.scalar:
                device = p.device if p.is_cuda else None
                with SimpleTimer(
                    "power",
                    device,
                    self._pending_timing_events,
                    enabled=self.debug_timing,
                ):
                    sig_max, pm_iters = self._estimate_spectral_norm(
                        C_op, psd=True, return_iters=True
                    )
                u_local = v_op / sig_max
            else:
                device = p.device if p.is_cuda else None
                with SimpleTimer(
                    "pinv",
                    device,
                    self._pending_timing_events,
                    enabled=self.debug_timing,
                ):
                    pinv_mat = torch.linalg.pinv(C_op, rtol=self.rcond)
                u_local = v_op @ pinv_mat

        return u_local.to(p.dtype), pm_iters, ns_iters

    def _precondition_cached(
        self,
        tensor: torch.Tensor,
        C: torch.Tensor,
        p: torch.Tensor,
        step: int,
        cache_key: str,
    ):
        """Wrapper over _precondition that optionally caches the result for a number of steps.
        If self.refresh_precond_iters is None, always recomputes.
        When caching, stores results under state[p][f"precond_cache_{cache_key}"] and associated step.
        Returns (u, pm_iters, ns_iters), where timing/iters are zero if reused from cache.
        """
        refresh = self.refresh_precond_iters
        if refresh is None:
            return self._precondition(tensor, C, p, step)

        state = self.state[p]
        cache_val_key = f"precond_cache_{cache_key}"
        cache_step_key = f"precond_cache_step_{cache_key}"
        last_step = state.get(cache_step_key, None)
        should_recompute = (last_step is None) or ((step - last_step) >= refresh)

        if should_recompute:
            u, pm_iters, ns_iters = self._precondition(tensor, C, p, step)
            # store a cloned tensor to avoid unexpected in-place mutations
            state[cache_val_key] = u.clone()
            state[cache_step_key] = step
            return u, pm_iters, ns_iters
        else:
            u = state[cache_val_key]
            return u, 0, 0

    def _apply_first_moment(
        self,
        tensor: torch.Tensor,
        state: dict,
        momentum: float,
        nesterov: bool,
        beta1: float,
        prefix: str = "",
    ):
        """Apply Adam beta1 EMA or classical momentum on the given tensor.
        Prefix controls state keys suffix (e.g., "", "_pre"). Returns the updated tensor.
        """
        if beta1 > 0.0:
            key = f"dap_moment1{prefix}"
            if key not in state:
                state[key] = torch.zeros_like(tensor)
            m = state[key]
            m.lerp_(tensor, 1 - beta1)
            return m
        else:
            key = f"momentum_buffer{prefix}"
            if key not in state:
                state[key] = torch.zeros_like(tensor)
            buf = state[key]
            buf.mul_(momentum).add_(tensor)
            if nesterov:
                return tensor.add(buf, alpha=momentum)
            else:
                return buf

    def _adam_numerator(
        self, m_t: torch.Tensor, beta1: float, step: int
    ) -> torch.Tensor:
        # m̂_t = m_t / (1 - β₁^t); if β₁==0, denominator=1 → returns m_t
        return m_t / (1.0 - (beta1**step))

    def _nadam_numerator(
        self, m_t: torch.Tensor, g_t: torch.Tensor, beta1: float, step: int
    ) -> torch.Tensor:
        # PyTorch-style Nadam (constant β₁): m̂_t = (β₁ m_t)/(1-β₁^{t+1}) + ((1-β₁) g_t)/(1-β₁^t)
        return (beta1 * m_t) / (1.0 - (beta1 ** (step + 1))) + ((1.0 - beta1) * g_t) / (
            1.0 - (beta1**step)
        )

    def _adam_denominator(
        self, v_t: torch.Tensor, beta2: float, step: int, eps: float
    ) -> torch.Tensor:
        # v̂_t = v_t / (1 - β₂^t)  → denom = sqrt(v̂_t) + eps
        # For β₂==0 we’ll skip this path entirely (see step()) to avoid unnecessary work.
        bc2 = 1.0 - (beta2**step)
        return v_t.sqrt() / (bc2**0.5) + eps

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.
        Args:
        closure (Callable, optional): A closure that reevaluates the model
            and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.debug_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            optimizer_step_t0 = time.perf_counter()
            total_power_method_iters = 0
            total_ns_iters = 0
            num_dap_params = 0

        for group in self.param_groups:
            ############################
            #           DAP           #
            ############################
            lr = group["lr"]
            wd = group["wd"]

            momentum = group["momentum"]
            damping = group["damping"]
            dap_beta1 = group["dap_beta1"]
            dap_beta2 = group["dap_beta2"]
            dap_eps = group["dap_eps"]

            params = [p for p in group["params"] if self.state[p]["use_dap"]]

            # Before applying updates, finalize accumulated covariance per parameter if present
            if not self.disable_preconditioning:
                self._step_cov.clear()
                for p, mod in self.param_to_module.items():
                    count = mod.C_accum_count.item()
                    if not count:
                        continue

                    C_sum = mod.C_accum_sum
                    C_mean = C_sum / count

                    state = self.state[p]

                    if self.ema_beta == 0.0:
                        self._step_cov[p] = C_mean
                    else:
                        C_prev = state.get("C_ema", None)

                        # Original diff metric: C_mean - C_prev (pre-update EMA)
                        if self.debug_timing and C_prev is not None:
                            diff = C_mean - C_prev
                            rel = (1.0 - self.ema_beta) * (
                                diff.norm("fro") / C_prev.norm("fro").clamp_min(1e-12)
                            )
                            self._C_ema_rel_change_accum += rel.item()
                            self._C_ema_update_count += 1

                        # Update EMA snapshot
                        if C_prev is None:
                            state["C_ema"] = C_mean.detach().clone()
                        else:
                            C_prev.mul_(self.ema_beta).add_(C_mean, alpha=(1.0 - self.ema_beta))

                    mod.C_accum_sum.zero_()
                    mod.C_accum_count.zero_()

            # apply weight updates
            for i, p in enumerate(params):
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                # calc momentum / first moment and preconditioning based on mode
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                step = state["step"]

                use_nadam = group["nesterov"] and dap_beta1 > 0.0

                # Determine if we can/should precondition this parameter this step
                C_eff = None
                if not self.disable_preconditioning:
                    C_cur = self._step_cov.get(p, self.state[p].get("C_ema", None))
                    if C_cur is not None:
                        if damping:
                            trace_C = torch.trace(C_cur)
                            damp_val = damping * trace_C / C_cur.shape[0]
                            I = torch.eye(
                                C_cur.shape[0], dtype=C_cur.dtype, device=C_cur.device
                            )
                            C_eff = C_cur + I * damp_val
                        else:
                            C_eff = C_cur

                # First compute momentum/EMA on raw gradient, then (optionally) precondition
                m_t = self._apply_first_moment(
                    g,
                    state,
                    momentum,
                    group["nesterov"],
                    dap_beta1,
                    prefix="",
                )

                if use_nadam:
                    num_grad = self._nadam_numerator(m_t, g, dap_beta1, step)
                elif dap_beta1 > 0.0:
                    num_grad = self._adam_numerator(m_t, dap_beta1, step)
                else:
                    num_grad = m_t

                if C_eff is not None:
                    u, pm_iters, ns_iters = self._precondition_cached(
                        num_grad, C_eff, p, step, cache_key="numgrad"
                    )
                    if self.debug_timing:
                        total_power_method_iters += pm_iters
                        total_ns_iters += ns_iters
                else:
                    u, pm_iters, ns_iters = num_grad, 0, 0

                if self.debug_timing:
                    num_dap_params += 1

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # Build the raw update O (before learning-rate scaling)
                if dap_beta2 > 0.0:
                    # Adam-style per-parameter normalization and bias correction (epsilon in denominator)
                    if "dap_moment2" not in state:
                        state["dap_moment2"] = torch.zeros_like(u)
                    v = state["dap_moment2"]
                    v.lerp_(u.square(), 1 - dap_beta2)
                    denom = self._adam_denominator(v, dap_beta2, step, dap_eps)
                    o = u / denom
                else:
                    # No second-moment normalization.
                    o = u

                # Optionally normalize O: operator norm or RMS
                if group["update_opnorm"]:
                    # Use power iteration on O^T O to estimate the top singular value
                    OtO = (o.transpose(0, 1) @ o).to(torch.float32)
                    sig2, _ = power_method(OtO, max_iters=self.ns_pinv_steps, psd=True, return_iters=True)
                    sig = sig2.sqrt()
                    o = o / sig.clamp_min(1e-12)
                elif group["update_norm"]:
                    rms = o.pow(2).mean().sqrt()
                    o = o * (0.2 / rms.clamp_min(1e-12))

                # Apply update with learning rate
                p.data.add_(o, alpha=-lr)

            ############################
            #       AdamW backup       #
            ############################

            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]
            params = [p for p in group["params"] if not self.state[p]["use_dap"]]

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

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        if self.debug_timing:
            # If we recorded any CUDA events, sync the relevant devices
            if self._pending_timing_events:
                devices_to_sync = {
                    rec[4]
                    for rec in self._pending_timing_events
                    if rec[0] == "cuda" and rec[4] is not None
                }
                for dev in devices_to_sync:
                    torch.cuda.synchronize(dev)

            power_time_add = 0.0
            pinv_time_add = 0.0
            xtx_time_add = 0.0
            xtx_updates = 0

            # Drain all timing records (both CUDA-event and CPU-wall-clock)
            for rec in self._pending_timing_events:
                kind = rec[0]
                if kind == "cuda":
                    _, label, ev_start, ev_end, _dev = rec
                    try:
                        ms = ev_start.elapsed_time(ev_end)
                    except Exception:
                        ms = 0.0
                    sec = ms / 1000.0
                else:  # "cpu": ('cpu', label, seconds)
                    _, label, sec = rec

                if label == "power":
                    power_time_add += sec
                elif label == "pinv":
                    pinv_time_add += sec
                elif label == "xtx":
                    xtx_time_add += sec
                    xtx_updates += 1

            self._pending_timing_events.clear()

            # Roll into step accumulators used by the printout
            self._XtX_update_time_accum += xtx_time_add
            self._XtX_update_count += xtx_updates

            total_power_method_time = power_time_add
            total_pseudo_inverse_time = pinv_time_add

            optimizer_step_elapsed = time.perf_counter() - optimizer_step_t0

            avg_C_ema_rel_change = self._C_ema_rel_change_accum / max(
                self._num_dap_params_total, 1
            )
            dap_total = (
                self._XtX_update_time_accum
                + total_power_method_time
                + total_pseudo_inverse_time
            )

            xtx_mults_total = 0
            for mod in self.dap_modules:
                xtx_mults_total += mod._xtx_mults_this_step.item()

            print(
                f"[DAP] step={self._debug_step_idx} "
                f"XtX={self._XtX_update_time_accum:.4f}s/{xtx_mults_total:.2e} mults "
                f"power={total_power_method_time:.4f}s/{total_power_method_iters}it "
                f"pinv={total_pseudo_inverse_time:.4f}s/{total_ns_iters}it "
                f"Crel={avg_C_ema_rel_change:.6f} "
                f"opt_step={optimizer_step_elapsed:.4f}s "
                f"dap_total={dap_total:.4f}s"
            )

            self._debug_step_idx += 1

            # reset per-step accumulators
            self._C_ema_rel_change_accum = 0.0
            self._XtX_update_time_accum = 0.0
            self._XtX_update_count = 0

        for mod in self.dap_modules:
            mod._xtx_mults_this_step.zero_()

        self._step_cov.clear()

        return loss
