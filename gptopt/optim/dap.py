import torch
from torch import nn
from torch.optim import Optimizer


class DAP(Optimizer):
    def __init__(
        self,
        model,
        named_params,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        damping=0,
        ema_beta=0.99,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            damping=damping,
            ema_beta=ema_beta,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        dap_params, dap_params_names = [], []
        adamw_params, adamw_params_names = [], []

        self.dap_params   = dap_params
        self.adamw_params = adamw_params

        for name, p in named_params:
            if p.ndim >= 2 and not any(
                excluded in name
                for excluded in [
                    "embeddings",
                    "embed_tokens",
                    "wte",
                    "lm_head",
                    "wpe",
                    "weight_proj",
                ]
            ):
                dap_params.append(p)
                dap_params_names.append(name)
            else:
                adamw_params.append(p)
                adamw_params_names.append(name)

        params = list(dap_params)
        params.extend(adamw_params)
        super().__init__(params, defaults)
        self.ema_beta = ema_beta
            
        self._register_input_hooks(model, dap_params)

    def _register_input_hooks(self, model: nn.Module, dap_params):
        """Attach hooks **only** to modules whose weights are in dap_params."""
        # Build a fast identity‐based lookup for which weights to hook
        dap_param_ids = {id(p) for p in dap_params}

        # Map each parameter object to its owning nn.Linear module
        param_to_module = {}
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if id(module.weight) in dap_param_ids:
                    param_to_module[module.weight] = module

        # Sanity check: ensure every dap_param got assigned
        unclaimed = [p for p in dap_params if p not in param_to_module]
        if unclaimed:
            raise ValueError(
                f"Some DAP params not owned by any Linear module: {unclaimed}"
            )

        # Attach one forward‐hook per relevant module to capture XᵀX in state[p]["C"]
        for p_ref, module in param_to_module.items():
            def make_hook(p_ref):
                def hook(mod, inp, out):
                    X = inp[0].detach()
                    X_flat = X.reshape(-1, X.shape[-1])
                    # store covariance matrix C = Xᵀ X for this param
                    C_new = (X_flat.transpose(0, 1) @ X_flat) / X_flat.shape[0]
                    state = self.state[p_ref]

                    if "C_ema" not in state:
                        state["C_ema"] = C_new
                    else:
                        state["C_ema"].mul_(self.ema_beta).add_(
                            C_new, alpha=1 - self.ema_beta
                        )
                return hook

            module.register_forward_hook(make_hook(p_ref), prepend=False)

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

        for group in self.param_groups:
            ############################
            #           DAP           #
            ############################

            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            damping = group["damping"]

            # apply weight updates
            for i, p in enumerate(self.dap_params):

                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                # calc momentum.
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                # calc update. Note that we already computed and stored the momentum
                # term before, but we are re-computing the matrix sign. This is
                # suboptimal w.r.t.  time but doesn't use any extra memory. We can
                # always tweak this later.
                state = self.state[p]
                buf = state["momentum_buffer"]

                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                
                C32 = self.state[p]["C_ema"].float()

                if damping:
                    C32.diagonal().add_(damping)

                g32 = g.float()
                u32 = g32 @ torch.linalg.pinv(C32)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u32.to(p.dtype), alpha=-lr)

            ############################
            #       AdamW backup       #
            ############################

            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in self.adamw_params:
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

        return loss
    
    
