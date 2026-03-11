import torch

from .polar import zeropower_via_newtonschulz5, PolarExpress, SVDPolarFactor


class MuonMaxMomo(torch.optim.Optimizer):
    """
    Arguments:
        muon_params: Parameters to be optimized with Muon.
        adam_params: Parameters to be optimized with Adam.
        lr: The learning rate. (0.02 is a good default)
        muon_lr_scale: Multiple for lr of Muon params. (10 is a good default)
        wd: Weight decay.
        momentum: The momentum used for gradient accumulation. (0.95 is a good default)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        polar_method: Which method to compute polar factor. Choices: ["jordan",
            "polar_express", "svd"].
        betas: (beta1, beta2) for adam.
        eps: epsilon for adam.
        truncate_loss: Lower bound of loss, if using model truncation. Otherwise, set to None.
        stale_nuc: Whether to use nuclear norm from previous round to speed up update.
    """
    def __init__(
        self,
        muon_params,
        adam_params,
        lr=1e-3,
        muon_lr_scale=10.0,
        wd=0.1,
        momentum=0.95,
        ns_steps=5,
        polar_method="polar_express",
        betas=(0.95, 0.95),
        eps=1e-8,
        truncate_loss=0.0,
        stale_nuc=True,
    ):

        assert isinstance(muon_params, list)
        assert isinstance(adam_params, list)

        self.muon_lr_scale = muon_lr_scale
        self.ns_steps = ns_steps

        if polar_method == "polar_express":
            self.polar_fn = lambda g: PolarExpress(g, steps=ns_steps)
        elif polar_method == "jordan":
            self.polar_fn = lambda g: zeropower_via_newtonschulz5(g, steps=ns_steps)
        elif polar_method == "svd":
            self.polar_fn = lambda g: SVDPolarFactor(g)
        else:
            raise NotImplementedError

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            betas=betas,
            eps=eps
        )

        # Register all parameters.
        params = list(muon_params) + list(adam_params)
        super().__init__(params, defaults)

        for p in muon_params:
            assert p.ndim >= 2
            self.state[p]["muon"] = True

        for p in adam_params:
            self.state[p]["muon"] = False

        # Initalize model truncation.
        self.truncate_loss = truncate_loss
        self.use_truncation = self.truncate_loss is not None
        if self.use_truncation:
            assert momentum == betas[0]
            self.loss_model = None

        self.stale_nuc = stale_nuc

    def step(self, closure=None, loss=None):
        """Perform a single optimization step.
            Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
            loss (torch.Tensor, optional): Tensor holding the loss of the current iteration.
        """

        if self.use_truncation:
            assert (closure is not None) or (loss is not None), "Either loss tensor or closure must be passed."
            assert (closure is None) or (loss is None), "Pass either the loss tensor or the closure, not both."

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        wd = group["wd"]
        momentum = group["momentum"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        # First pass over parameters: Update momentum and model truncation variables.
        global_dual_norm = 0.0
        muon_dual_norm = 0.0
        adam_sq_dual_norm = 0.0
        current_loss_model = 0.0
        new_loss_model = 0.0
        for p in group["params"]:
            g = p.grad
            if g is None:
                continue

            # Update momentum.
            state = self.state[p]
            if state["muon"]:
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = g.clone()
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g, alpha=1.0-momentum)
            else:
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = g.clone()
                    state["sq_momentum_buffer"] = g.square()
                buf = state["momentum_buffer"]
                buf2 = state["sq_momentum_buffer"]
                buf.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

            # Update model truncation variables.
            if self.use_truncation:
                current_loss_model += torch.sum(torch.mul(p.data, p.grad.data))
                new_loss_model += torch.sum(torch.mul(p.data, buf.data))

            # Compute dual norm of layer momentum and add to running sums to compute
            # dual norm of global momentum.
            if state["muon"]:
                if not self.stale_nuc or "prev_nuc_norm" not in state:
                    # Compute nuclear norm from polar factor.
                    m = state["momentum_buffer"]
                    u = self.polar_fn(m.view(m.shape[0], -1)).view(m.shape)
                    muon_dual_norm += (m * u).sum()
                else:
                    # Reuse stale nuclear norm.
                    muon_dual_norm += state["prev_nuc_norm"]
            else:
                m = state["momentum_buffer"]
                v = state["sq_momentum_buffer"]
                adam_sq_dual_norm += torch.sum(m ** 2 / (eps + v.sqrt()))

        global_dual_norm = torch.sqrt(self.muon_lr_scale * muon_dual_norm ** 2 + adam_sq_dual_norm)

        # Update running average for truncated model and compute truncated_lr.
        current_lr = lr
        if self.use_truncation:
            loss_model_update = loss.item() - current_loss_model.item()
            if self.loss_model is None:
                self.loss_model = loss_model_update
            self.loss_model = momentum * self.loss_model + (1 - momentum) * loss_model_update
            truncated_lr = (self.loss_model - self.truncate_loss + new_loss_model.item()) / global_dual_norm ** 2
            current_lr = min(truncated_lr, lr)

        # Update Muon parameters.
        for p in group["params"]:
            state = self.state[p]
            g = p.grad
            if not state["muon"] or g is None:
                continue

            # Apply update.
            m = state["momentum_buffer"]
            u = self.polar_fn(m.view(m.shape[0], -1)).view(m.shape)
            p.data.mul_(1 - lr * wd)
            p.data.add_(u, alpha=-current_lr * self.muon_lr_scale * muon_dual_norm)

            # Store nuclear norm for next round, if necessary.
            if self.stale_nuc:
                state["prev_nuc_norm"] = (m * u).sum()

        # Update Adam parameters.
        for p in group["params"]:
            state = self.state[p]
            g = p.grad
            if state["muon"] or g is None:
                continue

            # Apply update.
            m = state["momentum_buffer"]
            v = state["sq_momentum_buffer"]
            g = m / (eps + v.sqrt())
            p.data.mul_(1 - lr * wd)
            p.data.add_(g, alpha=-current_lr)

        return loss
