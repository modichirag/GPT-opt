import torch

from .polar import zeropower_via_newtonschulz5, PolarExpress, SVDPolarFactor


class QwopAdam(torch.optim.Optimizer):
    """
    Arguments:
        muon_params: Parameters to be optimized with Muon.
        adam_params: Parameters to be optimized with Adam.
        lr: The learning rate. (0.02 is a good default)
        wd: Weight decay.
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        polar_method: Which method to compute polar factor. Choices: ["jordan",
            "polar_express", "svd"].
        betas: (beta1, beta2) for adam.
        eps: epsilon for adam.
    """
    def __init__(
        self,
        muon_params,
        adam_params,
        lr=1e-3,
        wd=0.1,
        ns_steps=5,
        polar_method="polar_express",
        betas=(0.95, 0.95),
        eps=1e-8,
    ):

        assert isinstance(muon_params, list)
        assert isinstance(adam_params, list)

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

    def step(self, closure=None, loss=None):
        """Perform a single optimization step.
            Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
            loss (torch.Tensor, optional): Tensor holding the loss of the current iteration.
        """

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        wd = group["wd"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        # Update Muon parameters.
        for p in group["params"]:
            state = self.state[p]
            g = p.grad
            if not state["muon"] or g is None:
                continue

            # Update momentum and second moment estimate.
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = g.clone()
                state["sq_momentum_buffer"] = g.square()
            buf = state["momentum_buffer"]
            buf2 = state["sq_momentum_buffer"]
            buf.mul_(beta1).add_(g, alpha=1.0-beta1)
            buf2.lerp_(g.square(), 1.0 - beta2)

            # Compute update and lr scale for layer.
            u = self.polar_fn(buf.view(buf.shape[0], -1)).view(buf.shape)
            d = (buf * u).sum()
            e = ((eps + buf2.sqrt()) * u * u).sum()
            layer_lr_scale = d / e

            # Apply update.
            p.data.mul_(1 - lr * wd)
            p.data.add_(u, alpha=-lr * layer_lr_scale)

        # Update Adam parameters.
        for p in group["params"]:
            state = self.state[p]
            g = p.grad
            if state["muon"] or g is None:
                continue

            # Update momentum and second moment estimate.
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = g.clone()
                state["sq_momentum_buffer"] = g.square()
            buf = state["momentum_buffer"]
            buf2 = state["sq_momentum_buffer"]
            buf.lerp_(g, 1.0 - beta1)
            buf2.lerp_(g.square(), 1.0 - beta2)

            # Apply update.
            g = buf / (eps + buf2.sqrt())
            p.data.mul_(1 - lr * wd)
            p.data.add_(g, alpha=-lr)

        return loss
