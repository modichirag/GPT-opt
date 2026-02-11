import torch
import math

from .polar import zeropower_via_newtonschulz5, PolarExpress, SVDPolarFactor


class MuonAdam(torch.optim.Optimizer):
    """
    Arguments:
        params: All parameters to be optimized, in two groups.
        lr: The learning rate. (0.02 is a good default)
        wd: Weight decay.
        momentum: The momentum used for gradient accumulation. (0.95 is a good default)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        polar_method: Which method to compute polar factor. Choices: ["jordan",
            "polar_express", "svd"].
        betas: (beta1, beta2) for adam.
        eps: epsilon for adam.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        ns_steps=5,
        polar_method="polar_express",
        betas=(0.95, 0.95),
        eps=1e-8,
    ):

        # We expect exactly two parameter groups, one with opt="muon" and one with
        # opt="adam".
        assert isinstance(params, list)
        assert len(params) == 2
        assert isinstance(params[0], dict)
        assert isinstance(params[1], dict)
        opts = [param_group["opt"] for param_group in params]
        assert "muon" in opts and "adam" in opts

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            betas=betas,
            eps=eps,
        )
        if polar_method == "polar_express":
            self.polar_fn = lambda g: PolarExpress(g, steps=ns_steps)
        elif polar_method == "jordan":
            self.polar_fn = lambda g: zeropower_via_newtonschulz5(g, steps=ns_steps)
        elif polar_method == "svd":
            self.polar_fn = lambda g: SVDPolarFactor(g)
        else:
            raise NotImplementedError

        # Register all parameters.
        super().__init__(params, defaults)

        # Index parameter groups by optimizer name.
        self.opt_param_groups = {}
        for param_group in self.param_groups:
            self.opt_param_groups[param_group["opt"]] = param_group

        # Check parameter sizes for Muon parameters.
        for p in self.opt_param_groups["muon"]["params"]:
            assert p.ndim == 2

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
        # Update Muon parameters.
        group = self.opt_param_groups["muon"]
        lr = group["lr"]
        wd = group["wd"]
        momentum = group["momentum"]

        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            state = self.state[p]

            # Update momentum.
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = g.clone()
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g, alpha=1.0-momentum)

            # Apply update.
            u = self.polar_fn(buf)
            p.data.mul_(1 - lr * wd)
            p.data.add_(u, alpha=-lr)

        # Update Adam parameters.
        group = self.opt_param_groups["adam"]
        lr = group["lr"]
        wd = group["wd"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            state = self.state[p]

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
