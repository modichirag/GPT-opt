import torch

from .polar import zeropower_via_newtonschulz5, PolarExpress, SVDPolarFactor


class QwopGN(torch.optim.Optimizer):
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
        gamma: EMA coefficient for layerwise-curvature matrix.
        ridge: Ridge regularization added to the layerwise-curvature matrix before
            inverting, relative to the mean of its diagonal (i.e. M + ridge *
            mean(diag(M)) * I).
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
        gamma=0.95,
        ridge=1e-2,
    ):

        assert isinstance(muon_params, list)
        assert isinstance(adam_params, list)

        self.ns_steps = ns_steps
        self.gamma = gamma
        self.ridge = ridge

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

        # Assign a fixed layer index to each parameter. Adam parameters share the last index.
        self.num_layers = len(muon_params) + 1
        for i, p in enumerate(muon_params):
            assert p.ndim >= 2
            self.state[p]["muon"] = True
            self.state[p]["layer"] = i

        for p in adam_params:
            self.state[p]["muon"] = False
            self.state[p]["layer"] = self.num_layers - 1

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

        # First pass over Muon parameters (update momentum, compute d and u).
        device = self.param_groups[0]["params"][0].device
        d = torch.zeros(self.num_layers, device=device)
        u = torch.zeros(self.num_layers, device=device)
        for p in group["params"]:
            state = self.state[p]
            g = p.grad
            if not state["muon"] or g is None:
                continue

            # Update momentum.
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = g.clone()
            m = state["momentum_buffer"]
            m.mul_(beta1).add_(g, alpha=1.0-beta1)

            # Compute polar factor, dual norm, and dot product of update with gradient.
            lmo = self.polar_fn(m.view(m.shape[0], -1)).view(m.shape)
            j = state["layer"]
            d[j] += (m * lmo).sum()
            u[j] += (g * lmo).sum()

        # First pass over Adam parameters (update buffers, compute d and u).
        lmo_scale_sq = torch.zeros((), device=device)
        for p in group["params"]:
            state = self.state[p]
            g = p.grad
            if state["muon"] or g is None:
                continue

            # Update momentum and second moment estimate.
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = g.clone()
                state["sq_momentum_buffer"] = g.square()
            m = state["momentum_buffer"]
            v = state["sq_momentum_buffer"]
            m.lerp_(g, 1.0 - beta1)
            v.lerp_(g.square(), 1.0 - beta2)

            # Compute update, dual norm, and dot product of update with gradient.
            lmo = m / (eps + v.sqrt())
            j = state["layer"]
            d[j] += (m * lmo).sum()
            u[j] += (g * lmo).sum()
            lmo_scale_sq += (m ** 2 / (eps + v.sqrt())).sum()

        # Correct d and u terms for Adam parameters.
        lmo_scale = lmo_scale_sq.sqrt()
        d[-1] /= lmo_scale
        u[-1] /= lmo_scale

        # Update running average of layerwise curvature matrix.
        state = self.state["global"]
        if "lr_precond" not in state:
            state["lr_precond"] = torch.diag(u * u)
        M = state["lr_precond"]
        M.mul_(self.gamma).add_(torch.outer(u, u), alpha=1.0-self.gamma)

        # Compute layer-wise learning rates. Ridge is relative to the mean diagonal.
        ridge = self.ridge * M.diagonal().mean()
        eta = torch.linalg.solve(M + ridge * torch.eye(self.num_layers, device=device), d)

        # Second pass over Muon parameters (apply update).
        for p in group["params"]:
            state = self.state[p]
            g = p.grad
            if not state["muon"] or g is None:
                continue

            # Compute update direction, apply update.
            m = state["momentum_buffer"]
            lmo = self.polar_fn(m.view(m.shape[0], -1)).view(m.shape)
            p.data.mul_(1 - lr * wd)
            p.data.add_(lmo, alpha=-lr*eta[state["layer"]])

        # Second pass over Adam parameters (apply update).
        for p in group["params"]:
            state = self.state[p]
            g = p.grad
            if state["muon"] or g is None:
                continue

            # Compute update direction, apply update.
            m = state["momentum_buffer"]
            v = state["sq_momentum_buffer"]
            lmo = m / (eps + v.sqrt())
            lmo /= lmo_scale
            p.data.mul_(1 - lr * wd)
            p.data.add_(lmo, alpha=-lr*eta[state["layer"]])

        return loss
