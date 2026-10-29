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
            inverting.
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
        ridge=1e-3,
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
            eps=eps,
            gamma=gamma,
            ridge=ridge,
        )

        # Register all parameters.
        params = list(muon_params) + list(adam_params)
        super().__init__(params, defaults)

        for p in muon_params:
            assert p.ndim >= 2
            self.state[p]["muon"] = True

        for p in adam_params:
            self.state[p]["muon"] = False

    def assign_layers(self, params):
        """
        Assign a fixed layer index to each parameter with a gradient. Each Muon parameter gets its
        own index, and all Adam parameters share the last index. Parameters without a gradient get
        no index, and are assumed to never receive a gradient.
        """
        self.num_layers = 0
        for p in params:
            state = self.state[p]
            state["layer"] = None
            if state["muon"] and p.grad is not None:
                state["layer"] = self.num_layers
                self.num_layers += 1
        for p in params:
            state = self.state[p]
            if not state["muon"] and p.grad is not None:
                state["layer"] = self.num_layers
        self.num_layers += 1 # adam parameters are treated as a single layer.

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
        gamma = group["gamma"]
        ridge = group["ridge"]

        # Assign layer indices on the first step, once we know which parameters have gradients.
        if not hasattr(self, "num_layers"):
            self.assign_layers(group["params"])

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
            m.lerp_(g, 1.0 - beta1)

            # Compute polar factor, dual norm, and dot product of update with gradient.
            delta = self.polar_fn(m.view(m.shape[0], -1)).view(m.shape)
            j = state["layer"]
            d[j] += (m * delta).sum()
            u[j] += (g * delta).sum()

        # First pass over Adam parameters (update buffers, compute d and u).
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
            delta = m / (eps + v.sqrt())
            j = state["layer"]
            d[j] += (m * delta).sum()
            u[j] += (g * delta).sum()

        # Update running average of layerwise curvature matrix.
        state = self.state["global"]
        if "lr_precond" not in state:
            state["lr_precond"] = torch.diag(u * u)
        M = state["lr_precond"]
        M.lerp_(torch.outer(u, u), 1.0-gamma)

        # Compute layer-wise learning rates.
        eta = torch.linalg.solve(M + ridge * torch.eye(self.num_layers, device=device), d)

        # Second pass over Muon parameters (apply update).
        for p in group["params"]:
            state = self.state[p]
            if not state["muon"] or p.grad is None:
                continue

            # Compute update direction, apply update.
            m = state["momentum_buffer"]
            delta = self.polar_fn(m.view(m.shape[0], -1)).view(m.shape)
            p.data.mul_(1 - lr * wd)
            p.data.add_(delta, alpha=-lr*eta[state["layer"]])

        # Second pass over Adam parameters (apply update).
        for p in group["params"]:
            state = self.state[p]
            if state["muon"] or p.grad is None:
                continue

            # Compute update direction, apply update.
            m = state["momentum_buffer"]
            v = state["sq_momentum_buffer"]
            delta = m / (eps + v.sqrt())
            p.data.mul_(1 - lr * wd)
            p.data.add_(delta, alpha=-lr*eta[state["layer"]])

        return loss
