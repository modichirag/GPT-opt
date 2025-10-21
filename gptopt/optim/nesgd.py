import torch
import math

from .polar import zeropower_via_newtonschulz5, PolarExpress, SVDPolarFactor



class LinftyNorm:
    def __init__(self):
        pass

    def lmo(self, g, state):
        return torch.sign(g)

    def dual(self, g, state):
        return torch.linalg.vector_norm(g, ord=1)


class SpectralNorm:
    def __init__(self, spectral_scale=1.0, nuc_approx=None, polar_method="polar_express", ns_steps=5):
        assert nuc_approx in [None, "fro", "past"]
        self.spectral_scale = spectral_scale
        self.nuc_approx = nuc_approx
        self.ns_steps = 5
        if polar_method == "jordan":
            self.polar_fn = lambda g: zeropower_via_newtonschulz5(g, steps=self.ns_steps)
        elif polar_method == "polar_express":
            self.polar_fn = lambda g: PolarExpress(g, steps=self.ns_steps)
        elif polar_method == "svd":
            self.polar_fn = lambda g: SVDPolarFactor(g)
        else:
            raise NotImplementedError

    def lmo(self, g, state):
        return self.spectral_scale * self.polar_fn(g)

    def dual(self, g, state):
        if self.nuc_approx is None or (self.nuc_approx == "past" and "past_nuc" not in state):
            # If G = UDV^T, then nuc(G) = tr(G @ UV^T).
            u = self.polar_fn(g)
            nuc = (g.bfloat16() * u).sum()
        elif self.nuc_approx == "fro":
            nuc = torch.linalg.matrix_norm(g, ord="fro")
        elif self.nuc_approx == "past":
            nuc = state["past_nuc"]
        else:
            raise NotImplementedError

        return self.spectral_scale * nuc


class AdamLinftyNorm:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def lmo(self, g, state):
        m = state["momentum_buffer"]
        v = state["sq_momentum_buffer"]
        return torch.sign(g) * torch.abs(m) / (self.eps + v.sqrt())

    def dual(self, g, state):
        m = state["momentum_buffer"]
        v = state["sq_momentum_buffer"]
        return torch.linalg.vector_norm(g * torch.abs(m) / (self.eps + v.sqrt()), ord=1)


class AdamL2Norm:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def lmo(self, g, state):
        v = state["sq_momentum_buffer"]
        return g / ((self.eps + v.sqrt()) * self.dual(g, state))

    def dual(self, g, state):
        v = state["sq_momentum_buffer"]
        return torch.linalg.vector_norm(g / (self.eps + v.sqrt()).sqrt(), ord=2)


class LinftyProductNorm:
    def lmo(muon_dual_norms, other_dual_norms):
        lmo_dict = {}
        layer_dual_norms = muon_dual_norms | other_dual_norms
        for p in layer_dual_norms:
            lmo_dict[p] = torch.sign(layer_dual_norms[p])
        return lmo_dict

    def dual(muon_dual_norms, other_dual_norms):
        layer_dual_norms = torch.stack(
            [muon_dual_norms[p] for p in muon_dual_norms] +
            [other_dual_norms[p] for p in other_dual_norms]
        )
        return torch.linalg.vector_norm(layer_dual_norms, ord=1)


class L2ProductNorm:
    def lmo(muon_dual_norms, other_dual_norms):
        global_dual_norm = L2ProductNorm.dual(muon_dual_norms, other_dual_norms)
        layer_dual_norms = muon_dual_norms | other_dual_norms
        lmo_dict = {}
        for p in layer_dual_norms:
            lmo_dict[p] = layer_dual_norms[p] / global_dual_norm
        return lmo_dict

    def dual(muon_dual_norms, other_dual_norms):
        layer_dual_norms = torch.stack(
            [muon_dual_norms[p] for p in muon_dual_norms] +
            [other_dual_norms[p] for p in other_dual_norms]
        )
        return torch.linalg.vector_norm(layer_dual_norms, ord=2)


class HybridProductNorm:
    def lmo(muon_dual_norms, other_dual_norms):
        muon_vec = torch.stack([muon_dual_norms[p] for p in muon_dual_norms])
        other_vec = torch.stack([other_dual_norms[p] for p in other_dual_norms])
        muon_dual_norm = torch.linalg.vector_norm(muon_vec, ord=1)
        other_dual_norm = torch.linalg.vector_norm(other_vec, ord=2)
        global_dual_norm = (muon_dual_norm.square() + other_dual_norm.square()).sqrt()

        lmo_dict = {}
        for p in muon_dual_norms:
            lmo_dict[p] = muon_dual_norm / global_dual_norm
        for p in other_dual_norms:
            lmo_dict[p] = other_dual_norms[p] / global_dual_norm
        return lmo_dict

    def dual(muon_dual_norms, other_dual_norms):
        muon_vec = torch.stack([muon_dual_norms[p] for p in muon_dual_norms])
        other_vec = torch.stack([other_dual_norms[p] for p in other_dual_norms])
        muon_dual_norm = torch.linalg.vector_norm(muon_vec, ord=1)
        other_dual_norm = torch.linalg.vector_norm(other_vec, ord=2)
        return (muon_dual_norm.square() + other_dual_norm.square()).sqrt()


norm_obj_dict = {
    "linfty": LinftyNorm,
    "spectral": SpectralNorm,
    "adam_infty": AdamLinftyNorm,
    "adam_2": AdamL2Norm,
}
product_norm_obj_dict = {
    "linfty": LinftyProductNorm,
    "l2": L2ProductNorm,
    "hybrid": HybridProductNorm,
}


class NESGD(torch.optim.Optimizer):
    """
    NESGD - Non-Euclidean Stochastic Gradient Descent

    Non-Euclidean SGD according to a norm on the space of neural network parameters. The
    neural network parameters are considered as a Cartesian product of matrices (linear
    layer weights) and vectors (biases, embedding layers, and everything else); we endow
    each matrix or vector parameter with a norm, then construct a norm of the entire
    parameter space as a product norm over all parameter norms. We then run gradient
    descent with respect to this norm.

    Arguments:
        named_params: All parameters to be optimized, with names included.
        lr: The learning rate. (0.02 is a good default)
        wd: Weight decay.
        momentum: The momentum used for gradient accumulation. (0.95 is a good default)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        lmo: Whether to use LMO instead of variational viewpoint of gradient descent to
            derive update rule. If lmo=False, update is additionally scaled by the dual
            norm of the gradient.
        prod_norm: Which product norm to use. Choices ["linfty", "l2", "hybrid"].
            instead of the max norm, which scales each layer's LR by the nuclear norm of
            the gradient. "hybrid" applies the linfty norm to the product of all muon
            layers, the l2 norm to the product of all non-muon layers, then takes the l2
            norm of the resulting two-coordinate vector.
        nuc_approx: How to approximate the gradient nuclear norm. Choices: [None, 'fro',
            'past']
        spectral_scale: Scales spectral norm by `1/spectral_scale`. Will scale the
            layer-wise dual and lmo of these layers by `spectral_scale`, which affects
            the effective LR for these layers in different ways depending on the product
            norm. If using the linfty product norm and lmo=True (as in Muon and Scion),
            this scales the Muon LR by `spectral_scale`. If using the l2 or hybrid
            product norm and lmo=False (as in MuonMax and PolarGrad), this scales the
            Muon LR by `spectral_scale**2`.
        polar_method: Which method to compute polar factor. Choices: ["jordan",
            "polar_express", "svd"].
        embed_norm: Which norm to use on embedding layer parameters. Choices: ["linfty",
            "adam_infty", "adam_2"]. Note that "adam_infty" will essentially induce Adam
            when lmo=True, and an unnormalized version of Adam when lmo=False, while
            "adam_2" will induce Adam when lmo=False, and a normalized version of Adam
            when lmo=True.
        adamw_betas: (beta1, beta2) for adam.
        adamw_eps: epsilon for adam.
        truncate_loss: Lower bound of loss, if using a truncated model.
    """
    def __init__(
        self,
        named_params,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        ns_steps=5,
        lmo=False,
        prod_norm="linfty",
        nuc_approx=None,
        spectral_scale=1.0,
        polar_method="polar_express",
        embed_norm="linfty",
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        truncate_loss=None,
    ):

        assert prod_norm in ["linfty", "l2", "hybrid"]
        assert embed_norm in ["linfty", "adam_infty", "adam_2"]
        if truncate_loss is not None:
            assert momentum == adamw_betas[0]

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            adamw_betas=adamw_betas,
        )
        self.lmo = lmo
        self.nuc_approx = nuc_approx
        self.spectral_scale = spectral_scale
        self.polar_method = polar_method
        self.truncate_loss = truncate_loss
        if self.nuc_approx is not None:
            print(f"Using {self.nuc_approx} approximation for nuclear norm.")

        # Assign a norm to each parameter.
        self.embed_norm = embed_norm
        sorted_params = {}
        for name, p in named_params:
            if p.ndim >= 2 and not any(excluded in name for excluded in ["embeddings", "embed_tokens", "wte", "lm_head", "wpe"]):
                assert p.ndim == 2 # sanity check that we aren't applying Muon for any parameters with more than 2 axes
                current_norm = "spectral"
            else:
                current_norm = self.embed_norm
            if current_norm not in sorted_params:
                sorted_params[current_norm] = []
            sorted_params[current_norm].append(p)

        # Register all parameters.
        params = []
        for norm in sorted_params:
            params += sorted_params[norm]
        super().__init__(params, defaults)

        # Encode parameter norms in optimizer state.
        for norm in sorted_params:

            norm_kwargs = {}
            if norm == "spectral":
                norm_kwargs["spectral_scale"] = spectral_scale
                norm_kwargs["nuc_approx"] = nuc_approx
                norm_kwargs["polar_method"] = polar_method
                norm_kwargs["ns_steps"] = ns_steps
            elif norm == ["adam_infty", "adam_2"]:
                norm_kwargs["eps"] = adamw_eps

            for p in sorted_params[norm]:
                self.state[p]["norm"] = norm
                self.state[p]["norm_obj"] = norm_obj_dict[norm](**norm_kwargs)

        self.product_norm = prod_norm
        self.product_norm_obj = product_norm_obj_dict[prod_norm]

        # Set up model truncation.
        self.use_truncation = self.truncate_loss is not None
        if self.use_truncation:
            self.loss_model = None
        self.step_size_list = list()

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

        # Warning for the future: if we ever use more than one param group, the LR
        # scalings are not going to behave exactly right. Inside the following loop we
        # compute scaling factors that depend on all layers of the network, so we assume
        # that all layers of the network are inside the current param group.
        assert len(self.param_groups) == 1

        for group in self.param_groups:

            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            beta1, beta2 = group["adamw_betas"]

            # First pass over parameters: Compute momentum/Adam buffers, model
            # truncation variables, and per-layer dual norm of momentum.
            current_loss_model = 0.0
            new_loss_model = 0.0
            need_dual_norms = not (self.lmo and self.product_norm == "linfty") or self.use_truncation
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if state["norm"] in ["adam_infty", "adam_2"]:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = g.clone()
                        state["sq_momentum_buffer"] = g.square()
                    buf = state["momentum_buffer"]
                    buf2 = state["sq_momentum_buffer"]
                    buf.lerp_(g, 1 - beta1)
                    buf2.lerp_(g.square(), 1 - beta2)

                elif state["norm"] in ["spectral", "linfty"]:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = g.clone()
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g, alpha=1.0-momentum)

                else:
                    raise NotImplementedError

                if not need_dual_norms:
                    continue

                # Update model truncation variables.
                if self.use_truncation:
                    current_loss_model += torch.sum(torch.mul(p.data, p.grad.data))
                    new_loss_model += torch.sum(torch.mul(p.data, buf.data))

                # Compute dual norm of layer momentum.
                if "layer_dual_norm" not in state:
                    state["layer_dual_norm"] = torch.zeros(1, device=p.device)
                state["layer_dual_norm"] = state["norm_obj"].dual(buf, state)

            # Compute dual norm of overall momentum, and per-layer LR scalings.
            global_dual_norm = None
            if need_dual_norms:
                muon_dual_norms = {
                    p: self.state[p]["layer_dual_norm"] for p in group["params"]
                    if p.grad is not None and self.state[p]["norm"] == "spectral"
                }
                other_dual_norms = {
                    p: self.state[p]["layer_dual_norm"] for p in group["params"]
                    if p.grad is not None and self.state[p]["norm"] == self.embed_norm
                }
                global_dual_norm = self.product_norm_obj.dual(muon_dual_norms, other_dual_norms)
                lr_scalings = self.product_norm_obj.lmo(muon_dual_norms, other_dual_norms)
            else:
                lr_scalings = {p: 1.0 for p in group["params"] if p.grad is not None}

            # Update running average for truncated model and compute truncated lr.
            current_lr = lr
            if self.use_truncation:
                loss_model_update = loss.item() - current_loss_model.item()
                if self.loss_model is None:
                    self.loss_model = loss_model_update
                self.loss_model = momentum * self.loss_model + (1 - momentum) * loss_model_update
                truncated_lr = (self.loss_model - self.truncate_loss + new_loss_model.item()) / global_dual_norm ** 2
                if self.lmo:
                    truncated_lr *= global_dual_norm
                current_lr = min(truncated_lr, lr)

            self.step_size_list.append(float(current_lr))

            # Second pass over parameters: apply weight updates.
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                pre_lmo = state["momentum_buffer"]
                post_lmo = state["norm_obj"].lmo(pre_lmo, state)

                # Compute and store nuclear norm if necessary.
                if state["norm"] == "spectral" and self.nuc_approx == "past":
                    if "past_nuc" not in state:
                        state["past_nuc"] = torch.zeros(1, device=p.device)
                    # If G = UDV^T, then nuc(G) = tr(G @ UV^T).
                    state["past_nuc"] = (pre_lmo.bfloat16() * post_lmo).sum()

                # Apply layer-wise scaling to lr.
                lr_scale = lr_scalings[p]
                if not self.lmo:
                    lr_scale *= global_dual_norm
                adjusted_lr = lr_scale * current_lr

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(post_lmo, alpha=-adjusted_lr)

        return loss
