import torch
import math

from .polar import PolarExpress


norm_options = ["spectral", "linfty", "adam_infty", "adam_2"]


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
        derive update rule. If lmo=False, update is additionally scaled by the dual norm
        of the gradient.
        l2_prod_norm: Whether to use the L2 norm for the product space over layers
        instead of the max norm, which scales each layer's LR by the nuclear norm of the
        gradient.
        nuc_approx: How to approximate the gradient nuclear norm. Choices: [None, 'fro', 'past']
        linfty_scale: Coefficient for norm of layers with "linfty" norm. Will scale the
        learning rate of these layers by `1/linfty_scale` for LMO and
        `1/linfty_scale**2` for GD.
        embed_norm: Which norm to use on embedding layer parameters. Choices: ["linfty",
        "adam_infty", "adam_2"]. Note that "adam_infty" will essentially induce Adam when
        lmo=True, and an unnormalized version of Adam when lmo=False, while "adam_2"
        will induce Adam when lmo=False, and a normalized version of Adam when lmo=True.
        adamw_betas:
        rms_scaling: Whether to use the RMS norm the input/output space of each
        layer, which scale each layer's LR by sqrt(fan_out/fan_in).
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
        l2_prod_norm=False,
        nuc_approx=None,
        linfty_scale=1.0,
        embed_norm="linfty",
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        rms_scaling=False,
        truncate_loss=None,
    ):

        assert embed_norm in ["linfty", "adam_infty", "adam_2"]

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            ns_steps=ns_steps,
            lmo=lmo,
            l2_prod_norm=l2_prod_norm,
            nuc_approx=nuc_approx,
            linfty_scale=linfty_scale,
            embed_norm=embed_norm,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            rms_scaling=rms_scaling,
            truncate_loss=truncate_loss,
        )

        # Assign a norm to each parameter.
        sorted_params = {norm: [] for norm in norm_options}
        for name, p in named_params:
            if p.ndim >= 2 and not any(excluded in name for excluded in ["embeddings", "embed_tokens", "wte", "lm_head", "wpe"]):
                assert p.ndim == 2 # sanity check that we aren't applying Muon for any parameters with more than 2 axes
                current_norm = "spectral"
            else:
                current_norm = embed_norm
            sorted_params[current_norm].append(p)

        # Register all parameters.
        params = []
        for norm in sorted_params:
            params += sorted_params[norm]
        super().__init__(params, defaults)

        # Encode parameter norms in optimizer state.
        for norm in sorted_params:
            for p in sorted_params[norm]:
                self.state[p]["norm"] = norm

        # Set up model truncation.
        self.use_truncation = truncate_loss is not None
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
            lmo = group["lmo"]
            l2_prod_norm = group["l2_prod_norm"]
            nuc_approx = group["nuc_approx"]
            linfty_scale = group["linfty_scale"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            rms_scaling = group["rms_scaling"]
            truncate_loss = group["truncate_loss"]

            # First pass over parameters: Compute momentum/Adam buffers and model
            # truncation variables.
            current_loss_model = 0.0
            new_loss_model = 0.0
            for i, p in enumerate(group["params"]):
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

                else:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = g.clone()
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g, alpha=1.0-momentum)

                if self.use_truncation:
                    current_loss_model += torch.sum(torch.mul(p.data, p.grad.data))
                    new_loss_model += torch.sum(torch.mul(p.data, buf.data))

            # Second pass over parameters: Compute dual norm of layer gradients, if
            # needed.
            layer_dual_norms = None
            need_dual_norms = (not lmo) or l2_prod_norm or self.use_truncation
            for i, p in enumerate(group["params"]):
                g = p.grad
                if g is None:
                    continue

                # quit now if update doesn't depend on dual norm of layer gradients.
                if layer_dual_norms is None:
                    layer_dual_norms = torch.zeros(len(group["params"]), device=p.device)
                if not need_dual_norms:
                    break

                state = self.state[p]
                pre_lmo = state["momentum_buffer"]

                # Compute dual norm of layer gradient.
                if state["norm"] == "linfty":
                    layer_dual_norms[i] = torch.sum(torch.abs(pre_lmo)) / linfty_scale

                elif state["norm"] == "adam_infty":
                    buf1 = state["momentum_buffer"]
                    buf2 = state["sq_momentum_buffer"]
                    layer_dual_norms[i] = torch.sum(torch.abs(buf1 / (eps + buf2.sqrt()) * pre_lmo))

                elif state["norm"] == "adam_2":
                    buf1 = state["momentum_buffer"]
                    buf2 = state["sq_momentum_buffer"]
                    layer_dual_norms[i] = torch.linalg.vector_norm(buf1 / (eps + buf2.sqrt()).sqrt())

                elif state["norm"] == "spectral":

                    # Compute or approximate nuclear norm of layer gradient.
                    if nuc_approx is None or (nuc_approx == "past" and "past_nuc" not in state):
                        # If G = UDV^T, then nuc(G) = tr(G @ UV^T).
                        u = PolarExpress(pre_lmo, steps=group["ns_steps"])
                        layer_dual_norms[i] = torch.trace(pre_lmo.bfloat16().T @ u)

                    elif nuc_approx == "fro":
                        layer_dual_norms[i] = torch.linalg.matrix_norm(pre_lmo, ord="fro")
                    elif nuc_approx == "past":
                        layer_dual_norms[i] = state["past_nuc"]
                    else:
                        raise NotImplementedError

                    # Apply RMS scaling to nuclear norms.
                    if rms_scaling:
                        fan_out, fan_in = p.shape[:2]
                        layer_dual_norms[i] *= math.sqrt(fan_out / fan_in)

                else:
                    raise NotImplementedError

            # Compute dual norm of gradient, which is used to scale LR.
            global_dual_norm = None
            if need_dual_norms:
                if l2_prod_norm:
                    global_dual_norm = torch.linalg.vector_norm(layer_dual_norms, ord=2)
                else:
                    global_dual_norm = torch.sum(layer_dual_norms)

            # Update running average for truncated model and compute truncated lr.
            current_lr = lr
            if self.use_truncation:
                loss_model_update = loss.item() - current_loss_model.item()
                if self.loss_model is None:
                    self.loss_model = loss_model_update
                self.loss_model = momentum * self.loss_model + (1 - momentum) * loss_model_update
                current_lr = min((self.loss_model - truncate_loss + new_loss_model.item()) / global_dual_norm ** 2, lr)
            self.step_size_list.append(current_lr)

            # Third pass over parameters: apply weight updates.
            for i, p in enumerate(group["params"]):
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                pre_lmo = state["momentum_buffer"]

                # Compute update direction.
                if state["norm"] == "linfty":
                    post_lmo = torch.sign(pre_lmo) / linfty_scale

                elif state["norm"] == "adam_infty":
                    post_lmo = pre_lmo / (eps + state["sq_momentum_buffer"].sqrt())

                elif state["norm"] == "adam_2":
                    v = state["sq_momentum_buffer"]
                    dual_norm = torch.linalg.vector_norm(pre_lmo / (eps + v.sqrt()).sqrt())
                    post_lmo = pre_lmo / ((eps + v.sqrt()) * dual_norm)

                elif state["norm"] == "spectral":
                    post_lmo = PolarExpress(pre_lmo, steps=group["ns_steps"])

                    # Compute and store nuclear norm of pre_lmo if necessary.
                    if nuc_approx == "past":
                        if "past_nuc" not in state:
                            state["past_nuc"] = torch.zeros(1, device=p.device)
                        state["past_nuc"] = torch.trace(pre_lmo.bfloat16().T @ post_lmo)

                else:
                    raise NotImplementedError

                # Apply scaling factors to lr depending on steepest descent variations
                lr_scale = 1.0
                if lmo and not l2_prod_norm:
                    if rms_scaling:
                        fan_out, fan_in = p.shape[:2]
                        lr_scale = math.sqrt(fan_out / fan_in)
                if lmo and l2_prod_norm:
                    lr_scale = layer_dual_norms[i] / global_dual_norm
                if not lmo and not l2_prod_norm:
                    lr_scale = global_dual_norm
                if not lmo and l2_prod_norm:
                    lr_scale = layer_dual_norms[i]
                adjusted_lr = lr_scale * current_lr

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(post_lmo, alpha=-adjusted_lr)
