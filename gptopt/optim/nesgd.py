import torch
import math

from .polar import PolarExpress


norm_options = ["spectral", "infty"]


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
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        lmo: Whether to use LMO instead of variational viewpoint of gradient descent to
        derive update rule. If lmo=False, update is additionally scaled by the dual norm
        of the gradient.
        l2_prod_norm: Whether to use the L2 norm for the product space over layers
        instead of the max norm, which scales each layer's LR by the nuclear norm of the
        gradient.
        nuc_approx: How to approximate the gradient nuclear norm. Choices: [None, 'fro', 'past']
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
        nesterov=False,
        ns_steps=5,
        lmo=False,
        l2_prod_norm=False,
        nuc_approx=None,
        rms_scaling=False,
        truncate_loss=None,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            lmo=lmo,
            l2_prod_norm=l2_prod_norm,
            nuc_approx=nuc_approx,
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
                current_norm = "infty"
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

        for group in self.param_groups:

            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            lmo = group["lmo"]
            l2_prod_norm = group["l2_prod_norm"]
            nuc_approx = group["nuc_approx"]
            rms_scaling = group["rms_scaling"]
            truncate_loss = group["truncate_loss"]

            # initial pass over parameters to compute momentum and dual norm of
            # parameter gradients, which are used to scale the learning rate.
            # Warning for the future: if we ever use more than one param group, these
            # scalings are not going to behave exactly right. Here we compute scaling
            # factors that depend on all layers of the network, so we assume that all
            # layers of the network are inside the current param group.
            layer_dual_norms = None
            need_dual_norms = (not lmo) or l2_prod_norm or self.use_truncation
            current_loss_model = 0.0
            new_loss_model = 0.0
            for i, p in enumerate(group["params"]):

                g = p.grad
                if g is None:
                    continue

                # calc momentum.
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = g.clone()
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g, alpha=1.0-momentum)

                # Compute inner product term of running average for truncated model.
                if self.use_truncation:
                    current_loss_model += torch.sum(torch.mul(p.data, p.grad.data))
                    new_loss_model += torch.sum(torch.mul(p.data, buf.data))

                # quit now if update doesn't depend on dual norm of layer gradients.
                if layer_dual_norms is None:
                    layer_dual_norms = torch.zeros(len(group["params"]), device=p.device)
                if not need_dual_norms:
                    continue

                # Compute dual norm of layer gradient.
                if state["norm"] == "infty":
                    layer_dual_norms[i] = torch.sum(torch.abs(g))

                elif state["norm"] == "spectral":

                    # temp: remove this after things are running
                    if g.ndim > 2:
                        assert False

                    # Compute or approximate nuclear norm of layer gradient.
                    if nuc_approx is None or (nuc_approx == "past" and "past_nuc" not in state):

                        # calc update.
                        if nesterov:
                            g = g.add(buf, alpha=momentum)
                        else:
                            g = buf
                        u = PolarExpress(g, steps=group["ns_steps"])

                        # If G = UDV^T, then nuc(G) = tr(G @ UV^T).
                        layer_dual_norms[i] = torch.trace(g.bfloat16().T @ u)

                    elif nuc_approx == "fro":
                        layer_dual_norms[i] = torch.linalg.matrix_norm(g, ord="fro")
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

            # apply weight updates
            for i, p in enumerate(group["params"]):

                g = p.grad
                if g is None:
                    continue

                # Calc update. Note that we already computed and stored the momentum
                # term before, but we are re-computing the matrix sign. This is
                # suboptimal w.r.t.  time but doesn't use any extra memory. We can
                # always tweak this later.
                state = self.state[p]
                buf = state["momentum_buffer"]
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Compute update direction.
                if state["norm"] == "infty":
                    u = torch.sign(g)

                elif state["norm"] == "spectral":

                    # temp: remove this after things are running
                    if g.ndim > 2:
                        assert False

                    u = PolarExpress(g, steps=group["ns_steps"])

                    # Compute and store nuclear norm of u if necessary.
                    if nuc_approx == "past":
                        if "past_nuc" not in state:
                            state["past_nuc"] = torch.zeros(1, device=p.device)
                        state["past_nuc"] = torch.trace(g.bfloat16().T @ u)

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
                p.data.add_(u, alpha=-adjusted_lr)
