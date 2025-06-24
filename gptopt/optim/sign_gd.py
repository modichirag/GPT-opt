import warnings

import torch

class SignGD(torch.optim.Optimizer):
    """
    Simple implementation of Sign GD in two variations (LMO and GD).

    Arguments:
        params: Parameters to optimize.
        lr: The learning rate.
        wd: Weight decay.
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        lmo: Whether to use LMO instead variational viewpoint of gradient descent to derive
        update rule. If lmo=False, update is additionally scaled by the dual norm of the
        gradient.
    """
    def __init__(self, params, lr=1e-3, wd=0.1, momentum=0.95, nesterov=True, lmo=True):

        defaults = dict(
                lr=lr,
                wd=wd,
                momentum=momentum,
                nesterov=nesterov,
                lmo=lmo,
        )

        super().__init__(params, defaults)

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
                        
        if len(self.param_groups) > 1:
            warnings.warn("More than one param group. This may cause issues with the update scaling.")

        for group in self.param_groups:

            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            lmo = group["lmo"]

            if lmo:
                grad_dual_norm = None
            else:
                grad_dual_norm = 0.0

            # Compute momentum and learning rate scaling.
            for p in group["params"]:

                g = p.grad
                if g is None:
                    continue

                # Compute momentum.
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # If necessary, accumulate gradient L1 norm.
                if not lmo:
                    grad_dual_norm += float(torch.sum(torch.abs(g)))

            # Apply weight updates.
            for p in group["params"]:

                g = p.grad
                if g is None:
                    continue

                buf = self.state[p]["momentum_buffer"]
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Apply weight decay.
                p.data.mul_(1 - lr * wd)

                # Apply update.
                sign_g = torch.sign(g)
                adjusted_lr = lr if lmo else lr * grad_dual_norm
                p.data.add_(sign_g, alpha=-adjusted_lr)

        return loss
