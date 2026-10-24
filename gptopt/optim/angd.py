import torch
import torch.distributed as dist


class ANGD(torch.optim.Optimizer):
    """
    Diagonal Attention Natural Gradient Descent.
    The query and key weights of each attention module are updated like Adam, except
    that the second moment is an EMA of the diagonal Fisher information of the attention
    distributions, instead of an EMA of squared gradients. All other parameters are
    updated with Adam.

    Arguments:
        attn_modules: CausalSelfAttention modules (built with split_qkv=True). The weights c_q.weight
            and c_k.weight of each module are optimized with ANGD.
        adam_params: Parameters to be optimized with Adam.
        lr: The learning rate.
        angd_lr_scale: Multiple for lr of ANGD params.
        wd: Weight decay.
        betas: (beta1, beta2). beta1 is the momentum for all params, beta2 is the EMA parameter for the
            Fisher (ANGD params) and squared gradients (Adam params).
        eps: epsilon added to the square root of the second moment.
        precond_every: Recompute the Fisher every this many steps.
        fisher_batch: Number of sequences (per rank) from the last microbatch used to compute the
            Fisher. None uses the whole microbatch.
        fisher_chunk: Number of sequences processed at once when computing the Fisher.
    """
    def __init__(
        self,
        attn_modules,
        adam_params,
        lr=1e-3,
        angd_lr_scale=1.0,
        wd=0.1,
        betas=(0.95, 0.95),
        eps=1e-8,
        precond_every=1,
        fisher_batch=None,
        fisher_chunk=4,
    ):

        assert isinstance(attn_modules, list)
        assert isinstance(adam_params, list)

        self.attn_modules = attn_modules
        self.angd_lr_scale = angd_lr_scale
        self.precond_every = precond_every
        self.fisher_batch = fisher_batch
        self.fisher_chunk = fisher_chunk
        self.step_count = 0

        defaults = dict(
            lr=lr,
            wd=wd,
            betas=betas,
            eps=eps
        )

        # Register all parameters.
        angd_params = [p for m in attn_modules for p in (m.c_q.weight, m.c_k.weight)]
        params = angd_params + list(adam_params)
        assert len(set(map(id, params))) == len(params)
        super().__init__(params, defaults)

        for p in angd_params:
            self.state[p]["angd"] = True

        for p in adam_params:
            self.state[p]["angd"] = False

        # Ask attention modules to store their inputs during training forward passes.
        for m in attn_modules:
            m.angd_capture = True

    @torch.no_grad()
    def update_fisher(self, beta2):
        """Update the EMA of the diagonal Fisher for each ANGD parameter."""
        for m in self.attn_modules:
            x = m.angd_input
            assert x is not None, "No stored attention input. Was there a training forward pass?"
            if self.fisher_batch is not None:
                x = x[:self.fisher_batch]
            D_Q, D_K = m.angd_fisher_diag(x, chunk_size=self.fisher_chunk)
            m.angd_input = None

            # Average over ranks, so that all replicas apply the same update.
            if dist.is_initialized():
                dist.all_reduce(D_Q, op=dist.ReduceOp.AVG)
                dist.all_reduce(D_K, op=dist.ReduceOp.AVG)

            for p, D in ((m.c_q.weight, D_Q), (m.c_k.weight, D_K)):
                state = self.state[p]
                D = D.to(p.dtype)
                if "fisher_buffer" not in state:
                    state["fisher_buffer"] = D
                else:
                    state["fisher_buffer"].lerp_(D, 1.0 - beta2)

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

        group = self.param_groups[0]
        lr = group["lr"]
        wd = group["wd"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        # Update Fisher estimates.
        if self.step_count % self.precond_every == 0:
            self.update_fisher(beta2)
        self.step_count += 1

        for p in group["params"]:
            state = self.state[p]
            g = p.grad
            if g is None:
                continue

            # Update momentum and second moment estimate. ANGD params get their second moment
            # from update_fisher().
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = g.clone()
                if not state["angd"]:
                    state["sq_momentum_buffer"] = g.square()
            elif not state["angd"]:
                state["sq_momentum_buffer"].lerp_(g.square(), 1.0 - beta2)
            buf = state["momentum_buffer"]
            buf.lerp_(g, 1.0 - beta1)
            buf2 = state["fisher_buffer"] if state["angd"] else state["sq_momentum_buffer"]

            # Apply update.
            u = buf / (eps + buf2.sqrt())
            p_lr = lr * self.angd_lr_scale if state["angd"] else lr
            p.data.mul_(1 - p_lr * wd)
            p.data.add_(u, alpha=-p_lr)

        return loss
