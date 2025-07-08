import torch
from torch.optim.optimizer import Optimizer

class ClipSGD(Optimizer):
    def __init__(self, params, lr=1e-2,  weight_decay=0, clip_type = 'L2', clip_value=0.111, huber_mu = 0.5):
        """
        Implements stochastic gradient descent (optionally with momentum).

        Parameters
        ----------
        params : iterable
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr : float
            Learning rate.
        weight_decay : float, optional
            Weight decay (L2 penalty) (default: 0).
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        if clip_type not in ['L2', 'L1', 'huber']:
            raise ValueError(f"Invalid clip_type: {clip_type}. Supported values are 'L2', 'L1', 'huber'.")
        defaults = dict(lr=lr,  weight_decay=weight_decay, clip_type = clip_type, clip_value=clip_value, huber_mu = huber_mu)
        super(ClipSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        loss : torch.Tensor, optional
            The loss value if `closure` is provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            clip_type = group['clip_type']
            clip_value = group['clip_value']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = state['momentum_buffer']
                    epsilon = 1e-8
                    if clip_type == 'L2':
                        oneminusmom = clip_value/ (torch.norm(buf-grad)+ epsilon) if torch.norm(buf-grad) > clip_value else 1.0
                        momentum = 1 - oneminusmom
                        buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                    elif clip_type == 'L1':
                        # Coordinate-wise clipping
                        diff = grad - buf
                        clipped_diff = torch.clamp(diff, min=-clip_value, max=clip_value)
                        buf.add_(clipped_diff)
                    elif clip_type == 'huber':
                        huber_mu = group['huber_mu']
                        # Compute the difference between the momentum buffer and the gradient
                        diff = buf - grad
                        norm_diff = torch.norm(diff)
                        # Compute beta_t
                        beta_t = 1 - (clip_value * huber_mu) / (max(norm_diff, huber_mu * (1 + clip_value)+ epsilon ))
                        # Update the momentum buffer using the huber update rule
                        buf.mul_(beta_t).add_(grad, alpha=1 - beta_t)
                    

                p.data.add_(buf, alpha=-lr)

        return loss