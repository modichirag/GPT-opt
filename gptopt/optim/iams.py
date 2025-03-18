import torch
import warnings
from math import sqrt
from typing import Union
class IAMS(torch.optim.Optimizer):
    def __init__(self,
                 params: 0.0,
                 lr: float=1.0,
                 lmbda: Union[float,None]=9.0,
                 weight_decay: float=0.0,
                 lb: float=0.0,
                 ) -> None:
        """
        IAM optimizer
        Parameters
        ----------
        params : Params
            Model parameters.
        lr : float, optional
            Learning rate cap, by default 1.0.
        lmbda : float or None, optional
            lambda_t from paper, by default 9.0. If set to None, use lambda_t=t
        weight_decay : float, optional
            Weight decay parameter, by default 0.0.
        lb : float, optional
            Lower bound for loss. Zero is often a good guess.
            By default 0.0.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        # if lmbda=None, we use lmbda_t = t
        if lmbda is not None:
            if lmbda < 0.0:
                raise ValueError("Invalid negative lambda value: {}".format(lmbda))
            self._theoretical_lmbda = False
        else:
            self._theoretical_lmbda = True
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr,
                        lmbda=lmbda,
                        weight_decay=weight_decay,
                        weight_sum=0.0
        )
        super(IAMS, self).__init__(params, defaults)
        self.lb = lb
        # Initialization
        self._number_steps = 0
        self.state['step_size_list'] = list() # for storing the adaptive step size term
        self.state['beta_list'] = list()
        return
    def step(self, closure =None, loss: torch.Tensor=None, lb: float=None):
        """
        Performs a single optimization step.
        Parameters
        ----------
        closure : LossClosure, optional
            A callable that evaluates the model (possibly with backprop) and returns the loss, by default None.
        loss : torch.tensor, optional
            The loss tensor. Use this when the backward step has already been performed. By default None.
        lb : float, optional
            The optimal value for this batch of data. If None, the use the general lower bound from initialization.
        Returns
        -------
        (Stochastic) Loss function value.
        """
        assert (closure is not None) or (loss is not None), "Either loss tensor or closure must be passed."
        assert (closure is None) or (loss is None), "Pass either the loss tensor or the closure, not both."
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if len(self.param_groups) > 1:
            warnings.warn("More than one param group. step_size_list contains adaptive term of last group.")
            warnings.warn("More than one param group. This might cause issues for the step method.")
        _norm = 0.
        _dot = 0.
        ############################################################
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                grad = p.grad.data.detach()
                state = self.state[p]
                # Initialize Averaging Variables
                if self._number_steps == 0:
                    state['z'] = p.detach().clone().to(p.device)
                z = state['z']
                _dot += torch.sum(torch.mul(grad, z-p.data))
                _norm += torch.sum(torch.mul(grad, grad))
        #################
        # Update
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['lmbda']
            weight_decay = group['weight_decay']
            # compute lmbda_t
            if self._theoretical_lmbda:
                lmbda = self._number_steps +1     # lmbda_t = t
            ### Compute adaptive step size
            this_lb = self.lb if not lb else lb
            t1 = loss.item() - this_lb + _dot
            eta = max(t1, 0) / _norm
            eta = eta.item() # make scalar
            tau = min(lr, eta)
            ### Update params
            for p in group['params']:
                grad = p.grad.data.detach()
                state = self.state[p]
                z = state['z']
                if weight_decay > 0.0:
                    z.add_(p.data, alpha= (-lr*weight_decay))  # z = z - lr*wd*x
                # z Update
                z.add_(grad, alpha=-tau)
                # x Update
                p.data.mul_(lmbda/(1+lmbda)).add_(other=z, alpha=1/(1+lmbda))
        self._number_steps += 1
        ############################################################
        self.state['step_size_list'].append(tau/(1+lmbda))
        return loss