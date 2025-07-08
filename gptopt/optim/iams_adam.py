import torch
import warnings
from math import sqrt
from typing import Union
class IAMSAdam(torch.optim.Optimizer):
    def __init__(self,
                 params: 0.0,
                 lr: float=1.0,
                 lmbda: Union[float,None]=9.0,
                 beta2:float=0.999, 
                 eps:float=1e-8,
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
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
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
                        beta2=beta2,
                        eps=eps,
                        weight_decay=weight_decay,
                        weight_sum=0.0
        )
        super(IAMSAdam, self).__init__(params, defaults)
        self.lb = lb
        # Initialization
        self._number_steps = 0
        self.state['step_size_list'] = list() # for storing the adaptive step size term
        return
    def step(self, closure =None, loss: torch.Tensor=None, teacher_loss: float=None):
        """
        Performs a single optimization step.
        Parameters
        ----------
        closure : LossClosure, optional
            A callable that evaluates the model (possibly with backprop) and returns the loss, by default None.
        loss : torch.tensor, optional
            The loss tensor. Use this when the backward step has already been performed. By default None.
        teacher_loss : float, optional
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
        self._number_steps += 1
        average_precon =0
        ############################################################
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            beta2 = group['beta2']
            for p in group['params']:
                grad = p.grad.data.detach()
                state = self.state[p]

                 # Adam State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of squared gradient values
                    state['grad_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Initialize Averaging Variables
                    state['z'] = p.detach().clone().to(p.device)
                self._number_steps +=1
                state['step'] += 1 
                grad_avg_sq =  state['grad_avg_sq']

                # Adam EMA updates
                grad_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2) # = v_k
                # grad_dot_w.mul_(beta1).add_(torch.sum(torch.mul(p.data, grad)), alpha=1-beta1)
                bias_correction2 = 1 - beta2 ** self._number_steps
                Dk = grad_avg_sq.div(bias_correction2).sqrt().add(eps) # = D_k
                z = state['z']
                _dot += torch.sum(torch.mul(grad, z-p.data))
                _norm += torch.sum(grad.mul(grad.div(Dk)))

        num_params = sum(len(group['params']) for group in self.param_groups)
        #################
        # Update
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['lmbda']
            weight_decay = group['weight_decay']
            beta2 = group['beta2']
            bias_correction2 = 1 - beta2 ** self._number_steps

            # compute lmbda_t
            if self._theoretical_lmbda:
                lmbda = self._number_steps +1     # lmbda_t = t
            ### Compute adaptive step size
            this_teacher_loss = self.lb if not teacher_loss else teacher_loss
            t1 = loss.item() - this_teacher_loss + _dot
            eta = max(t1, 0) / _norm
            eta = eta.item() # make scalar
            tau = min(lr, eta)
            ### Update params
            for p in group['params']:
                grad = p.grad.data.detach()
                state = self.state[p]
                grad_avg_sq =  state['grad_avg_sq']
                Dk = grad_avg_sq.div(bias_correction2).sqrt().add(eps)
                # average_precon += (torch.mean(1/Dk)/num_params).item()
                z = state['z']
                if weight_decay > 0.0:
                    z.add_(p.data, alpha= (-lr*weight_decay))  # z = z - lr*wd*x
                # z Update
                z.add_(grad.div(Dk), alpha=-tau)
                # x Update
                p.data.mul_(lmbda/(1+lmbda)).add_(other=z, alpha=1/(1+lmbda))
        ############################################################
        self.state['step_size_list'].append(tau/(1+lmbda))
        return loss