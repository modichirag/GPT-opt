import sys
import os

# Add KL-Methods to path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'KL-Methods'))

import torch
from torch.optim.optimizer import Optimizer
from optim.kl_opt import KLOpt


class KLShampooWrapper(Optimizer):
    """Wrapper around Lin et al.'s KLOpt that matches our named_params interface.

    2D non-embedding params -> KLOpt (KL-Shampoo), everything else -> AdamW.
    """

    def __init__(
        self,
        named_params,
        lr=0.003,
        wd=0.1,
        beta1=0.95,
        beta2=0.95,
        shampoo_beta=-1,
        eps=1e-8,
        precondition_frequency=10,
        using_klsoap=False,
        normalize_grads=False,
        init_factor=0.1,
        using_damping=False,
        using_clamping=True,
        max_clamp_value=4000,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):
        excluded = ["embeddings", "embed_tokens", "wte", "lm_head", "weight_proj", "wpe"]
        shampoo_params, shampoo_names = [], []
        adamw_params, adamw_names = [], []

        for name, p in named_params:
            if p.ndim >= 2 and not any(ex in name for ex in excluded):
                shampoo_params.append(p)
                shampoo_names.append(name)
            else:
                adamw_params.append(p)
                adamw_names.append(name)

        print(f"\n=== KLShampooWrapper Parameter Classification ===")
        print(f"KL-Shampoo Parameters ({len(shampoo_params)}):")
        for name in shampoo_names:
            print(f"  - {name}")
        print(f"\nAdamW Parameters ({len(adamw_params)}):")
        for name in adamw_names:
            print(f"  - {name}")
        print(f"==================================================\n")

        self.kl_opt = KLOpt(
            shampoo_params,
            lr=lr,
            betas=(beta1, beta2),
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=wd,
            precondition_frequency=precondition_frequency,
            using_klsoap=using_klsoap,
            normalize_grads=normalize_grads,
            init_factor=init_factor,
            using_damping=using_damping,
            using_clamping=using_clamping,
            max_clamp_value=max_clamp_value,
        )

        if adamw_params:
            self.adamw_opt = torch.optim.AdamW(
                adamw_params,
                lr=lr,
                weight_decay=wd,
                betas=adamw_betas,
                eps=adamw_eps,
                fused=True,
            )
        else:
            self.adamw_opt = None

        self.defaults = {"lr": lr}
        self.state = {}
        self._param_groups = self.kl_opt.param_groups + (self.adamw_opt.param_groups if self.adamw_opt else [])

    @property
    def param_groups(self):
        return self.kl_opt.param_groups + (self.adamw_opt.param_groups if self.adamw_opt else [])

    @param_groups.setter
    def param_groups(self, value):
        self._param_groups = value

    def step(self, closure=None):
        self.kl_opt.step(closure)
        if self.adamw_opt:
            self.adamw_opt.step(closure)

    def zero_grad(self, set_to_none=True):
        self.kl_opt.zero_grad(set_to_none)
        if self.adamw_opt:
            self.adamw_opt.zero_grad(set_to_none)
