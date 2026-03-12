import sys
import os

# The reference DistributedShampoo requires Python 3.10+ (match/case syntax).
# Add the optimizers directory to sys.path for import.
sys.path.insert(0, os.path.expanduser("~/optimizers"))

import torch
from torch.optim.optimizer import Optimizer
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    RootInvShampooPreconditionerConfig,
    SingleDeviceDistributedConfig,
    WeightDecayType,
)


class DistShampooWrapper(Optimizer):
    """Thin wrapper around Meta's DistributedShampoo that mirrors ShampooClean's param split.

    2D non-embedding params -> DistributedShampoo, everything else -> AdamW.
    Ungrafted, no block partitioning, eigendecomposition-based root inverse.

    Inherits from Optimizer so PyTorch LR schedulers work with it.
    """

    def __init__(
        self,
        named_params,
        lr=0.016,
        wd=0.1,
        momentum=0.95,
        beta2=0.8,
        epsilon=1e-15,
        use_bias_correction=True,
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

        print(f"\n=== DistShampooWrapper Parameter Classification ===")
        print(f"Shampoo Parameters ({len(shampoo_params)}):")
        for name in shampoo_names:
            print(f"  - {name}")
        print(f"\nAdamW Parameters ({len(adamw_params)}):")
        for name in adamw_names:
            print(f"  - {name}")
        print(f"=====================================================\n")

        self.shampoo_opt = DistributedShampoo(
            shampoo_params,
            lr=lr,
            betas=(momentum, beta2),
            epsilon=epsilon,
            weight_decay=wd,
            weight_decay_type=WeightDecayType.DECOUPLED,
            max_preconditioner_dim=float("inf"),
            use_bias_correction=use_bias_correction,
            grafting_config=None,
            preconditioner_config=RootInvShampooPreconditionerConfig(
                inverse_exponent_override={2: 0.5},
            ),
            distributed_config=SingleDeviceDistributedConfig(target_parameter_dimensionality=2),
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

        # Initialize Optimizer base attributes directly so isinstance checks pass
        # and LR schedulers can find param_groups. We skip super().__init__() to
        # avoid interfering with the inner optimizers' state.
        self.defaults = {"lr": lr}
        self.state = {}
        self._param_groups = self.shampoo_opt.param_groups + (self.adamw_opt.param_groups if self.adamw_opt else [])

    @property
    def param_groups(self):
        return self.shampoo_opt.param_groups + (self.adamw_opt.param_groups if self.adamw_opt else [])

    @param_groups.setter
    def param_groups(self, value):
        self._param_groups = value

    def step(self, closure=None):
        self.shampoo_opt.step(closure)
        if self.adamw_opt:
            self.adamw_opt.step(closure)

    def zero_grad(self, set_to_none=True):
        self.shampoo_opt.zero_grad(set_to_none)
        if self.adamw_opt:
            self.adamw_opt.zero_grad(set_to_none)
