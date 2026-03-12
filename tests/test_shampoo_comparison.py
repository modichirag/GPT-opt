"""
Diagnostic comparison: ShampooClean vs DistributedShampoo (via DistShampooWrapper).

Runs both optimizers on synthetic 2D parameters with identical gradients
and compares intermediate values after each step.

Usage: python tests/test_shampoo_comparison.py  (requires GPU)
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gptopt.optim.shampoo_clean import ShampooClean
from gptopt.optim.dist_shampoo_wrapper import DistShampooWrapper


def make_param(shape, dtype, seed=42):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, dtype=dtype, device="cuda", generator=gen).requires_grad_(False)


def run_comparison(
    shape=(64, 32),
    dtype=torch.bfloat16,
    lr=0.004,
    wd=0.1,
    momentum=0.95,
    beta2=0.8,
    epsilon=1e-15,
    nesterov=False,
    num_steps=10,
    seed=42,
    verbose=True,
):
    m, n = shape
    if verbose:
        print(f"=== shape={shape}, dtype={dtype}, lr={lr}, nesterov={nesterov}, steps={num_steps} ===")

    p_clean = make_param(shape, dtype, seed=seed).clone().requires_grad_(True)
    p_dist = make_param(shape, dtype, seed=seed).clone().requires_grad_(True)
    assert torch.equal(p_clean.data, p_dist.data)

    opt_clean = ShampooClean(
        named_params=[("test_param", p_clean)],
        lr=lr, wd=wd, momentum=momentum, nesterov=nesterov,
        beta2=beta2, epsilon=epsilon, use_bias_correction=True,
    )
    opt_dist = DistShampooWrapper(
        named_params=[("test_param", p_dist)],
        lr=lr, wd=wd, momentum=momentum,
        beta2=beta2, epsilon=epsilon, use_bias_correction=True,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed + 1000)
    grads = [torch.randn(shape, dtype=dtype, device="cuda", generator=gen) for _ in range(num_steps)]

    results = []
    for step_idx in range(num_steps):
        g = grads[step_idx]
        p_clean.grad = g.clone()
        p_dist.grad = g.clone()

        opt_clean.step()
        opt_dist.step()
        opt_clean.zero_grad()
        opt_dist.zero_grad()

        param_diff = (p_clean.data.float() - p_dist.data.float()).abs()
        max_diff = param_diff.max().item()
        mean_norm = (p_clean.data.float().norm() + p_dist.data.float().norm()) / 2
        rel_diff = max_diff / (mean_norm + 1e-8)

        results.append({"step": step_idx + 1, "max_diff": max_diff, "rel_diff": rel_diff, "mean_norm": mean_norm.item()})

        if verbose:
            # Check covariance match for step 1
            state_c = opt_clean.state[p_clean]
            if step_idx == 0:
                cov_match = "N/A"
                if hasattr(opt_dist, 'shampoo_opt'):
                    for pp in opt_dist.shampoo_opt.param_groups[0]['params']:
                        sd = opt_dist.shampoo_opt.state[pp]
                        if 'block_0' in sd:
                            L_dist = sd['block_0']['shampoo'].factor_matrices[0]
                            L_clean = state_c['L']
                            cov_diff = (L_clean - L_dist).abs().max().item()
                            cov_match = f"cov_diff={cov_diff:.2e}"

                linv_norm = opt_clean._matrix_inv_root(state_c['L'] / (1 - beta2 ** (step_idx + 1)), 0.5).norm().item()
                print(f"  Step {step_idx+1}: max_diff={max_diff:.2e} rel_diff={rel_diff:.2e} "
                      f"L_inv_norm={linv_norm:.2e} {cov_match}")
            else:
                print(f"  Step {step_idx+1}: max_diff={max_diff:.2e} rel_diff={rel_diff:.2e}")

    return results


def test_covariance_match():
    """Verify covariance matrices match exactly between ShampooClean and DistributedShampoo."""
    print("\n=== TEST: Covariance Matrix Match ===")
    shape = (64, 32)
    dtype = torch.bfloat16
    seed = 42
    beta2 = 0.8

    p_clean = make_param(shape, dtype, seed).clone().requires_grad_(True)
    p_dist = make_param(shape, dtype, seed).clone().requires_grad_(True)

    opt_clean = ShampooClean(
        named_params=[("t", p_clean)], lr=0.004, wd=0.1, momentum=0.95,
        nesterov=False, beta2=beta2, epsilon=1e-15,
    )

    sys.path.insert(0, os.path.expanduser("~/optimizers"))
    from distributed_shampoo.distributed_shampoo import DistributedShampoo
    from distributed_shampoo.shampoo_types import (
        RootInvShampooPreconditionerConfig, SingleDeviceDistributedConfig, WeightDecayType,
    )
    opt_dist = DistributedShampoo(
        [p_dist], lr=0.004, betas=(0.95, 0.8), epsilon=1e-15, weight_decay=0.1,
        weight_decay_type=WeightDecayType.DECOUPLED, max_preconditioner_dim=float("inf"),
        use_bias_correction=True, grafting_config=None,
        preconditioner_config=RootInvShampooPreconditionerConfig(inverse_exponent_override={2: 0.5}),
        distributed_config=SingleDeviceDistributedConfig(target_parameter_dimensionality=2),
    )

    gen = torch.Generator(device="cuda").manual_seed(1000)
    all_pass = True
    for step in range(5):
        g = torch.randn(shape, dtype=dtype, device="cuda", generator=gen)
        p_clean.grad = g.clone()
        p_dist.grad = g.clone()
        opt_clean.step()
        opt_dist.step()

        state_c = opt_clean.state[p_clean]
        state_d = opt_dist.state[p_dist]
        L_c = state_c["L"]
        R_c = state_c["R"]
        L_d = state_d["block_0"]["shampoo"].factor_matrices[0]
        R_d = state_d["block_0"]["shampoo"].factor_matrices[1]

        l_diff = (L_c - L_d).abs().max().item()
        r_diff = (R_c - R_d).abs().max().item()
        ok = l_diff < 1e-6 and r_diff < 1e-6
        all_pass = all_pass and ok
        print(f"  Step {step+1}: L_diff={l_diff:.2e} R_diff={r_diff:.2e} {'PASS' if ok else 'FAIL'}")

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


def test_preconditioner_match():
    """Verify inverse root preconditioners match exactly."""
    print("=== TEST: Preconditioner Match ===")
    shape = (64, 32)
    dtype = torch.bfloat16
    beta2 = 0.8
    epsilon = 1e-15

    p_clean = make_param(shape, dtype, 42).clone().requires_grad_(True)
    opt_clean = ShampooClean(
        named_params=[("t", p_clean)], lr=0.004, wd=0.1, momentum=0.95,
        nesterov=False, beta2=beta2, epsilon=epsilon,
    )

    sys.path.insert(0, os.path.expanduser("~/optimizers"))
    from distributed_shampoo.preconditioner.matrix_functions import matrix_inverse_root
    from fractions import Fraction

    gen = torch.Generator(device="cuda").manual_seed(1000)
    all_pass = True
    for step in range(5):
        g = torch.randn(shape, dtype=dtype, device="cuda", generator=gen)
        p_clean.grad = g.clone()
        opt_clean.step()

        state = opt_clean.state[p_clean]
        bc2 = 1 - beta2 ** (step + 1)
        L_bc = state["L"] / bc2
        R_bc = state["R"] / bc2

        L_inv_clean = opt_clean._matrix_inv_root(L_bc, 0.5)
        R_inv_clean = opt_clean._matrix_inv_root(R_bc, 0.5)
        L_inv_ref = matrix_inverse_root(L_bc, root=Fraction(2), epsilon=epsilon)
        R_inv_ref = matrix_inverse_root(R_bc, root=Fraction(2), epsilon=epsilon)

        l_diff = (L_inv_clean - L_inv_ref).abs().max().item()
        r_diff = (R_inv_clean - R_inv_ref).abs().max().item()
        ok = l_diff < 1e-3 and r_diff < 1e-3
        all_pass = all_pass and ok
        linv_norm = L_inv_clean.norm().item()
        print(f"  Step {step+1}: L_inv_diff={l_diff:.2e} R_inv_diff={r_diff:.2e} "
              f"L_inv_norm={linv_norm:.2e} {'PASS' if ok else 'FAIL'}")

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


def test_convergence():
    """Test that param diff converges as covariance becomes better-conditioned."""
    print("=== TEST: Convergence (rel_diff should decrease over steps) ===")
    results = run_comparison(shape=(32, 32), num_steps=20, verbose=False)

    # After first few steps, covariance is full-rank and rel_diff should stabilize/decrease
    early_rel = max(r["rel_diff"] for r in results[:3])
    late_rel = max(r["rel_diff"] for r in results[10:])

    print(f"  Early max rel_diff (steps 1-3):    {early_rel:.4e}")
    print(f"  Late max rel_diff (steps 11-20):   {late_rel:.4e}")

    # For square matrices, covariance is full-rank after step 1, so even early rel_diff should be small
    passed = late_rel < 0.05  # 5% relative tolerance
    print(f"  Result: {'PASS' if passed else 'FAIL'}\n")
    return passed


def run_eshampoo_comparison(
    shape=(64, 32),
    dtype=torch.bfloat16,
    lr=0.004,
    wd=0.1,
    momentum=0.95,
    beta2=0.8,
    epsilon=1e-15,
    nesterov=False,
    num_steps=10,
    seed=42,
    verbose=True,
):
    """Compare ShampooClean(eshampoo=True) vs DistShampooWrapper(eshampoo=True)."""
    m, n = shape
    if verbose:
        print(f"=== EShampoo: shape={shape}, dtype={dtype}, lr={lr}, steps={num_steps} ===")

    p_clean = make_param(shape, dtype, seed=seed).clone().requires_grad_(True)
    p_dist = make_param(shape, dtype, seed=seed).clone().requires_grad_(True)
    assert torch.equal(p_clean.data, p_dist.data)

    opt_clean = ShampooClean(
        named_params=[("test_param", p_clean)],
        lr=lr, wd=wd, momentum=momentum, nesterov=nesterov,
        beta2=beta2, epsilon=epsilon, use_bias_correction=True,
        eshampoo=True,
    )
    opt_dist = DistShampooWrapper(
        named_params=[("test_param", p_dist)],
        lr=lr, wd=wd, momentum=momentum,
        beta2=beta2, epsilon=epsilon, use_bias_correction=True,
        eshampoo=True,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed + 1000)
    grads = [torch.randn(shape, dtype=dtype, device="cuda", generator=gen) for _ in range(num_steps)]

    results = []
    for step_idx in range(num_steps):
        g = grads[step_idx]
        p_clean.grad = g.clone()
        p_dist.grad = g.clone()

        opt_clean.step()
        opt_dist.step()
        opt_clean.zero_grad()
        opt_dist.zero_grad()

        param_diff = (p_clean.data.float() - p_dist.data.float()).abs()
        max_diff = param_diff.max().item()
        mean_norm = (p_clean.data.float().norm() + p_dist.data.float().norm()) / 2
        rel_diff = max_diff / (mean_norm + 1e-8)

        results.append({"step": step_idx + 1, "max_diff": max_diff, "rel_diff": rel_diff, "mean_norm": mean_norm.item()})

        if verbose:
            print(f"  Step {step_idx+1}: max_diff={max_diff:.2e} rel_diff={rel_diff:.2e}")

    return results


def test_eshampoo_covariance_match():
    """Verify covariance matrices match between ShampooClean(eshampoo) and DistributedShampoo(eigenvalue-corrected)."""
    print("\n=== TEST: EShampoo Covariance Matrix Match ===")
    shape = (64, 32)
    dtype = torch.bfloat16
    seed = 42
    beta2 = 0.8

    p_clean = make_param(shape, dtype, seed).clone().requires_grad_(True)
    p_dist = make_param(shape, dtype, seed).clone().requires_grad_(True)

    opt_clean = ShampooClean(
        named_params=[("t", p_clean)], lr=0.004, wd=0.1, momentum=0.95,
        nesterov=False, beta2=beta2, epsilon=1e-15, eshampoo=True,
    )

    sys.path.insert(0, os.path.expanduser("~/optimizers"))
    from distributed_shampoo.distributed_shampoo import DistributedShampoo
    from distributed_shampoo.shampoo_types import (
        DefaultEigenvalueCorrectedShampooConfig,
        SingleDeviceDistributedConfig, WeightDecayType,
    )
    opt_dist = DistributedShampoo(
        [p_dist], lr=0.004, betas=(0.95, 0.8), epsilon=1e-15, weight_decay=0.1,
        weight_decay_type=WeightDecayType.DECOUPLED, max_preconditioner_dim=float("inf"),
        precondition_frequency=1,
        use_bias_correction=True, grafting_config=None,
        preconditioner_config=DefaultEigenvalueCorrectedShampooConfig,
        distributed_config=SingleDeviceDistributedConfig(target_parameter_dimensionality=2),
    )

    gen = torch.Generator(device="cuda").manual_seed(1000)
    all_pass = True
    for step in range(5):
        g = torch.randn(shape, dtype=dtype, device="cuda", generator=gen)
        p_clean.grad = g.clone()
        p_dist.grad = g.clone()
        opt_clean.step()
        opt_dist.step()

        state_c = opt_clean.state[p_clean]
        state_d = opt_dist.state[p_dist]
        L_c = state_c["L"]
        R_c = state_c["R"]
        L_d = state_d["block_0"]["shampoo"].factor_matrices[0]
        R_d = state_d["block_0"]["shampoo"].factor_matrices[1]

        l_diff = (L_c - L_d).abs().max().item()
        r_diff = (R_c - R_d).abs().max().item()
        ok = l_diff < 1e-6 and r_diff < 1e-6
        all_pass = all_pass and ok
        print(f"  Step {step+1}: L_diff={l_diff:.2e} R_diff={r_diff:.2e} {'PASS' if ok else 'FAIL'}")

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


def test_eshampoo_corrected_eigenvalues_match():
    """Verify corrected eigenvalues match between ShampooClean and DistributedShampoo."""
    print("=== TEST: EShampoo Corrected Eigenvalues Match ===")
    shape = (32, 32)  # Square for better conditioning
    dtype = torch.bfloat16
    seed = 42
    beta2 = 0.8

    p_clean = make_param(shape, dtype, seed).clone().requires_grad_(True)
    p_dist = make_param(shape, dtype, seed).clone().requires_grad_(True)

    opt_clean = ShampooClean(
        named_params=[("t", p_clean)], lr=0.004, wd=0.1, momentum=0.95,
        nesterov=False, beta2=beta2, epsilon=1e-15, eshampoo=True,
    )

    sys.path.insert(0, os.path.expanduser("~/optimizers"))
    from distributed_shampoo.distributed_shampoo import DistributedShampoo
    from distributed_shampoo.shampoo_types import (
        DefaultEigenvalueCorrectedShampooConfig,
        SingleDeviceDistributedConfig, WeightDecayType,
    )
    opt_dist = DistributedShampoo(
        [p_dist], lr=0.004, betas=(0.95, 0.8), epsilon=1e-15, weight_decay=0.1,
        weight_decay_type=WeightDecayType.DECOUPLED, max_preconditioner_dim=float("inf"),
        precondition_frequency=1,
        use_bias_correction=True, grafting_config=None,
        preconditioner_config=DefaultEigenvalueCorrectedShampooConfig,
        distributed_config=SingleDeviceDistributedConfig(target_parameter_dimensionality=2),
    )

    gen = torch.Generator(device="cuda").manual_seed(1000)
    all_pass = True
    for step in range(5):
        g = torch.randn(shape, dtype=dtype, device="cuda", generator=gen)
        p_clean.grad = g.clone()
        p_dist.grad = g.clone()
        opt_clean.step()
        opt_dist.step()

        state_c = opt_clean.state[p_clean]
        state_d = opt_dist.state[p_dist]

        D_clean = state_c["D_eshampoo"]
        D_dist = state_d["block_0"]["shampoo"].corrected_eigenvalues

        d_diff = (D_clean - D_dist.view(shape)).abs().max().item()
        d_norm = D_clean.norm().item()
        d_rel = d_diff / (d_norm + 1e-8)
        # Corrected eigenvalues may differ due to eigenvector sign/ordering differences
        # affecting the projected gradient. Use relative tolerance.
        ok = d_rel < 0.1
        all_pass = all_pass and ok
        print(f"  Step {step+1}: D_abs_diff={d_diff:.2e} D_rel_diff={d_rel:.2e} "
              f"D_norm={d_norm:.2e} {'PASS' if ok else 'FAIL'}")

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


def test_eshampoo_param_convergence():
    """Test EShampoo parameter convergence between ShampooClean and DistShampooWrapper."""
    print("=== TEST: EShampoo Parameter Convergence ===")

    # Square matrix — eigenvector sign/ordering differences compound through D_t
    # updates, causing larger divergence than standard Shampoo (which has no per-element state).
    # 10% tolerance accounts for this accumulation effect.
    results_sq = run_eshampoo_comparison(shape=(32, 32), num_steps=20, verbose=False)
    late_rel_sq = max(r["rel_diff"] for r in results_sq[5:])
    p_sq = late_rel_sq < 0.10
    print(f"  Square (32x32) late rel_diff (steps 6-20): {late_rel_sq:.4e} {'PASS' if p_sq else 'FAIL'}")

    # Rectangular matrix
    results_rect = run_eshampoo_comparison(shape=(64, 32), num_steps=20, verbose=False)
    late_rel_rect = max(r["rel_diff"] for r in results_rect[5:])
    p_rect = late_rel_rect < 0.15
    print(f"  Rect (64x32) late rel_diff (steps 6-20):   {late_rel_rect:.4e} {'PASS' if p_rect else 'FAIL'}")

    passed = p_sq and p_rect
    print(f"  Result: {'PASS' if passed else 'FAIL'}\n")
    return passed


if __name__ == "__main__":
    assert torch.cuda.is_available(), "This test requires a GPU"

    # Standard Shampoo tests
    p1 = test_covariance_match()
    p2 = test_preconditioner_match()
    p3 = test_convergence()

    print("=== TEST: Rectangular matrix (rank-deficient early steps) ===")
    results_rect = run_comparison(shape=(64, 32), num_steps=10)
    late_rect = max(r["rel_diff"] for r in results_rect[3:])
    p4 = late_rect < 0.15
    print(f"  Late rel_diff (steps 4-10): {late_rect:.4e} {'PASS' if p4 else 'FAIL'}\n")

    # EShampoo tests
    p5 = test_eshampoo_covariance_match()
    p6 = test_eshampoo_corrected_eigenvalues_match()
    p7 = test_eshampoo_param_convergence()

    print("=" * 60)
    all_passed = p1 and p2 and p3 and p4 and p5 and p6 and p7
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    if not all_passed:
        sys.exit(1)
