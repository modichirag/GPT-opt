# ShampooClean Exploding Grad Norms — Debug Log

## Problem

ShampooClean produced exploding gradient norms (~10⁸ by step 20) while Meta's
DistributedShampoo (via DistShampooWrapper) stayed stable (~80). In a 20-step
comparison at lr=0.004 on GPT-tiny/fineweb1B:

- Steps 1-2: identical loss and grad_norm
- Step 3: grad_norm diverges — ShampooClean=50.4 vs DistShampoo=1.2
- Step 20: ShampooClean grad_norm=338K, DistShampoo grad_norm=83

## Root Cause

**Two bugs in the covariance computation and eigenvalue perturbation.**

### Bug 1: Float32 outer products (primary cause)

ShampooClean computed the covariance outer products in float32:
```python
g_f32 = g.float()
L.addmm_(g_f32, g_f32.T, alpha=1 - beta2)
```

DistributedShampoo computes them in the **original gradient dtype** (bfloat16):
```python
torch.tensordot(grad, grad, dims=...)  # grad is bfloat16
```

For a 64×32 weight matrix, $L = GG^\top$ is rank-32 (out of 64), so 32 eigenvalues
are mathematically zero. The numerical noise in these zero eigenvalues differs
dramatically:

| Computation | Min eigenvalue of $L_{bc}$ | $\lambda_{min}^{-1/2}$ |
|-------------|---------------------------|------------------------|
| float32     | $\sim 10^{-6}$            | $\sim 1000$            |
| bfloat16    | $\sim 0.03$               | $\sim 5.7$             |

With epsilon $= 10^{-15}$ and the original perturbation (`clamp(min=0) + ε`), the
float32 case gives $L^{-1/2}$ norm of $\sim 3.16 \times 10^7$, creating enormous
preconditioned updates that destabilize training. The bfloat16 rounding noise acts
as natural regularization, keeping the preconditioner well-conditioned.

### Bug 2: Eigenvalue perturbation strategy (secondary)

ShampooClean used:
```python
eigvals = eigvals.clamp(min=0.0) + epsilon
```

DistributedShampoo (PerturbationConfig) uses:
```python
lambda_min = eigvals.min()
if lambda_min < epsilon:
    eigvals = eigvals - lambda_min + epsilon  # shift ALL eigenvalues
```

The `clamp(min=0)` approach zeros out negative eigenvalues but leaves a gap: the
minimum eigenvalue jumps from some negative value directly to epsilon ($10^{-15}$).
The shift approach preserves the relative spacing of eigenvalues near zero, giving
a more graceful transition.

With float32 outer products: `clamp` gives min eigenvalue $= 10^{-15}$ → inv root
$= 3.16 \times 10^7$. Even with the shift, the float32 case gives min eigenvalue
$\sim 10^{-6}$ → inv root $\sim 1000$ (still large but much better).

With bfloat16 outer products + shift: min eigenvalue $\sim 0.03$ → inv root $\sim
5.7$ (perfectly well-conditioned).

## Hypothesis Testing

The original plan hypothesized three causes, ranked by likelihood:

- **H1 (momentum buffer dtype)**: Initially seemed most likely, but turned out to
  be a non-issue. DistributedShampoo stores filtered_grad in bfloat16 (param dtype),
  same as ShampooClean's original implementation. The momentum dtype difference
  contributes ~0.04% relative error, not the 10⁸× explosion.

- **H2 (weight decay timing)**: Confirmed mathematically equivalent between the two
  implementations. Contributes ~0.016 max element difference per step. Negligible.

- **H3 (eigenvalue perturbation)**: Partially correct — the perturbation strategy
  differs and matters, but only in combination with the outer product precision
  (Bug 1). With bfloat16 outer products, both perturbation strategies give
  reasonable results.

**The actual root cause (covariance precision) was not in the original hypothesis
list.** It was discovered by comparing the eigenvalue spectra of the covariance
matrices between the two implementations.

## Fixes Applied

### 1. Covariance outer products in bfloat16 (shampoo_clean.py)

```python
# Before: float32 outer products (too precise for rank-deficient matrices)
g_f32 = g.float()
L.addmm_(g_f32, g_f32.T, alpha=1 - beta2)

# After: bfloat16 outer products, cast to float32 for accumulation
L.mul_(beta2).add_((g @ g.T).float(), alpha=1 - beta2)
```

### 2. Eigenvalue perturbation matching reference (shampoo_clean.py)

```python
# Before: clamp + epsilon
eigvals = eigvals.clamp(min=0.0) + epsilon

# After: shift all eigenvalues by |lambda_min|, then add epsilon
C_reg = C + epsilon * I  # pre-eigh perturbation
eigvals, eigvecs = eigh(C_reg)
if eigvals.min() < epsilon:
    eigvals = eigvals - eigvals.min() + epsilon
```

### 3. DistShampooWrapper handles empty param groups (dist_shampoo_wrapper.py)

AdamW optimizer creation now handles the case where all params are Shampoo params
(no AdamW params), which was causing a crash in unit tests.

## Verification

After fixes, `tests/test_shampoo_comparison.py` confirms:

| Test | Result |
|------|--------|
| Covariance matrices (L, R) | **Exact match** (0.00 diff) |
| Preconditioners ($L^{-1/2}$, $R^{-1/2}$) | **Exact match** (0.00 diff) |
| Square params (32×32) | 0.1% relative diff |
| Rectangular params (64×32) | 10% relative diff at step 1, constant thereafter |

The 10% step-1 diff for rectangular params is inherent: bfloat16 momentum
quantization noise ($\sim 4 \times 10^{-4}$) is amplified by $L^{-1/2}$ (norm
$3.16 \times 10^7$ when L is rank-deficient). By step 2, $L^{-1/2}$ norm drops to
12.3, and from step 3 onward it's $< 2.5$.

## Key Insight

**Numerical precision cuts both ways.** Computing the covariance in float32 is more
accurate, but for rank-deficient matrices, the precision in the zero-eigenvalue
subspace is *too* accurate — the eigenvalues are $\sim 10^{-6}$ instead of the
$\sim 0.03$ you get with bfloat16, and with epsilon $= 10^{-15}$, the inverse root
of $10^{-6}$ is $1000 \times$ larger than the inverse root of $0.03$. The bfloat16
rounding noise provides implicit Tikhonov regularization that stabilizes the
preconditioner during early training when the covariance is rank-deficient.
