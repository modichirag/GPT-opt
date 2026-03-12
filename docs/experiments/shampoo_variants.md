# Shampoo Variants: fineweb1B Experiments

## Setup

**GPT-tiny** (256d, 2L, 4-head, ~6M params): 1 epoch, ~3814 optimizer steps,
$B=256$ sequences/step ($2^{18}$ tokens/step), single GPU, warmup 10% (381 steps) + cosine decay.

All Shampoo variants use `ShampooClean` (no grafting, no block partitioning, eigendecomposition-based inverse root). Non-2D params (embeddings, biases, layernorms) fall back to AdamW($\beta_1=0.95, \beta_2=0.95$).

## Baselines

| Method | Best val loss | lr | Notes |
|--------|-------------|------|-------|
| AdamW  | 4.273       | 0.003 | $\beta=(0.9, 0.95)$ |
| Muon   | 4.239       | 0.02  | NS steps=5 |

## Standard Shampoo ($p=0.5$)

$L^{-1/2} G R^{-1/2}$, no grafting. Update: $L_t = \beta_2 L_{t-1} + (1-\beta_2) G G^\top$.

| lr | $\beta_2$ | val loss | Notes |
|----|-----------|----------|-------|
| 0.016 | 0.8 | ~10.97 | Diverges without grafting |

Standard ungrafted Shampoo fails on GPT-tiny — loss stuck near initialization.

## EShampoo / SOAP ($p=0.5$)

Adam in the eigenbasis of Shampoo's preconditioner. Per-element second moments replace Kronecker eigenvalues.

| lr | $\beta_2$ | val loss | Notes |
|----|-----------|----------|-------|
| 0.004 | 0.8 | **4.217** | Best overall |
| 0.008 | 0.8 | 4.228 | |
| 0.016 | 0.8 | 4.295 | |

EShampoo works without grafting but sidesteps the damping question — it's Adam in the eigenbasis, not true Shampoo preconditioning.

## KL-Shampoo ($p=0.25$)

Two-sided covariance conditioned on the other factor's inverse root (Lin et al., 2025):
$$L_t = \beta_2 L_{t-1} + (1-\beta_2) (G R_t^{-p})(G R_t^{-p})^\top$$

Default $p=0.25$, $\beta_2=0.99$.

| lr | $\beta_2$ | val loss | Notes |
|----|-----------|----------|-------|
| 0.001 | 0.99 | | |
| 0.003 | 0.99 | | |
| 0.005 | 0.99 | | |
| 0.01  | 0.99 | | |
| 0.02  | 0.99 | | |
| 0.04  | 0.99 | | |

## Purifying Shampoo ($p=0.25$ + trace scaling)

Standard Shampoo with $p=0.25$ and update scaled by $1/(\text{Tr}(L) \cdot \text{Tr}(R))$ (Eschenhagen et al., 2025):
$$\Delta W = \frac{L^{-p} M R^{-p}}{\text{Tr}(L) \cdot \text{Tr}(R)}$$

Default $p=0.25$, $\beta_2=0.99$.

| lr | $\beta_2$ | val loss | Notes |
|----|-----------|----------|-------|
| 0.001 | 0.99 | | |
| 0.003 | 0.99 | | |
| 0.005 | 0.99 | | |
| 0.01  | 0.99 | | |
| 0.02  | 0.99 | | |
| 0.04  | 0.99 | | |

## Implementation Notes

- **Parity**: `ShampooClean` matches Meta's `DistributedShampoo` reference exactly (covariance diff = 0.0) for standard, EShampoo, and KL modes. See `tests/test_shampoo_comparison.py`.
- **KL feedback sensitivity**: KL-Shampoo inverse roots feed back into the covariance update, amplifying any floating-point difference exponentially. Exact parity required matching float32 bias correction and bf16 outer product dtype.
- **Trace scaling**: No reference implementation to compare against — validated via sanity test (with/without produces different params).
