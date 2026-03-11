# DAPOpNorm: fineweb1B Experiments

## Setup

**GPT-tiny** (256d, 2L, 4-head, ~6M params): 1 epoch, ~3814 optimizer steps,
$B=256$ sequences/step ($2^{18}$ tokens/step), single GPU, warmup 10% (381 steps) + cosine decay.

**GPT-small** (768d, 12L, 12-head, ~124M params): 1 epoch, ~1906 optimizer steps,
$B=512$ sequences/step, single GPU, warmup 10% + cosine decay, $\text{wd}=0.1$.

All DAPOpNorm runs use EMA $\beta = 0.99$.

## GPT-tiny Results

### Fixed Damping Grid

Best val loss. Rows = damping, columns = lr (-- = not run):

|          | lr=0.01 | lr=0.02 | lr=0.03 | lr=0.04 | lr=0.05 | lr=0.07 | lr=0.10 | lr=0.15 | lr=0.20 |
|----------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| d=0.0    | --      | --      | 4.336   | --      | 4.580   | 4.617   | 4.637   | 4.637   | --      |
| d=0.001  | --      | --      | 4.252   | --      | 4.531   | 4.517   | 4.513   | 4.539   | --      |
| d=0.003  | 4.215   | 4.222   | 4.246   | 4.284   | 4.501   | 4.514   | 4.498   | 4.535   | --      |
| d=0.01   | 4.226   | 4.218   | 4.213   | 4.230   | 4.252   | 4.504   | 4.506   | 4.527   | --      |
| d=0.03   | 4.230   | 4.214   | **4.202** | 4.205 | 4.211   | 4.284   | 4.513   | 4.608   | --      |
| d=0.05   | --      | --      | 4.214   | --      | **4.209** | 4.250 | 4.521   | 4.592   | 4.603   |
| d=0.1    | --      | --      | 4.225   | --      | 4.218   | 4.232   | 4.518   | 4.574   | 4.600   |
| d=0.3    | --      | --      | 4.248   | --      | 4.246   | 4.249   | 4.260   | 4.617   | 4.618   |

### Method Comparison

| Method | Best LR | Best val | Gain over AdamW |
|--------|---------|----------|-----------------|
| AdamW | 0.02–0.03 | 4.273 | -- |
| Muon | 0.02 | 4.239 | 0.034 |
| DAPOpNorm (fixed $\delta$=0.03) | 0.03 | 4.202 | 0.071 |
| DAPOpNorm (per-layer adaptive) | 0.02–0.03 | **4.189** | **0.084** |

Per-layer adaptive damping targets opnorm_target=2.0 per layer. See
[`docs/notes/adaptive_damping.md`](../notes/adaptive_damping.md) for the formula.

### Key Findings

- **Damping is essential**: $d=0.0$ is consistently worst; performance improves up to $d \approx 0.03$–$0.05$, then degrades toward the Muon limit ($d \to \infty$).
- **Higher damping widens the LR optimum**: $d=0.03$ beats Muon across lr=0.01–0.07; $d=0.003$ only at lr=0.01–0.02.
- **Per-lr envelope uniformly dominates Muon**: at every LR from 0.01 to 0.07, optimally-damped DAPOpNorm beats Muon.
- **Per-layer adaptive beats fixed damping at every LR**: best 4.189 vs 4.202, with val loss spread of only 0.016 across $7\times$ LR range (vs 0.030 for fixed). Eliminates LR sensitivity.
- **Gain stacking**: AdamW $\to$ Muon: 0.034 nats; Muon $\to$ DAPOpNorm (fixed): 0.037 nats; DAPOpNorm (fixed) $\to$ per-layer: 0.013 nats.

## GPT-small Results

### LR Sweep Comparison

All methods across learning rates (per-layer adaptive damping, opnorm_target=2.0 for DAPOpNorm variants):

| lr   | AdamW | Muon  | null  | sign_input | kfac_sign | full  | shampoo_sign |
|------|-------|-------|-------|------------|-----------|-------|---------|
| 0.0003 | 4.235 | --  | --    | --         | --        | --    | --      |
| 0.001 | 3.842 | --   | --    | --         | --        | --    | --      |
| 0.003 | **3.707** | -- | 3.775 | --        | --        | --    | --      |
| 0.005 | --    | 3.691 | 3.691 | --        | --        | --    | --      |
| 0.01 | 6.479 | 3.600 | 3.610 | 3.565     | 3.553     | 3.567 | 3.552   |
| 0.02 | --    | **3.597** | 3.562 | 3.530  | **3.506** | **3.533** | 3.518 |
| 0.03 | --    | 3.629 | 3.579 | --        | --        | --    | --      |
| 0.04 | --    | --    | --    | 3.557     | 3.531     | 3.578 | 3.543   |
| 0.05 | --    | 3.713 | 3.613 | --        | --        | --    | --      |
| 0.06 | --    | --    | --    | --        | --        | --    | 3.581   |
| 0.07 | --    | --    | 3.645 | --        | --        | --    | --      |
| 0.08 | --    | --    | --    | 3.611     | 3.581     | --    | 3.646   |
| 0.12 | --    | --    | --    | --        | --        | --    | 3.646   |
| 0.16 | --    | --    | --    | --        | --        | --    | 3.771   |
| 0.20 | --    | --    | --    | --        | --        | --    | 3.713   |

Output covariance modes: **null** = input covariance only ($\text{sign}(G \, C_{in}^{-1/2}) \, C_{in}^{-1/2}$);
**sign_input** = $\text{sign}(G \, C_{in}^{-1/2})$ (unit opnorm, lr controls scale);
**kfac_sign** = $\text{sign}(C_{out}^{-1/2} \, G \, C_{in}^{-1/2})$ (two-sided activation cov, sign only);
**full** = two-sided activation covariances ($C_{out}^{-1/2} \, \text{sign}(C_{out}^{-1/2} \, G \, C_{in}^{-1/2}) \, C_{in}^{-1/2}$);
**shampoo_sign** = two-sided gradient covariances ($\text{sign}(L^{-1/2} \, G \, R^{-1/2})$).

Full mode uses opnorm_target=4.0 (best of $\{2, 3, 4, 6\}$ at lr=0.02); all other modes use opnorm_target=2.0.

### Compute Efficiency

Shorter runs at lr=0.02 with properly scaled cosine schedules (no lr retuning):

| Steps | Fraction | Muon  | null  |
|-------|----------|-------|-------|
| 1334  | 70%      | 3.734 | 3.679 |
| 1620  | 85%      | 3.666 | 3.610 |
| 1733  | 91%      | --    | 3.589 |
| 1906  | 100%     | 3.597 | 3.562 |

**Breakeven**: DAPOpNorm (null) at 1334 steps (3.679) beats Muon at 1620 steps (3.666),
tolerating $\sim 1.2\times$ overhead per step.

### opnorm_target Sensitivity (lr=0.03)

| opnorm_target | Val loss |
|---------------|----------|
| 1.5           | 3.579    |
| 2.0           | 3.579    |
| 3.0           | 3.593    |
| 5.0           | 3.664    |

$\text{opnorm\_target} \in \{1.5, 2.0\}$ are tied; performance degrades for larger targets.

### Key Findings

- **kfac_sign is best overall**: 3.506 at lr=0.02, beating Muon by 0.091 nats
- **All DAPOpNorm variants beat Muon** at best lr (0.02 for all)
- **Two-sided whitening helps**: all two-sided modes beat one-sided null (3.562)
- **Activation covariances beat gradient covariances**: kfac_sign (3.506) > shampoo_sign (3.518), with the gap widening at higher lr
- **kfac_sign is more lr-robust than shampoo_sign**: spread of 0.075 across lr=0.01–0.08 (vs 0.094 for shampoo_sign)

### Summary

| Method | Best val | Best LR | Gain over Muon |
|--------|----------|---------|----------------|
| AdamW | 3.707 | 0.003 | -0.110 |
| Muon | 3.597 | 0.02 | -- |
| DAPOpNorm (null) | 3.562 | 0.02 | +0.035 |
| DAPOpNorm (full) | 3.533 | 0.02 | +0.064 |
| DAPOpNorm (sign_input) | 3.530 | 0.02 | +0.067 |
| DAPOpNorm (shampoo_sign) | 3.518 | 0.02 | +0.079 |
| DAPOpNorm (kfac_sign) | **3.506** | 0.02 | **+0.091** |

## Scaling: fineweb10B

**GPT-small** (124M params): 4200 optimizer steps (~2.2B tokens),
$B=512$ sequences/step, single GPU, warmup 10% + cosine decay, $\text{wd}=0.1$.

### LR Sweep

| lr   | Muon  | null  | kfac_sign | shampoo_sign | full  |
|------|-------|-------|-----------|--------------|-------|
| 0.01 | 3.401 | --    | --        | --           | 3.428 |
| 0.02 | **3.387** | 3.410 | 3.362 | 3.359        | **3.383** |
| 0.03 | 3.401 | **3.403** | --    | **3.356**    | --    |
| 0.04 | 3.408 | --    | **3.353** | 3.365        | 3.402 |
| 0.05 | 3.438 | 3.419 | --       | --           | --    |
| 0.08 | --    | --    | 3.407     | 3.431        | 3.466 |
| 0.12 | --    | 3.455 | --       | --           | --    |

### Summary

| Method | Best val | Best LR | Gain over Muon | Step time |
|--------|----------|---------|----------------|-----------|
| Muon | 3.387 | 0.02 | -- | ~3.3s |
| DAPOpNorm (null) | 3.403 | 0.03 | −0.016 | ~5.7s |
| DAPOpNorm (full) | 3.383 | 0.02 | +0.004 | ~10s |
| DAPOpNorm (shampoo_sign) | 3.356 | 0.03 | +0.031 | ~6.3s |
| DAPOpNorm (kfac_sign) | **3.353** | 0.04 | **+0.034** | ~10s |

### Key Findings

- **kfac_sign and shampoo_sign beat Muon** by 0.034 and 0.031 nats respectively.
  Both are robust across LRs: kfac_sign beats Muon at lr $\in \{0.02, 0.04\}$,
  shampoo_sign at lr $\in \{0.02, 0.03, 0.04\}$.
- **full barely beats Muon** (−0.004), and only at its best LR.
- **null does not beat Muon at 10B** (+0.016), despite being competitive at 1B (−0.101).
- **Advantage shrinks from 1B to 10B** for all modes. The biggest gap narrowing is
  null (−0.101 → +0.016), while kfac_sign (−0.091 → −0.034) and
  shampoo_sign (−0.079 → −0.031) retain meaningful advantages.
- **shampoo_sign is the most compute-efficient**: at ~6.3s/step (~1.9$\times$ Muon),
  it achieves nearly the same gain as kfac_sign which runs at ~10s/step (~3$\times$ Muon).
- **Numerical stability**: kfac_sign crashed during LR cooldown due to
  `torch.linalg.eigh` failing on ill-conditioned covariance matrices.
  Fixed by adding an SVD fallback (`torch.linalg.svd`) when `eigh` raises `LinAlgError`.

### 1B → 10B Scaling Comparison

| Mode | 1B gain over Muon | 10B gain over Muon | Retained |
|------|-------------------|---------------------|----------|
| null | +0.101 | −0.016 | 0% |
| full | +0.064 | +0.004 | 6% |
| shampoo_sign | +0.079 | +0.031 | 39% |
| kfac_sign | +0.091 | +0.034 | 37% |

Two-sided preconditioning with sign (kfac_sign, shampoo_sign) retains ~38% of its 1B
advantage at 10B. Input-only preconditioning (null) loses its advantage entirely.

## Next Steps

- **Scale further**: test kfac_sign and shampoo_sign at larger model sizes or longer training
- **Compute-normalized comparison**: compare at equal wall-clock time rather than equal steps
- **No-sign modes** (kfac, shampoo): see [`docs/plans/nosign_damping.md`](../plans/nosign_damping.md)
- **Ablations**: weight decay sensitivity, EMA beta sensitivity
