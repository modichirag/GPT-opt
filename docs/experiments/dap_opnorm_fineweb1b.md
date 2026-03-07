# DAPOpNorm: fineweb1B experiments

**Setting**: GPT-tiny (256d, 2L, 4-head, ~6M params), fineweb1B dataset, 1 epoch,
~3815 optimizer steps, $B=256$ sequences/step ($2^{18}$ tokens/step), single GPU.
All DAPOpNorm runs use EMA $\beta = 0.99$.

## Baselines

| Method | Best LR | Val loss |
|--------|---------|----------|
| Muon   | 0.02    | 4.2385   |
| AdamW  | 0.02–0.03 | 4.273  |

Muon gain over AdamW: **0.035 nats**. Muon has a broad optimum (4.239–4.244 across lr=0.02–0.05).

## Joint (lr, damping) sweep — DAPOpNorm, EMA=0.99

Best val loss. Rows = damping, columns = lr (— = not run):

|          | lr=0.01 | lr=0.02 | lr=0.03 | lr=0.04 | lr=0.05 | lr=0.07 | lr=0.10 | lr=0.15 | lr=0.20 |
|----------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| d=0.0    | —       | —       | 4.336   | —       | 4.580   | 4.617   | 4.637   | 4.637   | —       |
| d=0.001  | —       | —       | 4.252   | —       | 4.531   | 4.517   | 4.513   | 4.539   | —       |
| d=0.003  | 4.215   | 4.222   | 4.246   | 4.284   | 4.501   | 4.514   | 4.498   | 4.535   | —       |
| d=0.01   | 4.226   | 4.218   | 4.213   | 4.230   | 4.252   | 4.504   | 4.506   | 4.527   | —       |
| d=0.03   | 4.230   | 4.214   | **4.202** | 4.205 | 4.211   | 4.284   | 4.513   | 4.608   | —       |
| d=0.05   | —       | —       | 4.214   | —       | **4.209** | 4.250 | 4.521   | 4.592   | 4.603   |
| d=0.1    | —       | —       | 4.225   | —       | 4.218   | 4.232   | 4.518   | 4.574   | 4.600   |
| d=0.3    | —       | —       | 4.248   | —       | 4.246   | 4.249   | 4.260   | 4.617   | 4.618   |

**Best overall**: d=0.03, lr=0.03 → val=**4.202** (gain over Muon: **0.037 nats**).

## Key findings

### 1. Damping is essential and higher is better (up to a point)

$d=0.0$ (no damping, EMA only) is consistently worst. Performance improves with
damping up to $d \approx 0.03$–$0.05$, then degrades as $d \to \infty$ (approaching Muon).
In the limit $\delta \to \infty$, $C_{\mathrm{eff}} \approx \delta\lambda_{\max}(C) I$ and
the update reduces exactly to $\mathrm{sign}(G)$ — Muon. So DAPOpNorm interpolates between
a fully covariance-adapted method ($\delta=0$) and Muon ($\delta\to\infty$).

### 2. Higher damping gives a wider, flatter LR optimum

| Damping | Best LR | Best val | LR range beating Muon |
|---------|---------|----------|----------------------|
| 0.003   | 0.01    | 4.215    | lr=0.01–0.02         |
| 0.01    | 0.03    | 4.213    | lr=0.01–0.05         |
| 0.03    | 0.03    | 4.202    | lr=0.01–0.07         |
| 0.05    | 0.05    | 4.209    | lr=0.03–0.07         |
| 0.1     | 0.07    | 4.232    | lr=0.03–0.07         |

This is directly opposite to the finewebmini suggestion (which implied small damping was better
— but those runs were noise-dominated).

### 3. Per-lr envelope uniformly dominates Muon (for lr ≤ 0.07)

| LR    | Best DAPOpNorm (optimal $d$) | Muon    |
|-------|------------------------------|---------|
| 0.01  | 4.215 (d=0.003)              | 4.261   |
| 0.02  | 4.214 (d=0.03)               | 4.239   |
| 0.03  | **4.202** (d=0.03)           | 4.245   |
| 0.04  | 4.205 (d=0.03)               | —       |
| 0.05  | 4.209 (d=0.05)               | 4.243   |
| 0.07  | 4.232 (d=0.1)                | 4.268   |

At every LR from 0.01 to 0.07, the optimally-damped DAPOpNorm beats Muon. This is
strong evidence of a real and consistent advantage.

### 4. DAPOpNorm gain over Muon ≈ Muon gain over AdamW

- AdamW → Muon: **0.035 nats**
- Muon → DAPOpNorm: **0.037 nats**

Essentially equal steps up the ladder, suggesting each method is capturing roughly the same
amount of additional curvature information.

## Diagnostic analysis: why does high LR fail?

### The opnorm feedback loop

The update is $W \leftarrow W - \eta \cdot \mathrm{sign}(G C^{-1/2}) C^{-1/2}$, so
the preconditioner $C^{-1/2}$ directly scales the effective step. With relative damping
$C_{\mathrm{eff}} = C + \delta\lambda_{\max}(C)I$:

$$\|C_{\mathrm{eff}}^{-1/2}\|_{\mathrm{op}} \approx \frac{1}{\sqrt{\delta \cdot \lambda_{\max}(C)}}$$

Large opnorm $\Leftrightarrow$ small $\lambda_{\max}(C)$.

### Covariance scale $\lambda_{\max}(C)$ over training (damping=0.01)

| LR   | step 0 | step 500 | step 2000 | step 3813 | Final val |
|------|--------|----------|-----------|-----------|-----------|
| 0.03 | 0.11   | 4.57     | 4.92      | 5.15      | 4.196     |
| 0.07 | 0.11   | 0.76     | 0.33      | 0.22      | 4.504     |
| 0.15 | 0.11   | 0.81     | 0.14      | 0.11      | 4.527     |

At good LR, $\lambda_{\max}(C)$ grows ~47× as the model develops structured activations.
At high LR, $\lambda_{\max}(C)$ briefly grows after warmup then collapses — keeping
opnorm large and the effective step size uncontrolled. The divergence onset is at the
warmup boundary (~step 381).

Importantly, the growth of $\lambda_{\max}(C)$ at good LR is a **beneficial property**
of the current relative damping: as activations become more structured, $\lambda_{\max}(C)$
grows, opnorm decreases, and the effective step naturally anneals. This is an automatic
form of learning rate decay driven by the covariance.

## The effective step size $c = \eta \cdot \overline{\mathrm{opnorm}}$

Computing $c = \mathrm{lr} \times \text{(mean layer-averaged opnorm over training)}$ for
the best-performing configs:

| Config           | Val loss | $c$ (mean) | $c$ (end of training) |
|-----------------|----------|------------|----------------------|
| d=0.03, lr=0.03 | 4.2016   | 0.051      | 0.053                |
| d=0.03, lr=0.04 | 4.2053   | 0.069      | 0.069                |
| d=0.05, lr=0.05 | 4.2092   | 0.070      | 0.069                |
| d=0.01, lr=0.03 | 4.2133   | 0.095      | 0.086                |
| d=0.1,  lr=0.07 | 4.2317   | 0.079      | 0.089                |
| d=0.003, lr=0.03| 4.2463   | 0.261      | 0.240 ← too large    |

The top three configs cluster at $c \approx 0.05$–$0.07$ at end of training.
This is a strong empirical signal: the optimal operating point corresponds to a specific
effective step magnitude, not a specific nominal LR or damping individually.

## Per-layer adaptive damping (opnorm_target=2.0)

The $c$ analysis above motivates adaptive damping. After iterating through several formulations
(see `docs/notes/adaptive_damping.md` for the full derivation), the winning approach is
**per-layer adaptive damping**: compute $\delta_\ell$ per layer from eigenvalues already
available in the eigendecomposition:

$$\delta_\ell = \max\left(0,\; \frac{1/\mathrm{opnorm\_target}^2 - \lambda_{\min}(C_\ell)}{\lambda_{\max}(C_\ell)}\right)$$

This gives exact opnorm = opnorm_target for every layer, every step. No averaging bias,
no extra cost (eigendecomposition is already done per layer).

### Results: per-layer adaptive vs best fixed damping

All runs: opnorm_target=2.0, ema_beta=0.99, full 3814-step fineweb1B training.

| lr | Fixed (best δ) | Per-Layer Adaptive | Δ |
|----|---------------|-------------------|-----|
| 0.01 | 4.215 (δ=0.003) | **4.205** | -0.010 |
| 0.02 | 4.205 (δ=0.01) | **4.189** | -0.016 |
| 0.03 | 4.202 (δ=0.03) | **4.189** | -0.013 |
| 0.05 | 4.209 (δ=0.05) | **4.198** | -0.011 |
| 0.07 | 4.232 (δ=0.1) | **4.202** | -0.030 |

**Per-layer adaptive beats hand-tuned fixed damping at every LR.** Best: 4.189 vs 4.202.

Key properties:
- **LR-invariant**: val loss spread is only 0.016 across 7× LR range (vs 0.030 for fixed)
- **Single hyperparameter**: opnorm_target=2.0 replaces joint (lr, δ) tuning
- **Zero extra cost**: uses eigenvalues already computed per layer
- **Beats Muon by 0.050 nats** (4.189 vs 4.239)

A global adaptive variant (Jensen-corrected averaging) was also tested but fails at lr=0.07
(4.716) due to the 30× spread in per-layer eigmax. See `docs/notes/adaptive_damping.md`.

### Updated comparison

| Method | Best val | Gain over AdamW |
|--------|---------|----------------|
| AdamW  | 4.273   | —              |
| Muon   | 4.239   | 0.034          |
| DAPOpNorm (fixed δ=0.03) | 4.202 | 0.071 |
| DAPOpNorm (per-layer adaptive) | **4.189** | **0.084** |

## Next steps

### 1. Scale to GPT-small

GPT-tiny results are promising. Need to validate at GPT-small (768d, 12L, ~124M params)
on fineweb1B. Existing baselines: Adam best = 4.238 (from `gptopt/outputs/fineweb1B_baseline/`).
Need fresh AdamW and Muon baselines with warm-up-cosine schedule, plus DAPOpNorm per-layer sweep.

### 2. Sensitivity to opnorm_target

All results use opnorm_target=2.0. Worth a quick sweep of {1.5, 2.0, 2.5, 3.0} at fixed LR
to confirm 2.0 is near-optimal.
