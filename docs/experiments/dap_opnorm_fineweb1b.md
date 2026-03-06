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

## Open questions and next steps

### 1. Adaptive damping via target $c$

The $c$ analysis motivates a direct adaptive scheme: choose $\delta$ per step to maintain
$\eta \cdot \|C_{\mathrm{eff}}^{-1/2}\|_{\mathrm{op}} \approx c^*$. Since
$\|C_{\mathrm{eff}}^{-1/2}\|_{\mathrm{op}} \approx 1/\sqrt{\delta \cdot \lambda_{\max}(C)}$,
this gives:

$$\delta = \frac{\eta^2}{(c^*)^2 \cdot \lambda_{\max}(C)}$$

All quantities ($\eta$, $\lambda_{\max}(C)$) are available at each step. One hyperparameter
($c^* \approx 0.06$) replaces the current two ($\eta$, $\delta$). This would make the method
automatically robust to LR choice — a key practical advantage. See
`docs/notes/dap_opnorm_diagnostics_and_adaptive_damping.md` for details.

### 2. Scale to larger models

All results are at GPT-tiny. The gain over Muon and the optimal $c$ may both change at
larger scale.

### 3. Confirm fillin runs (d=0.003 smaller LRs, d=0.05/0.1 at lr=0.02)

Pending.
