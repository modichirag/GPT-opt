# No-Sign Damping Experiments (GPT-tiny, fineweb1B)

## Motivation

The current DAPOpNorm update uses the matrix sign function:

$$W \leftarrow W - \eta \cdot \mathrm{sign}(G C^{-1/2}) \cdot C^{-1/2}$$

The sign normalizes singular values to 1, so $\|\mathrm{update}\|_{\mathrm{op}} = \|C^{-1/2}\|_{\mathrm{op}}$ — independent of gradient magnitude. Damping controls update scale purely through the preconditioner.

**Without the sign** (kfac and shampoo modes), the update is the raw whitened gradient:

$$W \leftarrow W - \eta \cdot C_{\mathrm{out}}^{-1/2} \, G \, C_{\mathrm{in}}^{-1/2}$$

Now $\|\mathrm{update}\|_{\mathrm{op}} \approx \|C_{\mathrm{out}}^{-1/2}\| \cdot \|G\|_{\mathrm{op}} \cdot \|C_{\mathrm{in}}^{-1/2}\|$, so gradient magnitude directly influences step size. This changes the damping problem fundamentally: the right damping depends on both the covariance spectrum *and* the gradient scale.

### Why explore no-sign modes?

1. **Theoretical**: Sign discards curvature magnitude information. KFAC-style updates preserve the natural gradient structure $F^{-1} g$, which has convergence guarantees that sign-based updates lack.
2. **Practical**: Sign requires Newton-Schulz iterations (5 steps per layer per optimizer step). Removing sign saves ~30% of optimizer compute.
3. **Empirical**: Eschenhagen et al. 2025 ("Clarifying Shampoo") show that ungrafted Shampoo$^{1/2}$ (no sign, no Adam grafting) matches grafted Shampoo when damping is tuned correctly.

## Experimental Design

### Two covariance sources

| Mode | $C_{\mathrm{in}}$ | $C_{\mathrm{out}}$ | Sign? |
|------|-------------------|---------------------|-------|
| **kfac** | Activation cov $X^T X$ | Output-gradient cov $S^T S$ | No |
| **shampoo** | Gradient cov $G^T G$ | Gradient cov $G G^T$ | No |

Both are already implemented in `dap_opnorm.py`.

### Four damping strategies

1. **opnorm\_target + per\_layer\_damping** (gradient-aware, default):
   Per-side target $= \sqrt{\mathrm{opnorm\_target} / \|G\|_{\mathrm{op}}}$, then solve for $\delta_\ell$ per layer to hit this target exactly. Scale-invariant and adaptive to both covariance spectrum and gradient magnitude.

2. **precond\_only\_opnorm** (ablation):
   Per-side target $= \sqrt{\mathrm{opnorm\_target}}$, ignoring $\|G\|_{\mathrm{op}}$.
   Tests whether gradient-awareness matters or just preconditioner control suffices.

3. **Relative damping**: $C_{\mathrm{eff}} = C + \delta \cdot \lambda_{\max}(C) \cdot I$.
   Scale-invariant but fixed across layers. Standard in many second-order optimizers.

4. **Trace-scaled damping** (Ishikawa & Karakida 2023): $C_{\mathrm{eff}} = C + \rho \cdot \frac{\mathrm{tr}(C)}{d} \cdot I$.
   Scales with the mean eigenvalue rather than $\lambda_{\max}$. More robust to outlier eigenvalues than relative damping. Width-transferable under $\mu$P.

### Hyperparameter grids

All experiments use GPT-tiny on fineweb1B (3814 steps, warmup 10%).

**Phase 1: kfac (36 runs)**

| Config | lr | Damping param | output\_cov\_mode | Runs |
|--------|-----|--------------|-------------------|------|
| kfac\_opnorm | {0.005, 0.01, 0.02, 0.04} | opnorm\_target ∈ {1.0, 2.0, 4.0} | kfac | 12 |
| kfac\_rel | {0.005, 0.01, 0.02, 0.04} | damping ∈ {0.003, 0.01, 0.03} | kfac | 12 |
| kfac\_trace | {0.005, 0.01, 0.02, 0.04} | trace\_damping ∈ {0.03, 0.1, 0.3} | kfac | 12 |

**Phase 2: shampoo (36 runs, pending Phase 1 results)**

Same 3 damping strategies with `output_cov_mode=shampoo`.

### Key questions

1. **Does gradient-aware damping (opnorm\_target) outperform fixed relative damping?**
   On GPT-small with sign mode, per-layer adaptive damping beat all fixed schemes (val 4.189 vs 4.202). Does this hold without sign?

2. **kfac vs shampoo**: Activation covariances (kfac) are "forward-looking" curvature; gradient covariances (shampoo) are "backward-looking". Which is better?

3. **Does precond\_only\_opnorm suffice?** If gradient-awareness doesn't help, the simpler preconditioner-only formula avoids SVD computation of $G$ each step.

4. **Can no-sign modes match sign mode quality?** Best sign-mode (DAPOpNorm null, per-layer, opnorm\_target=2.0) val loss on GPT-small: 4.189.

## Submission

```bash
# Phase 1: kfac experiments (36 runs)
bash slurm_scripts/submit.sh train_scripts/dap_opnorm_tiny_fineweb1B.sh \
    param_configs/dap_opnorm_tiny_fineweb1B_kfac_opnorm.json kfac_opnorm_tiny 1

bash slurm_scripts/submit.sh train_scripts/dap_opnorm_tiny_fineweb1B.sh \
    param_configs/dap_opnorm_tiny_fineweb1B_kfac_rel.json kfac_rel_tiny 1

bash slurm_scripts/submit.sh train_scripts/dap_opnorm_tiny_fineweb1B.sh \
    param_configs/dap_opnorm_tiny_fineweb1B_kfac_trace.json kfac_trace_tiny 1
```

## File inventory

- `train_scripts/dap_opnorm_tiny_fineweb1B.sh` — train script with output\_cov\_mode, abs\_damping, precond\_only\_opnorm args
- `param_configs/dap_opnorm_tiny_fineweb1B_kfac_opnorm.json` — kfac × opnorm\_target grid
- `param_configs/dap_opnorm_tiny_fineweb1B_kfac_rel.json` — kfac × relative damping grid
- `param_configs/dap_opnorm_tiny_fineweb1B_kfac_trace.json` — kfac × trace-scaled damping grid
