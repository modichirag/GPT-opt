# DAPOpNorm: finewebmini short-run experiments

**Setting**: GPT-tiny, finewebmini, ~381 optimizer steps (~100M tokens), batch size 256
(global), single GPU. This is a very short run — results are noisy and should be treated
as directional only.

## Baselines (3 seeds each, confirmed)

| Method | Best LR | Val loss (mean ± std) |
|--------|---------|----------------------|
| Muon | 0.05–0.1 | 4.742 ± 0.002 |
| DAPOpNorm (no fixes) | 0.02 | 4.690 ± 0.018 |
| AdamW | 0.003 | ~5.17 |

DAPOpNorm beats Muon at its optimal LR (~0.02) but degrades badly at lr≥0.05.
Muon is flat across a wide LR range (0.05–0.1); DAPOpNorm is sharply peaked.

## Hypothesis: $C^{-1/2}$ instability at high LR

Diagnostics confirm that the operator norm of $C^{-1/2}$ grows much larger at lr≥0.05
than at lr=0.02, consistent with small eigenvalues of $C$ causing the effective step size
to blow up. Two fixes were explored: EMA of C (variance reduction) and damping
(additive regularization before inversion).

## EMA=0.99 sweep (3 seeds at best point)

| LR | Val loss |
|----|---------|
| 0.005–0.015 | ≥4.68 (worse than no-fix) |
| 0.02 | **4.678 ± 0.005** |
| 0.03–0.1 | 4.67–4.84 |

EMA=0.99 gives a modest improvement at lr=0.02 (4.678 vs 4.690 baseline) but the
optimum LR doesn't shift. Low variance across seeds. Doesn't fix the high-LR degradation.

## Damping-only sweeps (single runs)

Small damping (0.001–0.005, no EMA) shifts the optimal LR from 0.02 toward 0.03–0.05
but doesn't give uniform improvements. The sensitivity to damping value is high and
non-monotone across single runs — likely noise-dominated at this training length.

## Damping + EMA=0.99 (single runs + seeds at best point)

| Config | Best LR | Val loss |
|--------|---------|---------|
| d=0.001 + EMA=0.99 | 0.07 | **4.512 ± 0.009** (3 seeds) |
| d=0.005 + EMA=0.99 | 0.03 | 4.731 (single run) |

d=0.001 + EMA=0.99 is the strongest result found: 4.512 mean over 3 seeds at lr=0.07,
a clear improvement over baseline DAPOpNorm and Muon.

## Interpretation in light of fineweb1B results

The fineweb1B experiments (10× longer runs, ~3815 steps) significantly revise the
conclusions drawn here:

1. **Optimal damping direction was misleading.** The finewebmini results appeared to
   favor small damping (d=0.001) — but in fineweb1B with a reliable signal, *higher*
   damping (d=0.01–0.03) is clearly better. The apparent non-monotone behavior of
   damping in finewebmini was noise, not signal.

2. **"Large damping → behaves like Muon" is true in the limit but the threshold is
   higher than finewebmini suggested.** As $\delta \to \infty$, $C_{\mathrm{eff}} \approx \delta\lambda_{\max}(C) I$
   and the update reduces to $\mathrm{sign}(G)$, which is exactly Muon. But $d=0.03$
   in fineweb1B still substantially outperforms Muon — finewebmini was too short to
   reliably distinguish DAPOpNorm from Muon at moderate damping values.

3. **Higher damping → wider LR optimum.** This pattern is clear in fineweb1B and is
   consistent with (but not visible in) finewebmini. The key advantage of larger damping
   is robustness: the method works well across a broader LR range.

4. **The ~0.23 nat gap over Muon in finewebmini likely overstates the advantage.**
   In fineweb1B the gap at best tuning is ~0.04 nats. Both methods improve substantially
   in absolute terms in longer runs; the relative gap shrinks.

**Bottom line**: finewebmini is too short to draw reliable conclusions about damping.
Use fineweb1B (~3815 steps) as the primary experimental setting going forward.
