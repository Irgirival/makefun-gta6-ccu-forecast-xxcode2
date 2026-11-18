# Model — hierarchical Bayesian forecaster for GTA VI launch-day Twitch CCU

## 1. What is being forecast

The Twitch category "Grand Theft Auto VI" (Twitch game id 1867279146)
concurrent viewers, hour by hour over the UTC release day 2026-11-19.
An hour's print is the mean of the house recorder's minute reads in that
hour. The program sees only the case file: hour 1's print (viewers and live
channels), the category's pre-release history, and `comparables` — hourly
curves of 38–42 past big launches from SullyGnome (14 days before to 3 days
after each launch's release day). It must output, for hours 2–24, five
quantiles (0.05, 0.25, 0.50, 0.75, 0.95). Scoring is mean pinball loss on
the log scale: `2 * max(tau * d, (tau - 1) * d)` with `d = ln(y) - ln(q)`.

## 2. The empirical facts that drive the model

Computed from the case files themselves (kit `input/p00.in`–`p03.in`,
comparables only) plus the practice answers (SullyGnome hourly averages):

1. **Hour 1 is not representative.** All four practice launches are
   console games with rolling midnight-local unlocks, like GTA VI. Their
   hour-1 prints (00:00–01:00 UTC) cover only the regions already unlocked
   (New Zealand, Australia, Asia, Europe); the US unlocks at 05:00Z
   (Eastern) and 08:00Z (Pacific). True peak ratios (peak / hour 1):
   Wolverine 9.09×, RDR2 3.17×, God of War Ragnarök 1.71×, Spider-Man 2
   2.21×. The house persistence baseline, which assumes every hour equals
   hour 1, scores 0.387 on the practice — its typical miss is 83%.
2. **Peak ratio is predictable from hour 1.** Across the 37 comparables
   with a usable hour 1, corr(log hour-1 viewers, log peak ratio) = **−0.94**;
   corr(log hour-1 channels, log peak ratio) = −0.89. A launch that is
   already huge in hour 1 has already unlocked most of its audience; a
   launch with a tiny hour 1 (e.g. Black Myth: Wukong, hour 1 = 4,941
   because China unlocked at 02:00Z; Silksong, hour 1 = 164, global unlock
   14:00Z) grows 90–2,000×. This is a regression-to-the-mean structure,
   not a fixed growth curve — a pooled median ratio curve scores only
   0.311 on the practice.
3. **The anchoring fades over the day.** The fitted slope on log hour-1
   viewers falls monotonically with hour: ≈0.92 at hour 2, ≈0.65 at hour
   12, ≈0.15 at hour 20. Early hours track hour 1 almost one-for-one;
   the evening peak is nearly independent of hour 1 and is set by the
   intercept — the typical absolute peak of a big launch (≈200–900k
   viewers in the comparables). The channels slope is positive early
   (≈0.13–0.26 at hours 2–7) and ≈0 later: how broad the launch already
   is matters most in the first hours.
4. **High-print launches are tightly banded.** Among comparables with
   hour-1 prints ≥ 100k (n = 16), hourly ratios (viewers_h / viewers_1)
   never fell below 0.70 in the evening (hours 18–24, 5th percentile
   0.62–0.72) and never exceeded 2.68 (95th percentile 2.53–2.74). The
   night dip (hours 4–12) bottoms at 0.31–0.40. These bands are the
   structural prior of §4.

## 3. The hierarchy

For comparable launch c and hour h, with p1_c its hour-1 viewers and
ch1_c its hour-1 channels, in log space:

```
log y_{c,h} = a_h + b1_h * (log p1_c - m1) + b2_h * (log ch1_c - m2) + e_{c,h}
b_j_h ~ Normal(pooled_j, tau_j)          # level 2: partial pooling
e_{c,h} ~ Normal(0, s_h^2)               # level 1: launch noise
```

- `pooled_j` is the mean over hours of feature j's univariate OLS slope
  (the level-2 prior mean, estimated from the data — empirical Bayes).
- Each hour's slope is shrunk toward it with pseudo-count 8:
  `b = (n * b_ols + 8 * pooled) / (n + 8)`, n ≈ 37 launches per hour.
  With 37 launches the shrinkage is gentle (≈18% pull); it exists to
  stabilise hours where a few launches are missing.
- The intercept is the OLS intercept at the shrunk slopes (equivalently,
  the mean log viewers at the geometric-mean hour-1 print/channels).

## 4. Posterior predictive quantiles and structural priors

For the target, with its hour-1 print p1 and channels ch1:

```
mu_h    = a_h + b1_h * tx1 + b2_h * tx2
s_h     = max(0.20, 1.4826 * MAD(residuals_h) * 0.50) * sqrt(1 + 1/n_h)
q_tau,h = exp(mu_h + z_tau * s_h)
```

- **tx1, tx2 are clamped** to the training range of log p1 / log channels
  (± 0.5 in log space). GTA VI's hour-1 print will exceed every
  comparable's (largest: ELDEN RING 875,155; the category already showed
  2,452,216 viewers / 11,781 channels on 2026-11-18, pre-release). A
  log-linear slope extrapolated past the data would predict an evening
  collapse to ≈0.23× hour 1; clamping stops that.
- **Ratio bands.** When the target's own hour-1 print is at/above the
  60th percentile of the comparables' hour-1 prints, mu_h is clamped into
  [log p1 + log band_low_h, log p1 + log band_high_h], where the bands are
  the 5% / 95% quantiles of hourly ratios among high-print comparables
  (§2.4). The clamp does not bind for small-print targets (e.g. Wolverine,
  whose true evening ratio is 8.9× — outside any high-print band), so it
  costs nothing on the practice and only guards the GTA VI extrapolation.
- **Calibration of s_h.** The MAD scale (robust to the simultaneous-unlock
  outliers whose ratios reach 2,000×) times 0.50 was chosen on the four
  practice launches: multiplier 0.45 → 0.1652, 0.50 → 0.1659, 0.55 →
  0.1675, 0.70 → 0.1772, 0.90 → ≈0.19 (mean pinball). The floor 0.20 keeps
  hours with spuriously tight residuals honest; sqrt(1 + 1/n_h) adds
  parameter uncertainty. 0.50 was picked over 0.45 as the less aggressive
  of two near-equal optima (leave-one-out over all 37 comparables prefers
  wider spreads, practice prefers narrower — 0.50 splits the difference).

## 5. Backtest

Practice cases scored with the grader's exact pinball rule, answers =
SullyGnome hourly category averages (the source the kit's cases use;
hour-1 prints match the cases exactly). `scripts/backtest.py` reproduces
this table from the kit's input directory.

| case | launch | loss | typical miss | 90% coverage |
| --- | --- | --- | --- | --- |
| p00 | Marvel's Wolverine (p1 23,357) | 0.146 | 21.6% | 0.87 |
| p01 | Red Dead Redemption 2 (p1 116,550) | 0.170 | 30.0% | 0.83 |
| p02 | God of War Ragnarök (p1 269,517) | 0.145 | 27.4% | 1.00 |
| p03 | Marvel's Spider-Man 2 (p1 119,087) | 0.203 | 33.0% | 0.74 |
| **mean** | | **0.166** | | **0.86** |

House persistence baseline on the same cases: mean loss **0.387**,
typical miss 82.8%, coverage 0.783. Pooled median-ratio curve: 0.311.
The model beats the baseline on every practice case, most on the case
where the baseline fails hardest (Wolverine: 0.146 vs 0.744).

Leave-one-out over the 37 comparables (each launch as target, fitted on
the other 36) gives mean loss ≈ 0.45–0.46 — worse, because that pool
includes simultaneous-global-unlock and early-access launches whose
curves are far more dispersed; the final target is a rolling-midnight
console launch, the practice regime, where the model scores 0.166.

## 6. What the model predicts for GTA VI

Illustrative medians (training = the 37 comparables in a practice case;
final training will also include the four practice launches):

| hour-1 print | median h2 | median h5 | median h12 | median h20 | median peak |
| --- | --- | --- | --- | --- | --- |
| 500,000 | 438,935 | 330,377 | 240,066 | 397,450 | 438,935 |
| 1,000,000 | 846,066 | 613,416 | 398,354 | 794,900 | 846,066 |
| 2,000,000 | 1,574,758 | 930,261 | 796,708 | 1,589,800 | 1,597,533 |
| 2,452,216 | 1,930,823 | 1,140,600 | 976,850 | 1,949,266 | 1,958,748 |

Shape: a night dip to ~0.4–0.55× hour 1 around hours 5–12, then a climb
into the US afternoon / European evening, peaking at hours 19–21 at
≈0.8× hour 1 for a very large hour 1 (the ratio-band floor), with a
90% band roughly 0.4×–2.1× the median at hour 20. If hour 1 instead
comes in small (e.g. 300k), the unclamped regression takes over and the
evening median rises above hour 1 (≈1.05×).

## 7. Failure modes and fallbacks

- **Comparables too thin** (< 8 launches, or an hour unfittable): fall
  back to the pooled median-ratio curve, then to persistence. The program
  never crashes and always writes a valid 23-row forecast.
- **Hour 1 missing**: anchor on the last pre-checkpoint hourly mean.
- **A launch unlike any comparable** (e.g. Twitch itself down, or the
  category flooded by non-gameplay streams): the predictive spread is
  wide (log-scale ≈ 0.3–0.6) and the ratio bands bound the damage; the
  median can still be off by ~30% on the worst practice case, which the
  pinball loss prices at ≈0.20, still well under the baseline.
- **Practice ≠ final.** The practice is graded on SullyGnome hourly
  averages; the final on minute-mean prints. The model uses only
  hour-level aggregates, so the measure change should be second-order,
  but the absolute level (GTA VI is bigger than any comparable) is the
  main residual risk — hence the structural priors of §4 rather than
  blind regression.
