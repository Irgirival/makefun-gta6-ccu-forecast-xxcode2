# GTA VI Twitch CCU Forecast — hierarchical Bayesian forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Irgirival/makefun-gta6-ccu-forecast-xxcode2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Irgirival/makefun-gta6-ccu-forecast-xxcode2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for make.fun's ["Forecast GTA VI's launch-day Twitch viewers hour by hour"](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty ($250 in USDC).

`main.py` (Python 3.13, standard library only, no network) reads one case
(`makefun/twitch-ccu-case/v1`) and writes five quantiles (5/25/50/75/95%) of
the Twitch "Grand Theft Auto VI" category's concurrent viewers for each hour
2–24 of release day 2026-11-19 (UTC), given only hour 1's print.

## Method in one paragraph

A hierarchical Bayesian regression in log space, fitted at run time from the
case's own comparables (past big launches' hourly curves). Each hour's log
viewers are regressed on log hour-1 viewers and log hour-1 channels, with
each hour's slopes partially pooled toward the cross-hour mean (empirical
Bayes, pseudo-count 8). The key empirical fact: across 37 comparable
launches, log hour-1 viewers correlates −0.94 with log peak ratio — a launch
already huge in hour 1 grows little, a launch with a tiny hour 1 grows
10–1000× — and the slope on hour 1 falls from ≈0.9 at hour 2 to ≈0.15 at
hour 20, so the evening peak is set by the intercept, not by hour 1.
Predictive quantiles come from a robust (MAD) residual scale, calibrated on
the practice launches. Two structural priors guard extrapolation to GTA VI's
unprecedented scale: covariates are clamped to the training range, and for a
high-print target the forecast is kept inside the 5–95% band of hourly
ratios ever observed among high-print launches (no evening ratio below 0.70
or above 2.68 there). Full derivation, priors, calibration and backtest:
[`MODEL.md`](MODEL.md).

## Backtest (practice cases, scored exactly as the grader scores)

| case | launch | this model | house baseline |
| --- | --- | --- | --- |
| p00 | Marvel's Wolverine | 0.146 | 0.744 |
| p01 | Red Dead Redemption 2 | 0.170 | 0.287 |
| p02 | God of War Ragnarök | 0.145 | 0.215 |
| p03 | Marvel's Spider-Man 2 | 0.203 | 0.303 |
| **mean pinball loss** | | **0.166** | **0.387** |

Typical miss 21–33% per case; 90% coverage 0.74–1.00 (mean ≈0.86, nominal
0.90). Reproduce with `python3 scripts/backtest.py <kit input dir>`.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: two runs on the same case write identical bytes. One case
runs in well under the 8-second cap (four practice cases in ~3 s total
including interpreter start-up). Always emits a valid forecast: if the
comparables are too thin it falls back to a pooled median-ratio curve, then
to a persistence forecast.

## Live data behind the design

Twitch public GQL (`POST https://gql.twitch.tv/gql`, Client-Id
`kimne78kx3ncx6brgo4mv6wki5h1ko`), probed 2026-11-18 ~13:02 UTC:
Grand Theft Auto VI (id 1867279146) showed **2,452,216 viewers /
11,781 channels** the day before launch — pre-release hype already an order
of magnitude above any comparable launch's hour-1 print (largest: ELDEN
RING, 875,155). That is why the model clamps extrapolation instead of
trusting a log-linear slope past the training range. Practice answers are
SullyGnome hourly category averages, the same source the kit's cases use.
