"""GTA VI launch-day Twitch CCU forecaster -- hierarchical Bayesian model.

Forecasts, for each hour h = 2..24 of the release day (UTC), five quantiles
of the Twitch "Grand Theft Auto VI" category's concurrent viewers, given only
the case file: hour 1's print, the category's pre-release history, and the
hourly curves of past big game launches (the case's ``comparables``).

Model (see MODEL.md for the full write-up)
------------------------------------------
Work in log space, y = log viewers. For each comparable launch c and hour h
let p1_c be its hour-1 print and ch1_c its hour-1 live channels. The model
is a two-level hierarchy:

  level 1 (launch level):  log y_{c,h} = a_h + b1_h * (log p1_c - m1)
                                        + b2_h * (log ch1_c - m2) + e_{c,h}
  level 2 (population):    b_j_h ~ Normal(pooled_j, tau_j)  -- slopes are
                           partially pooled across hours: each hour's slope
                           is shrunk toward the cross-hour mean with a
                           pseudo-count of 8 launches (empirical Bayes).

Why this shape: across 37 comparable launches the correlation between
log hour-1 viewers and log peak ratio is -0.94 -- a launch that is already
huge in hour 1 (most of the world already unlocked) grows little, while a
launch with a tiny hour 1 (unlock still hours away) grows 10-1000x. The
slope b1_h itself falls with h (about 0.9 at h=2, about 0.15 at h=20):
early hours track hour 1 almost one-for-one, the evening peak is nearly
independent of hour 1 and is set by the intercept a_h, i.e. by the typical
absolute peak of a big launch. Channels carry independent signal about how
broad the launch already is (corr -0.89 with peak ratio on their own).

Predictive distribution: for hour h the predictive log-viewers are
Normal(mu_h, s_h^2) with s_h from a robust (MAD) residual scale, floored at
0.20 and multiplied by 0.55 (calibrated on the four practice launches; see
MODEL.md), inflated by sqrt(1 + 1/n_h) for parameter uncertainty. The five
reported quantiles are exp(mu_h + z_tau * s_h) at the standard normal
quantiles. Two structural priors guard extrapolation, since GTA VI's hour-1
print will be far above every comparable's (largest: 875k):

  * covariates are clamped to the training range in log space (+/- 0.5);
  * when the target's own hour-1 print is itself in the high-print range,
    mu_h is clamped into the 5%-95% band of observed hour-h ratios
    (viewers_h / viewers_1) among high-print comparables. Among launches
    with hour-1 prints above 100k, no evening ratio ever fell below 0.70
    or above 2.68, so a naive log-linear extrapolation that would predict
    an evening collapse to 0.23x hour 1 is ruled out a priori.

Everything is recomputed from the case file at run time: no fitted numbers
are hard-coded, no network, no randomness, standard library only. If the
comparables are too thin to fit, the program falls back to a pooled
median-ratio curve, and then to a persistence forecast, so it always emits
a valid forecast. Two runs on the same case write identical bytes.

Run:  MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; put it back so
# a helper shipped beside main.py (e.g. scripts/backtest.py's import) works.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at LEVELS.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
HOURS = list(range(2, 25))

# --- tuned hyper-parameters (calibration in MODEL.md) -----------------------
SHRINK_PSEUDO_COUNT = 8.0   # launches' worth of pull toward the pooled slope
SIGMA_MULT = 0.5            # shrink the robust residual scale for pinball
SIGMA_FLOOR = 0.20          # minimum predictive log-scale, per hour
BAND_FRACTION = 0.6         # "high-print" = at/above this quantile of p1
BAND_LOW_Q = 0.05           # ratio-band quantiles for the structural prior
BAND_HIGH_Q = 0.95
COVARIATE_SLACK = 0.5       # log-space slack beyond the training range
MIN_CURVES = 5              # comparables needed per hour to fit that hour
MIN_CURVES_TOTAL = 8        # comparables needed overall, else fallback


def _log(x):
    """log of a viewer count, floored at 1 viewer."""
    return math.log(max(1.0, float(x)))


def _quantile(sorted_values, p):
    """Linear-interpolation quantile of an already sorted list."""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return sorted_values[0]
    pos = p * (n - 1)
    lo = int(math.floor(pos))
    hi = min(n - 1, lo + 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _round2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = math.floor(x * 100.0 + 0.5)
    value = scaled / 100.0
    return int(value) if float(value).is_integer() else value


def _solve(matrix, rhs):
    """Solve a small dense linear system by Gauss-Jordan elimination.

    Returns None when the matrix is (numerically) singular.
    """
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col] / pv
                for cc in range(col, n + 1):
                    aug[r][cc] -= factor * aug[col][cc]
    return [aug[i][n] / aug[i][i] if abs(aug[i][i]) > 1e-12 else 0.0
            for i in range(n)]


def _parse_time(text):
    """Parse an ISO-8601 UTC timestamp; None when unparsable."""
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _extract_curves(case):
    """Turn the case's comparables into launch curves.

    Returns a list of dicts {name, p1, ch1, y: {hour: viewers}} for every
    comparable that has a usable hour-1 print and at least five other
    release-day hours. The target game's own pre-release curve (Grand Theft
    Auto VI) has no release-day hours and drops out by itself.
    """
    curves = []
    for comp in case.get("comparables") or []:
        day_start = _parse_time(comp.get("day_start"))
        hourly = comp.get("hours") or comp.get("hourly") or []
        if day_start is None or not hourly:
            continue
        by_hour = {}
        for point in hourly:
            at = _parse_time(point.get("from") or point.get("at"))
            viewers = point.get("viewers")
            if at is None or viewers is None:
                continue
            try:
                viewers = float(viewers)
            except (TypeError, ValueError):
                continue
            if viewers <= 0:
                continue
            hour = int((at - day_start).total_seconds() // 3600) + 1
            if 1 <= hour <= 24 and hour not in by_hour:
                channels = point.get("channels")
                by_hour[hour] = (viewers, channels)
        if 1 not in by_hour or len(by_hour) < 6:
            continue
        p1, ch1 = by_hour[1]
        curves.append({
            "name": str(comp.get("name") or "launch"),
            "p1": p1,
            "ch1": ch1 if isinstance(ch1, (int, float)) and ch1 > 0 else 0.0,
            "y": {h: v[0] for h, v in by_hour.items() if h >= 2},
        })
    curves.sort(key=lambda c: c["name"])
    return curves


def _fit_slopes(curves):
    """Per-hour OLS of log viewers on centred log p1 and log channels.

    Returns (fits, pooled) where fits[h] = (intercept, b1, b2, n) with the
    slopes already shrunk toward the cross-hour pooled slopes, and pooled
    is the pair of pooled slopes. The pooled slope for a feature is the
    mean over hours of that feature's univariate OLS slope -- the level-2
    prior mean of the hierarchy.
    """
    if len(curves) < MIN_CURVES_TOTAL:
        return {}, (0.0, 0.0)
    mean_log_p1 = sum(_log(c["p1"]) for c in curves) / len(curves)
    # _log floors a missing/zero channel count at 1 viewer, so the mean is
    # over every curve, exactly as the backtest in scripts/backtest.py does.
    mean_log_ch = sum(_log(c["ch1"]) for c in curves) / len(curves)

    def features(c):
        return [_log(c["p1"]) - mean_log_p1, _log(c["ch1"]) - mean_log_ch]

    raw = {}
    for h in HOURS:
        rows = [(features(c), _log(c["y"][h]))
                for c in curves if h in c["y"]]
        if len(rows) < MIN_CURVES:
            continue
        # Univariate slopes for the pooled prior, then the joint OLS.
        pooled_here = []
        for j in range(2):
            xs = [r[0][j] for r in rows]
            ys = [r[1] for r in rows]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            pooled_here.append(sxy / sxx if sxx > 0 else 0.0)
        n = len(rows)
        dim = 3
        # Build X'X / X'y directly.
        ata = [[0.0] * dim for _ in range(dim)]
        aty = [0.0] * dim
        for f, y in rows:
            x = [1.0, f[0], f[1]]
            for a in range(dim):
                aty[a] += x[a] * y
                for b in range(dim):
                    ata[a][b] += x[a] * x[b]
        for a in range(1, dim):
            ata[a][a] += 1e-6
        beta = _solve(ata, aty)
        if beta is None:
            continue
        raw[h] = (beta, pooled_here, n)
    if not raw:
        return {}, (0.0, 0.0)
    pooled = [sum(raw[h][1][j] for h in raw) / len(raw) for j in range(2)]
    fits = {}
    for h, (beta, _pooled_here, n) in raw.items():
        shrunk = [beta[0]]
        for j in range(2):
            shrunk.append((n * beta[j + 1] +
                           SHRINK_PSEUDO_COUNT * pooled[j]) /
                          (n + SHRINK_PSEUDO_COUNT))
        fits[h] = (shrunk, n)
    return fits, pooled


def _robust_scale(residuals):
    """MAD-based scale of residuals (1.4826 * median |r - median r|)."""
    if not residuals:
        return None
    med = statistics.median(residuals)
    mad = statistics.median([abs(r - med) for r in residuals])
    return 1.4826 * mad


def _ratio_bands(curves):
    """5%-95% bands of hourly ratios among high-print comparables.

    Returns (bands, threshold): bands[h] = (low, high) ratio, threshold the
    hour-1 print a target must reach for the bands to apply to it.
    """
    if len(curves) < MIN_CURVES_TOTAL:
        return {}, 0.0
    prints = sorted(c["p1"] for c in curves)
    threshold = prints[min(len(prints) - 1, int(BAND_FRACTION * len(prints)))]
    big = [c for c in curves if c["p1"] >= threshold]
    bands = {}
    for h in HOURS:
        ratios = sorted(c["y"][h] / c["p1"] for c in big if h in c["y"])
        if len(ratios) >= MIN_CURVES:
            bands[h] = (_quantile(ratios, BAND_LOW_Q),
                        _quantile(ratios, BAND_HIGH_Q))
    return bands, threshold


def _forecast_hierarchical(curves, p1, ch1):
    """The hierarchical forecast: {hour: [q05, q25, q50, q75, q95]}."""
    fits, _pooled = _fit_slopes(curves)
    if not fits:
        return None
    mean_log_p1 = sum(_log(c["p1"]) for c in curves) / len(curves)
    mean_log_ch = sum(_log(c["ch1"]) for c in curves) / len(curves)
    log_p1_range = [_log(c["p1"]) for c in curves]
    log_ch_range = [_log(c["ch1"]) for c in curves]
    # Clamp the target's covariates to the training range (plus slack):
    # GTA VI's hour-1 print will exceed every comparable's, and a
    # log-linear slope must not be extrapolated without bound.
    tx1 = (min(max(_log(p1), min(log_p1_range) - COVARIATE_SLACK),
               max(log_p1_range) + COVARIATE_SLACK) - mean_log_p1)
    tx2 = (min(max(_log(ch1), min(log_ch_range) - COVARIATE_SLACK),
               max(log_ch_range) + COVARIATE_SLACK) - mean_log_ch)
    bands, threshold = _ratio_bands(curves)
    out = {}
    for h in HOURS:
        if h not in fits:
            continue
        (intercept, b1, b2), n = fits[h]
        mu = intercept + b1 * tx1 + b2 * tx2
        residuals = []
        for c in curves:
            if h not in c["y"]:
                continue
            f1 = _log(c["p1"]) - mean_log_p1
            f2 = _log(c["ch1"]) - mean_log_ch
            residuals.append(_log(c["y"][h]) - (intercept + b1 * f1 + b2 * f2))
        scale = _robust_scale(residuals)
        sigma = SIGMA_FLOOR
        if scale is not None:
            sigma = max(SIGMA_FLOOR, scale * SIGMA_MULT)
        sigma *= math.sqrt(1.0 + 1.0 / n)
        # Structural prior: for a high-print target, the evening audience
        # does not collapse far below hour 1 (and does not explode far
        # above the biggest ratios high-print launches ever showed).
        if h in bands and p1 >= threshold:
            low, high = bands[h]
            mu = min(max(mu, _log(p1) + math.log(max(low, 0.05))),
                     _log(p1) + math.log(max(high, low)))
        out[h] = [math.exp(mu + z * sigma) for z in Z]
    return out if len(out) == len(HOURS) else None


def _forecast_pooled(curves, p1):
    """Fallback: pooled median log-ratio curve with IQR spread."""
    out = {}
    for h in HOURS:
        ratios = sorted(_log(c["y"][h]) - _log(c["p1"])
                        for c in curves if h in c["y"])
        if len(ratios) < 3:
            continue
        med = statistics.median(ratios)
        q25 = _quantile(ratios, 0.25)
        q75 = _quantile(ratios, 0.75)
        sigma = max(SIGMA_FLOOR, (q75 - q25) / 1.349)
        out[h] = [p1 * math.exp(med + z * sigma) for z in Z]
    return out if len(out) == len(HOURS) else None


def _forecast_persistence(p1):
    """Last-resort fallback: every hour stays at hour 1's print, with a
    random-walk spread that widens like sqrt(h - 1) in log space."""
    return {h: [max(1.0, p1) * math.exp(z * 0.15 * math.sqrt(h - 1))
                for z in Z] for h in HOURS}


def _clean_row(quantiles):
    """Round, enforce > 0, <= 1e9 and non-decreasing, as the grader needs."""
    cleaned = []
    prev = 0.0
    for q in quantiles:
        value = _round2(min(max(float(q), 0.01), 1e9))
        if value < prev:
            value = prev
        cleaned.append(value)
        prev = value
    if cleaned[0] <= 0:
        cleaned[0] = 0.01
    return cleaned


def forecast_case(case):
    """Forecast one case dict -> {"hours": [{"hour": h, "q": [...]}, ...]}."""
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r"
                         % (case.get("levels"), LEVELS))
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        p1 = max(1.0, float(first["viewers"]))
        ch1 = first.get("channels") or 0
    else:
        # No hour-1 print: anchor on the last pre-checkpoint hourly mean.
        hourly = case.get("history_hourly") or []
        p1 = max(1.0, float(hourly[-1]["viewers"])) if hourly else 1.0
        ch1 = (hourly[-1].get("channels") or 0) if hourly else 0
    try:
        ch1 = float(ch1)
    except (TypeError, ValueError):
        ch1 = 0.0
    curves = _extract_curves(case)
    forecast = None
    if len(curves) >= MIN_CURVES_TOTAL:
        forecast = _forecast_hierarchical(curves, p1, ch1)
    if forecast is None and curves:
        forecast = _forecast_pooled(curves, p1)
    if forecast is None:
        forecast = _forecast_persistence(p1)
    hours = [{"hour": h, "q": _clean_row(forecast[h])} for h in HOURS]
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / ("%s.out" % path.stem)
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
