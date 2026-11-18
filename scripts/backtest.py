"""Backtest for the hierarchical Bayesian forecaster (main.py).

Scores main.forecast_case on the kit's four practice cases with the
grader's exact pinball rule on the log scale, against the practice
answers: SullyGnome hourly category averages for each launch's release
day (the same source the kit's cases use; the hour-1 prints match the
cases exactly). Also scores the house persistence baseline for comparison.

Usage:  python3 scripts/backtest.py <kit input directory>
        (default: the kit's input/ next to this repository's parent,
        overridable with MAKEFUN_INPUT_DIR)
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Practice answers: SullyGnome "Average viewers" per hour of the release
# day (UTC), hours 1..24. Source: https://sullygnome.com, GameViewers
# chart data, read 2026-11-18 for the release month of each game.
ANSWERS = {
    # Marvel's Wolverine, 2026-09-15 (SullyGnome 117671)
    "p00": [23357, 19397, 20292, 18329, 35760, 40523, 38196, 30468,
            37322, 32884, 36770, 41223, 45078, 58758, 78329, 91382,
            133910, 189396, 175604, 206982, 212243, 196416, 153926, 147358],
    # Red Dead Redemption 2, 2018-10-26 (SullyGnome 28264)
    "p01": [116550, 102778, 99986, 84459, 152428, 176319, 163532, 148485,
            134888, 120686, 119921, 138073, 165093, 195646, 231529, 264043,
            330969, 369358, 322991, 290598, 281596, 258394, 223948, 192584],
    # God of War Ragnarok, 2022-11-09 (SullyGnome 65645)
    "p02": [269517, 220787, 146900, 141669, 156301, 281917, 273454, 258613,
            196345, 186781, 189204, 152840, 179704, 217655, 271761, 310384,
            376075, 435038, 444580, 460387, 445097, 434600, 411192, 383725],
    # Marvel's Spider-Man 2, 2023-10-20 (SullyGnome 97296)
    "p03": [119087, 79573, 51615, 50272, 166214, 146818, 125021, 109152,
            93203, 83997, 71214, 43730, 68551, 100971, 131596, 164722,
            176018, 225745, 250538, 240849, 258378, 263140, 253088, 216387],
}

Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]


def pinball(answer_by_hour, forecast_by_hour):
    """Mean pinball loss, typical miss %, and 90% coverage."""
    total = 0.0
    count = 0
    misses = []
    covered = 0
    scored = 0
    for hour, y in answer_by_hour.items():
        if hour < 2 or hour not in forecast_by_hour:
            continue
        q = forecast_by_hour[hour]
        for tau, qq in zip(LEVELS, q):
            d = math.log(max(1.0, y)) - math.log(qq)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
        misses.append(abs(math.log(y) - math.log(q[2])))
        scored += 1
        if q[0] <= y <= q[4]:
            covered += 1
    typical = 100.0 * (math.exp(sum(misses) / len(misses)) - 1.0)
    return total / count, typical, covered / scored


def baseline_forecast(p1):
    """The house persistence baseline, for comparison."""
    return {h: [max(1.0, p1) * math.exp(z * 0.15 * math.sqrt(h - 1))
                for z in Z] for h in range(2, 25)}


def main_backtest(input_dir):
    input_dir = Path(input_dir)
    losses = []
    base_losses = []
    for case_id in sorted(ANSWERS):
        path = input_dir / ("%s.in" % case_id)
        if not path.exists():
            print("%s: case file missing at %s, skipped" % (case_id, path))
            continue
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = main.forecast_case(case)
        by_hour = {row["hour"]: row["q"] for row in forecast["hours"]}
        answer = {h + 1: v for h, v in enumerate(ANSWERS[case_id])}
        loss, miss, cov = pinball(answer, by_hour)
        p1 = case["known"][0]["viewers"]
        base = baseline_forecast(p1)
        base_loss, _, _ = pinball(answer, base)
        losses.append(loss)
        base_losses.append(base_loss)
        print("%s %-28s loss=%.4f (baseline %.4f) miss=%.1f%% cov90=%.2f"
              % (case_id, case["game"]["name"], loss, base_loss, miss, cov))
    if losses:
        print("MEAN loss=%.4f (baseline %.4f over %d cases)"
              % (sum(losses) / len(losses),
                 sum(base_losses) / len(base_losses), len(losses)))


if __name__ == "__main__":
    directory = (sys.argv[1] if len(sys.argv) > 1
                 else os.environ.get("MAKEFUN_INPUT_DIR", "input"))
    main_backtest(directory)
