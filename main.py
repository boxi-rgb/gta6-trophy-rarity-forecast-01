"""gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's Exophase earn rate
at the case target, as five quantiles, from the house's recorded reads of the page.

Method (see the prep plan for the reasoning; this file is the whole model):

  * Decompose the rate into earners E and players tracked N.  Exophase refreshes
    its counts on its own schedule, so the page is a staircase: long plateaus,
    then a jump.  A rate extrapolated directly cannot see that structure; E and
    N extrapolated separately and divided can.
  * Compress the history to refresh states: consecutive reads with the same
    players-tracked count and the same rates carry no information, so they are
    folded into one state.  E is recovered as round(rate * N / 100).
  * Extrapolate ln N and ln E against ln(1 + hours since release) by weighted
    least squares over the recent states (last 40 hours), with weights that
    grow quadratically toward the present.  Small-E trophies have their slope
    shrunk toward a pooled slope (by tier group when there are few states,
    across all trophies when there are fewer still).  The E extrapolation
    blends the last-state-anchored line with the regression line's own value
    at the target, so one dipped refresh cannot drag the forecast down.
  * Median rate r50 = E(target) / N(target), blended in logit space with a
    curve-style median (logit rate fit on the last 24 hours of states) where
    that fit exists: the decomposition and the direct curve fit make
    different mistakes, and the blend is steadier than either alone.  The
    story chain Ch.1 >= Ch.2 >= Ch.3 >= Ch.4 >= final >= 100% >= platinum is
    enforced on the medians, Gone Fishing (id 2) is pinned at the print
    ceiling 99.99% -- it has shown 100.00% at every observation, and the
    scorer clamps the answer to 99.99, so a median even a little below the
    ceiling pays full logit distance -- and the platinum is capped low:
    almost nobody legitimately platinums the game in week one.
  * When the history has seen no refresh at all (the practice cases), E/N
    fitting is flat, so the measured first refresh (638 -> 3,868 players
    tracked) is used as a dilution prior: N is grown toward the target by the
    refresh cadence the launch night showed, and E is allowed only a partial
    catch-up among the new players.
  * Spread is built in logit space from three components -- a random-walk term
    (house's 0.5 * sqrt(days)), a refresh-jump term (cross-trophy spread of the
    log-rate jumps seen at refreshes, times the square root of the refreshes
    still to come), and a fit-residual term (regression residual plus a
    1/sqrt(E) counting term) -- then multiplied by a calibration factor and
    shifted by a bias found by a rolling-origin backtest run inside this
    program on the case's own history.  The backtest standardises each error
    against the full spread the model would have used at that origin (all
    three components at that horizon), not the fit term alone; standardising
    against the fit term alone inflates the multiplier to its cap on every
    history.  With too few origins the calibration shrinks to its defaults,
    and long histories thin their origins so the backtest stays fast.
  * How much of the model runs depends on how many refresh states the case
    holds: >= 5 states, per-trophy fits; 3-4, tier-group fits; <= 2, pooled
    slope plus the dilution prior only.

Deterministic throughout: no randomness, fixed iteration order, fixed
rounding (4 decimals, half up, as the starter rounds).  Standard library only.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release of GTA VI, the origin of the log-time axis: ln(1 + hours since here).
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- Measured first Exophase refresh (RULES.txt) ---------------------------------
# Launch night the page sat unchanged at 638 players tracked; at the 04:30 UTC
# read it refreshed to 3,868 players and every rate but the 100% one fell.
# Used as the dilution prior whenever a case's history has seen no refresh.
FIRST_N0 = 638.0
FIRST_N1 = 3868.0
FIRST_GROWTH = FIRST_N1 / FIRST_N0  # ~6.06x players tracked per early refresh
FIRST_INTERVAL_H = 4.7  # 23:48 -> 04:30 UTC, hours between the flat page and the refresh
PRIOR_JUMP_SD = 0.5  # cross-trophy sd of the log-rate jump at a refresh, logit units
PRIOR_FIT_SD = 0.6  # fit-residual spread when there is nothing to fit on

# --- Model constants -------------------------------------------------------------
RW_SIGMA = 0.5  # random-walk spread, logit units per sqrt(day), same as the house
SHRINK_K = 20.0  # earners at which a trophy's own slope gets weight 1/2
EARNER_CATCHUP = 0.25  # exponent: new players earn at a lower rate than old ones
PLATINUM_CAP = 0.4  # platinum median cap, percent
GONE_FISHING_Q50 = 99.99  # id 2 median, percent: every tracked player has it
SPREAD_MIN = 0.05
SPREAD_MAX = 4.0
WINDOW_H = 40.0  # fits use the states within this many hours of the last one


def _recent(states):
    """The states a trend fit should use: the recent window, so the frozen
    launch regime and the first dilution jumps do not set the slopes that
    get extrapolated.  The last state is always kept."""
    if len(states) <= 4:
        return states
    cut = states[-1]["h"] - WINDOW_H
    win = [s for s in states if s["h"] >= cut]
    return win if len(win) >= 3 else states[-4:]

# Story progression chain (ids): chapter order must not rise along the chain.
CHAIN = [3, 4, 5, 6, 7, 14, 1]
GONE_FISHING_ID = 2
PLATINUM_ID = 1


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as the starter rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment: datetime) -> float:
    return (moment - RELEASE).total_seconds() / 3600.0


def fit_line(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x. Returns (a, b, resid_sd).

    resid_sd is the weighted root-mean-square residual (0 when the fit is
    exact or has fewer than 2 points); with 2 points a line fits exactly.
    """
    sw = sum(ws)
    if sw <= 0 or not xs:
        return 0.0, 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(
        sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    )
    return a, b, resid


def compress_history(case):
    """Fold the history into refresh states.

    A new state starts when the players-tracked count or any rate changes;
    consecutive identical reads are one state.  Each state carries its time
    (hours since release), N, and the earners E per trophy recovered from the
    printed rates.  Returns (states, last_rates) where last_rates is the
    last read's rate vector (the fallback the rules score an invalid case by).
    """
    trophies = case["trophies"]
    states = []
    last_rates = None
    for read in case["history"]:
        rates = read["rates"]
        last_rates = rates
        n = read["players_tracked"]
        if states:
            prev = states[-1]
            if n == prev["n"] and rates == prev["rates"]:
                continue
        earners = []
        for i in range(len(trophies)):
            r = rates[i]
            earners.append(None if r is None else round(r * n / 100.0))
        states.append(
            {
                "h": hours_since_release(parse_time(read["at"])),
                "n": n,
                "rates": list(rates),
                "e": earners,
            }
        )
    return states, last_rates


def forecast_n(states, target_h):
    """Extrapolate players tracked N to the target hour.

    With >= 2 states, ln N is fit against ln(1 + h) with recency weights,
    then leaned toward the slope of the latest interval when that slope is
    steeper (weekend sync can accelerate N).  With 1 state there is no slope
    to fit, so N grows by the measured first-refresh factor per refresh
    interval still to come, capped at two refreshes' worth.
    """
    n_last = states[-1]["n"]
    states = _recent(states)
    if len(states) < 2:
        horizon_h = max(0.0, target_h - states[-1]["h"])
        refreshes = min(horizon_h / FIRST_INTERVAL_H, 2.0)
        return n_last * (FIRST_GROWTH ** refreshes)
    xs = [math.log1p(s["h"]) for s in states]
    ys = [math.log(s["n"]) for s in states]
    m = len(states)
    ws = [((i + 1) / m) ** 2 * 2.0 for i in range(m)] if m > 1 else [1.0]
    a, b, _ = fit_line(xs, ys, ws)
    n_fit = math.exp(a + b * math.log1p(target_h))
    # Lean toward the latest interval's slope when it runs hotter.
    s0, s1 = states[-2], states[-1]
    dx = math.log1p(s1["h"]) - math.log1p(s0["h"])
    if dx > 0:
        b_recent = (math.log(s1["n"]) - math.log(s0["n"])) / dx
        if b_recent > b:
            n_recent = s1["n"] * math.exp(
                b_recent * (math.log1p(target_h) - math.log1p(s1["h"]))
            )
            n_fit = math.exp(0.7 * math.log(n_fit) + 0.3 * math.log(n_recent))
    return max(n_fit, float(n_last))


def trophy_slopes(states):
    """Per-trophy ln E slopes against ln(1 + h), plus residual sds.

    Returns (slopes, resids, last_e, intercepts): slope and intercept None
    where the trophy never had a positive earner count to fit on.  Zero
    earners are read as a pseudo-count of 0.5 so the log stays finite.
    """
    states = _recent(states)
    k = len(states[0]["e"])
    xs = [math.log1p(s["h"]) for s in states]
    m = len(states)
    ws = [((i + 1) / m) ** 2 * 2.0 for i in range(m)] if m > 1 else [1.0]
    slopes, resids, last_e, intercepts = [], [], [], []
    for j in range(k):
        ys = []
        ok = True
        for s in states:
            e = s["e"][j]
            if e is None:
                ok = False
                break
            ys.append(math.log(max(e, 0.5)))
        if not ok or len(states) < 2:
            slopes.append(None)
            resids.append(0.0)
            intercepts.append(None)
        else:
            a, b, r = fit_line(xs, ys, ws)
            slopes.append(b)
            resids.append(r)
            intercepts.append(a)
        last_e.append(states[-1]["e"][j])
    return slopes, resids, last_e, intercepts


def pooled_slope(slopes, last_e, idxs):
    """Earners-weighted mean slope over the given trophy indices."""
    num = 0.0
    den = 0.0
    for j in idxs:
        if slopes[j] is None:
            continue
        w = max(last_e[j] if last_e[j] is not None else 0, 1)
        num += w * slopes[j]
        den += w
    return (num / den) if den > 0 else 0.0


def forecast_medians(states, trophies, target_h, calib=(0.0, 1.0)):
    """Median earn rates at target_h, plus per-trophy base spreads.

    calib is (logit bias, spread multiplier) from the rolling backtest.
    Returns (medians, spreads, n_target): medians in percent, spreads in
    logit units, before the calibration multiplier is applied to the spread.
    """
    n_states = len(states)
    n_target = forecast_n(states, target_h)
    x_target = math.log1p(target_h)
    slopes, resids, last_e, intercepts = trophy_slopes(states)
    all_idx = list(range(len(trophies)))

    # Pooled slope, and tier-group slopes for the 3-4 state regime.
    pooled = pooled_slope(slopes, last_e, all_idx)
    tier_groups = {}
    for j, t in enumerate(trophies):
        tier_groups.setdefault(t["tier"], []).append(j)
    tier_slope = {
        tier: pooled_slope(slopes, last_e, idxs)
        for tier, idxs in tier_groups.items()
    }

    bias, mult = calib
    medians = []
    spreads = []
    for j, trophy in enumerate(trophies):
        e_last = last_e[j]
        if e_last is None:
            e_last = 0
        x_last = math.log1p(states[-1]["h"])
        # --- E at target ---------------------------------------------------
        if n_states >= 5:
            own = slopes[j]
            w = e_last / (e_last + SHRINK_K) if e_last > 0 else 0.0
            beta = w * own + (1.0 - w) * pooled if own is not None else pooled
        elif n_states >= 3:
            own = slopes[j]
            group = tier_slope[trophy["tier"]]
            w = e_last / (e_last + SHRINK_K) if e_last > 0 else 0.0
            beta = w * own + (1.0 - w) * group if own is not None else group
        else:
            beta = pooled  # <= 2 states: pooled slope only
        if n_states >= 2:
            # Anchor the line at the last state's earners (pseudo-count
            # floor), then blend in log space with the regression line's own
            # value at the target: anchoring alone lets one dipped refresh
            # drag the whole extrapolation down with it.
            e_anchor = max(e_last, 0.5)
            e_loglin = math.exp(math.log(e_anchor) + beta * (x_target - x_last))
            e_target = e_loglin
            if intercepts[j] is not None and n_states >= 5:
                beta_j = slopes[j] if slopes[j] is not None else beta
                e_line = math.exp(intercepts[j] + beta_j * x_target)
                e_target = math.exp(0.5 * math.log(max(e_loglin, 0.5))
                                    + 0.5 * math.log(max(e_line, 0.5)))
            e_target = max(e_target, 0.5)
        else:
            # One state: no slope exists.  Earners grow only part-way with
            # the new players the dilution prior brings in.
            e_target = max(e_last, 0.5) * (n_target / states[-1]["n"]) ** EARNER_CATCHUP
        r = 100.0 * e_target / n_target
        medians.append(r)
        # --- base spread (before calibration) ------------------------------
        # The counting term uses the earners actually observed at the last
        # state, not the extrapolated count: a pseudo-count floor of 0.5 in
        # the extrapolation must not inflate the spread by itself.
        resid = resids[j] if n_states >= 2 else PRIOR_FIT_SD
        s_fit = math.sqrt(resid * resid + 1.0 / max(e_last, 1))
        spreads.append(s_fit)
    # Blend with a curve-style median: logit rate fit on the last 24 hours
    # of states, read at the target.  Only where that fit itself ran -- with
    # too few states there is no curve estimate to blend with, and the
    # decomposition stands alone.
    x_t = math.log1p(target_h)
    h_last = states[-1]["h"]
    for j in range(len(trophies)):
        pts_x, pts_y = [], []
        for s in states:
            if s["h"] >= h_last - 24.0 and s["rates"][j] is not None:
                pts_x.append(math.log1p(s["h"]))
                pts_y.append(logit(s["rates"][j]))
        if len(pts_x) < 3 or len(set(pts_x)) < 2:
            continue
        a_c, b_c, _ = fit_line(pts_x, pts_y, [1.0] * len(pts_x))
        lg_c = a_c + b_c * x_t
        medians[j] = inv_logit(0.5 * logit(medians[j]) + 0.5 * lg_c)
    return medians, spreads, n_target


def jump_spread(states, trophies, target_h):
    """Refresh-jump spread component, in logit units.

    The cross-trophy standard deviation of the log-rate change at each
    refresh in the history, times sqrt(expected refreshes to the target).
    With no refresh in the history, the measured first refresh's jump spread
    is the prior.
    """
    horizon_h = max(0.0, target_h - states[-1]["h"])
    if len(states) < 2:
        refreshes = min(horizon_h / FIRST_INTERVAL_H, 2.0)
        return PRIOR_JUMP_SD * math.sqrt(max(refreshes, 0.0))
    jumps = []
    # Only the latest refreshes set the jump scale: early jumps are large
    # while the page is young, and the jumps still to come look like the
    # recent ones, not the launch-night ones.
    recent_pairs = list(zip(states, states[1:]))[-3:]
    for a, b in recent_pairs:
        for j, t in enumerate(trophies):
            if t["id"] == GONE_FISHING_ID:
                continue
            ra, rb = a["rates"][j], b["rates"][j]
            if ra is None or rb is None:
                continue
            jumps.append(logit(rb) - logit(ra))
    if not jumps:
        return PRIOR_JUMP_SD
    mean = sum(jumps) / len(jumps)
    var = sum((x - mean) ** 2 for x in jumps) / len(jumps)
    sd = math.sqrt(var)
    # Refresh cadence from the history's own state times.
    gaps = [b["h"] - a["h"] for a, b in zip(states, states[1:]) if b["h"] > a["h"]]
    interval = sum(gaps) / len(gaps) if gaps else FIRST_INTERVAL_H
    refreshes = horizon_h / max(interval, 1.0)
    return sd * math.sqrt(max(refreshes, 0.0))


def backtest_calibration(states, trophies):
    """Rolling-origin backtest on the case's own history.

    Each earlier state is an origin: fit the model on states up to it and
    predict the first later state at least 12 hours ahead, then compare the
    logit error against the spread the model would have used.  Returns
    (bias, multiplier), both shrunk toward their defaults when origins are
    few, so a short history calibrates weakly.
    """
    if len(states) < 3:
        return (0.0, 1.0)
    errors = []
    step = max(1, len(states) // 24)
    for origin in range(1, len(states) - 1, step):
        o_h = states[origin]["h"]
        later = None
        for s in states[origin + 1:]:
            if s["h"] - o_h >= 12.0:
                later = s
                break
        if later is None:
            continue
        med, s_fit, _ = forecast_medians(
            states[: origin + 1], trophies, later["h"], calib=(0.0, 1.0)
        )
        s_rw_o = RW_SIGMA * math.sqrt(max((later["h"] - o_h) / 24.0, 0.0))
        s_jump_o = jump_spread(states[: origin + 1], trophies, later["h"])
        for j, t in enumerate(trophies):
            actual = later["rates"][j]
            if actual is None:
                continue
            err = logit(actual) - logit(med[j])
            s_full = math.sqrt(s_rw_o * s_rw_o + s_jump_o * s_jump_o
                               + s_fit[j] * s_fit[j])
            errors.append((err, s_full))
    if len(errors) < 5:
        return (0.0, 1.0)
    n = len(errors)
    bias_raw = sum(e for e, _ in errors) / n
    bias = bias_raw * n / (n + 3.0)
    # Standardised squared error: 1 means the base spread is honest.
    mean_sq = sum((e / max(s, 0.05)) ** 2 for e, s in errors) / n
    mult_raw = math.sqrt(mean_sq)
    mult_raw = min(1.5, max(0.7, mult_raw))
    mult = 1.0 + (mult_raw - 1.0) * n / (n + 4.0)
    return (bias, mult)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    states, last_rates = compress_history(case)
    target_h = hours_since_release(parse_time(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    s_rw = RW_SIGMA * math.sqrt(max(horizon_days, 0.0))
    s_jump = jump_spread(states, trophies, target_h)
    calib = backtest_calibration(states, trophies)
    bias, mult = calib
    medians, s_fit_list, _ = forecast_medians(states, trophies, target_h, calib)
    # Story-chain monotonicity on the medians (backward min pass).
    by_id = {t["id"]: i for i, t in enumerate(trophies)}
    chain_idx = [by_id[i] for i in CHAIN if i in by_id]
    for a, b in zip(chain_idx, chain_idx[1:]):
        if medians[b] > medians[a]:
            medians[b] = medians[a]
    rows = []
    for j, trophy in enumerate(trophies):
        r = medians[j]
        if trophy["id"] == GONE_FISHING_ID:
            # Pinned at the print ceiling: every tracked player already has
            # it, and the backtest bias and the refresh spread do not apply
            # to a trophy that cannot move.  The spread only prices the
            # small chance that a few newly tracked players have not yet
            # finished the opening mission when the page is read.
            r = GONE_FISHING_Q50
            s = 0.3
            lg = logit(r)
        elif trophy["id"] == PLATINUM_ID:
            r = min(r, PLATINUM_CAP)
            r = clamp_pct(r)
            s = math.sqrt(s_rw * s_rw + s_jump * s_jump + s_fit_list[j] ** 2)
            s = min(SPREAD_MAX, max(SPREAD_MIN, s * mult))
            lg = logit(r) + bias
        else:
            r = clamp_pct(r)
            s = math.sqrt(s_rw * s_rw + s_jump * s_jump + s_fit_list[j] ** 2)
            s = min(SPREAD_MAX, max(SPREAD_MIN, s * mult))
            lg = logit(r) + bias
        qs = []
        prev = None
        for z in Z:
            q = round4(clamp_pct(inv_logit(lg + z * s)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
