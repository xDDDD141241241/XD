"""
Trust layer for the risk monitor.

Two jobs, because "is this number believable?" has two halves.

1. quality_report() runs EVERY night and asks whether tonight's readings are
   internally coherent, fresh, and physically possible. It produces a single
   status the page shows before the score itself. The failure that matters is
   never a number that looks obviously wrong -- it is a number that looks
   completely normal and isn't. So these checks target plausibility, not
   formatting.

2. event_report() runs against the backfilled history and asks the only
   question that establishes the tool works: did the score actually rise
   during known stress, and how often did it cry wolf? A monitor that never
   lit up during a real crisis is decoration, however tidy its arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Physically possible ranges. Not "unusual" -- impossible. Anything outside
# these is a broken feed, not a market event.
HARD_BANDS = {
    "vix": (5, 200), "vvix": (40, 300), "move": (20, 400),
    "ts_vix_vix3m": (0.35, 3.0), "ts_vix9d_vix": (0.35, 3.0),
    "vvix_vix": (1.0, 20.0), "hy_oas": (1.0, 30.0), "hy_oas_chg": (-15, 15),
    "pct_above_20": (0, 100), "nh_nl": (-505, 505),
    "breadth_ad_z": (-8, 8), "corr_spy_tlt": (-1.0, 1.0),
    "concentration": (-60, 60),
    "quality_spread": (0.0, 25.0), "nfci": (-3.0, 8.0),
    "net_liquidity": (-40, 40), "defensive_rotation": (-40, 40),
    "breadth_divergence": (0, 63), "credit_divergence": (0, 63),
}

# Known stress episodes. Deliberately chosen BEFORE looking at any output,
# and dated to the start of each move rather than its low, so the test is
# "did it warn", not "did it notice afterwards".
EPISODES = [
    ("2018-02-05", "Volmageddon"),
    ("2018-12-19", "Q4 2018 selloff"),
    ("2020-02-24", "Covid crash"),
    ("2022-01-21", "2022 bear begins"),
    ("2022-06-13", "2022 June leg"),
    ("2022-10-11", "2022 October low"),
    ("2023-03-10", "SVB / regional banks"),
    ("2024-08-05", "Yen carry unwind"),
    ("2025-04-07", "Tariff selloff"),
]


def _chk(out, name, ok, detail, hard=False):
    out.append({"name": name, "status": "pass" if ok else ("fail" if hard else "warn"),
                "detail": detail})


def quality_report(payload: dict, prev: dict | None = None) -> dict:
    """Nightly self-audit. Returns status OK / WARN / FAIL plus the reasons."""
    checks: list = []
    ind = {i["id"]: i for i in payload.get("indicators", [])}

    # --- 1. physically possible?
    bad = []
    for key, i in ind.items():
        lo, hi = HARD_BANDS.get(key, (-np.inf, np.inf))
        if not (lo <= i["value"] <= hi):
            bad.append(f"{key}={i['value']}")
    _chk(checks, "Values within possible ranges", not bad,
         "all gauges physically plausible" if not bad else "impossible: " + ", ".join(bad),
         hard=True)

    # --- 2. the failure that already bit us once
    pinned = []
    if "pct_above_20" in ind and ind["pct_above_20"]["value"] in (0.0, 100.0):
        pinned.append("pct_above_20 pinned at an extreme")
    if "nh_nl" in ind and ind["nh_nl"]["value"] == 0 and "pct_above_20" in ind:
        pinned.append("net new highs exactly zero")
    _chk(checks, "Breadth not pinned at a suspicious extreme", not pinned,
         "breadth readings look like real data" if not pinned else "; ".join(pinned),
         hard=True)

    # --- 3. do the derived ratios reconcile with their parts?
    recon = True
    if all(k in ind for k in ("vvix", "vix", "vvix_vix")):
        implied = ind["vvix"]["value"] / ind["vix"]["value"]
        recon = abs(implied - ind["vvix_vix"]["value"]) < 0.05
    _chk(checks, "Derived ratios reconcile", recon,
         "VVIX/VIX matches its components" if recon
         else "VVIX/VIX disagrees with VVIX and VIX -- likely a date misalignment",
         hard=True)

    # --- 4. do the breadth gauges agree with each other?
    agree = True
    if "pct_above_20" in ind and "breadth_ad_z" in ind:
        thin = ind["pct_above_20"]["value"] < 25
        adz = ind["breadth_ad_z"]["value"]
        agree = not (thin and adz > 0.5)
    _chk(checks, "Breadth gauges agree", agree,
         "participation measures tell the same story" if agree
         else "very few stocks above their 20-day, yet the A/D line is strong -- one of them is wrong")

    # --- 5. freshness
    stale = [i["id"] for i in payload.get("indicators", []) if i.get("stale")]
    _chk(checks, "Feeds current", not stale,
         "all feeds current" if not stale
         else f"{len(stale)} stale: {', '.join(stale)}")

    # --- 6. is the score doing something it cannot justify?
    jump_ok, jump_detail = True, "no prior reading to compare"
    if prev and isinstance(prev.get("score"), (int, float)):
        d = payload["score"] - prev["score"]
        vixp = ind.get("vix", {}).get("percentile")
        pv = {i["id"]: i for i in prev.get("indicators", [])}.get("vix", {}).get("percentile")
        moved = (vixp is not None and pv is not None and abs(vixp - pv) > 15)
        jump_ok = abs(d) < 20 or moved
        jump_detail = (f"score moved {d:+.1f} overnight"
                       + ("" if jump_ok else " with no matching move in volatility"))
    _chk(checks, "Score change explainable", jump_ok, jump_detail)

    # --- 7. enough gauges to mean anything
    n = len(ind)
    _chk(checks, "Enough gauges reporting", n >= 9,
         f"{n} of 13 gauges reporting", hard=n < 7)

    # --- 8. history integrity
    h = payload.get("history", [])
    dates = [x["date"] for x in h]
    ok_hist = len(h) > 100 and len(dates) == len(set(dates)) and dates == sorted(dates)
    _chk(checks, "History intact", ok_hist,
         f"{len(h)} sessions, no gaps or duplicates" if ok_hist
         else "history is short, duplicated or out of order")

    fails = [c for c in checks if c["status"] == "fail"]
    warns = [c for c in checks if c["status"] == "warn"]
    status = "FAIL" if fails else ("WARN" if warns else "OK")
    return {
        "status": status,
        "headline": ("Readings look trustworthy." if status == "OK" else
                     "Something is off — read the notes before trusting the score."
                     if status == "WARN" else
                     "Do not trust tonight's score. A feed is broken."),
        "checks": checks,
        "n_failed": len(fails), "n_warned": len(warns),
    }


def event_report(history: list, episodes=EPISODES, window: int = 10) -> dict:
    """
    Did the score rise during known stress, and how often did it fire without
    any? Both halves matter -- a gauge pinned at 90 every day would 'catch'
    every crisis and be useless.
    """
    if not history or len(history) < 250:
        return {"status": "insufficient history"}

    s = pd.Series({pd.Timestamp(h["date"]): h["score"] for h in history
                   if isinstance(h.get("score"), (int, float))}).sort_index()
    if s.empty:
        return {"status": "no usable history"}

    p80, p90 = s.quantile(0.80), s.quantile(0.90)
    rows, covered = [], []
    for date, name in episodes:
        d = pd.Timestamp(date)
        if d < s.index[0] or d > s.index[-1]:
            continue
        w = s.loc[d - pd.Timedelta(days=window): d + pd.Timedelta(days=window)]
        if w.empty:
            continue
        peak = float(w.max())
        rows.append({"date": date, "name": name, "peak_score": round(peak, 1),
                     "percentile": round(float((s <= peak).mean() * 100), 1),
                     "flagged": bool(peak >= p80)})
        covered.append(d)

    hits = sum(r["flagged"] for r in rows)

    # false alarms: days above the 90th percentile that sit far from any episode
    if covered:
        near = pd.Series(False, index=s.index)
        for d in covered:
            near |= (s.index >= d - pd.Timedelta(days=45)) & (s.index <= d + pd.Timedelta(days=45))
        high = s >= p90
        false_days = int((high & ~near).sum())
        high_days = int(high.sum())
    else:
        false_days = high_days = 0

    return {
        "status": "ok",
        "window_days": window,
        "history_from": str(s.index[0].date()), "history_to": str(s.index[-1].date()),
        "threshold_p80": round(float(p80), 1), "threshold_p90": round(float(p90), 1),
        "episodes_testable": len(rows), "episodes_flagged": hits,
        "episodes": rows,
        "high_days": high_days,
        "high_days_away_from_stress": false_days,
        "false_alarm_share": round(false_days / high_days, 2) if high_days else None,
    }


# ---------------------------------------------------------------------------
# Turning points
# ---------------------------------------------------------------------------
# Confirmed pivots, fixed in advance. Lows and tops are judged by DIFFERENT
# standards on purpose: at a low, a stress gauge is supposed to be pegged high
# (vol spikes at lows almost mechanically -- an easy test it should never
# fail). At a top it is supposed to be rising, or at least not asleep, which
# is genuinely hard and is where this kind of tool usually fails. Reporting
# them together, with tops judged on lead behaviour rather than level, is what
# stops the easy half flattering the hard half.

PIVOT_LOWS = [
    ("2018-12-24", "Q4 2018 low"),
    ("2020-03-23", "Covid low"),
    ("2022-10-12", "Oct 2022 bear low"),
    ("2023-03-13", "SVB low"),
    ("2024-08-05", "Yen carry low"),
    ("2025-04-08", "Tariff low"),
]

PIVOT_TOPS = [
    ("2018-09-20", "Sep 2018 top"),
    ("2020-02-19", "Pre-Covid top"),
    ("2021-11-08", "Nov 2021 breadth top"),
    ("2022-01-03", "Jan 2022 index top"),
    ("2025-02-19", "Feb 2025 top"),
]


def pivot_report(history: list, lead: int = 42) -> dict:
    """
    For each confirmed turning point: where was the score, and had it been
    moving beforehand? The second question is the one that matters -- a gauge
    that only reacts after the fact tells you nothing you could have traded.
    """
    if not history or len(history) < 250:
        return {"status": "insufficient history"}

    s = pd.Series({pd.Timestamp(h["date"]): h["score"] for h in history
                   if isinstance(h.get("score"), (int, float))}).sort_index()
    if s.empty:
        return {"status": "no usable history"}

    def rank(v):
        return round(float((s <= v).mean() * 100), 1)

    def at(date, name, kind):
        d = pd.Timestamp(date)
        if d < s.index[0] or d > s.index[-1]:
            return None
        w = s.loc[d - pd.Timedelta(days=7): d + pd.Timedelta(days=7)]
        if w.empty:
            return None
        here = float(w.mean())
        before = s.loc[d - pd.Timedelta(days=lead): d - pd.Timedelta(days=7)]
        prior = float(before.mean()) if not before.empty else float("nan")
        chg = here - prior if prior == prior else float("nan")

        if kind == "low":
            # a stress gauge should be near its extreme when price bottoms
            useful = rank(here) >= 75
            verdict = "pegged high at the low" if useful else "did not register the low"
        else:
            # at a top, level is not the test -- direction is
            useful = bool(chg == chg and chg > 4) or rank(here) >= 60
            verdict = ("rising into the top" if chg == chg and chg > 4
                       else "elevated at the top" if rank(here) >= 60
                       else "asleep at the top")
        return {"date": date, "name": name, "kind": kind,
                "score": round(here, 1), "percentile": rank(here),
                "change_before": None if chg != chg else round(chg, 1),
                "useful": bool(useful), "verdict": verdict}

    rows = [r for r in (
        [at(d, n, "low") for d, n in PIVOT_LOWS] +
        [at(d, n, "top") for d, n in PIVOT_TOPS]) if r]

    lows = [r for r in rows if r["kind"] == "low"]
    tops = [r for r in rows if r["kind"] == "top"]
    return {
        "status": "ok",
        "history_from": str(s.index[0].date()),
        "lows_tested": len(lows), "lows_useful": sum(r["useful"] for r in lows),
        "tops_tested": len(tops), "tops_useful": sum(r["useful"] for r in tops),
        "pivots": rows,
        "note": ("Lows are the easy test and should be near-perfect. Tops are the "
                 "hard test — judge the tool on that column."),
    }
