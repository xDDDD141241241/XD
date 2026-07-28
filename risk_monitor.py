"""
Market Risk Monitor
===================
One refreshable panel that fuses everything we had scattered across separate
tools: VIX term structure, VVIX, MOVE, credit spreads, breadth, concentration
and cross-asset correlation.

Runs daily after the US close via GitHub Actions, writes docs/data.json,
which the dashboard in docs/index.html reads. No API keys anywhere.

Design:
  - every raw series is converted to its PERCENTILE over a trailing 3-year
    window, so a VIX of 22 and an OAS of 340bp are on the same footing
  - percentiles are oriented so that 100 always means "more dangerous"
  - the composite is a weighted mean of whatever survived the fetch;
    weights renormalise when a source fails, so one dead feed doesn't
    silently drag the score toward zero
  - regime is kept SEPARATE from the score. Score = stress level,
    regime = trend health. Regime multiplies, it does not add.

Run:
    python risk_monitor.py                 # full daily run
    python risk_monitor.py --no-breadth    # skip the 500-ticker download
    python risk_monitor.py --selftest      # offline, synthetic, no network
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import validate as V

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "docs", "data.json")

PCTL_WINDOW = 756        # ~3 years of sessions for percentile ranking
HISTORY_MAX = 1300
SP500_LIST = ("https://raw.githubusercontent.com/Ate329/top-us-stock-tickers"
              "/main/tickers/sp500.csv")
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRED_SERIES = {
    "hy_oas": "BAMLH0A0HYM2",      # high yield spread
    "ccc_oas": "BAMLH0A3HYC",      # CCC and lower -- the junk end
    "bb_oas": "BAMLH0A1HYBB",      # BB -- the quality end of junk
    "nfci": "NFCI",                # Chicago Fed financial conditions, weekly
    "fed_assets": "WALCL",         # balance sheet, weekly
    "tga": "WTREGEN",              # Treasury General Account
    "rrp": "RRPONTSYD",            # overnight reverse repo
}

YAHOO_SERIES = {
    "vix9d": "^VIX9D", "vix": "^VIX", "vix3m": "^VIX3M", "vvix": "^VVIX",
    "move": "^MOVE", "spy": "SPY", "rsp": "RSP", "tlt": "TLT",
    "xlu": "XLU", "xlp": "XLP",
}


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _yahoo(tickers: list[str], start: str = "2015-01-01") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(tickers, start=start, progress=False,
                     auto_adjust=False, threads=True)
    close = df["Close"] if "Close" in df else df
    return close if isinstance(close, pd.DataFrame) else close.to_frame()


def fetch_market(errors: list) -> dict[str, pd.Series]:
    out = {}
    try:
        px = _yahoo(list(YAHOO_SERIES.values()))
        for key, tic in YAHOO_SERIES.items():
            if tic in px.columns and px[tic].notna().sum() > 100:
                out[key] = px[tic].dropna()
            else:
                errors.append(f"empty series: {tic}")
    except Exception as exc:
        errors.append(f"yahoo failed: {exc}")
    return out


def _fred(series_id: str, errors: list) -> pd.Series | None:
    try:
        with urllib.request.urlopen(FRED.format(series_id), timeout=30) as r:
            raw = r.read().decode()
        df = pd.read_csv(io.StringIO(raw))
        df.columns = ["date", "value"]
        s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                      index=pd.to_datetime(df["date"]))
        return s.dropna()
    except Exception as exc:
        errors.append(f"fred {series_id} failed: {exc}")
        return None


def fetch_fred(errors: list) -> dict[str, pd.Series]:
    """Every FRED series in one place. A miss drops that gauge, nothing more."""
    return {k: v for k, v in
            ((k, _fred(sid, errors)) for k, sid in FRED_SERIES.items())
            if v is not None}


def fetch_breadth(errors: list) -> dict[str, pd.Series]:
    """
    Breadth built directly from S&P 500 constituents -- no paid ADD/S5TW feed.
    Constituent list is a daily-updated public CSV.
    """
    try:
        with urllib.request.urlopen(SP500_LIST, timeout=30) as r:
            tickers = pd.read_csv(io.StringIO(r.read().decode()))["symbol"]
        tickers = [t for t in tickers.dropna().astype(str)
                   if t.isalpha() or "." in t][:505]
        px = _yahoo(tickers, start="2016-01-01").dropna(how="all", axis=1)
        if px.shape[1] < 100:
            raise RuntimeError(f"only {px.shape[1]} constituents returned")

        # A batch pull can end on a partial row -- a handful of names printed,
        # the rest still empty. Every breadth measure then divides by that
        # handful and returns nonsense (0% above the 20-day, exactly zero net
        # new highs) which looks like a market event rather than a gap in the
        # data. Drop any date without broad coverage before computing anything.
        cov = px.notna().sum(axis=1)
        keep = cov >= max(100, int(cov.median() * 0.8))
        dropped = int((~keep).sum())
        px = px[keep]
        if px.empty:
            raise RuntimeError("no dates with sufficient constituent coverage")
        if dropped:
            errors.append(f"breadth: skipped {dropped} thinly-covered date(s), "
                          f"latest good {px.index[-1].date()}")

        ret = px.pct_change()
        adv = (ret > 0).sum(axis=1)
        dec = (ret < 0).sum(axis=1)
        ad_line = (adv - dec).cumsum()

        above20 = (px > px.rolling(20).mean()).sum(axis=1) / px.notna().sum(axis=1) * 100
        hi52 = (px >= px.rolling(252).max()).sum(axis=1)
        lo52 = (px <= px.rolling(252).min()).sum(axis=1)

        return {
            "ad_line": ad_line.dropna(),
            "pct_above_20": above20.dropna(),
            "nh_nl": (hi52 - lo52).dropna(),
            "adv_ratio": (adv / (adv + dec).replace(0, np.nan)).dropna(),
        }
    except Exception as exc:
        errors.append(f"breadth failed: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Indicator construction
# ---------------------------------------------------------------------------

def pctl(series: pd.Series, window: int = PCTL_WINDOW) -> float:
    """Rank of the latest value within the trailing window, 0-100."""
    s = series.dropna().iloc[-window:]
    if len(s) < 60:
        return float("nan")
    v = s.iloc[-1]
    # Midrank: the average of the strict and inclusive ranks. Matters for the
    # count gauges, which are mostly zero -- an inclusive rank would score a
    # perfectly healthy zero at the 90th percentile of danger. This also
    # matches pandas' rolling rank, so today's dot sits on the same scale as
    # the history curve behind it.
    return round(float(((s < v).mean() + (s <= v).mean()) / 2 * 100), 1)


def _spec(out, key, label, series, weight, invert=False, fmt="{:.2f}",
          note="", check="", scale=None, max_lag=4):
    """
    `scale=(lo, hi)` reads the gauge on a fixed absolute scale rather than a
    percentile. Percentiles are wrong for two shapes: a mostly-zero count
    (where any non-zero value ranks near the top) and a trending series
    (where the newest value ranks near the top by construction). Both produce
    an alarming number from an unremarkable reading.
    """
    if series is None:
        return
    ser = series.dropna()
    if ser.empty:
        return
    out.append({"id": key, "label": label, "series": ser, "weight": weight,
                "invert": invert, "fmt": fmt, "note": note, "check": check,
                "scale": scale, "max_lag": max_lag})


def _near_high(spy: pd.Series, lookback: int = 63, tol: float = 0.01) -> pd.Series:
    """True on days the index sat within tol of its own recent high."""
    return spy >= spy.rolling(lookback).max() * (1 - tol)


def indicator_specs(m: dict, fr: dict, br: dict) -> list[dict]:
    """
    Single source of truth for what a gauge IS. Both the daily reading and the
    historical backfill consume this, so the number on the page and the number
    in the history curve can never be computed two different ways.

    Every gauge carries a `check`: what to compare it against on the chart.
    A number with no stated comparison is decoration.
    """
    if fr is None:                        # every FRED series failed
        fr = {}
    elif isinstance(fr, pd.Series):       # tolerate the old single-series call
        fr = {"hy_oas": fr}
    out: list[dict] = []
    g, f = m.get, fr.get
    spy = g("spy")

    # ---- volatility -------------------------------------------------------
    if g("vix") is not None and g("vix3m") is not None:
        _spec(out, "ts_vix_vix3m", "VIX / VIX3M term structure",
              (m["vix"] / m["vix3m"]), 12, fmt="{:.3f}", scale=(0.85, 1.10),
              note="above 1.00 = backwardation, the classic risk-off trigger",
              check="Crossing 1.00 while the index is still near its highs is the "
                    "meaningful case. Above 1.00 during an active selloff is normal "
                    "and usually marks the panic, not the start of one.")
    if g("vix9d") is not None and g("vix") is not None:
        _spec(out, "ts_vix9d_vix", "VIX9D / VIX near-term stress",
              (m["vix9d"] / m["vix"]), 8, fmt="{:.3f}", scale=(0.85, 1.15),
              check="Rising while price is flat means a dated event is being priced "
                    "-- look for what is on the calendar rather than on the chart.")
    _spec(out, "vix", "VIX level", g("vix"), 8,
          check="Direction against price is what counts. VIX rising while the index "
                "also rises is rare and is worth stopping on.")
    _spec(out, "vvix", "VVIX (vol of vol)", g("vvix"), 8, fmt="{:.1f}",
          note="rises when people start paying up for tail protection",
          check="Rising VVIX with a flat VIX means someone is buying protection "
                "before any move shows in price. This leads more often than VIX does.")
    if g("vvix") is not None and g("vix") is not None:
        _spec(out, "vvix_vix", "VVIX / VIX", (m["vvix"] / m["vix"]), 6, fmt="{:.2f}",
              check="High ratio with a low VIX = complacent spot, expensive tails. "
                    "Protection is cheapest to own exactly here.")
    _spec(out, "move", "MOVE (rates volatility)", g("move"), 8, fmt="{:.1f}",
          check="If this leads VIX higher, the shock is coming from bonds or policy "
                "rather than from equities -- a different playbook.")

    # ---- credit -----------------------------------------------------------
    if f("hy_oas") is not None:
        _spec(out, "hy_oas", "High yield OAS", fr["hy_oas"], 12, fmt="{:.2f}",
              max_lag=6,
              note="credit is usually early to price real trouble",
              check="The level matters less than the direction. Check it every time "
                    "the index prints a new high.")
        _spec(out, "hy_oas_chg", "HY OAS, 20-day change", fr["hy_oas"].diff(20), 8,
              fmt="{:+.2f}", max_lag=6,
              check="Positive while the index makes new highs is the single most "
                    "reliable non-confirmation in this whole panel.")
    if f("ccc_oas") is not None and f("bb_oas") is not None:
        _spec(out, "quality_spread", "Quality spread (CCC minus BB)",
              (fr["ccc_oas"] - fr["bb_oas"]), 8, fmt="{:.2f}",
              # FRED truncated these series to three years in April 2026, so a
              # percentile here is ranked against a window that cannot see 2022
              # -- every drift wider reads as a record. Anchored instead:
              # ~3-5 is normal, 8+ is genuinely wide, 15+ is crisis.
              scale=(2.0, 14.0), max_lag=6,
              note="the junk end versus the decent end of high yield",
              check="Widening while equities hold up means risk appetite is leaving "
                    "from the bottom. The weakest borrowers crack first, and this "
                    "gap opens before the index-level spread does.")

    # ---- financial conditions and liquidity -------------------------------
    if f("nfci") is not None:
        _spec(out, "nfci", "Financial conditions (NFCI)", fr["nfci"], 6, fmt="{:+.2f}",
              max_lag=10,
              note="above zero = tighter than average",
              check="Tightening while the index is at highs is a warning. Loosening "
                    "during a selloff usually marks the end of it. Weekly, so treat "
                    "it as background rather than a trigger.")
    if all(f(k) is not None for k in ("fed_assets", "tga", "rrp")):
        # Forward-fill only as far as the slowest component actually reports.
        # Filling to today would stamp weekly data with today's date, making
        # this gauge look the freshest on the page and every other feed look
        # stale against it.
        last = min(fr[k].index[-1] for k in ("fed_assets", "tga", "rrp"))
        idx = pd.date_range(fr["fed_assets"].index[0], last, freq="D")
        parts = [fr[k].reindex(idx).ffill() for k in ("fed_assets", "tga", "rrp")]
        netliq = (parts[0] - parts[1] / 1000 - parts[2] / 1000).dropna()
        _spec(out, "net_liquidity", "Fed net liquidity, 63d change",
              netliq.pct_change(63) * 100, 4, invert=True, fmt="{:+.1f}%",
              max_lag=12,
              note="reserves less the Treasury account less reverse repo",
              check="Falling while the index makes highs means the advance is running "
                    "on positioning rather than on new money. Short history and "
                    "contested -- treat as supporting evidence, never as a trigger.")

    # ---- structure --------------------------------------------------------
    if spy is not None and g("rsp") is not None:
        _spec(out, "concentration", "Cap-weight vs equal-weight, 63d",
              (m["spy"] / m["rsp"]).pct_change(63) * 100, 8, fmt="{:+.1f}%",
              note="rising = the index is being carried by fewer names",
              check="Compare directly with the index. Both rising together means the "
                    "gain is narrowing into a handful of names.")
    if spy is not None and g("tlt") is not None:
        _spec(out, "corr_spy_tlt", "Stock/bond correlation, 63d",
              m["spy"].pct_change().rolling(63).corr(m["tlt"].pct_change()), 6,
              fmt="{:+.2f}", scale=(-1.0, 1.0), note="above zero means bonds stop cushioning equities",
              check="Check this before you need the hedge, not after. It says nothing "
                    "about whether a drawdown is coming, only what it will cost you.")
    if spy is not None and g("xlu") is not None and g("xlp") is not None:
        defens = ((m["xlu"] / m["xlu"].iloc[0] + m["xlp"] / m["xlp"].iloc[0]) / 2)
        _spec(out, "defensive_rotation", "Defensives vs index, 63d",
              (defens / (m["spy"] / m["spy"].iloc[0])).pct_change(63) * 100, 4,
              fmt="{:+.1f}%", note="utilities and staples relative to the index",
              check="Rising while the index makes highs means money is moving to "
                    "safety inside the rally. Confirms more often than it leads.")

    # ---- breadth ----------------------------------------------------------
    if br:
        ad = br["ad_line"]
        adz = (ad - ad.rolling(63).mean()) / ad.rolling(63).std()
        _spec(out, "breadth_ad_z", "Advance/decline line, 63d z-score", adz, 12,
              invert=True, fmt="{:+.2f}", scale=(-2.5, 2.5),
              note="low = participation narrowing under the surface",
              check="Negative while the index sits at highs is the divergence that "
                    "matters. Negative during a pullback is just a pullback.")
        _spec(out, "pct_above_20", "% of S&P 500 above 20-day average",
              br["pct_above_20"], 8, invert=True, fmt="{:.0f}%", scale=(0, 100),
              check="Short-horizon. Under 15% is washout territory -- check whether "
                    "price has actually broken structure or merely dipped.")
        _spec(out, "nh_nl", "New 52w highs minus new lows", br["nh_nl"], 6,
              invert=True, fmt="{:+.0f}", scale=(-150, 150),
              check="Negative while the index is at a new high is the textbook "
                    "non-confirmation. It was present at 1972, 2000, 2007 and 2021.")

        # ---- the divergence constructs: price and internals disagreeing ----
        if spy is not None:
            hi = _near_high(spy)
            a = adz.reindex(hi.index).ffill()
            _spec(out, "breadth_divergence", "Breadth non-confirmation, 63d count",
                  (hi & (a < 0)).rolling(63).sum(), 10, fmt="{:.0f}",
                  scale=(0, 25),
                  note="days near a high with breadth already negative",
                  check="This IS the price comparison, counted for you. A rising "
                        "count is the November 2021 pattern: index still making "
                        "highs, participation already gone. Zero is healthy.")
            if f("hy_oas") is not None:
                oc = fr["hy_oas"].diff(20).reindex(hi.index).ffill()
                _spec(out, "credit_divergence", "Credit non-confirmation, 63d count",
                      (hi & (oc > 0)).rolling(63).sum(), 10, fmt="{:.0f}",
                      scale=(0, 25),
                      note="days near a high with credit spreads widening",
                      check="Counts the days equities made highs while credit "
                            "disagreed. This led 2000, 2007, 2018 and 2021-22. "
                            "A count climbing off zero is the thing to watch.")
    return out


def build_indicators(m: dict, oas, br: dict) -> list[dict]:
    """Today's reading for each gauge, with a freshness stamp."""
    specs = indicator_specs(m, oas, br)
    if not specs:
        return []
    newest = max(sp["series"].index[-1] for sp in specs)

    out = []
    for sp in specs:
        s = sp["series"]
        if sp.get("scale"):
            lo, hi = sp["scale"]
            p = float(np.clip((float(s.iloc[-1]) - lo) / (hi - lo), 0, 1) * 100)
        else:
            p = pctl(s)
        if p != p:
            continue
        as_of = s.index[-1]
        val = float(s.iloc[-1])
        # A pinned 0% / 100% breadth print alongside a calm VIX is a data
        # artifact, not a market state. Drop it rather than let it dominate.
        if sp["id"] == "pct_above_20" and val in (0.0, 100.0):
            continue
        out.append({
            "id": sp["id"], "label": sp["label"],
            "value": round(float(s.iloc[-1]), 4),
            "display": sp["fmt"].format(float(s.iloc[-1])),
            "percentile": round(100 - p, 1) if sp["invert"] else round(p, 1),
            "weight": sp["weight"], "note": sp["note"],
            "check": sp["check"],
            "as_of": str(as_of.date()),
            "basis": ("absolute scale" if sp.get("scale")
                      else f"ranked vs {min(len(s), PCTL_WINDOW)} sessions"),
            "inverted": bool(sp["invert"]),
            "thin_history": bool(not sp.get("scale")
                                 and len(s) < PCTL_WINDOW),
            # a feed that has quietly stopped updating still returns a number;
            # this is what stops that number being read as current
            # Staleness is judged against each gauge's OWN update cadence.
            # A weekly series is not stale at six days -- it is weekly. A flat
            # daily rule flagged NFCI and net liquidity every single run, and
            # a warning that fires every day gets ignored on the day it counts.
            "stale": bool((newest - as_of).days > sp["max_lag"]),
            # A gauge that stopped updating a fortnight ago is not a reading,
            # it is a fossil. Keep showing it, but stop it voting.
            "excluded": bool((newest - as_of).days > 3 * sp["max_lag"]),
        })
    return out


def historical_scores(m: dict, oas, br: dict,
                      window: int = PCTL_WINDOW) -> list[dict]:
    """
    Recompute the composite for every day in the past, using only the
    information available on that day: each gauge is ranked inside its own
    trailing window as it stood then. No lookahead.
    """
    specs = indicator_specs(m, oas, br)
    if not specs:
        return []

    ranks, weights = {}, {}
    for sp in specs:
        s = sp["series"].dropna()
        if len(s) < window // 4:
            continue
        if sp.get("scale"):
            lo, hi = sp["scale"]
            r = ((s - lo) / (hi - lo)).clip(0, 1) * 100
        else:
            r = s.rolling(window, min_periods=120).rank(pct=True) * 100
        ranks[sp["id"]] = (100 - r) if sp["invert"] else r
        weights[sp["id"]] = sp["weight"]
    if not ranks:
        return []

    R = pd.DataFrame(ranks)
    # Score only on real trading days. One gauge (net liquidity) is built on a
    # calendar index, and its weekend rows were producing scores computed from
    # a single input -- which is what turned the history curve into noise.
    if m.get("spy") is not None:
        R = R.reindex(m["spy"].index.intersection(R.index))
    R = R.dropna(how="all")

    W = pd.Series(weights)
    present = R.notna()
    den = (present * W).sum(axis=1)
    # and never publish a score built on a thin slice of the panel
    den = den.where(den >= 0.6 * W.sum())
    num = (R.fillna(0) * W).sum(axis=1)
    score = (num / den).dropna()

    n_live = present.sum(axis=1).reindex(score.index)
    return [{"date": str(d.date()), "score": round(float(v), 1),
             "n": int(n_live.loc[d])}
            for d, v in score.items()]


def composite_score(indicators: list[dict]) -> float:
    """Weighted mean of percentiles, renormalised over surviving inputs."""
    ok = [i for i in indicators
          if i["percentile"] == i["percentile"] and not i.get("excluded")]
    if not ok:
        return float("nan")
    w = sum(i["weight"] for i in ok)
    return round(sum(i["percentile"] * i["weight"] for i in ok) / w, 1)


# ---------------------------------------------------------------------------
# Regime and signals -- kept separate from the score on purpose
# ---------------------------------------------------------------------------

def trend_table(spy: pd.Series | None) -> list[dict]:
    """
    Price against several horizons rather than one. 'Above trend' is
    meaningless without saying which trend -- a market can sit above its
    200-day and below its 9-day at the same time, and usually does during
    an ordinary pullback.
    """
    if spy is None or len(spy) < 210:
        return []
    last = float(spy.iloc[-1])
    out = []
    for span, kind in ((9, "ema"), (21, "ema"), (50, "sma"), (200, "sma")):
        ref = (spy.ewm(span=span, adjust=False).mean() if kind == "ema"
               else spy.rolling(span).mean())
        r = float(ref.iloc[-1])
        out.append({"span": span, "kind": kind, "above": bool(last > r),
                    "dist_pct": round((last / r - 1) * 100, 2)})
    return out


def classify_regime(spy: pd.Series | None, br: dict) -> dict:
    trend_up = ad_ok = True
    if spy is not None and len(spy) > 200:
        trend_up = bool(spy.iloc[-1] > spy.rolling(200).mean().iloc[-1])
    if br:
        ad = br["ad_line"]
        z = ((ad - ad.rolling(63).mean()) / ad.rolling(63).std()).dropna()
        ad_ok = bool(z.iloc[-1] > -0.5) if len(z) else True

    if trend_up and ad_ok:
        name = "BULL"
        desc = "above the 200-day and participation is broad"
    elif trend_up and not ad_ok:
        name = "DETERIORATING"
        desc = "above the 200-day, but participation is narrowing"
    elif not trend_up and ad_ok:
        name = "CORRECTION"
        desc = "below the 200-day, breadth not confirming damage"
    else:
        name = "BEAR"
        desc = "below the 200-day and participation is broken"
    return {"name": name, "description": desc,
            "trend_up": trend_up, "breadth_ok": ad_ok,
            "horizon": "200-day"}


def build_signals(m: dict, br: dict) -> list[dict]:
    sig, spy = [], m.get("spy")

    if m.get("vix") is not None and m.get("vix3m") is not None:
        ts = (m["vix"] / m["vix3m"]).dropna()
        if len(ts) > 25:
            back = bool((ts.iloc[-20:] > 1.0).any())
            sig.append({"id": "ts_swing_low", "label": "Volatility term structure reset",
                        "fired": bool(back and ts.iloc[-1] < 0.95),
                        "meaning": "panic priced in and now unwinding — historically a buy window"})
            sig.append({"id": "ts_swing_high", "label": "Term structure inverted",
                        "fired": bool(ts.iloc[-1] > 1.0),
                        "meaning": "near-term risk priced above 3-month — risk-off trigger"})

    if br:
        pa = br["pct_above_20"]
        sig.append({"id": "washout_buy", "label": "Breadth washout",
                    "fired": bool(pa.iloc[-10:].min() < 15 and pa.iloc[-1] > pa.iloc[-4]),
                    "meaning": "selling exhausted, breadth turning up off the floor"})
        ar = br["adv_ratio"].ewm(span=10, adjust=False).mean()
        if len(ar) > 12:
            sig.append({"id": "zweig_thrust", "label": "Breadth thrust",
                        "fired": bool(ar.iloc[-11:-1].min() < 0.40 and ar.iloc[-1] > 0.615),
                        "meaning": "rare, powerful buying surge — strongly bullish"})

    if all(k in m for k in ("vvix", "spy")) and br:
        v = m["vvix"]
        vz = ((v - v.rolling(252).mean()) / v.rolling(252).std()).dropna()
        near_high = bool(spy.iloc[-1] >= spy.rolling(63).max().iloc[-1] * 0.98)
        corr_up = False
        if m.get("tlt") is not None:
            c = m["spy"].pct_change().rolling(63).corr(m["tlt"].pct_change()).dropna()
            corr_up = bool(len(c) > 25 and c.iloc[-1] > 0 and c.iloc[-1] > c.iloc[-21])
        sig.append({"id": "major_top", "label": "Major top cluster",
                    "fired": bool(len(vz) and vz.iloc[-1] > 1.5 and near_high and corr_up),
                    "meaning": "tail hedging bid, bonds no longer cushioning, price at highs"})
    return sig


# ---------------------------------------------------------------------------

def prev_payload() -> dict | None:
    try:
        with open(DATA_PATH, encoding="utf-8") as f:
            p = json.load(f)
        return None if p.get("demo") else p
    except Exception:
        return None


def load_history() -> list:
    """
    Previous readings, unless the file still holds the shipped sample. That
    sample carries 180 fabricated points; inheriting them would have drawn a
    curve of invented history under a real reading.
    """
    try:
        with open(DATA_PATH, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("demo"):
            return []
        return prev.get("history", [])
    except Exception:
        return []


def assemble(m, oas, br, errors, backfill: bool = False) -> dict:
    notes: list = []
    ind = build_indicators(m, oas, br)
    score = composite_score(ind)
    regime = classify_regime(m.get("spy"), br)
    sigs = build_signals(m, br)

    # The coverage gate firing is the system working, not a feed failing.
    # Leaving it in `errors` painted a red banner over a routine, correct
    # outcome -- and a warning that cries wolf daily gets ignored when it
    # matters.
    routine = [e for e in errors if e.startswith("breadth: skipped")]
    for e in routine:
        errors.remove(e)
        notes.append(e.replace("breadth: skipped", "breadth data lagged by"))

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hist = load_history()

    # The history is ALWAYS recomputed from source rather than accumulated.
    # Accumulating meant any bad entry -- notably the fabricated points in the
    # shipped sample file -- stayed in the curve forever, and nothing in the
    # stored record showed it was fake. Recomputing is cheap, deterministic,
    # and self-healing: a wrong past reading can only survive one run.
    computed = historical_scores(m, oas, br)
    if computed:
        hist = computed
        notes.append(f"score history rebuilt from {len(computed)} sessions")
    elif backfill:
        notes.append("history rebuild produced nothing -- kept stored readings")

    hist = [h for h in hist if h.get("date") != today]
    hist.append({"date": today, "score": score, "regime": regime["name"]})

    spy = m.get("spy")
    payload = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "score": score,
        "band": ("CALM" if score < 35 else "ELEVATED" if score < 60
                 else "STRESSED" if score < 80 else "ACUTE"),
        "regime": regime,
        "trend": trend_table(m.get("spy")),
        "indicators": sorted(ind, key=lambda i: -i["percentile"]),
        "signals": sigs,
        "spx_proxy": None if spy is None else {
            "last": round(float(spy.iloc[-1]), 2),
            "pct_from_high": round(float(spy.iloc[-1] / spy.cummax().iloc[-1] - 1) * 100, 2),
        },
        "history": hist[-HISTORY_MAX:],
        "notes": notes,
        "errors": errors,
    }
    payload["quality"] = V.quality_report(payload, prev_payload())
    payload["events"] = V.event_report(payload["history"])
    payload["pivots"] = V.pivot_report(payload["history"])
    _h = [x["score"] for x in payload["history"]
          if isinstance(x.get("score"), (int, float))]
    payload["score_percentile"] = (
        round(float(np.mean([v <= payload["score"] for v in _h]) * 100), 1)
        if _h else None)
    return payload


def write(payload: dict) -> None:
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-breadth", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--backfill", action="store_true",
                   help="recompute the whole score history from scratch")
    a = p.parse_args()

    if a.selftest:
        from selftest import run_selftest
        return run_selftest()

    errors: list = []
    m = fetch_market(errors)
    oas = fetch_fred(errors)
    br = {} if a.no_breadth else fetch_breadth(errors)

    if not m:
        print("no market data at all — aborting", file=sys.stderr)
        return 1

    payload = assemble(m, oas, br, errors, backfill=a.backfill)
    write(payload)

    print(f"score {payload['score']}  band {payload['band']}  "
          f"regime {payload['regime']['name']}")
    for i in payload["indicators"]:
        print(f"  {i['id']:<16} {i['display']:>9}   pctl {i['percentile']:>5}")
    for s in payload["signals"]:
        if s["fired"]:
            print(f"  SIGNAL: {s['label']}")
    for e in errors:
        print(f"  ! {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
