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

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "docs", "data.json")

PCTL_WINDOW = 756        # ~3 years of sessions for percentile ranking
HISTORY_MAX = 1300
SP500_LIST = ("https://raw.githubusercontent.com/Ate329/top-us-stock-tickers"
              "/main/tickers/sp500.csv")
FRED_HY_OAS = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
               "?id=BAMLH0A0HYM2")

YAHOO_SERIES = {
    "vix9d": "^VIX9D", "vix": "^VIX", "vix3m": "^VIX3M", "vvix": "^VVIX",
    "move": "^MOVE", "spy": "SPY", "rsp": "RSP", "tlt": "TLT",
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


def fetch_hy_oas(errors: list) -> pd.Series | None:
    try:
        with urllib.request.urlopen(FRED_HY_OAS, timeout=30) as r:
            raw = r.read().decode()
        df = pd.read_csv(io.StringIO(raw))
        df.columns = ["date", "value"]
        s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                      index=pd.to_datetime(df["date"]))
        return s.dropna()
    except Exception as exc:
        errors.append(f"fred hy oas failed: {exc}")
        return None


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
        px = _yahoo(tickers, start="2018-01-01").dropna(how="all", axis=1)
        if px.shape[1] < 100:
            raise RuntimeError(f"only {px.shape[1]} constituents returned")

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
    return round(float((s <= s.iloc[-1]).mean() * 100), 1)


def _spec(out, key, label, series, weight, invert=False, fmt="{:.2f}", note=""):
    if series is None:
        return
    s = series.dropna()
    if s.empty:
        return
    out.append({"id": key, "label": label, "series": s, "weight": weight,
                "invert": invert, "fmt": fmt, "note": note})


def indicator_specs(m: dict, oas: pd.Series | None, br: dict) -> list[dict]:
    """
    Single source of truth for what a gauge IS. Both the daily reading and the
    historical backfill consume this, so the number on the page and the number
    in the history curve can never be computed two different ways.
    """
    out: list[dict] = []
    g = m.get

    if g("vix") is not None and g("vix3m") is not None:
        _spec(out, "ts_vix_vix3m", "VIX / VIX3M term structure",
              (m["vix"] / m["vix3m"]), 12, fmt="{:.3f}",
              note="above 1.00 = backwardation, the classic risk-off trigger")
    if g("vix9d") is not None and g("vix") is not None:
        _spec(out, "ts_vix9d_vix", "VIX9D / VIX near-term stress",
              (m["vix9d"] / m["vix"]), 8, fmt="{:.3f}")
    _spec(out, "vix", "VIX level", g("vix"), 8)
    _spec(out, "vvix", "VVIX (vol of vol)", g("vvix"), 8, fmt="{:.1f}",
          note="rises when people start paying up for tail protection")
    if g("vvix") is not None and g("vix") is not None:
        _spec(out, "vvix_vix", "VVIX / VIX", (m["vvix"] / m["vix"]), 6, fmt="{:.2f}")
    _spec(out, "move", "MOVE (rates volatility)", g("move"), 8, fmt="{:.1f}")

    if oas is not None:
        _spec(out, "hy_oas", "High yield OAS", oas, 12, fmt="{:.2f}",
              note="credit is usually early to price real trouble")
        _spec(out, "hy_oas_chg", "HY OAS, 20-day change", oas.diff(20), 8,
              fmt="{:+.2f}")

    if g("spy") is not None and g("rsp") is not None:
        _spec(out, "concentration", "Cap-weight vs equal-weight, 63d",
              (m["spy"] / m["rsp"]).pct_change(63) * 100, 8, fmt="{:+.1f}%",
              note="rising = the index is being carried by fewer names")
    if g("spy") is not None and g("tlt") is not None:
        _spec(out, "corr_spy_tlt", "Stock/bond correlation, 63d",
              m["spy"].pct_change().rolling(63).corr(m["tlt"].pct_change()), 6,
              fmt="{:+.2f}", note="above zero means bonds stop cushioning equities")

    if br:
        ad = br["ad_line"]
        _spec(out, "breadth_ad_z", "Advance/decline line, 63d z-score",
              (ad - ad.rolling(63).mean()) / ad.rolling(63).std(), 12,
              invert=True, fmt="{:+.2f}",
              note="low = participation narrowing under the surface")
        _spec(out, "pct_above_20", "% of S&P 500 above 20-day average",
              br["pct_above_20"], 8, invert=True, fmt="{:.0f}%")
        _spec(out, "nh_nl", "New 52w highs minus new lows", br["nh_nl"], 6,
              invert=True, fmt="{:+.0f}")
    return out


def build_indicators(m: dict, oas: pd.Series | None, br: dict) -> list[dict]:
    """Today's reading for each gauge, with a freshness stamp."""
    specs = indicator_specs(m, oas, br)
    if not specs:
        return []
    newest = max(sp["series"].index[-1] for sp in specs)

    out = []
    for sp in specs:
        s = sp["series"]
        p = pctl(s)
        if p != p:
            continue
        as_of = s.index[-1]
        out.append({
            "id": sp["id"], "label": sp["label"],
            "value": round(float(s.iloc[-1]), 4),
            "display": sp["fmt"].format(float(s.iloc[-1])),
            "percentile": round(100 - p, 1) if sp["invert"] else p,
            "weight": sp["weight"], "note": sp["note"],
            "as_of": str(as_of.date()),
            # a feed that has quietly stopped updating still returns a number;
            # this is what stops that number being read as current
            "stale": bool((newest - as_of).days > 4),
        })
    return out


def historical_scores(m: dict, oas: pd.Series | None, br: dict,
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
        r = s.rolling(window, min_periods=120).rank(pct=True) * 100
        ranks[sp["id"]] = (100 - r) if sp["invert"] else r
        weights[sp["id"]] = sp["weight"]
    if not ranks:
        return []

    R = pd.DataFrame(ranks).dropna(how="all")
    W = pd.Series(weights)
    present = R.notna()
    num = (R.fillna(0) * W).sum(axis=1)
    den = (present * W).sum(axis=1).replace(0, np.nan)
    score = (num / den).dropna()

    return [{"date": str(d.date()), "score": round(float(v), 1)}
            for d, v in score.items()]


def composite_score(indicators: list[dict]) -> float:
    """Weighted mean of percentiles, renormalised over surviving inputs."""
    ok = [i for i in indicators if i["percentile"] == i["percentile"]]
    if not ok:
        return float("nan")
    w = sum(i["weight"] for i in ok)
    return round(sum(i["percentile"] * i["weight"] for i in ok) / w, 1)


# ---------------------------------------------------------------------------
# Regime and signals -- kept separate from the score on purpose
# ---------------------------------------------------------------------------

def classify_regime(spy: pd.Series | None, br: dict) -> dict:
    trend_up = ad_ok = True
    if spy is not None and len(spy) > 200:
        trend_up = bool(spy.iloc[-1] > spy.rolling(200).mean().iloc[-1])
    if br:
        ad = br["ad_line"]
        z = ((ad - ad.rolling(63).mean()) / ad.rolling(63).std()).dropna()
        ad_ok = bool(z.iloc[-1] > -0.5) if len(z) else True

    if trend_up and ad_ok:
        name, desc = "BULL", "trend and participation both intact"
    elif trend_up and not ad_ok:
        name, desc = "DETERIORATING", "price still up, participation narrowing"
    elif not trend_up and ad_ok:
        name, desc = "CORRECTION", "price below trend, breadth not confirming damage"
    else:
        name, desc = "BEAR", "trend and participation both broken"
    return {"name": name, "description": desc,
            "trend_up": trend_up, "breadth_ok": ad_ok}


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

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hist = load_history()

    # Backfill when asked, and automatically whenever history is too thin to
    # be worth looking at -- a curve that starts the day you installed this
    # tells you nothing about whether today is unusual.
    if backfill or len(hist) < 30:
        computed = historical_scores(m, oas, br)
        if computed:
            hist = computed
            notes.append(f"score history rebuilt from {len(computed)} sessions")

    hist = [h for h in hist if h.get("date") != today]
    hist.append({"date": today, "score": score, "regime": regime["name"]})

    spy = m.get("spy")
    return {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "score": score,
        "band": ("CALM" if score < 35 else "ELEVATED" if score < 60
                 else "STRESSED" if score < 80 else "ACUTE"),
        "regime": regime,
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
    oas = fetch_hy_oas(errors)
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
