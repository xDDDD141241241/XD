"""
What happened next.

For every gauge, and for the index's own price state, this answers one
question: on past days when the reading was about as loud as it is today,
what did the index do over the following week, month and three months?

Rules that keep it honest -- each one exists because the obvious version
quietly flatters the result:

  - Readings are the same 0-100 values the page shows, rebuilt day by day
    from data available on that day. Nothing is ranked against the future.
  - Weekly series (financial conditions, net liquidity) are published days
    after the date they carry. They are shifted forward by a week before
    being compared with price, so the backtest never "knows" them early.
  - Neighbouring days are nearly the same event. Six loud weeks are thirty
    days but ONE episode, and every count on the page says how many
    separate episodes stand behind it.
  - Every result is shown next to the same figure for all days. The index
    rises most of the time; a gauge only earns attention if its row looks
    clearly worse than an ordinary day.
  - Nothing here feeds back into the score. This is a display, not a tuner.
    Adjusting thresholds until this table looks good is how a monitor gets
    fitted to the past and stops working.

Index = SPY closing price. Dividends are not included, which shaves a
little off every forward return equally and does not change comparisons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 21, 63)          # one week, one month, three months
DD_WINDOW = 63                  # worst dip measured over the next three months
EPISODE_GAP = 10                # sessions apart before two loud spells count as two
WEEKLY_LAG = 5                  # sessions a weekly series is held back
BUCKETS = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100.01))
BUCKET_NAMES = ("0–20", "20–40", "40–60", "60–80", "80–100")
MIN_EPISODES = 3                # below this, say so rather than show a rate


# ---------------------------------------------------------------------------

def forward_paths(spy: pd.Series) -> pd.DataFrame:
    """For each session: index change after 5/21/63 sessions and the worst
    dip along the next 63. NaN where the future has not happened yet."""
    s = spy.dropna().astype(float)
    out = pd.DataFrame(index=s.index)
    for h in HORIZONS:
        out[f"r{h}"] = (s.shift(-h) / s - 1) * 100
    fut_min = s.rolling(DD_WINDOW).min().shift(-DD_WINDOW)
    out["dd"] = ((fut_min / s - 1) * 100).clip(upper=0)
    return out


def split_episodes(mask: pd.Series, gap: int = EPISODE_GAP) -> list[tuple[int, int]]:
    """Positions (start, end) of separate spells where mask is True."""
    pos = np.flatnonzero(mask.to_numpy(dtype=bool))
    if not len(pos):
        return []
    breaks = np.flatnonzero(np.diff(pos) > gap)
    starts = np.r_[pos[0], pos[breaks + 1]]
    ends = np.r_[pos[breaks], pos[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _r(v, nd=1):
    return None if v is None or v != v else round(float(v), nd)


def summarize(mask: pd.Series, fwd: pd.DataFrame) -> dict:
    """Day-based and episode-based outcomes for the days where mask is True."""
    mask = mask.reindex(fwd.index).fillna(False).astype(bool)
    done = fwd["r63"].notna()
    d = fwd[mask & done]
    eps = split_episodes(mask)
    starts = [fwd.index[a] for a, _ in eps]
    ep_done = [t for t in starts if done.loc[t]]
    first = fwd.loc[ep_done] if ep_done else fwd.iloc[0:0]
    return {
        "days": int(mask.sum()),
        "days_done": int(len(d)),
        "episodes": len(eps),
        "episodes_done": len(ep_done),
        # day-based: every qualifying session counts once
        "lower_1m": _r((d["r21"] < 0).mean() * 100 if len(d) else None, 0),
        "med_1w": _r(d["r5"].median() if len(d) else None),
        "med_1m": _r(d["r21"].median() if len(d) else None),
        "med_3m": _r(d["r63"].median() if len(d) else None),
        "med_dd": _r(d["dd"].median() if len(d) else None),
        "worst_dd": _r(d["dd"].min() if len(d) else None),
        # episode-based: judged from the day each spell began
        "ep_lower_1m": int((first["r21"] < 0).sum()) if len(first) else 0,
        "ep_med_dd": _r(first["dd"].median() if len(first) else None),
    }


def _bucket(reading: float) -> int:
    for k, (lo, hi) in enumerate(BUCKETS):
        if lo <= reading < hi:
            return k
    return len(BUCKETS) - 1


def _episode_rows(mask: pd.Series, fwd: pd.DataFrame, reading: pd.Series | None,
                  limit: int = 40) -> list[dict]:
    mask = mask.reindex(fwd.index).fillna(False).astype(bool)
    rows = []
    for a, b in split_episodes(mask):
        t0 = fwd.index[a]
        f = fwd.iloc[a]
        rows.append({
            "start": str(t0.date()), "end": str(fwd.index[b].date()),
            "sessions": int(b - a + 1),
            "reading": None if reading is None else _r(reading.get(t0), 0),
            "r21": _r(f["r21"]), "r63": _r(f["r63"]), "dd": _r(f["dd"]),
            "open": bool(f["r63"] != f["r63"]),
        })
    return rows[-limit:]


# ---------------------------------------------------------------------------

def gauge_history(specs: list[dict], readings: dict[str, pd.Series],
                  spy: pd.Series, today: dict[str, float]) -> tuple[dict, dict]:
    """
    Returns (summaries, detail).

    summaries[id] is small and rides along in data.json: today's bucket, the
    outcome after past spells in it, and the all-days baseline.
    detail is the full per-day record for gauges.json, loaded by the page
    only when a history panel is opened.
    """
    if spy is None or spy.dropna().empty or not readings:
        return {}, {}
    spy = spy.dropna()
    fwd = forward_paths(spy)
    base = summarize(pd.Series(True, index=fwd.index), fwd)

    start = min(r.dropna().index[0] for r in readings.values() if not r.dropna().empty)
    idx = fwd.index[fwd.index >= start]
    fwd = fwd.loc[idx]
    base_here = summarize(pd.Series(True, index=idx), fwd)

    by_id = {sp["id"]: sp for sp in specs}
    summaries, gauges = {}, {}
    for gid, r in readings.items():
        sp = by_id.get(gid)
        if sp is None:
            continue
        lag = WEEKLY_LAG if sp.get("max_lag", 4) >= 10 else 0
        rd = r.dropna()
        # carry weekly/irregular series across trading days, then hold back
        # by the publication lag so the past never sees a number early
        rd = rd.reindex(rd.index.union(idx)).ffill(limit=10).reindex(idx)
        raw = sp["series"].dropna()
        raw = raw.reindex(raw.index.union(idx)).ffill(limit=10).reindex(idx)
        if lag:
            rd, raw = rd.shift(lag), raw.shift(lag)
        if rd.notna().sum() < 250:
            continue

        now = today.get(gid)
        if now is None or now != now:
            now = float(rd.dropna().iloc[-1])
        k = _bucket(now)
        valid = rd.notna()
        bk = rd.apply(lambda v: _bucket(v) if v == v else -1)

        table = []
        for j, name in enumerate(BUCKET_NAMES):
            s = summarize((bk == j) & valid, fwd)
            s["bucket"] = name
            s["share"] = _r((bk == j).sum() / max(valid.sum(), 1) * 100, 0)
            table.append(s)
        here = table[k]

        summaries[gid] = {
            "bucket": BUCKET_NAMES[k], "bucket_index": k,
            "episodes": here["episodes"], "episodes_done": here["episodes_done"],
            "ep_lower_1m": here["ep_lower_1m"],
            "lower_1m": here["lower_1m"], "med_3m": here["med_3m"],
            "med_dd": here["med_dd"], "worst_dd": here["worst_dd"],
            "base_lower_1m": base_here["lower_1m"], "base_med_3m": base_here["med_3m"],
            "base_med_dd": base_here["med_dd"],
            "enough": here["episodes_done"] >= MIN_EPISODES,
            "since": str(rd.dropna().index[0].date()),
            "lag": lag,
        }
        gauges[gid] = {
            "label": sp["label"],
            "reading": [None if v != v else int(round(v)) for v in rd.to_numpy()],
            "value": [None if v != v else float(f"{v:.4g}") for v in raw.to_numpy()],
            "buckets": table, "today_bucket": k,
            "episodes": _episode_rows((bk == k) & valid, fwd, rd),
            "lag": lag,
        }

    detail = {
        "dates": [str(t.date()) for t in idx],
        "spy": [round(float(v), 2) for v in spy.reindex(idx).to_numpy()],
        "fwd21": [_r(v) for v in fwd["r21"].to_numpy()],
        "fwd63": [_r(v) for v in fwd["r63"].to_numpy()],
        "baseline": base_here,
        "baseline_all": base,
        "gauges": gauges,
    }
    return summaries, detail


def state_history(states: pd.Series, spy: pd.Series, current: str,
                  labels: dict[str, str]) -> dict:
    """The same what-happened-next table, split by price state instead."""
    if spy is None or states is None or states.dropna().empty:
        return {}
    fwd = forward_paths(spy.dropna())
    st = states.reindex(fwd.index)
    idx = st.dropna().index
    fwd = fwd.loc[idx]
    st = st.loc[idx]
    rows = []
    for key, label in labels.items():
        s = summarize(st == key, fwd)
        s.update({"state": key, "label": label,
                  "share": _r((st == key).mean() * 100, 0)})
        rows.append(s)
    base = summarize(pd.Series(True, index=idx), fwd)
    return {"table": rows, "baseline": base,
            "current": current,
            "episodes": _episode_rows(st == current, fwd, None, limit=20),
            "since": str(idx[0].date())}
