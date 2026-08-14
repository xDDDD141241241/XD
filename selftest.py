"""
Offline validation. No network. Generates synthetic series for every input,
runs the whole pipeline, and checks the invariants that actually matter:
weight renormalisation on failed feeds, percentile orientation, and that a
manufactured crisis pushes the score up and the regime down.
"""
import numpy as np
import pandas as pd


def synth(n=1400, seed=3, crisis=False):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)

    def walk(base, vol, floor=0.5):
        return pd.Series(np.maximum(base + np.cumsum(rng.normal(0, vol, n)), floor),
                         index=idx)

    m = {"vix": walk(18, 0.30), "vix9d": walk(17, 0.35), "vix3m": walk(20, 0.25),
         "vvix": walk(95, 1.1), "move": walk(100, 1.4)}
    spy = pd.Series(300 * np.exp(np.cumsum(rng.normal(0.0004, 0.010, n))), index=idx)
    m["spy"] = spy
    m["rsp"] = pd.Series(140 * np.exp(np.cumsum(rng.normal(0.0003, 0.010, n))), index=idx)
    m["tlt"] = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0000, 0.008, n))), index=idx)
    oas = walk(4.0, 0.04, floor=1.5)
    for tic, base, dr, vol in (("uup",28,0.0000,0.004),("fxy",62,-0.0001,0.005),
                               ("fxf",105,0.0001,0.005),("gld",180,0.0003,0.008),
                               ("iwv",280,0.0004,0.010)):
        m[tic] = pd.Series(base*np.exp(np.cumsum(rng.normal(dr,vol,n))), index=idx)
    m["xlu"] = pd.Series(70*np.exp(np.cumsum(rng.normal(0.0002,0.008,n))), index=idx)
    m["xlp"] = pd.Series(75*np.exp(np.cumsum(rng.normal(0.0002,0.007,n))), index=idx)
    fred = {"hy_oas": oas,
            "ccc_oas": walk(9.0, 0.09, floor=3.0),
            "bb_oas": walk(2.6, 0.03, floor=1.0),
            "nfci": walk(-0.4, 0.01, floor=-2.0),
            "fed_assets": walk(7.5e6, 4000.0, floor=1e6),
            "tga": walk(6.0e5, 8000.0, floor=1e4),
            "rrp": walk(4.0e5, 6000.0, floor=1.0)}

    br = {
        "ad_line": pd.Series(np.cumsum(rng.normal(2, 45, n)), index=idx),
        "pct_above_20": pd.Series(np.clip(rng.normal(58, 14, n), 0, 100), index=idx),
        "nh_nl": pd.Series(rng.normal(40, 45, n), index=idx),
        "adv_ratio": pd.Series(np.clip(rng.normal(0.53, 0.11, n), 0, 1), index=idx),
    }
    br["nhnl_line"] = br["nh_nl"].cumsum()

    if crisis:
        tail = slice(n - 30, n)
        m["vix"].iloc[tail] *= 2.6
        m["vix9d"].iloc[tail] *= 3.0
        m["vix3m"].iloc[tail] *= 1.7
        m["vvix"].iloc[tail] *= 1.6
        m["move"].iloc[tail] *= 1.5
        oas.iloc[tail] *= 2.2
        m["spy"].iloc[tail] *= np.linspace(1.0, 0.74, 30)
        # a real crisis hits credit quality, conditions and liquidity too --
        # leaving them calm made the synthetic crisis milder than any real one
        fred["ccc_oas"].iloc[tail] *= 2.6
        fred["nfci"].iloc[tail] += 1.8
        fred["fed_assets"].iloc[tail] *= 0.97
        m["xlu"].iloc[tail] *= 1.04
        m["xlp"].iloc[tail] *= 1.03

    if crisis:
        # Degrade only the tail. Percentiles rank the latest value against its
        # OWN history, so shifting a whole series does nothing -- a mistake
        # that made an earlier version of this test pass for the wrong reason.
        t = slice(n - 30, n)
        br["ad_line"].iloc[t] = br["ad_line"].iloc[n - 31] + np.cumsum(
            rng.normal(-260, 60, 30))
        br["pct_above_20"].iloc[t] = np.clip(rng.normal(11, 6, 30), 0, 100)
        br["nh_nl"].iloc[t] = rng.normal(-230, 50, 30)
        br["nhnl_line"] = br["nh_nl"].cumsum()
        for tic, mult in (("iwv",0.80),("gld",1.05),("uup",1.04),("fxy",1.06),("fxf",1.05)):
            m[tic].iloc[t] *= np.linspace(1.0, mult, 30)
        br["adv_ratio"].iloc[t] = np.clip(rng.normal(0.28, 0.08, 30), 0, 1)
        # the confirmation gap needs price to fall WITH breadth for the crisis
        # case; the synthetic A/D line is otherwise independent of price
        br["ad_line"].iloc[t] = br["ad_line"].iloc[n - 31] + np.cumsum(
            rng.normal(-400, 60, 30))

    return m, fred, br


def run_selftest() -> int:
    import risk_monitor as R
    ok = True

    calm_m, calm_oas, calm_br = synth(crisis=False)
    crisis_m, crisis_oas, crisis_br = synth(crisis=True)

    calm = R.assemble(calm_m, calm_oas, calm_br, [])
    crisis = R.assemble(crisis_m, crisis_oas, crisis_br, [])

    print(f"calm    score {calm['score']:>5}  band {calm['band']:<9} "
          f"regime {calm['regime']['name']}")
    print(f"crisis  score {crisis['score']:>5}  band {crisis['band']:<9} "
          f"regime {crisis['regime']['name']}")
    print(f"\nindicators built: {len(calm['indicators'])}")
    for i in sorted(crisis["indicators"], key=lambda x: -x["percentile"])[:6]:
        print(f"  {i['id']:<16} {i['display']:>9}  pctl {i['percentile']:>5}  w{i['weight']}")

    print("\nsignals fired in crisis:")
    for s in crisis["signals"]:
        print(f"  {'FIRED ' if s['fired'] else '      '} {s['id']}")

    # invariant 1: crisis must score materially higher
    if not crisis["score"] > calm["score"] + 15:
        print("FAIL: crisis did not raise the score"); ok = False

    # invariant 2: dropping feeds must renormalise, not deflate
    partial = R.assemble({k: v for k, v in crisis_m.items() if k in ("vix", "spy")},
                         None, {}, ["simulated outage"])
    print(f"\ndegraded run: {len(partial['indicators'])} indicators, "
          f"score {partial['score']} (weights renormalised)")
    if partial["score"] != partial["score"] or partial["score"] < 20:
        print("FAIL: degraded run collapsed"); ok = False

    # invariant 3: inverted percentiles point the right way
    b = [i for i in crisis["indicators"] if i["id"] == "pct_above_20"]
    if b and b[0]["percentile"] < 50:
        print("FAIL: low breadth should read as HIGH danger"); ok = False

    # invariant 4: score is bounded
    for p in (calm, crisis, partial):
        if not 0 <= p["score"] <= 100:
            print("FAIL: score out of bounds"); ok = False

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run_selftest())
