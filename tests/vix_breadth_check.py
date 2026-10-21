"""
Do market-wide features separate the bad escape trades (2022) from the good ones?
Runs the applied rules (+ GARCH gate) over six years once, then scores every trade
by the market state on its entry date and reports profit factor per feature bucket
and for a handful of candidate gates, by year.  Post-hoc filter: removing a trade
does not re-open its ticker for a later entry, so a promising gate still needs a
proper engine run before it is applied.

Features (all known at the entry bar's close):
  vix_chg5      VIX vs five sessions earlier (%)
  vix_vs_ma20   VIX vs its 20-session mean (%)
  vix_pct252    VIX percentile over the past 252 sessions (0–100)
  breadth50     share of focus tickers above their 50-day MA (0–1)
  breadth200    share above their 200-day MA
  breadth50_chg5  change in breadth50 over five sessions
  spy_ret5 / spy_ret20   SPY return over 5 / 20 sessions (%)

    py tests/vix_breadth_check.py        → tests/vix_breadth_report.json
"""
import json, os, sys, time, warnings
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)
import options_bot as ob                              # noqa: E402
from escape_longrun import run, pf_of                 # noqa: E402

ob.BT_HISTORY_PERIOD = "6y"
BASE = {**ob.BT_DEFAULT_PARAMS, "entry_mode": "escape", "escape_normal": "skip", "escape_breakout_vol": 2.0, "garch_mode": "gate",
        "max_spread_pct": 15, "otm_factor_call": 0.97, "dte_target": 30, "stop_loss_pct": 0, "profit_target_pct": 0,
        "trail_activate_pct": 0, "stagnant_days": 0, "hold_days": 21, "stock_stop_pct": 0, "stock_trail_pct": 7}
CASES = ("sig_escape_reversion", "sig_escape_breakout", "sig_escape_momentum")
FEATURES = ("vix_chg5", "vix_vs_ma20", "vix_pct252", "breadth50", "breadth200", "breadth50_chg5", "spy_ret5", "spy_ret20")
# candidate gates: (name, feature, op, threshold) — a trade is KEPT when feature op threshold holds
GATES = [
    ("VIX not spiking (chg5 ≤ +15%)",      "vix_chg5",      "le", 15),
    ("VIX not spiking (chg5 ≤ +30%)",      "vix_chg5",      "le", 30),
    ("VIX spiking (chg5 > +15%)",          "vix_chg5",      "gt", 15),
    ("VIX ≤ 20-day mean +20%",             "vix_vs_ma20",   "le", 20),
    ("VIX below 1y median (pct ≤ 50)",     "vix_pct252",    "le", 50),
    ("VIX above 1y median (pct > 50)",     "vix_pct252",    "gt", 50),
    ("breadth50 ≥ 0.30",                   "breadth50",     "ge", 0.30),
    ("breadth50 ≥ 0.50",                   "breadth50",     "ge", 0.50),
    ("breadth50 < 0.50",                   "breadth50",     "lt", 0.50),
    ("breadth200 ≥ 0.50",                  "breadth200",    "ge", 0.50),
    ("breadth improving (chg5 ≥ 0)",       "breadth50_chg5", "ge", 0.0),
    ("SPY not in 20d drawdown (ret20 ≥ -5%)", "spy_ret20",  "ge", -5),
    ("SPY 5d down (ret5 < 0)",             "spy_ret5",      "lt", 0),
    ("SPY 5d up (ret5 ≥ 0)",               "spy_ret5",      "ge", 0),
    # "crash, not dip" candidates from the quintile view (bottom quintile of each loses in every year it has trades)
    ("SPY not crashing (ret5 ≥ -4%)",      "spy_ret5",      "ge", -4),
    ("SPY not crashing (ret5 ≥ -3%)",      "spy_ret5",      "ge", -3),
    ("breadth not collapsing (chg5 ≥ -0.20)", "breadth50_chg5", "ge", -0.20),
    ("breadth not collapsing (chg5 ≥ -0.15)", "breadth50_chg5", "ge", -0.15),
    ("no crash: ret5 ≥ -4% AND breadth chg5 ≥ -0.20", [("spy_ret5", "ge", -4), ("breadth50_chg5", "ge", -0.20)], None, None),
    ("no crash: ret5 ≥ -3% AND breadth chg5 ≥ -0.15", [("spy_ret5", "ge", -3), ("breadth50_chg5", "ge", -0.15)], None, None),
    ("no crash (either): ret5 ≥ -4% OR breadth chg5 ≥ -0.20", [("spy_ret5", "ge", -4), ("breadth50_chg5", "ge", -0.20)], "any", None),
]
OPS = {"le": lambda a, b: a <= b, "lt": lambda a, b: a < b, "ge": lambda a, b: a >= b, "gt": lambda a, b: a > b}


def keep(t, k, op, th):
    """A gate is one (feature, op, threshold) or a list of them (all must hold; op='any' → at least one)."""
    if isinstance(k, list):
        tests = [OPS[o](t[f], v) for f, o, v in k]
        return any(tests) if op == "any" else all(tests)
    return OPS[op](t[k], th)


def features(precomp):
    ctx = ob.market_context_history()
    vix = ctx["vix"].astype(float)
    spy = ctx["spy_close"].astype(float)
    f = pd.DataFrame(index=ctx.index)
    f["vix_chg5"]    = (vix / vix.shift(5) - 1) * 100
    f["vix_vs_ma20"] = (vix / vix.rolling(20).mean() - 1) * 100
    f["vix_pct252"]  = vix.rolling(252, min_periods=120).rank(pct=True) * 100
    f["spy_ret5"]    = (spy / spy.shift(5) - 1) * 100
    f["spy_ret20"]   = (spy / spy.shift(20) - 1) * 100
    above50, above200 = {}, {}
    for sym, pc in precomp.items():
        idx = pd.DatetimeIndex(pc["index"])
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        px, m50, m200 = pc["price"], pc["ma50"], pc["ma200"]
        with np.errstate(invalid="ignore"):
            above50[sym]  = pd.Series(np.where(np.isnan(m50), np.nan, px > m50), index=idx.normalize())
            above200[sym] = pd.Series(np.where(np.isnan(m200), np.nan, px > m200), index=idx.normalize())
    b50  = pd.DataFrame(above50).reindex(f.index).mean(axis=1)
    b200 = pd.DataFrame(above200).reindex(f.index).mean(axis=1)
    f["breadth50"], f["breadth200"] = b50, b200
    f["breadth50_chg5"] = b50 - b50.shift(5)
    return f


def summarise(trades):
    if not trades:
        return {"n": 0, "net": 0.0, "pf": 0.0}
    p = [t["pnl_pct"] for t in trades]
    return {"n": len(p), "net": round(float(np.mean(p)), 2), "pf": pf_of(p)}


def by_year(trades):
    g = defaultdict(list)
    for t in trades:
        g[t["entry_date"][:4]].append(t)
    return {y: summarise(v) for y, v in sorted(g.items())}


def fmt_years(d):
    return "  ".join(f"{y[2:]}:{v['pf']}({v['n']})" for y, v in d.items())


def main():
    t0 = time.time()
    ob._load_garch_cache()
    precomp, _, _ = ob._wfo_prepare(list(dict.fromkeys(ob.FOCUS_LIST)), label="VIX/breadth")
    trades = run(precomp, BASE)
    f = features(precomp)
    for t in trades:
        row = f.loc[pd.Timestamp(t["entry_date"])] if pd.Timestamp(t["entry_date"]) in f.index else None
        for k in FEATURES:
            v = None if row is None else row[k]
            t[k] = None if v is None or v != v else float(v)
        t["case"] = next((k[11:] for k in CASES if t.get(k)), "?")
    scored = [t for t in trades if all(t[k] is not None for k in FEATURES)]
    base = summarise(scored)
    print(f"\napplied + GARCH gate, 6y: n={base['n']} net {base['net']:+.2f}% PF {base['pf']}   {fmt_years(by_year(scored))}")
    print(f"  ({len(trades) - len(scored)} trades without a full feature row dropped)")
    report = {"base": {**base, "by_year": by_year(scored)}, "buckets": {}, "gates": {}}

    # ── quintile buckets per feature, overall and by case ──
    for k in FEATURES:
        vals = np.array([t[k] for t in scored])
        edges = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])
        buckets = defaultdict(list)
        for t, v in zip(scored, vals):
            buckets[int(np.searchsorted(edges, v))].append(t)
        print(f"\n{k}: quintile edges {np.round(edges, 2).tolist()}")
        report["buckets"][k] = {"edges": np.round(edges, 3).tolist(), "q": {}}
        for q in range(5):
            s = summarise(buckets[q]); cases = {c: summarise([t for t in buckets[q] if t["case"] == c]) for c in ("reversion", "breakout")}
            print(f"  Q{q+1}: n={s['n']:4d} net {s['net']:+6.2f}% PF {s['pf']:<5}  rev PF {cases['reversion']['pf']:<5} (n={cases['reversion']['n']})  brk PF {cases['breakout']['pf']:<5} (n={cases['breakout']['n']})   {fmt_years(by_year(buckets[q]))}")
            report["buckets"][k]["q"][q + 1] = {**s, "by_case": cases, "by_year": by_year(buckets[q])}

    # ── candidate gates: kept vs removed, by year ──
    print("\nGates (kept | removed), by year PF(n):")
    for name, k, op, th in GATES:
        kept = [t for t in scored if keep(t, k, op, th)]
        gone = [t for t in scored if not keep(t, k, op, th)]
        ks, gs = summarise(kept), summarise(gone)
        rev = summarise([t for t in kept if t["case"] == "reversion"]); brk = summarise([t for t in kept if t["case"] == "breakout"])
        print(f"  {name:<52} kept n={ks['n']:4d} net {ks['net']:+6.2f}% PF {ks['pf']:<5} (rev {rev['pf']} / brk {brk['pf']}) | removed n={gs['n']:4d} PF {gs['pf']:<5}   {fmt_years(by_year(kept))}")
        report["gates"][name] = {"feature": k, "op": op, "threshold": th, "kept": {**ks, "by_year": by_year(kept), "reversion": rev, "breakout": brk}, "removed": gs}

    with open(os.path.join(ROOT, "tests", "vix_breadth_report.json"), "w", encoding="utf-8") as fh:
        json.dump(ob._json_clean({**report, "completed_at": datetime.now().isoformat(), "seconds": round(time.time() - t0)}), fh, indent=2)
    print(f"\n({time.time()-t0:.0f}s) → tests/vix_breadth_report.json")


if __name__ == "__main__":
    main()
