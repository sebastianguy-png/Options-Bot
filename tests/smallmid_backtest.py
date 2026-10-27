"""
Small/mid-cap universe for the escape rules — step 2: the six-year backtest.

Takes the screened universe (tests/smallmid_universe.json from smallmid_screen.py),
keeps the names whose measured ~30-DTE spread is inside the live 15% cap (top
--max by dollar volume), charges each trade its own measured spread, and runs
the APPLIED rules (reversion calls only, GARCH gate, breadth gate −0.15, ITM
0.97, 30 DTE, 21-day hold, 7% stock trail) over six years with the vol model
refit every bar.  Breadth is the new universe's own.  Calls only, by year, by
index (S&P 400 / 600) and by spread band; the focus list's numbers for the same
rules are in handoff 7k (PF 1.76 on 203 trades).

    py tests/smallmid_backtest.py [--max 200]      → tests/smallmid_report.json
"""
import json, os, sys, time, warnings
from collections import defaultdict
from datetime import datetime
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)
import options_bot as ob                              # noqa: E402
from escape_longrun import run, stats, pf_of, bootstrap_pf   # noqa: E402

ob.BT_HISTORY_PERIOD = "6y"
APPLIED = {**ob.BT_DEFAULT_PARAMS, "entry_mode": "escape", "escape_normal": "skip", "escape_breakout_vol": 99.0, "garch_mode": "gate",
           "max_spread_pct": 15, "otm_factor_call": 0.97, "dte_target": 30, "stop_loss_pct": 0, "profit_target_pct": 0,
           "trail_activate_pct": 0, "stagnant_days": 0, "hold_days": 21, "stock_stop_pct": 0, "stock_trail_pct": 7, "breadth_chg5_min": -0.15}


def by_year(trades):
    g = defaultdict(list)
    for t in trades:
        g[t["entry_date"][:4]].append(t)
    return {y: stats(v) for y, v in sorted(g.items())}


def fmt(trades):
    s = stats(trades); p = [t["pnl_pct"] for t in trades]
    ci = bootstrap_pf(p) if len(p) > 20 else (0, 0, 0)
    return f"n={s.get('n', 0):4d} net {s.get('net', 0):+6.2f}% PF {s.get('pf', 0):<5} CI90 [{ci[0]}, {ci[2]}]  " + \
           "  ".join(f"{y[2:]}:{v['pf']}({v['n']})" for y, v in by_year(trades).items())


def main():
    t0 = time.time()
    maxn = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 200
    cap = float(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 15.0     # measured-spread cap for inclusion
    uni = json.load(open(os.path.join(ROOT, "tests", "smallmid_universe.json"), encoding="utf-8"))
    rows = uni["tickers"]
    with_spread = {s: r for s, r in rows.items() if r.get("spread_pct") is not None}
    if with_spread:
        keep = {s: r for s, r in with_spread.items() if r["spread_pct"] <= cap}
        print(f"{len(with_spread)} measured, {len(keep)} inside the {cap:g}% cap")
    else:
        keep = dict(rows)
        print(f"no spreads measured (run smallmid_screen.py in market hours) — using all {len(keep)} screened names with fallback costs")
    syms = sorted(keep, key=lambda s: -keep[s]["dollar_vol_m"])[:maxn]
    # cost model: each name's own measured spread; the focus list's entry-window buckets for the rest
    focus = ob._get_spread_model()
    small_by = {s: {"spread_pct": keep[s]["spread_pct"], "n": max(3, int(keep[s].get("n_contracts") or 3))}
                for s in syms if keep[s].get("spread_pct") is not None}
    with_focus = "--with-focus" in sys.argv          # the combined universe: focus list + the small/mid names (shared breadth)
    if with_focus:
        focus_syms = list(dict.fromkeys(ob.FOCUS_LIST))
        syms = focus_syms + [s for s in syms if s not in focus_syms]
        for s in focus_syms:
            keep.setdefault(s, {"index": "focus", "spread_pct": None, "dollar_vol_m": 0})
    ob._spread_model = {"by_ticker": {**(focus.get("by_ticker") or {}), **small_by},
                        "buckets": focus.get("buckets") or ob.SPREAD_FALLBACK}
    ob._load_garch_cache()
    precomp, _, _ = ob._wfo_prepare(syms, label="SmallMid")
    if not precomp:
        print("no data"); return
    print(f"{len(precomp)} tickers ready ({time.time()-t0:.0f}s)")
    trades = run(precomp, APPLIED)
    calls = [t for t in trades if t["strategy"] == "buy_call"]
    print("\n=== small/mid-cap universe, applied rules, calls only, six years, measured spreads ===")
    print("  ALL     " + fmt(calls))
    for idx in (("focus", "sp400", "sp600") if with_focus else ("sp400", "sp600")):
        sel = [t for t in calls if keep.get(t["ticker"], {}).get("index") == idx]
        print(f"  {idx:<7} " + fmt(sel))
    for lo, hi, lab in ((0, 8, "spread ≤8%"), (8, 15, "spread 8–15%")):
        sel = [t for t in calls if lo < (keep.get(t["ticker"], {}).get("spread_pct") or 99) <= hi]
        print(f"  {lab:<12} " + fmt(sel))
    per = defaultdict(list)
    for t in calls:
        per[t["ticker"]].append(t["pnl_pct"])
    top = sorted(per.items(), key=lambda kv: -sum(kv[1]))[:10]; bot = sorted(per.items(), key=lambda kv: sum(kv[1]))[:10]
    print("  best names: " + ", ".join(f"{s} {sum(v):+.0f}%({len(v)})" for s, v in top))
    print("  worst names: " + ", ".join(f"{s} {sum(v):+.0f}%({len(v)})" for s, v in bot))
    exits = defaultdict(int)
    for t in calls:
        exits[t["exit_reason"]] += 1
    rep = {"universe": syms, "n_tickers": len(precomp), "params": APPLIED, "calls": stats(calls), "by_year": by_year(calls),
           "by_index": {i: stats([t for t in calls if keep.get(t["ticker"], {}).get("index") == i]) for i in ("focus", "sp400", "sp600")},
           "with_focus": with_focus,
           "exits": dict(exits), "per_ticker": {s: {"n": len(v), "net_sum": round(sum(v), 1)} for s, v in per.items()},
           "all_strategies": stats(trades), "completed_at": datetime.now().isoformat(), "seconds": round(time.time() - t0),
           "trades": [{**{k: t.get(k) for k in ("ticker", "strategy", "entry_date", "exit_date", "entry_price", "strike", "entry_opt", "exit_opt",
                                                 "pnl_pct", "cost_pct", "exit_reason", "iv_est", "sig_escape_reversion", "sig_escape_breakout")},
                       "index": keep.get(t["ticker"], {}).get("index")} for t in trades]}
    with open(os.path.join(ROOT, "tests", "smallmid_report.json"), "w", encoding="utf-8") as f:
        json.dump(ob._json_clean(rep), f, indent=1)
    print(f"  exits: {dict(exits)}\n({time.time()-t0:.0f}s) → tests/smallmid_report.json")


if __name__ == "__main__":
    main()
