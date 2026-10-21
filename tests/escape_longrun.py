"""
The escape rules over SIX years (so the 2022 bear market is in the sample):
  - per-year results of the applied configuration, net of spread, by escape case
  - rolling walk-forward: 18-month train / 6-month test windows stepping 6 months;
    (a) the fixed applied config tested in every window, (b) a small grid re-chosen
    per window on training profit factor and tested out of sample
  - bootstrap 90% confidence interval of the profit factor

    py tests/escape_longrun.py          → tests/escape_longrun_report.json
"""
import json, os, random, sys, time, warnings
from collections import defaultdict
from datetime import datetime
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import options_bot as ob                              # noqa: E402

ob.BT_HISTORY_PERIOD = "6y"
APPLIED = {**ob.BT_DEFAULT_PARAMS, "entry_mode": "escape", "escape_normal": "skip", "escape_breakout_vol": 2.0, "garch_mode": "off",
           "max_spread_pct": 15, "stop_loss_pct": 0, "profit_target_pct": 0, "trail_activate_pct": 0, "stagnant_days": 0,
           "hold_days": 10, "stock_stop_pct": 0}
GRID = {"escape_breakout_vol": [1.5, 2.0, 99.0], "garch_mode": ["off", "gate"], "hold_days": [5, 10, 21], "stock_stop_pct": [0, 4]}


def pf_of(pnls):
    gw, gl = sum(p for p in pnls if p > 0), -sum(p for p in pnls if p < 0)
    return round(gw / gl, 2) if gl > 0 else (99.0 if gw > 0 else 0.0)


def stats(trades):
    if not trades:
        return {"n": 0}
    p = [t["pnl_pct"] for t in trades]
    return {"n": len(p), "net": round(float(np.mean(p)), 2), "win": round(100 * np.mean([x > 0 for x in p]), 1), "pf": pf_of(p),
            "bars": round(float(np.mean([t["bars_held"] for t in trades])), 1)}


def run(precomp, combo, lo=None, hi=None):
    params, exits = ob._split_combo(combo)
    out = []
    for sym, pc in precomp.items():
        kw = {}
        if lo is not None:
            kw["idx_start"] = int(pc["index"].searchsorted(pd.Timestamp(lo)))
        if hi is not None:
            kw["idx_end"] = int(pc["index"].searchsorted(pd.Timestamp(hi)))
        out += ob.run_backtest_ticker(sym, params=params, exits=exits, precomp=pc, **kw)
    return out


def bootstrap_pf(pnls, n=2000, seed=7):
    rng = random.Random(seed)
    pfs = []
    for _ in range(n):
        s = [pnls[rng.randrange(len(pnls))] for _ in pnls]
        pfs.append(pf_of(s))
    pfs.sort()
    return round(pfs[int(0.05 * n)], 2), round(pfs[int(0.5 * n)], 2), round(pfs[int(0.95 * n)], 2)


def main():
    t0 = time.time()
    ob._load_garch_cache()
    precomp, split_map, split_date = ob._wfo_prepare(list(dict.fromkeys(ob.FOCUS_LIST)), label="6y")
    first = min(pc["index"][0] for pc in precomp.values()); last = max(pc["index"][-1] for pc in precomp.values())
    print(f"{len(precomp)} tickers, {str(first)[:10]} → {str(last)[:10]} ({time.time()-t0:.0f}s)")
    report = {"period": "6y", "tickers": len(precomp), "first": str(first)[:10], "last": str(last)[:10]}

    # ── per year, applied config vs shipped rules ──
    trades = run(precomp, APPLIED)
    shipped = run(precomp, {**ob.BT_DEFAULT_PARAMS, **ob.BT_DEFAULT_EXITS})
    by_year = defaultdict(list); by_case = defaultdict(list); sh_year = defaultdict(list)
    for t in trades:
        by_year[t["entry_date"][:4]].append(t)
        by_case[next((k for k in ("sig_escape_reversion", "sig_escape_momentum", "sig_escape_breakout") if t.get(k)), "?")].append(t)
    for t in shipped:
        sh_year[t["entry_date"][:4]].append(t)
    print("\n=== applied escape rules by entry year (net of spread) — shipped rules alongside ===")
    for y in sorted(set(by_year) | set(sh_year)):
        a, s = stats(by_year.get(y, [])), stats(sh_year.get(y, []))
        print(f"  {y}: escape n={a.get('n', 0):4d} net={a.get('net', 0):+6.2f}% PF={a.get('pf', 0):<5} win={a.get('win', 0)}%   |   shipped n={s.get('n', 0):5d} net={s.get('net', 0):+6.2f}% PF={s.get('pf', 0)}")
    print("  by case: " + ", ".join(f"{k} n={len(v)} net {np.mean([t['pnl_pct'] for t in v]):+.1f}% PF {pf_of([t['pnl_pct'] for t in v])}" for k, v in by_case.items()))
    allst = stats(trades); lo, md, hi = bootstrap_pf([t["pnl_pct"] for t in trades]) if trades else (0, 0, 0)
    print(f"  ALL 6y: n={allst['n']} net={allst['net']:+.2f}% PF={allst['pf']}  bootstrap 90% CI of PF [{lo}, {hi}] (median {md})")
    report["by_year"] = {y: stats(v) for y, v in by_year.items()}
    report["shipped_by_year"] = {y: stats(v) for y, v in sh_year.items()}
    report["by_case"] = {k: stats(v) for k, v in by_case.items()}
    report["all"] = {**allst, "pf_ci90": [lo, hi]}

    # ── rolling walk-forward ──
    print("\n=== rolling walk-forward: train 18 months → test 6 months ===")
    combos = [{**APPLIED, **c} for c in ob._expand_grid(GRID)]
    start = pd.Timestamp(first) + pd.DateOffset(months=14)       # warm-up + first training window
    rows, fixed_test, chosen_test = [], [], []
    cur = start
    while cur + pd.DateOffset(months=24) <= pd.Timestamp(last):
        tr_lo, tr_hi, te_hi = cur, cur + pd.DateOffset(months=18), cur + pd.DateOffset(months=24)
        fixed = run(precomp, APPLIED, tr_hi, te_hi)
        best, best_pf = None, -1
        for c in combos:
            s = stats(run(precomp, c, tr_lo, tr_hi))
            if s["n"] >= 50 and s["pf"] > best_pf:
                best, best_pf = c, s["pf"]
        chosen = run(precomp, best, tr_hi, te_hi) if best else []
        fs, cs = stats(fixed), stats(chosen)
        rows.append({"train": f"{tr_lo.date()}→{tr_hi.date()}", "test": f"{tr_hi.date()}→{te_hi.date()}", "fixed": fs, "chosen": cs,
                     "chosen_params": {k: best[k] for k in GRID} if best else None, "chosen_train_pf": best_pf})
        fixed_test += fixed; chosen_test += chosen
        print(f"  test {tr_hi.date()}→{te_hi.date()}: fixed n={fs.get('n', 0):3d} PF={fs.get('pf', 0):<5} net={fs.get('net', 0):+6.2f}%  |  "
              f"re-chosen {json.dumps({k: best[k] for k in GRID}) if best else '—'} (train PF {best_pf}) → n={cs.get('n', 0):3d} PF={cs.get('pf', 0)} net={cs.get('net', 0):+.2f}%")
        cur += pd.DateOffset(months=6)
    print(f"  union of test windows: fixed {stats(fixed_test)} | re-chosen {stats(chosen_test)}")
    report["walk_forward"] = rows
    report["walk_forward_union"] = {"fixed": stats(fixed_test), "chosen": stats(chosen_test)}
    with open(os.path.join(ROOT, "tests", "escape_longrun_report.json"), "w", encoding="utf-8") as f:
        json.dump(ob._json_clean({**report, "completed_at": datetime.now().isoformat(), "seconds": round(time.time() - t0)}), f, indent=2)
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
