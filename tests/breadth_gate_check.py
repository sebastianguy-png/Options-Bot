"""
In-engine test of the breadth gate (`breadth_chg5_min`): no escape entry while the
share of focus tickers above their 50-day MA has fallen by more than X points over
five sessions.  tests/vix_breadth_check.py found it as a post-hoc filter (PF 1.18 →
1.65 over six years, every year ≥ 1.19); here the gate runs inside evaluate_entry,
so a blocked ticker is free to enter again once breadth stabilises.

  1. six years, applied rules + GARCH gate, breadth_chg5_min ∈ {off, -0.20, -0.15, -0.10}: by year, by case, bootstrap CI
  2. rolling walk-forward (train 18 months → test 6 months) with the gate in the grid: is it chosen out of sample?

    py tests/breadth_gate_check.py       → tests/breadth_gate_report.json
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
from escape_longrun import run, stats, pf_of, bootstrap_pf   # noqa: E402

ob.BT_HISTORY_PERIOD = "6y"
BASE = {**ob.BT_DEFAULT_PARAMS, "entry_mode": "escape", "escape_normal": "skip", "escape_breakout_vol": 2.0, "garch_mode": "gate",
        "max_spread_pct": 15, "otm_factor_call": 0.97, "dte_target": 30, "stop_loss_pct": 0, "profit_target_pct": 0,
        "trail_activate_pct": 0, "stagnant_days": 0, "hold_days": 21, "stock_stop_pct": 0, "stock_trail_pct": 7}
VARIANTS = {"gate off": None, "breadth_chg5_min -0.20": -0.20, "breadth_chg5_min -0.15": -0.15, "breadth_chg5_min -0.10": -0.10}
CASES = ("sig_escape_reversion", "sig_escape_breakout", "sig_escape_momentum")
GRID = {"breadth_chg5_min": [None, -0.20, -0.15], "hold_days": [10, 21], "escape_breakout_vol": [2.0, 99.0]}


def main():
    t0 = time.time()
    ob._load_garch_cache()
    precomp, _, _ = ob._wfo_prepare(list(dict.fromkeys(ob.FOCUS_LIST)), label="Breadth")
    ctx = ob.market_context_history()
    assert "breadth50_chg5" in ctx.columns and ctx["breadth50_chg5"].notna().sum() > 500, "breadth columns missing from the context"
    first = min(pc["index"][0] for pc in precomp.values()); last = max(pc["index"][-1] for pc in precomp.values())
    report = {"first": str(first)[:10], "last": str(last)[:10], "variants": {}}

    print("\n=== six years, applied rules + GARCH gate ===")
    for name, thr in VARIANTS.items():
        trades = run(precomp, {**BASE, "breadth_chg5_min": thr})
        by_year = defaultdict(list); by_case = defaultdict(list)
        for t in trades:
            by_year[t["entry_date"][:4]].append(t)
            by_case[next((k[11:] for k in CASES if t.get(k)), "?")].append(t)
        p = [t["pnl_pct"] for t in trades]
        lo, md, hi = bootstrap_pf(p) if len(p) > 20 else (0, 0, 0)
        yrs = "  ".join(f"{y[2:]}:{stats(v)['pf']}({len(v)})" for y, v in sorted(by_year.items()))
        cases = "  ".join(f"{c} {stats(v)['pf']}({len(v)})" for c, v in by_case.items())
        print(f"  {name:<24} n={len(p):4d} net {np.mean(p) if p else 0:+6.2f}% PF {pf_of(p):<5} CI90 [{lo}, {hi}]   {yrs}   [{cases}]")
        report["variants"][name] = {"threshold": thr, "all": {**stats(trades), "pf_ci90": [lo, hi]},
                                    "by_year": {y: stats(v) for y, v in by_year.items()}, "by_case": {c: stats(v) for c, v in by_case.items()}}

    print("\n=== rolling walk-forward: train 18 months → test 6 months, gate in the grid ===")
    combos = [{**BASE, **c} for c in ob._expand_grid(GRID)]
    cur = pd.Timestamp(first) + pd.DateOffset(months=14)
    rows, fixed_test, gated_test, chosen_test = [], [], [], []
    while cur + pd.DateOffset(months=24) <= pd.Timestamp(last):
        tr_lo, tr_hi, te_hi = cur, cur + pd.DateOffset(months=18), cur + pd.DateOffset(months=24)
        fixed = run(precomp, BASE, tr_hi, te_hi)
        gated = run(precomp, {**BASE, "breadth_chg5_min": -0.15}, tr_hi, te_hi)
        best, best_pf = None, -1
        for c in combos:
            s = stats(run(precomp, c, tr_lo, tr_hi))
            if s["n"] >= 40 and s["pf"] > best_pf:
                best, best_pf = c, s["pf"]
        chosen = run(precomp, best, tr_hi, te_hi) if best else []
        fs, gs, cs = stats(fixed), stats(gated), stats(chosen)
        pick = {k: best[k] for k in GRID} if best else None
        rows.append({"train": f"{tr_lo.date()}→{tr_hi.date()}", "test": f"{tr_hi.date()}→{te_hi.date()}", "fixed": fs, "gated": gs,
                     "chosen": cs, "chosen_params": pick, "chosen_train_pf": best_pf})
        fixed_test += fixed; gated_test += gated; chosen_test += chosen
        print(f"  test {tr_hi.date()}→{te_hi.date()}: off n={fs.get('n', 0):3d} PF {fs.get('pf', 0):<5} | gate -0.15 n={gs.get('n', 0):3d} PF {gs.get('pf', 0):<5} | "
              f"re-chosen {json.dumps(pick)} (train PF {best_pf}) → n={cs.get('n', 0):3d} PF {cs.get('pf', 0)}")
        cur += pd.DateOffset(months=6)
    print(f"  union of test windows: off {stats(fixed_test)} | gate -0.15 {stats(gated_test)} | re-chosen {stats(chosen_test)}")
    report["walk_forward"] = rows
    report["walk_forward_union"] = {"off": stats(fixed_test), "gate_-0.15": stats(gated_test), "chosen": stats(chosen_test)}
    with open(os.path.join(ROOT, "tests", "breadth_gate_report.json"), "w", encoding="utf-8") as f:
        json.dump(ob._json_clean({**report, "completed_at": datetime.now().isoformat(), "seconds": round(time.time() - t0)}), f, indent=2)
    print(f"\n({time.time()-t0:.0f}s) → tests/breadth_gate_report.json")


if __name__ == "__main__":
    main()
