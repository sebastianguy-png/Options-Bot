"""
The mirror of the breadth gate: when breadth is collapsing (the gate blocks the
call), does buying a PUT on the first-day lower escape pay?  Six years, applied
rules + GARCH gate + breadth gate, measured spreads; `escape_put_on_collapse` on
vs off.  Only the put trades are judged (the call trades are identical).

Bar to add it (agreed with Sebastian): PF > 1.3 over six years net of costs,
positive in at least four of five full years, 2025 not a disaster.

    py tests/collapse_put_check.py       → tests/collapse_put_report.json
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
BASE = {**ob.BT_DEFAULT_PARAMS, "entry_mode": "escape", "escape_normal": "skip", "escape_breakout_vol": 2.0, "garch_mode": "gate",
        "max_spread_pct": 15, "otm_factor_call": 0.97, "otm_factor_put": 1.03, "dte_target": 30, "stop_loss_pct": 0, "profit_target_pct": 0,
        "trail_activate_pct": 0, "stagnant_days": 0, "hold_days": 21, "stock_stop_pct": 0, "stock_trail_pct": 7, "breadth_chg5_min": -0.15}
VARIANTS = {
    "collapse puts, GARCH gate on puts too": {"escape_put_on_collapse": True},
    "collapse puts, no GARCH gate":          {"escape_put_on_collapse": True, "garch_mode": "off"},
    "collapse puts, 10-day hold":            {"escape_put_on_collapse": True, "hold_days": 10},
    "collapse puts, ATM (1.00)":             {"escape_put_on_collapse": True, "otm_factor_put": 1.0},
}


def by_year(trades):
    g = defaultdict(list)
    for t in trades:
        g[t["entry_date"][:4]].append(t)
    return {y: stats(v) for y, v in sorted(g.items())}


def main():
    t0 = time.time()
    ob._load_garch_cache()
    precomp, _, _ = ob._wfo_prepare(list(dict.fromkeys(ob.FOCUS_LIST)), label="CollapsePut")
    report = {}
    base_calls = [t for t in run(precomp, BASE) if t["strategy"] == "buy_call"]
    print(f"\ncalls under the applied rules (reference): {stats(base_calls)}")
    for name, patch in VARIANTS.items():
        trades = run(precomp, {**BASE, **patch})
        puts = [t for t in trades if t["strategy"] == "buy_put"]
        p = [t["pnl_pct"] for t in puts]
        lo, md, hi = bootstrap_pf(p) if len(p) > 20 else (0, 0, 0)
        yrs = by_year(puts)
        pos_years = sum(1 for y, s in yrs.items() if y >= "2022" and s["n"] >= 5 and s["pf"] > 1)
        verdict = "PASS" if (pf_of(p) > 1.3 and pos_years >= 4 and yrs.get("2025", {}).get("pf", 1) >= 0.7) else "fail"
        print(f"  {name:<40} puts n={len(p):3d} net {np.mean(p) if p else 0:+6.2f}% PF {pf_of(p):<5} CI90 [{lo}, {hi}]  "
              + "  ".join(f"{y[2:]}:{s['pf']}({s['n']})" for y, s in yrs.items()) + f"   → {verdict}")
        exits = defaultdict(int)
        for t in puts:
            exits[t["exit_reason"]] += 1
        report[name] = {"puts": {**stats(puts), "pf_ci90": [lo, hi]}, "by_year": yrs, "exits": dict(exits), "verdict": verdict,
                        "calls_unchanged": stats([t for t in trades if t["strategy"] == "buy_call"])}
    with open(os.path.join(ROOT, "tests", "collapse_put_report.json"), "w", encoding="utf-8") as f:
        json.dump(ob._json_clean({**report, "reference_calls": stats(base_calls), "completed_at": datetime.now().isoformat(),
                                  "seconds": round(time.time() - t0)}), f, indent=2)
    print(f"\n({time.time()-t0:.0f}s) → tests/collapse_put_report.json")


if __name__ == "__main__":
    main()
