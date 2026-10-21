"""
Six-year, per-year × escape-case view of the APPLIED rule set and two variants:
breakout calls only in a SPY uptrend, and reversion-only (no breakouts).  The
question from escape_longrun.py: does the breakout leg's 2022 loss go away when
it is gated by the market trend?

    py tests/escape_regime_check.py      → tests/escape_regime_report.json
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
APPLIED = {**ob.BT_DEFAULT_PARAMS, "entry_mode": "escape", "escape_normal": "skip", "escape_breakout_vol": 2.0, "garch_mode": "off",
           "max_spread_pct": 15, "otm_factor_call": 0.97, "dte_target": 30, "stop_loss_pct": 0, "profit_target_pct": 0,
           "trail_activate_pct": 0, "stagnant_days": 0, "hold_days": 21, "stock_stop_pct": 0, "stock_trail_pct": 7}
VARIANTS = {
    "applied (breakouts any trend)":      APPLIED,
    "breakouts only in SPY uptrend":      {**APPLIED, "escape_breakout_trend": "bull_only"},
    "reversion only (no breakouts)":      {**APPLIED, "escape_breakout_vol": 99.0},
    "applied + GARCH gate":               {**APPLIED, "garch_mode": "gate"},
}
CASES = ("sig_escape_reversion", "sig_escape_breakout", "sig_escape_momentum")


def main():
    t0 = time.time()
    ob._load_garch_cache()
    precomp, _, _ = ob._wfo_prepare(list(dict.fromkeys(ob.FOCUS_LIST)), label="Regime")
    report = {}
    for name, combo in VARIANTS.items():
        trades = run(precomp, combo)
        by_year = defaultdict(list); by_yc = defaultdict(list); by_case = defaultdict(list)
        for t in trades:
            y = t["entry_date"][:4]; c = next((k[11:] for k in CASES if t.get(k)), "?")
            by_year[y].append(t); by_yc[(y, c)].append(t); by_case[c].append(t)
        pnls = [t["pnl_pct"] for t in trades]
        lo, md, hi = bootstrap_pf(pnls) if len(pnls) > 20 else (0, 0, 0)
        print(f"\n=== {name}: n={len(trades)} net {np.mean(pnls) if pnls else 0:+.2f}% PF {pf_of(pnls)} (90% CI {lo}–{hi}) ===")
        for y in sorted(by_year):
            s = stats(by_year[y])
            cases = "  ".join(f"{c[:4]} n={len(by_yc[(y, c)])} PF={pf_of([t['pnl_pct'] for t in by_yc[(y, c)]])}" for c in ("reversion", "breakout", "momentum") if by_yc.get((y, c)))
            print(f"  {y}: n={s['n']:4d} net={s['net']:+6.2f}% PF={s['pf']:<5} win={s['win']}%   [{cases}]")
        print("  by case: " + ", ".join(f"{c} n={len(v)} net {np.mean([t['pnl_pct'] for t in v]):+.1f}% PF {pf_of([t['pnl_pct'] for t in v])}" for c, v in by_case.items()))
        report[name] = {"all": {**stats(trades), "pf_ci90": [lo, hi]}, "by_year": {y: stats(v) for y, v in by_year.items()},
                        "by_case": {c: stats(v) for c, v in by_case.items()},
                        "by_year_case": {f"{y}/{c}": stats(v) for (y, c), v in by_yc.items()}}
    with open(os.path.join(ROOT, "tests", "escape_regime_report.json"), "w", encoding="utf-8") as f:
        json.dump(ob._json_clean({**report, "completed_at": datetime.now().isoformat(), "seconds": round(time.time() - t0)}), f, indent=2)
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
