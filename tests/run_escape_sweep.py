"""
Walk-forward sweep of the band-escape classifier (entry_mode="escape") through
the full engine — real exits, spread costs, train/validation split — against
the live rule set as the baseline.

    py tests/run_escape_sweep.py        → tests/escape_sweep_report.json
"""
import json, os, sys, time, warnings
from datetime import datetime
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

import pandas as pd                                   # noqa: E402
import options_bot as ob                              # noqa: E402
from run_exit_sweep import summarize, run_combo, fmt_row   # noqa: E402

GRID = {
    "entry_mode":          ["escape"],
    "escape_normal":       ["skip", "call"],
    "escape_breakout_vol": [1.5, 2.0, 99.0],            # 99 = never trade an upper escape
    "garch_mode":          ["off", "gate"],
    "max_spread_pct":      [100, 15],
    "stop_loss_pct":       [0, 50],
    "profit_target_pct":   [0, 100],
    "trail_activate_pct":  [0, 15],
    "stagnant_days":       [0],
    "hold_days":           [5, 10, 21],
    "stock_stop_pct":      [0, 4],
}
LIVE = {**ob.BT_DEFAULT_PARAMS, "garch_mode": "gate", "max_spread_pct": 15, "stop_loss_pct": 0, "profit_target_pct": 0,
        "trail_activate_pct": 0, "stagnant_days": 0, "hold_days": 21, "stock_stop_pct": 0}


def main():
    t0 = time.time()
    ob._load_garch_cache()
    precomp, split_map, split_date = ob._wfo_prepare(list(dict.fromkeys(ob.FOCUS_LIST)), label="Escape")
    combos = ob._expand_grid(GRID)
    print(f"\n=== escape sweep: {len(combos)} combos ===")
    lb = ob._wfo_sweep(combos, precomp, split_map, top_n=30,
                       progress=lambda d, t: print(f"    {d}/{t} ({time.time()-t0:.0f}s)", flush=True) if d % 100 == 0 or d == t else None,
                       label="Escape")
    robust = sorted([r for r in lb if r["robust"]], key=lambda r: -r["val_sharpe"])
    best, best_rob = (lb[0] if lb else None), (robust[0] if robust else None)

    runs = [("LIVE rule set (reversion, GARCH gate, no stop, hold 21)", LIVE),
            ("escape default (skip normal, breakout vol 2, garch off, no stop, hold 10)",
             {**ob.BT_DEFAULT_PARAMS, "entry_mode": "escape", "stop_loss_pct": 0, "profit_target_pct": 0, "trail_activate_pct": 0,
              "stagnant_days": 0, "hold_days": 10, "stock_stop_pct": 0})]
    if best:
        runs.append(("escape sweep best: " + json.dumps(best["params"], separators=(",", ":")), best["params"]))
    if best_rob and best_rob is not best:
        runs.append(("escape best ROBUST: " + json.dumps(best_rob["params"], separators=(",", ":")), best_rob["params"]))
    baselines = [summarize(run_combo(precomp, split_map, combo, w), f"{label} [{w}]")
                 for label, combo in runs for w in ("train", "val", "all")]
    pick = best_rob or best
    sig_quality = ob.signal_quality_analysis(run_combo(precomp, split_map, pick["params"], "all")) if pick else []

    report = {"split_date": str(pd.Timestamp(split_date).date()), "tickers": len(precomp), "grid": GRID, "leaderboard": lb,
              "robust": robust, "baselines": baselines, "signal_quality_best": sig_quality,
              "completed_at": datetime.now().isoformat(), "seconds": round(time.time() - t0)}
    out = os.path.join(ROOT, "tests", "escape_sweep_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ob._json_clean(report), f, indent=2)

    print("\n=== top 10 (validation Sharpe, net of spread) ===")
    for r in lb[:10]:
        print(f"  val_sh={r['val_sharpe']:<6} valPF={r['val_pf']:<5} trPF={r['train_pf']:<5} val_n={r['val_trades']:<4} tr_n={r['train_trades']:<5} "
              f"bars={r['val_avg_bars_held']:<4} robust={str(r['robust']):<5} {json.dumps(r['params'], separators=(',', ':'))}")
    print(f"\nrobust combos: {len(robust)}")
    print("\n=== SUMMARY (net of spread) ===")
    for b in baselines:
        if b.get("n"):
            print(fmt_row(b))
    if sig_quality:
        print("\n=== signal value inside the best escape rule set (net) ===")
        for r in sig_quality[:14]:
            print(f"  {r['signal']:<28} n={r['count']:5d} win={r['win_rate']:5.1f}% avg={r['avg_pnl']:+6.2f}%")
    print(f"\nreport → {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
