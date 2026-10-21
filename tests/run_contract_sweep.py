"""
Contract selection + momentum exit sweep on top of the applied escape rules:
strike (otm_factor_call), expiry (dte_target), stock-based trailing exit,
max hold, stock stop — net of spread, train/validation.

    py tests/run_contract_sweep.py       → tests/contract_sweep_report.json
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

BASE = {**ob.BT_DEFAULT_PARAMS, "entry_mode": "escape", "escape_normal": "skip", "escape_breakout_vol": 2.0, "garch_mode": "off",
        "max_spread_pct": 15, "stop_loss_pct": 0, "profit_target_pct": 0, "trail_activate_pct": 0, "stagnant_days": 0,
        "hold_days": 10, "stock_stop_pct": 0, "stock_trail_pct": 0}
GRID = {"otm_factor_call": [0.95, 0.97, 1.0, 1.03], "dte_target": [30, 45, 60], "stock_trail_pct": [0, 4, 7],
        "hold_days": [10, 21], "stock_stop_pct": [0, 4]}


def main():
    t0 = time.time()
    ob._load_garch_cache()
    precomp, split_map, split_date = ob._wfo_prepare(list(dict.fromkeys(ob.FOCUS_LIST)), label="Contract")
    combos = [{**BASE, **c} for c in ob._expand_grid(GRID)]
    print(f"\n=== contract/exit sweep: {len(combos)} combos ===")
    lb = ob._wfo_sweep(combos, precomp, split_map, top_n=40,
                       progress=lambda d, t: print(f"    {d}/{t} ({time.time()-t0:.0f}s)", flush=True) if d % 50 == 0 or d == t else None,
                       label="Contract")
    robust = sorted([r for r in lb if r["robust"]], key=lambda r: -r["val_sharpe"])
    baselines = []
    runs = [("applied (ATM, 30 DTE, hold 10, no trail)", BASE)]
    if lb:
        runs.append(("sweep best: " + json.dumps({k: lb[0]["params"][k] for k in GRID}), lb[0]["params"]))
    if robust and robust[0] is not lb[0]:
        runs.append(("best ROBUST: " + json.dumps({k: robust[0]["params"][k] for k in GRID}), robust[0]["params"]))
    for label, combo in runs:
        for w in ("train", "val", "all"):
            baselines.append(summarize(run_combo(precomp, split_map, combo, w), f"{label} [{w}]"))
    # one-at-a-time view: each knob alone on top of the applied config
    singles = []
    for k, vals in GRID.items():
        for v in vals:
            if v == BASE.get(k):
                continue
            c = {**BASE, k: v}
            s = {w: summarize(run_combo(precomp, split_map, c, w), f"{k}={v} [{w}]") for w in ("train", "val")}
            singles.append({"knob": k, "value": v, "train": s["train"], "val": s["val"]})
    report = {"split_date": str(pd.Timestamp(split_date).date()), "tickers": len(precomp), "grid": GRID, "base": BASE,
              "leaderboard": lb, "robust": robust, "baselines": baselines, "singles": singles,
              "completed_at": datetime.now().isoformat(), "seconds": round(time.time() - t0)}
    with open(os.path.join(ROOT, "tests", "contract_sweep_report.json"), "w", encoding="utf-8") as f:
        json.dump(ob._json_clean(report), f, indent=2)
    print("\n=== top 10 (validation Sharpe, net) ===")
    for r in lb[:10]:
        print(f"  val_sh={r['val_sharpe']:<6} valPF={r['val_pf']:<5} trPF={r['train_pf']:<5} val_n={r['val_trades']:<4} tr_n={r['train_trades']:<5} "
              f"bars={r['val_avg_bars_held']:<4} robust={str(r['robust']):<5} {json.dumps({k: r['params'][k] for k in GRID})}")
    print(f"robust: {len(robust)}")
    print("\n=== one knob at a time (train → validation, net PF / avg) ===")
    for s in singles:
        tr, va = s["train"], s["val"]
        print(f"  {s['knob']:<18}={str(s['value']):<5} train n={tr.get('n', 0):4d} PF={tr.get('profit_factor', 0):<5} avg={tr.get('avg_pnl', 0):+6.2f}%   "
              f"val n={va.get('n', 0):4d} PF={va.get('profit_factor', 0):<5} avg={va.get('avg_pnl', 0):+6.2f}%")
    print("\n=== SUMMARY ===")
    for b in baselines:
        if b.get("n"):
            print(fmt_row(b))
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
