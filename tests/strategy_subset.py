"""
Which legs carry the value?  Runs the best exit rule set from the sweep (and the
shipped rules for comparison) and reports NET-of-cost metrics for strategy
subsets: all / calls only / sell puts only / calls + sell puts / puts only.

    py tests/strategy_subset.py
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import options_bot as ob                              # noqa: E402

RULESETS = {
    "shipped rules":              {**ob.BT_DEFAULT_PARAMS, **ob.BT_DEFAULT_EXITS},
    "garch gate, no stop/target, hold 21": {**ob.BT_DEFAULT_PARAMS, "garch_mode": "gate", "stop_loss_pct": 0, "profit_target_pct": 0,
                                   "trail_activate_pct": 0, "stagnant_days": 0, "hold_days": 21, "stock_stop_pct": 0},
    "garch gate, no stop, target 100, hold 10": {**ob.BT_DEFAULT_PARAMS, "garch_mode": "gate", "stop_loss_pct": 0, "profit_target_pct": 100,
                                   "trail_activate_pct": 0, "stagnant_days": 0, "hold_days": 10, "stock_stop_pct": 0},
    "garch OFF, no stop/target, hold 21": {**ob.BT_DEFAULT_PARAMS, "garch_mode": "off", "stop_loss_pct": 0, "profit_target_pct": 0,
                                   "trail_activate_pct": 0, "stagnant_days": 0, "hold_days": 21, "stock_stop_pct": 0},
}
SUBSETS = {"all": ("buy_call", "buy_put", "sell_put"), "calls only": ("buy_call",), "sell puts only": ("sell_put",),
           "calls + sell puts": ("buy_call", "sell_put"), "puts only": ("buy_put",)}


def metrics(trades):
    if len(trades) < 5:
        return None
    m = ob._bt_metrics(trades)
    n = len(trades)
    return {"n": n, "win": m["win_rate"], "net": m["avg_pnl"], "gross": round(sum(t["pnl_gross_pct"] for t in trades) / n, 2),
            "pf": m["profit_factor"], "sharpe": m["sharpe"], "bars": m["avg_bars_held"]}


def main():
    t0 = time.time()
    ob._load_garch_cache()
    tickers = list(dict.fromkeys(ob.FOCUS_LIST))
    precomp, split_map, split_date = ob._wfo_prepare(tickers, label="Subset")
    out = {}
    for name, combo in RULESETS.items():
        params, exits = ob._split_combo(combo)
        trades = []
        for sym, pc in precomp.items():
            for t in ob.run_backtest_ticker(sym, params=params, exits=exits, precomp=pc):
                t["_val"] = pc["dates"].index(t["entry_date"]) >= split_map[sym]
                trades.append(t)
        print(f"\n=== {name} ===  (net of spread cost; split {str(split_date)[:10]})")
        out[name] = {}
        for sub, strats in SUBSETS.items():
            st = [t for t in trades if t["strategy"] in strats]
            row = {w: metrics([t for t in st if t["_val"] == (w == "val")]) for w in ("train", "val")}
            row["all"] = metrics(st)
            out[name][sub] = row
            def f(m): return "   —   " if not m else f"n={m['n']:4d} win={m['win']:4.1f}% net={m['net']:+6.2f}% gross={m['gross']:+6.2f}% PF={m['pf']:<5} bars={m['bars']}"
            print(f"  {sub:<18} train: {f(row['train'])}\n  {'':<18} val:   {f(row['val'])}")
    with open(os.path.join(ROOT, "tests", "strategy_subset_report.json"), "w", encoding="utf-8") as fh:
        json.dump(ob._json_clean(out), fh, indent=2)
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
