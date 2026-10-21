"""
Does the applied escape rule set hold on the FULL S&P 500 universe (the dashboard's
"full" scan) the way it did on the 99-ticker focus list?  Same engine, same costs,
same train/validation split; per leg.

    py tests/escape_universe_check.py          → tests/escape_universe_report.json
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

APPLIED = {**ob.BT_DEFAULT_PARAMS, "entry_mode": "escape", "escape_normal": "skip", "escape_breakout_vol": 2.0, "garch_mode": "off",
           "max_spread_pct": 15, "stop_loss_pct": 0, "profit_target_pct": 0, "trail_activate_pct": 0, "stagnant_days": 0,
           "hold_days": 10, "stock_stop_pct": 0}
SHIPPED = {**ob.BT_DEFAULT_PARAMS, **ob.BT_DEFAULT_EXITS}


def main():
    t0 = time.time()
    ob._load_garch_cache()
    out = {}
    for label, tickers in (("focus", list(dict.fromkeys(ob.FOCUS_LIST))), ("full S&P 500", ob.get_universe(force_mode="full"))):
        prep = ob._wfo_prepare(tickers, label=label[:5])
        if prep is None:
            print(f"{label}: no data"); continue
        precomp, split_map, split_date = prep
        out[label] = {"tickers": len(precomp), "split": str(pd.Timestamp(split_date).date()), "runs": []}
        print(f"\n=== {label}: {len(precomp)} tickers, split {str(split_date)[:10]} ({time.time()-t0:.0f}s) ===")
        for name, combo in (("APPLIED escape rules (calls only counted below)", APPLIED), ("shipped rules", SHIPPED)):
            for w in ("train", "val", "all"):
                trades = run_combo(precomp, split_map, combo, w)
                b = summarize(trades, f"{name} [{w}]")
                calls = summarize([t for t in trades if t["strategy"] == "buy_call"], f"{name} CALLS ONLY [{w}]")
                out[label]["runs"] += [b, calls]
                if b.get("n"):
                    print(fmt_row(b))
                if calls.get("n"):
                    print("   " + fmt_row(calls).split("\n")[0])
    with open(os.path.join(ROOT, "tests", "escape_universe_report.json"), "w", encoding="utf-8") as f:
        json.dump(ob._json_clean({"completed_at": datetime.now().isoformat(), "seconds": round(time.time() - t0), **out}), f, indent=2)
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
