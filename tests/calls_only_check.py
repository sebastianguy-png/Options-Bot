"""
Reproducibility + calls-only view of the applied rules over six years.
Prints the stats of ALL trades and of the CALLS only (what the live bot trades),
by year and by escape case, and writes the trade list to
tests/calls_only_trades_<tag>.json so two runs can be diffed
(py tests/calls_only_check.py tagA ; py tests/calls_only_check.py tagB).
A fourth argument "live" runs the bot's universe (focus + small/mid) instead of the focus list.
A third argument "nobreak" switches the breakout leg off (escape_breakout_vol 99 = the
applied live rules since Oct 21); the dump then carries the escape-case flags.
"""
import json, os, sys, time, warnings
from collections import defaultdict
from datetime import datetime
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)
import options_bot as ob                              # noqa: E402
from escape_longrun import run, stats, pf_of          # noqa: E402

ob.BT_HISTORY_PERIOD = sys.argv[2] if len(sys.argv) > 2 else "6y"      # py tests/calls_only_check.py <tag> [period]
BASE = {**ob.BT_DEFAULT_PARAMS, "entry_mode": "escape", "escape_normal": "skip", "escape_breakout_vol": 2.0, "garch_mode": "gate",
        "max_spread_pct": 15, "otm_factor_call": 0.97, "dte_target": 30, "stop_loss_pct": 0, "profit_target_pct": 0,
        "trail_activate_pct": 0, "stagnant_days": 0, "hold_days": 21, "stock_stop_pct": 0, "stock_trail_pct": 7, "breadth_chg5_min": -0.15}
if len(sys.argv) > 3 and sys.argv[3] == "nobreak":
    BASE["escape_breakout_vol"] = 99.0                                 # reversion only (what the bot trades)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%H%M")
    t0 = time.time()
    ob._load_garch_cache()
    universe = ob.focus_universe() if (len(sys.argv) > 4 and sys.argv[4] == "live") else list(dict.fromkeys(ob.FOCUS_LIST))   # "live" = focus + small/mid, the bot's universe
    precomp, _, _ = ob._wfo_prepare(universe, label=f"CallsOnly-{tag}")
    ctx = ob.market_context_history()
    print(f"  context: {len(ctx)} dates, vix NaN {int(ctx['vix'].isna().sum())}, regime counts {ctx['regime'].value_counts().to_dict()}, "
          f"breadth NaN {int(ctx['breadth50_chg5'].isna().sum()) if 'breadth50_chg5' in ctx else 'n/a'}")
    bars = {s: (str(pc['index'][0])[:10], str(pc['index'][-1])[:10], int(pc['n'])) for s, pc in precomp.items()}
    tr = run(precomp, BASE)
    calls = [t for t in tr if t["strategy"] == "buy_call"]
    by = defaultdict(list)
    for t in calls:
        by[t["entry_date"][:4]].append(t["pnl_pct"])
    print(f"ALL strategies: {stats(tr)}")
    print(f"CALLS ONLY: {stats(calls)}  by year: " + "  ".join(f"{y[2:]}:{pf_of(p)}({len(p)})" for y, p in sorted(by.items())))
    for c in ("sig_escape_reversion", "sig_escape_breakout"):
        print(f"  {c[11:]}: {stats([t for t in calls if t.get(c)])}")
    out = os.path.join(ROOT, "tests", f"calls_only_trades_{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ob._json_clean({"tag": tag, "when": datetime.now().isoformat(), "bars": bars, "calls": stats(calls), "all": stats(tr),
                                  "trades": [{k: t.get(k) for k in ("ticker", "strategy", "entry_date", "exit_date", "entry_price", "strike",
                                                                    "entry_opt", "exit_opt", "pnl_pct", "cost_pct", "exit_reason", "iv_est",
                                                                    "sig_escape_reversion", "sig_escape_breakout")} for t in tr]}), f)
    print(f"({time.time()-t0:.0f}s) → {out}")


if __name__ == "__main__":
    main()
