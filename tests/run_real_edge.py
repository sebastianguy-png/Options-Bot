"""
Run the walk-forward optimizer on the focus list with the SHARED engine and
write a report of the real (out-of-sample) edge.

    py tests/run_real_edge.py                # focus list
    py tests/run_real_edge.py --full         # whole S&P 500 (slow)

Outputs:  wfo_results.json (if the best combo validates), and
          tests/real_edge_report.json with the leaderboard, the default-rule
          baseline (train / validation / full), and per-strategy stats.
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np, pandas as pd                      # noqa: E402
import options_bot as ob                              # noqa: E402


def summarize(trades, label):
    if not trades:
        return {"label": label, "n": 0}
    m = ob._bt_metrics(trades)
    by = {}
    for s in ("buy_call", "buy_put", "sell_put"):
        st = [t for t in trades if t["strategy"] == s]
        if st:
            by[s] = {k: ob._bt_metrics(st)[k] for k in ("count", "win_rate", "avg_pnl", "avg_pnl_trigger_fill",
                                                        "profit_factor", "sharpe", "avg_bars_held", "exit_mix")}
    return {"label": label, "n": m["count"], "win_rate": m["win_rate"], "avg_pnl": m["avg_pnl"],
            "avg_pnl_trigger_fill": m["avg_pnl_trigger_fill"], "profit_factor": m["profit_factor"],
            "sharpe": m["sharpe"], "sortino": m["sortino"], "max_drawdown_pct": m["max_drawdown_pct"],
            "avg_bars_held": m["avg_bars_held"], "exit_mix": m["exit_mix"], "by_strategy": by}


def run_params(precomp_map, split_map, params, hold_days, window):
    trades = []
    for sym, pc in precomp_map.items():
        kw = {"idx_end": split_map[sym]} if window == "train" else {"idx_start": split_map[sym]} if window == "val" else {}
        trades += ob.run_backtest_ticker(sym, hold_days, params=params, precomp=pc, **kw)
    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    tickers = ob.get_universe(force_mode="full") if args.full else list(dict.fromkeys(ob.FOCUS_LIST))

    ob._load_garch_cache()
    t0 = time.time()
    print(f"=== Walk-forward optimizer on {len(tickers)} tickers, {len(ob._expand_grid(ob.WFO_PARAM_GRID))} combos ===")
    ob._run_wfo_job_inner(tickers)
    res = ob._wfo_state.get("results") or {}
    print(f"WFO done in {time.time()-t0:.0f}s; split {res.get('split_date')}; best = {res.get('best')}")

    # Rebuild the same data for the baseline / per-strategy breakdown
    hist_map = ob._bulk_download(tickers, ob.BT_HISTORY_PERIOD, min_rows=ob.BT_WARMUP_BARS + ob.OPTION_DTE_BARS + 2)
    precomp  = ob._precompute_many(hist_map, label="Report")
    all_dates = sorted(set().union(*[set(h.index) for h in hist_map.values()]))
    split_date = all_dates[int(len(all_dates) * 0.75)]
    split_map = {s: int(h.index.searchsorted(split_date)) for s, h in hist_map.items() if s in precomp}

    report = {"universe": "full" if args.full else "focus", "tickers": len(precomp),
              "split_date": str(pd.Timestamp(split_date).date()),
              "grid": ob.WFO_PARAM_GRID, "leaderboard": res.get("leaderboard"), "best": res.get("best"),
              "baselines": []}
    base = dict(ob.BT_DEFAULT_PARAMS)
    for label, params, hold in [
        ("default rules (live defaults), garch off",        base, 21),
        ("default rules, garch confluence",                  {**base, "garch_mode": "confluence"}, 21),
        ("default rules, garch gate",                        {**base, "garch_mode": "gate"}, 21),
    ]:
        for window in ("train", "val", "all"):
            tr = run_params(precomp, split_map, params, hold, window)
            report["baselines"].append(summarize(tr, f"{label} [{window}]"))
    if res.get("best"):
        bp = dict(res["best"]["params"]); hold = bp.pop("hold_days", 21)
        for window in ("train", "val", "all"):
            tr = run_params(precomp, split_map, bp, hold, window)
            report["baselines"].append(summarize(tr, f"WFO best {res['best']['params']} [{window}]"))

    out = os.path.join(ROOT, "tests", "real_edge_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ob._json_clean(report), f, indent=2)
    print("\n=== SUMMARY ===")
    for b in report["baselines"]:
        if b.get("n"):
            print(f"{b['label']:<75} n={b['n']:5d} win={b['win_rate']:5.1f}% avg={b['avg_pnl']:+6.2f}% "
                  f"(trigger-fill {b['avg_pnl_trigger_fill']:+6.2f}%) PF={b['profit_factor']} sharpe={b['sharpe']}")
            for s, v in b["by_strategy"].items():
                print(f"      {s:<10} n={v['count']:5d} win={v['win_rate']:5.1f}% avg={v['avg_pnl']:+6.2f}% PF={v['profit_factor']} bars={v['avg_bars_held']}")
    print(f"report → {out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
