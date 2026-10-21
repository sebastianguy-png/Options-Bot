"""
Staged walk-forward sweep of the EXIT rules and the re-gated entries, with the
shared engine (the exact live rules).

  Stage A  exits × garch_mode, entry rules at the live defaults      (1,152 combos)
  Stage B  entry params × the 3 best exit sets from stage A           (576 combos)

Then the baselines — the old default rules, the new defaults, the stage-A and
stage-B winners and the best "robust" combo (profitable in BOTH windows) — on
train / validation / all bars, per strategy, with the exit mix.

    py tests/run_exit_sweep.py            # focus list → tests/exit_sweep_report.json
    py tests/run_exit_sweep.py --apply    # also save the stage-B winner as wfo_results.json
                                          # (the bot applies it at the next start if val Sharpe > 0)
"""
import argparse, json, os, sys, time, warnings
from collections import Counter
from datetime import datetime
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd                                   # noqa: E402
import options_bot as ob                              # noqa: E402

STAGE_A = {
    "garch_mode":         ["off", "gate"],
    "stop_loss_pct":      [15, 30, 50, 0],            # 0 = no premium stop (the premium is the max loss)
    "profit_target_pct":  [30, 60, 100, 0],           # 0 = no fixed target (trailing / RSI-BB / max hold decide)
    "trail_activate_pct": [15, 0],
    "stagnant_days":      [3, 0],
    "hold_days":          [5, 10, 21],
    "stock_stop_pct":     [0, 3, 6],
}
STAGE_B = {
    "rsi_oversold":       [30, 35],
    "rsi_overbought":     [65, 70],
    "bull_min":           [1, 2],
    "bear_min":           [2, 3],
    "adx_threshold":      [30, 35],
    "put_trend_gate":     ["off", "not_strong_bull", "bear_only"],
    "yield_block_calls":  [False, True],
}


def summarize(trades, label):
    if not trades:
        return {"label": label, "n": 0}
    m = ob._bt_metrics(trades)
    by = {}
    for s in ("buy_call", "buy_put", "sell_put"):
        st = [t for t in trades if t["strategy"] == s]
        if st:
            ms = ob._bt_metrics(st)
            by[s] = {k: ms[k] for k in ("count", "win_rate", "avg_pnl", "profit_factor", "sharpe", "avg_bars_held")}
    return {"label": label, "n": m["count"], "win_rate": m["win_rate"], "avg_pnl": m["avg_pnl"],
            "avg_pnl_trigger_fill": m["avg_pnl_trigger_fill"], "profit_factor": m["profit_factor"],
            "sharpe": m["sharpe"], "sortino": m["sortino"], "max_drawdown_pct": m["max_drawdown_pct"],
            "avg_bars_held": m["avg_bars_held"], "exit_mix": dict(Counter(t["exit_reason"] for t in trades)),
            "by_strategy": by}


def run_combo(precomp, split_map, combo, window):
    params, exits = ob._split_combo(combo)
    trades = []
    for sym, pc in precomp.items():
        kw = {"idx_end": split_map[sym]} if window == "train" else {"idx_start": split_map[sym]} if window == "val" else {}
        trades += ob.run_backtest_ticker(sym, params=params, exits=exits, precomp=pc, **kw)
    return trades


def fmt_row(b):
    s = (f"{b['label']:<60} n={b['n']:5d} win={b['win_rate']:5.1f}% avg={b['avg_pnl']:+6.2f}% "
         f"PF={b['profit_factor']:<5} sharpe={b['sharpe']:<6} bars={b['avg_bars_held']}")
    for st, v in b["by_strategy"].items():
        s += f"\n      {st:<10} n={v['count']:5d} win={v['win_rate']:5.1f}% avg={v['avg_pnl']:+6.2f}% PF={v['profit_factor']} bars={v['avg_bars_held']}"
    mix = ", ".join(f"{k} {v}" for k, v in sorted(b["exit_mix"].items(), key=lambda x: -x[1]))
    return s + f"\n      exits: {mix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--apply", action="store_true", help="save the stage-B winner as wfo_results.json")
    args = ap.parse_args()
    tickers = ob.get_universe(force_mode="full") if args.full else list(dict.fromkeys(ob.FOCUS_LIST))

    t0 = time.time()
    ob._load_garch_cache()
    prep = ob._wfo_prepare(tickers, label="Sweep")
    if prep is None:
        print("no data"); sys.exit(1)
    precomp, split_map, split_date = prep

    def prog(done, total):
        if done % 100 == 0 or done == total:
            print(f"    {done}/{total}  ({time.time()-t0:.0f}s)", flush=True)

    # ── Stage A ──
    combos_a = ob._expand_grid(STAGE_A)
    print(f"\n=== Stage A: {len(combos_a)} exit combos (entries at live defaults) ===")
    lb_a = ob._wfo_sweep(combos_a, precomp, split_map, top_n=30, progress=prog, label="StageA")
    seen, exit_sets = set(), []
    for row in sorted(lb_a, key=lambda r: (not r["robust"], -r["val_sharpe"])):   # robust ones first
        _, ex = ob._split_combo(row["params"])
        key = tuple(sorted(ex.items())) + (("garch_mode", row["params"]["garch_mode"]),)
        if key in seen:
            continue
        seen.add(key)
        exit_sets.append((ex, row["params"]["garch_mode"]))
        if len(exit_sets) == 3:
            break
    print(f"stage A done in {time.time()-t0:.0f}s; exit sets carried forward: {exit_sets}")

    # ── Stage B ──
    combos_b = [{**ent, "garch_mode": gm, **ex} for ent in ob._expand_grid(STAGE_B) for ex, gm in exit_sets]
    print(f"\n=== Stage B: {len(combos_b)} entry combos × carried exit sets ===")
    lb_b = ob._wfo_sweep(combos_b, precomp, split_map, top_n=30, progress=prog, label="StageB")
    print(f"stage B done in {time.time()-t0:.0f}s")

    best_a = lb_a[0] if lb_a else None
    best_b = lb_b[0] if lb_b else None
    robust = [r for r in lb_a + lb_b if r["robust"]]
    robust.sort(key=lambda r: r["val_sharpe"], reverse=True)
    best_robust = robust[0] if robust else None

    # ── Baselines ──
    old_default = {**ob.BT_DEFAULT_PARAMS, "yield_block_calls": True, **ob.BT_DEFAULT_EXITS}
    new_default = {**ob.BT_DEFAULT_PARAMS, **ob.BT_DEFAULT_EXITS}
    runs = [("OLD default rules (yield block on, exits as shipped)", old_default),
            ("default rules (yield block off, exits as shipped)", new_default)]
    if best_a:
        runs.append(("stage A best: " + json.dumps(best_a["params"], separators=(",", ":")), best_a["params"]))
    if best_b:
        runs.append(("stage B best: " + json.dumps(best_b["params"], separators=(",", ":")), best_b["params"]))
    if best_robust and best_robust is not best_b:
        runs.append(("best ROBUST: " + json.dumps(best_robust["params"], separators=(",", ":")), best_robust["params"]))
    baselines = []
    for label, combo in runs:
        for window in ("train", "val", "all"):
            baselines.append(summarize(run_combo(precomp, split_map, combo, window), f"{label} [{window}]"))

    report = {"universe": "full" if args.full else "focus", "tickers": len(precomp),
              "split_date": str(pd.Timestamp(split_date).date()), "stage_a_grid": STAGE_A, "stage_b_grid": STAGE_B,
              "exit_sets_carried": exit_sets, "stage_a": lb_a, "stage_b": lb_b,
              "best_robust": best_robust, "baselines": baselines, "completed_at": datetime.now().isoformat(),
              "seconds": round(time.time() - t0)}
    out = os.path.join(ROOT, "tests", "exit_sweep_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ob._json_clean(report), f, indent=2)

    print("\n=== STAGE A top 8 (by validation Sharpe) ===")
    for r in lb_a[:8]:
        print(f"  val_sharpe={r['val_sharpe']:<6} val_PF={r['val_pf']:<5} train_PF={r['train_pf']:<5} val_n={r['val_trades']:<5} "
              f"bars={r['val_avg_bars_held']:<4} robust={r['robust']}  {json.dumps(r['params'], separators=(',', ':'))}")
    print("\n=== STAGE B top 8 ===")
    for r in lb_b[:8]:
        print(f"  val_sharpe={r['val_sharpe']:<6} val_PF={r['val_pf']:<5} train_PF={r['train_pf']:<5} val_n={r['val_trades']:<5} "
              f"bars={r['val_avg_bars_held']:<4} robust={r['robust']}  {json.dumps(r['params'], separators=(',', ':'))}")
    print(f"\nrobust combos (PF > 1 in BOTH windows, n ≥ 200/100): {len(robust)}")
    print("\n=== SUMMARY ===")
    for b in baselines:
        if b.get("n"):
            print(fmt_row(b))
    print(f"\nreport → {out}  ({time.time()-t0:.0f}s total)")

    if args.apply and best_b:
        ob._save_wfo_results({"leaderboard": lb_b, "best": best_b, "suggestion": dict(best_b["params"]),
                              "split_date": str(pd.Timestamp(split_date).date()), "tickers_used": list(precomp),
                              "completed_at": datetime.now().isoformat()})
        print("saved as wfo_results.json — the bot applies it at the next start (val Sharpe > 0)")


if __name__ == "__main__":
    main()
