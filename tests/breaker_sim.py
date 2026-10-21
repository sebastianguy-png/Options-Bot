"""
Would a rolling-loss circuit breaker have protected the escape rules in 2022
without costing much in the good years?  Replays the six-year trade sequence
of the applied rules (+ GARCH gate) and pauses new entries for `pause_days`
after the last `window` closed trades net below `threshold` (% of premium).

    py tests/breaker_sim.py
"""
import os, sys, warnings
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)
import options_bot as ob                              # noqa: E402
from escape_longrun import run, pf_of                 # noqa: E402

ob.BT_HISTORY_PERIOD = "6y"
BASE = {**ob.BT_DEFAULT_PARAMS, "entry_mode": "escape", "escape_normal": "skip", "escape_breakout_vol": 2.0, "garch_mode": "gate",
        "max_spread_pct": 15, "otm_factor_call": 0.97, "dte_target": 30, "stop_loss_pct": 0, "profit_target_pct": 0,
        "trail_activate_pct": 0, "stagnant_days": 0, "hold_days": 21, "stock_stop_pct": 0, "stock_trail_pct": 7}
RULES = [(10, -30.0, 10), (10, -50.0, 10), (15, -40.0, 15), (20, -40.0, 20), (8, -40.0, 5)]   # (window, threshold %, pause days)


def replay(trades, window, threshold, pause_days):
    """Trades are taken in entry order; a trade is skipped while a pause is active; closes update the rolling window."""
    trades = sorted(trades, key=lambda t: t["entry_date"])
    closed = []           # (exit_date, pnl) of taken trades
    kept, pause_until, pauses = [], None, 0
    for t in trades:
        ed = datetime.strptime(t["entry_date"], "%Y-%m-%d")
        settled = [p for d, p in closed if d <= ed]           # only exits known at entry time
        if pause_until and ed < pause_until:
            continue
        if pause_until:                                           # pause over: start the window afresh (else it re-trips forever)
            closed = [c for c in closed if c[0] >= pause_until]
            pause_until = None
        if len(settled) >= window and sum(settled[-window:]) <= threshold:
            pause_until = ed + timedelta(days=pause_days); pauses += 1
            continue
        kept.append(t)
        closed.append((datetime.strptime(t["exit_date"], "%Y-%m-%d"), t["pnl_pct"]))
        closed.sort()
    return kept, pauses


def by_year(trades):
    g = defaultdict(list)
    for t in trades:
        g[t["entry_date"][:4]].append(t["pnl_pct"])
    return {y: (len(v), round(float(np.mean(v)), 1), pf_of(v)) for y, v in sorted(g.items())}


def main():
    ob._load_garch_cache()
    precomp, _, _ = ob._wfo_prepare(list(dict.fromkeys(ob.FOCUS_LIST)), label="Breaker")
    trades = run(precomp, BASE)
    p = [t["pnl_pct"] for t in trades]
    print(f"\nno breaker: n={len(p)} net {np.mean(p):+.2f}% PF {pf_of(p)}  by year {by_year(trades)}")
    for w, th, pd_ in RULES:
        kept, pauses = replay(trades, w, th, pd_)
        k = [t["pnl_pct"] for t in kept]
        print(f"window {w:2d} / net ≤ {th:+.0f}% / pause {pd_:2d}d: n={len(k)} net {np.mean(k):+.2f}% PF {pf_of(k)} pauses={pauses}  by year {by_year(kept)}")


if __name__ == "__main__":
    main()
