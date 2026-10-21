"""
The paper → real-money gate, written down before the results exist so the bar
can't move later.  Reads the autotrade fills in performance.json.

    py tests/live_gate.py

Every criterion must pass:
  fills      ≥ 60 closed autotrade trades
  PF         ≥ 1.2 net
  drawdown   max peak-to-trough on the cumulative P&L% curve ≤ 25% (of one trade's premium units)
  slippage   median entry+exit slippage within 1.0% of the model's spread cost
  win rate   ≥ 40%
"""
import os, statistics, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import options_bot as ob                              # noqa: E402

GATE = {"fills": 60, "pf": 1.2, "max_dd_pct": 25.0, "slippage_tolerance_pct": 1.0, "win_rate": 40.0}


def main():
    with ob._perf_lock:
        trades = [t for t in ob.load_performance() if t.get("source") == "autotrade"]
    n = len(trades)
    pnls = [float(t["pnl_pct"]) for t in trades]
    wins = [p for p in pnls if p > 0]
    gw, gl = sum(wins), -sum(p for p in pnls if p < 0)
    pf = (gw / gl) if gl > 0 else (99.0 if gw > 0 else 0.0)
    cum = peak = dd = 0.0
    for p in pnls:
        cum += p; peak = max(peak, cum); dd = max(dd, peak - cum)
    slips = [(t.get("entry_slippage_pct") or 0) + (t.get("exit_slippage_pct") or 0) for t in trades]
    model = [ob._spread_cost_pct(t.get("ticker", ""), float(t.get("entry_price") or 0)) for t in trades]
    slip_gap = (statistics.median(slips) - statistics.median(model)) if trades else 0.0
    wr = (len(wins) / n * 100) if n else 0.0
    checks = [
        ("fills ≥ 60",            n >= GATE["fills"],                          f"{n}"),
        ("profit factor ≥ 1.2",   pf >= GATE["pf"],                            f"{pf:.2f}"),
        ("max drawdown ≤ 25%",    dd <= GATE["max_dd_pct"],                    f"{dd:.1f}%"),
        ("slippage within model", slip_gap <= GATE["slippage_tolerance_pct"],  f"paid − model = {slip_gap:+.2f}%"),
        ("win rate ≥ 40%",        wr >= GATE["win_rate"],                      f"{wr:.1f}%"),
    ]
    for name, ok, val in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<24} {val}")
    print("\nGATE:", "OPEN — the strategy may be considered for real money" if all(c[1] for c in checks) else "CLOSED — keep paper trading")


if __name__ == "__main__":
    main()
