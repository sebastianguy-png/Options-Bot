"""
Quick end-to-end smoke test of the backtest engine after an engine change:
downloads a handful of tickers, pre-computes (process pool), runs the backtest
with the default rules at two hold lengths and prints what came out.

    py tests/engine_smoke.py
"""
import os, sys, time, warnings
from collections import Counter
from datetime import date, timedelta
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import options_bot as ob                              # noqa: E402

TICKERS = ["AAPL", "AMD", "PLUG", "PFE", "F", "SOFI", "INTC", "NVDA"]
fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def main():
    t0 = time.time()
    ob._load_garch_cache()
    hist = ob._bulk_download(TICKERS, ob.BT_HISTORY_PERIOD, min_rows=ob.BT_WARMUP_BARS + ob.OPTION_DTE_BARS + 2)
    print(f"downloaded {len(hist)} tickers in {time.time()-t0:.0f}s; bars: " + ", ".join(f"{s}={len(h)}" for s, h in hist.items()))
    check(len(hist) >= 6, "at least 6 tickers with a full 3y frame")
    pc = ob._precompute_many(hist, label="Smoke")
    check(len(pc) == len(hist), "precompute succeeded for every ticker (process pool)")
    sym, fr = next(iter(pc.items()))
    check(fr["n"] > ob.BT_WARMUP_BARS + 50, f"{sym}: frame has {fr['n']} bars")
    g = fr["garch_vol"]
    check(all(v == v for v in g[ob.BT_WARMUP_BARS:]), f"{sym}: GARCH forecast present on every evaluated bar")
    c = ob.ctx_at(fr, ob.BT_WARMUP_BARS)
    check(c["regime"] in ("low", "normal", "elevated", "high", "extreme") or isinstance(c["regime"], str),
          f"{sym}: context at the first evaluated bar = regime {c['regime']}, trend {c['market_trend']}")

    for hold in (7, 21):
        trades = []
        for s, f in pc.items():
            trades += ob.run_backtest_ticker(s, hold_days=hold, params=dict(ob.BT_DEFAULT_PARAMS), precomp=f)
        mix = Counter(t["exit_reason"] for t in trades)
        bars = [t["bars_held"] for t in trades]
        print(f"hold {hold:2d}: {len(trades)} trades, exit mix {dict(mix)}, max bars held {max(bars) if bars else 0}")
        check(len(trades) > 0, f"hold {hold}: produced trades")
        check(all(t["bars_held"] <= min(hold, ob.OPTION_DTE_BARS) for t in trades), f"hold {hold}: no trade held past the cap")
        first_idx = min((f["dates"].index(t["entry_date"]) for s, f in pc.items() for t in trades if t["ticker"] == s), default=10**9)
        check(first_idx >= ob.BT_WARMUP_BARS, f"hold {hold}: first entry at bar {first_idx} ≥ warm-up {ob.BT_WARMUP_BARS}")
        last_ok = all(f["dates"].index(t["exit_date"]) < f["n"] for s, f in pc.items() for t in trades if t["ticker"] == s)
        check(last_ok, f"hold {hold}: every exit inside the frame")
        if hold == 7:
            check("end_of_hold" in mix, "hold 7: max-hold exits appear")

    # walk-forward style split must not read past idx_end
    s, f = next(iter(pc.items()))
    split = f["n"] - 100
    tr = ob.run_backtest_ticker(s, hold_days=21, params=dict(ob.BT_DEFAULT_PARAMS), precomp=f, idx_end=split)
    check(all(f["dates"].index(t["exit_date"]) < split for t in tr), f"{s}: training trades exit before the split bar")

    # signal-log simulator: a signal 3 bars ago is 'pending' unless an exit already fired,
    # one 40 bars ago always resolves, one on the last bar cannot be simulated at all
    h = hist[s]
    for back, ok in ((3, lambda r: r == "pending" or isinstance(r, dict)), (40, lambda r: isinstance(r, dict)), (0, lambda r: r is None)):
        entry = {"strategy": "buy_call", "price_at_signal": float(h["Close"].iloc[-back - 1]),
                 "bar_date": str(h.index[-back - 1])[:10], "signal_date": str(h.index[-back - 1])[:10], "hold_days": 21, "ticker": s}
        res = ob._simulate_signal_exit(entry, h)
        check(ok(res), f"simulator {back} bars back → {res if isinstance(res, str) or res is None else 'exit ' + res['exit_reason'] + ' day ' + str(res['exit_day_idx'])}")

    print(f"\n{'SMOKE OK' if not fails else 'SMOKE FAILED: ' + str(fails)}  ({time.time()-t0:.0f}s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
