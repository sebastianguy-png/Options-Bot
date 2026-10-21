"""
Trade post-mortem: every autotrade fill vs what the model assumed — the spread
actually paid at entry and exit (fill vs quote mid), bars held, exit reason,
P&L by leg / escape case / sector.  Run it any time; it reads performance.json
and tracked_positions.json.

    py tests/postmortem.py
"""
import json, os, statistics, sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import options_bot as ob                              # noqa: E402


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def main():
    with ob._perf_lock:
        trades = [t for t in ob.load_performance() if t.get("source") == "autotrade"]
    with ob.track_lock:
        open_pos = [p for p in ob.load_tracked() if p.get("auto")]
    if not trades and not open_pos:
        print("No autotrade fills yet."); return
    print(f"{len(trades)} closed autotrade trade(s), {len(open_pos)} open\n")
    for t in trades:
        q_in, q_out = t.get("quote_at_entry") or {}, t.get("quote_at_exit") or {}
        model_cost = ob._spread_cost_pct(t.get("ticker", ""), float(t.get("entry_price") or 0))
        paid = (t.get("entry_slippage_pct") or 0) + (t.get("exit_slippage_pct") or 0)
        print(f"  {t.get('exit_date')} {t.get('ticker'):<6} {t.get('strategy'):<9} {str(t.get('escape_case') or ''):<10} "
              f"entry {t.get('entry_price')} (mid {q_in.get('mid')}, slip {t.get('entry_slippage_pct')}%)  "
              f"exit {t.get('exit_price')} (mid {q_out.get('mid')}, slip {t.get('exit_slippage_pct')}%)  "
              f"spread paid {paid:+.2f}% vs model {model_cost:.1f}%  held {t.get('bars_held')}d  "
              f"P&L {t.get('pnl_total'):+.2f} ({t.get('pnl_pct'):+.1f}%)  {str(t.get('notes') or '')[:60]}")
    if trades:
        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        gw, gl = sum(wins), -sum(p for p in pnls if p < 0)
        print(f"\nnet avg {statistics.mean(pnls):+.2f}%  win {len(wins)}/{len(pnls)}  PF {round(gw / gl, 2) if gl else 'inf'}  "
              f"median entry slip {med([t.get('entry_slippage_pct') for t in trades])}%  median exit slip {med([t.get('exit_slippage_pct') for t in trades])}%  "
              f"model cost {med([ob._spread_cost_pct(t.get('ticker', ''), float(t.get('entry_price') or 0)) for t in trades])}%")
        for key in ("strategy", "escape_case", "sector"):
            groups = defaultdict(list)
            for t in trades:
                groups[str(t.get(key))].append(t["pnl_pct"])
            print(f"  by {key}: " + ", ".join(f"{k} n={len(v)} avg {statistics.mean(v):+.1f}%" for k, v in groups.items()))
    for p in open_pos:
        sig = p.get("last_signal") or {}
        print(f"  OPEN {p.get('ticker'):<6} {p.get('strategy'):<9} {str(p.get('escape_case') or ''):<10} entry {p.get('entry_price')} "
              f"(slip {p.get('entry_slippage_pct')}%) now {sig.get('current_option_mid')} P&L {sig.get('pct_change')}%")


if __name__ == "__main__":
    main()
