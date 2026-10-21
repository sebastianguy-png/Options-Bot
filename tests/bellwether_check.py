"""
Do bellwether earnings (AAPL MSFT GOOGL AMZN META NVDA TSLA JPM) change what our
trades do?  Post-hoc study on the six-year trades of the applied rules (calls,
GARCH gate, breadth gate −0.15, measured spreads):

  anticipation  — entries made 1–2 trading days BEFORE a bellwether reports
  reaction      — entries made within 5 trading days AFTER a bellwether reported,
                  split by that stock's reaction (2-day move around the report:
                  < −3% bad, > +3% good, else flat)

Earnings dates come from yfinance (get_earnings_dates); coverage is printed —
if it falls short of six years the study runs on what is covered and says so.

    py tests/bellwether_check.py         → tests/bellwether_report.json
"""
import json, os, sys, time, warnings
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)
import options_bot as ob                              # noqa: E402
from escape_longrun import run, stats, pf_of          # noqa: E402

ob.BT_HISTORY_PERIOD = "6y"
BASE = {**ob.BT_DEFAULT_PARAMS, "entry_mode": "escape", "escape_normal": "skip", "escape_breakout_vol": 2.0, "garch_mode": "gate",
        "max_spread_pct": 15, "otm_factor_call": 0.97, "dte_target": 30, "stop_loss_pct": 0, "profit_target_pct": 0,
        "trail_activate_pct": 0, "stagnant_days": 0, "hold_days": 21, "stock_stop_pct": 0, "stock_trail_pct": 7, "breadth_chg5_min": -0.15}
BELLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM"]


def earnings_events(precomp):
    """[(date, sym, reaction_pct)] for every past bellwether report with price data around it."""
    import yfinance as yf
    out, coverage = [], {}
    for s in BELLS:
        try:
            df = ob._yf_call(lambda: yf.Ticker(s).get_earnings_dates(limit=60))
        except Exception as e:
            print(f"  {s}: earnings dates unavailable ({e})"); continue
        if df is None or len(df) == 0:
            continue
        idx = pd.DatetimeIndex(df.index)
        idx = (idx.tz_convert(None) if idx.tz is not None else idx).normalize()
        pc = precomp.get(s)
        if pc is None:
            continue
        pidx = pd.DatetimeIndex(pc["index"]); pidx = (pidx.tz_localize(None) if pidx.tz is not None else pidx).normalize()
        closes = pd.Series(pc["price"], index=pidx)
        past = sorted({d for d in idx if d <= pidx[-1] and d >= pidx[0]})
        coverage[s] = (str(past[0].date()) if past else None, len(past))
        for d in past:
            i = pidx.searchsorted(d)
            if 1 <= i < len(pidx) - 1:
                react = (closes.iloc[i + 1] / closes.iloc[i - 1] - 1) * 100     # 2-day move around the report (pre- or post-market)
                out.append((pidx[i], s, round(float(react), 2)))
    print("  coverage (first past report, count):", coverage)
    return sorted(out), coverage


def main():
    t0 = time.time()
    ob._load_garch_cache()
    precomp, _, _ = ob._wfo_prepare(list(dict.fromkeys(ob.FOCUS_LIST)), label="Bellwether")
    events, coverage = earnings_events(precomp)
    if not events:
        print("no earnings events — nothing to test"); return
    ev_dates = pd.DatetimeIndex([e[0] for e in events])
    first_cov = min(e[0] for e in events)
    trades = [t for t in run(precomp, BASE) if pd.Timestamp(t["entry_date"]) >= first_cov]
    spy_idx = pd.DatetimeIndex(ob.market_context_history().index)
    print(f"  {len(events)} bellwether reports from {first_cov.date()}; {len(trades)} trades in the covered window")

    def tdays(a, b):            # trading days from a to b (signed)
        return int(spy_idx.searchsorted(b) - spy_idx.searchsorted(a))

    groups = defaultdict(list)
    for t in trades:
        d = pd.Timestamp(t["entry_date"])
        nxt = [e for e in events if 0 < tdays(d, e[0]) <= 2]
        prev = [e for e in events if 0 <= tdays(e[0], d) <= 5]
        groups["all"].append(t)
        groups["1–2 days before a bellwether report" if nxt else "not before a report"].append(t)
        if prev:
            worst = min(e[2] for e in prev); best = max(e[2] for e in prev)
            tag = "≤5 days after a BAD bellwether reaction (<−3%)" if worst < -3 else \
                  "≤5 days after a GOOD bellwether reaction (>+3%)" if best > 3 else "≤5 days after a FLAT bellwether reaction"
            groups[tag].append(t)
            t["bell_after"] = tag
        else:
            groups["no bellwether report in the prior 5 days"].append(t)
    report = {"coverage": coverage, "events": len(events), "first_covered": str(first_cov.date()), "groups": {}}
    print()
    for name, v in groups.items():
        s = stats(v)
        by_year = defaultdict(list)
        for t in v:
            by_year[t["entry_date"][:4]].append(t["pnl_pct"])
        print(f"  {name:<52} n={s['n']:4d} net {s.get('net', 0):+6.2f}% PF {s.get('pf', 0):<5} win {s.get('win', 0)}%   "
              + "  ".join(f"{y[2:]}:{pf_of(p)}({len(p)})" for y, p in sorted(by_year.items())))
        report["groups"][name] = {**s, "by_year": {y: {"n": len(p), "pf": pf_of(p)} for y, p in by_year.items()}}
    # the market's own reaction: SPY over the 5 days after bad / good bellwether prints
    ctx = ob.market_context_history(); spy = ctx["spy_close"].astype(float)
    for tag, sel in (("bad", [e for e in events if e[2] < -3]), ("good", [e for e in events if e[2] > 3]), ("flat", [e for e in events if -3 <= e[2] <= 3])):
        rets = []
        for d, s_, r in sel:
            i = spy.index.searchsorted(d)
            if i + 5 < len(spy):
                rets.append((spy.iloc[i + 5] / spy.iloc[i] - 1) * 100)
        if rets:
            print(f"  SPY 5 days after a {tag} bellwether print: n={len(rets)} mean {np.mean(rets):+.2f}% median {np.median(rets):+.2f}% up {np.mean([x > 0 for x in rets])*100:.0f}%")
            report[f"spy_after_{tag}"] = {"n": len(rets), "mean": round(float(np.mean(rets)), 2), "median": round(float(np.median(rets)), 2)}
    with open(os.path.join(ROOT, "tests", "bellwether_report.json"), "w", encoding="utf-8") as f:
        json.dump(ob._json_clean({**report, "completed_at": datetime.now().isoformat(), "seconds": round(time.time() - t0)}), f, indent=2)
    print(f"\n({time.time()-t0:.0f}s) → tests/bellwether_report.json")


if __name__ == "__main__":
    main()
