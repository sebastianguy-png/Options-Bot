"""
Measure today's bid/ask spreads on the contracts the bot actually trades — the
~30-DTE options within ±10% of the stock price with real open interest — for
every focus-list ticker, and write spread_model.json for the backtest's cost
model (options_bot._spread_cost_pct).

    py tests/measure_spreads.py

The number that matters is the ROUND-TRIP cost as a % of the option premium:
buy at the ask, sell at the bid → (ask − bid) / mid.  No historical option
quotes exist, so today's spread per ticker stands in for the past; the
premium-bucket medians are the fallback for tickers without a measurement.
"""
import json, os, statistics, sys, time, warnings
from datetime import datetime
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import yfinance as yf                                 # noqa: E402
import options_bot as ob                              # noqa: E402

BUCKETS = [0.5, 1.0, 3.0, 10.0, 1e9]                  # premium upper bounds ($)
FALLBACK = dict(ob.SPREAD_FALLBACK)


def main():
    tickers = list(dict.fromkeys(ob.FOCUS_LIST))
    hist = ob._bulk_download(tickers, "5d")
    rows, by_ticker, t0 = [], {}, time.time()
    for i, sym in enumerate(tickers, 1):
        try:
            tkr = yf.Ticker(sym)
            exp, dte = ob._best_exp(tkr)
            if not exp or sym not in hist:
                print(f"  {sym:<6} no expiry / no price")
                continue
            price = float(hist[sym]["Close"].iloc[-1])
            chain = ob._yf_call(tkr.option_chain, exp)
            mine = []
            for side, df in (("call", chain.calls), ("put", chain.puts)):
                if df is None or df.empty:
                    continue
                df = df[(df["strike"] >= price * 0.9) & (df["strike"] <= price * 1.1)]
                df = df[df["openInterest"].fillna(0) >= ob.MIN_OPEN_INTEREST]
                for _, r in df.iterrows():
                    bid, ask = ob._as_float(r.get("bid")), ob._as_float(r.get("ask"))
                    if bid > 0 and ask > bid:
                        mid = (bid + ask) / 2
                        rows.append({"ticker": sym, "side": side, "strike": float(r["strike"]), "mid": round(mid, 3),
                                     "spread_pct": round((ask - bid) / mid * 100, 2), "dte": dte})
                        mine.append((ask - bid) / mid * 100)
            if mine:
                by_ticker[sym] = {"spread_pct": round(statistics.median(mine), 2), "n": len(mine),
                                  "price": round(price, 2), "dte": dte, "exp": exp}
                print(f"  {sym:<6} ${price:8.2f}  {len(mine):3d} contracts  median spread {statistics.median(mine):5.1f}%   ({i}/{len(tickers)}, {time.time()-t0:.0f}s)")
            else:
                print(f"  {sym:<6} ${price:8.2f}  no two-sided ATM quotes with OI ≥ {ob.MIN_OPEN_INTEREST}")
        except Exception as e:
            print(f"  {sym:<6} failed: {e}")
        time.sleep(0.25)

    if not rows:
        # Yahoo zeroes every bid/ask outside the regular session — nothing was measured.
        from broker import _nyse_regular_session_open
        print("\nNo two-sided quotes at all" + ("" if _nyse_regular_session_open() else " — the market is closed (Yahoo blanks bid/ask off-hours)")
              + f"; {ob.SPREAD_MODEL_FILE} left untouched. Run this between 9:45 and 16:00 ET.")
        sys.exit(2)

    buckets = []
    lo = 0.0
    for hi in BUCKETS:
        vals = [r["spread_pct"] for r in rows if lo <= r["mid"] < hi]
        med = round(statistics.median(vals), 2) if len(vals) >= 5 else FALLBACK[hi]
        buckets.append([hi, med, len(vals)])
        lo = hi
    overall = round(statistics.median([r["spread_pct"] for r in rows]), 2) if rows else None
    model = {"measured_at": datetime.now().isoformat(timespec="minutes"), "universe": "focus",
             "n_tickers": len(by_ticker), "n_contracts": len(rows), "overall_median_pct": overall,
             "buckets": [[hi, med] for hi, med, _ in buckets], "bucket_counts": {str(hi): n for hi, _, n in buckets},
             "by_ticker": by_ticker}
    ob._atomic_json_write(ob.SPREAD_MODEL_FILE, model)

    print(f"\n{len(rows)} contracts on {len(by_ticker)} tickers; overall median round-trip spread {overall}% of premium")
    print("by premium:  " + "   ".join(f"<${hi:g}: {med}% (n={n})" for hi, med, n in buckets))
    ranked = sorted(by_ticker.items(), key=lambda kv: kv[1]["spread_pct"])
    print("tightest:    " + ", ".join(f"{s} {v['spread_pct']}%" for s, v in ranked[:10]))
    print("widest:      " + ", ".join(f"{s} {v['spread_pct']}%" for s, v in ranked[-10:]))
    for lim in (8, 15, 25):
        print(f"tickers with median spread ≤ {lim}%: {sum(1 for _, v in ranked if v['spread_pct'] <= lim)} of {len(ranked)}")
    print(f"written → {ob.SPREAD_MODEL_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
