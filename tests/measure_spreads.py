"""
Measure today's bid/ask spreads on the contracts the bot actually trades — the
~30-DTE options within ±10% of the stock price with real open interest — for
every focus-list ticker, and write spread_model.json for the backtest's cost
model (options_bot._spread_cost_pct).

    py tests/measure_spreads.py [--source yahoo|tradier|alpaca] [--out file.json] [--compare]

--source picks the quote feed (Yahoo by default, as spread_model.json was built);
--out writes somewhere other than spread_model.json; --compare prints each
ticker's new median against the current spread_model.json.

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


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv and sys.argv.index(flag) + 1 < len(sys.argv) else default


def _chain(source, tkr, sym, exp):
    if source == "tradier":
        return ob._tradier_data().chain(sym, exp)
    if source == "alpaca":
        return ob._alpaca_data().chain(sym, exp)
    return ob._yf_call(tkr.option_chain, exp)


def main():
    source, out_file, compare = _arg("--source", "yahoo"), _arg("--out", ob.SPREAD_MODEL_FILE), "--compare" in sys.argv
    if source == "tradier" and ob._tradier_data() is None:
        print("Tradier feed not available (token / probe)"); sys.exit(2)
    if source == "alpaca" and ob._alpaca_data() is None:
        print("Alpaca feed not available"); sys.exit(2)
    old = dict((ob._get_spread_model().get("by_ticker") or {})) if compare else {}
    tickers = ob.focus_universe()          # the live universe (focus + small/mid extension)
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
            chain = _chain(source, tkr, sym, exp)
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
        time.sleep(0.25 if source == "yahoo" else 0.05)

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
    model = {"measured_at": datetime.now().isoformat(timespec="minutes"), "universe": "focus", "source": source,
             "n_tickers": len(by_ticker), "n_contracts": len(rows), "overall_median_pct": overall,
             "buckets": [[hi, med] for hi, med, _ in buckets], "bucket_counts": {str(hi): n for hi, _, n in buckets},
             "by_ticker": by_ticker}
    ob._atomic_json_write(out_file, model)

    print(f"\n{len(rows)} contracts on {len(by_ticker)} tickers; overall median round-trip spread {overall}% of premium")
    print("by premium:  " + "   ".join(f"<${hi:g}: {med}% (n={n})" for hi, med, n in buckets))
    ranked = sorted(by_ticker.items(), key=lambda kv: kv[1]["spread_pct"])
    print("tightest:    " + ", ".join(f"{s} {v['spread_pct']}%" for s, v in ranked[:10]))
    print("widest:      " + ", ".join(f"{s} {v['spread_pct']}%" for s, v in ranked[-10:]))
    for lim in (8, 15, 25):
        print(f"tickers with median spread ≤ {lim}%: {sum(1 for _, v in ranked if v['spread_pct'] <= lim)} of {len(ranked)}")
    print(f"written → {out_file}  ({time.time()-t0:.0f}s)")
    if compare and old:
        both = [(s, old[s]["spread_pct"], v["spread_pct"]) for s, v in by_ticker.items() if s in old]
        if both:
            diffs = [n - o for _, o, n in both]
            print(f"\nvs current {ob.SPREAD_MODEL_FILE} ({len(both)} tickers in both): old median {statistics.median(o for _, o, _ in both):.1f}%  new median {statistics.median(n for _, _, n in both):.1f}%  "
                  f"median change {statistics.median(diffs):+.1f} pts; wider on {sum(1 for d in diffs if d > 0)}, tighter on {sum(1 for d in diffs if d < 0)}")
            worst = sorted(both, key=lambda x: x[2] - x[1])
            print("most wider:  " + ", ".join(f"{s} {o:.1f}→{n:.1f}" for s, o, n in worst[-8:][::-1]))
            print("most tighter: " + ", ".join(f"{s} {o:.1f}→{n:.1f}" for s, o, n in worst[:8]))


if __name__ == "__main__":
    main()
