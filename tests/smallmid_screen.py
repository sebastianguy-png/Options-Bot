"""
Small/mid-cap universe for the escape rules — step 1: screen and measure.

  1. S&P 400 + S&P 600 constituents (tests/smallmid_universe_raw.json, from Wikipedia)
  2. keep: price $8–$150, 20-day average dollar volume ≥ $15M, ≥ 250 daily bars
  3. measure the ~30-DTE option spread (±10% strikes, OI ≥ MIN_OPEN_INTEREST) on the
     survivors — Alpaca feed when its keys are configured (closing quotes off-hours),
     Yahoo otherwise (only works 09:45–16:00 ET)
  4. write tests/smallmid_universe.json: every survivor with price, $volume, spread

    py tests/smallmid_screen.py [--max N]
"""
import json, os, statistics, sys, time, warnings
from datetime import datetime
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import yfinance as yf                                 # noqa: E402
import options_bot as ob                              # noqa: E402

MIN_PRICE, MAX_PRICE, MIN_DOLLAR_VOL, MIN_BARS = 8.0, 150.0, 15e6, 250


def main():
    t0 = time.time()
    raw = json.load(open(os.path.join(ROOT, "tests", "smallmid_universe_raw.json"), encoding="utf-8"))
    sectors = {}
    tickers = []
    for k in ("sp400", "sp600"):
        for t in raw[k]["tickers"]:
            if t not in tickers and t not in ob.FOCUS_LIST:
                tickers.append(t)
        sectors.update({t: (s if isinstance(s, str) else "N/A") for t, s in raw[k].get("sectors", {}).items()})
    print(f"{len(tickers)} S&P 400/600 tickers (focus names removed)")
    hist = ob._bulk_download(tickers, "1y", min_rows=MIN_BARS)
    print(f"{len(hist)} with ≥{MIN_BARS} bars ({time.time()-t0:.0f}s)")
    rows = {}
    for sym, h in hist.items():
        px = float(h["Close"].iloc[-1])
        dv = float((h["Close"] * h["Volume"]).tail(20).mean())
        if MIN_PRICE <= px <= MAX_PRICE and dv >= MIN_DOLLAR_VOL:
            rows[sym] = {"price": round(px, 2), "dollar_vol_m": round(dv / 1e6, 1), "sector": sectors.get(sym, "N/A"),
                         "index": "sp400" if sym in raw["sp400"]["tickers"] else "sp600"}
    print(f"{len(rows)} pass price ${MIN_PRICE:g}–${MAX_PRICE:g} and ≥ ${MIN_DOLLAR_VOL/1e6:g}M/day")
    maxn = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else None
    syms = sorted(rows, key=lambda s: -rows[s]["dollar_vol_m"])[:maxn]

    use_yahoo = "--yahoo" in sys.argv          # real delayed NBBO during market hours (Alpaca's off-hours quotes are closing marks)
    ad = None if use_yahoo else ob._alpaca_data()
    print("spread source:", "Yahoo chains (market hours)" if use_yahoo else ("Alpaca feed" if ad is not None else "Yahoo (needs market hours)"))
    measured = 0
    for i, sym in enumerate(syms, 1):
        try:
            tkr = yf.Ticker(sym)
            exp, dte = ob._best_exp(tkr)
            if not exp:
                rows[sym]["spread_pct"] = None; continue
            chain = ob._yf_call(tkr.option_chain, exp) if use_yahoo else ob._option_chain(tkr, exp, sym)
            price = rows[sym]["price"]
            mine = []
            for df in (chain.calls, chain.puts):
                if df is None or df.empty:
                    continue
                df = df[(df["strike"] >= price * 0.9) & (df["strike"] <= price * 1.1)]
                df = df[df["openInterest"].fillna(0) >= ob.MIN_OPEN_INTEREST]
                for _, r in df.iterrows():
                    bid, ask = ob._as_float(r.get("bid")), ob._as_float(r.get("ask"))
                    if bid > 0 and ask > bid:
                        mine.append((ask - bid) / (bid + ask) * 200)
            rows[sym].update(spread_pct=round(statistics.median(mine), 2) if mine else None, n_contracts=len(mine), dte=dte, exp=exp)
            if mine:
                measured += 1
            if i % 25 == 0:
                print(f"  {i}/{len(syms)} measured {measured} ({time.time()-t0:.0f}s)")
        except Exception as e:
            rows[sym]["spread_pct"] = None; rows[sym]["error"] = str(e)[:80]
        time.sleep(0.2)
    out = {"screened_at": datetime.now().isoformat(timespec="minutes"), "criteria": {"price": [MIN_PRICE, MAX_PRICE], "min_dollar_vol": MIN_DOLLAR_VOL, "min_bars": MIN_BARS},
           "spread_source": "yahoo-intraday" if use_yahoo else ("alpaca-close" if ad is not None else "yahoo"), "tickers": {s: rows[s] for s in syms}}
    out_name = "smallmid_universe_intraday.json" if use_yahoo else "smallmid_universe.json"
    with open(os.path.join(ROOT, "tests", out_name), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    sp = [r["spread_pct"] for r in out["tickers"].values() if r.get("spread_pct") is not None]
    print(f"\n{len(syms)} screened, {len(sp)} with a spread; median spread {statistics.median(sp) if sp else 'n/a'}%; "
          f"≤8%: {sum(1 for x in sp if x <= 8)}  ≤15%: {sum(1 for x in sp if x <= 15)}  ({time.time()-t0:.0f}s) → tests/smallmid_universe.json")


if __name__ == "__main__":
    main()
