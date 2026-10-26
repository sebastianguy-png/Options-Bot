"""
Daily vol-surface slice for every name in the live universe → surface_history.csv
(one row per name per day).  Nobody sells historical option surfaces for free, so
the bot records its own from the chains it already reads; after a few months this
is the dataset for the "don't overpay for vol" gate, the variance-premium study
and any spread / vol-selling strategy test.

Columns: date, ticker, price, hv30, iv_atm30, iv_atm60, iv_put95, iv_call105,
         skew (put95 − call105), term (atm60 − atm30), vrp (atm30 − hv30), dte30, dte60, source

    py tests/surface_log.py            (run it inside market hours; ~2–4 min for 140 names)
"""
import csv, os, sys, time, warnings
from datetime import datetime
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import yfinance as yf                                 # noqa: E402
import options_bot as ob                              # noqa: E402

OUT = os.path.join(ROOT, "surface_history.csv")
COLS = ["date", "ticker", "price", "hv30", "iv_atm30", "iv_atm60", "iv_put95", "iv_call105", "skew", "term", "vrp", "dte30", "dte60", "source"]


def _iv_at(df, strike_target):
    """IV (%) of the listed strike nearest the target, from rows with a solved IV."""
    if df is None or df.empty:
        return None
    d = df[df["impliedVolatility"].fillna(0) > 0]
    if d.empty:
        return None
    row = d.iloc[(d["strike"] - strike_target).abs().argmin()]
    iv = float(row["impliedVolatility"]) * 100
    return round(iv, 1) if 1 < iv < 400 else None


def main():
    t0 = time.time()
    tickers = ob.focus_universe()
    hist = ob._bulk_download(tickers, "3mo")
    ad = ob._alpaca_data()
    source = "alpaca" if ad is not None else "yahoo"
    today = datetime.now().strftime("%Y-%m-%d")
    rows, done = [], 0
    for sym in tickers:
        h = hist.get(sym)
        if h is None or len(h) < 35:
            continue
        try:
            closes = h["Close"]
            price = float(closes.iloc[-1])
            lr = np.log(closes / closes.shift(1)).dropna()
            hv30 = round(float(lr.tail(30).std() * np.sqrt(252) * 100), 1)
            tkr = yf.Ticker(sym)
            exp30, dte30 = ob._best_exp(tkr, 30)
            exp60, dte60 = ob._best_exp(tkr, 60)
            c30 = ob._option_chain(tkr, exp30, sym) if exp30 else None
            c60 = ob._option_chain(tkr, exp60, sym) if (exp60 and exp60 != exp30) else None
            iv_c = _iv_at(c30.calls, price) if c30 else None
            iv_p = _iv_at(c30.puts, price) if c30 else None
            iv_atm30 = round((iv_c + iv_p) / 2, 1) if (iv_c and iv_p) else (iv_c or iv_p)
            iv_put95 = _iv_at(c30.puts, price * 0.95) if c30 else None
            iv_call105 = _iv_at(c30.calls, price * 1.05) if c30 else None
            iv_atm60 = None
            if c60:
                a, b = _iv_at(c60.calls, price), _iv_at(c60.puts, price)
                iv_atm60 = round((a + b) / 2, 1) if (a and b) else (a or b)
            rows.append({"date": today, "ticker": sym, "price": round(price, 2), "hv30": hv30, "iv_atm30": iv_atm30, "iv_atm60": iv_atm60,
                         "iv_put95": iv_put95, "iv_call105": iv_call105,
                         "skew": round(iv_put95 - iv_call105, 1) if (iv_put95 and iv_call105) else None,
                         "term": round(iv_atm60 - iv_atm30, 1) if (iv_atm60 and iv_atm30) else None,
                         "vrp": round(iv_atm30 - hv30, 1) if iv_atm30 else None,
                         "dte30": dte30, "dte60": dte60, "source": source})
            done += 1
        except Exception as e:
            print(f"  {sym}: {e}")
        time.sleep(0.15)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerows(rows)
    vrp = [r["vrp"] for r in rows if r["vrp"] is not None]
    print(f"{today}: {done}/{len(tickers)} names logged from {source} in {time.time()-t0:.0f}s → {OUT}"
          + (f"; median IV−HV {np.median(vrp):+.1f} pts, {np.mean([v > 0 for v in vrp])*100:.0f}% of names have IV above realized" if vrp else ""))


if __name__ == "__main__":
    main()
