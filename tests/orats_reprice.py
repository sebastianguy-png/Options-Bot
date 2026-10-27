"""
Reprice the backtest's trades with REAL end-of-day option quotes from ORATS.

The backtest prices every option from HV30 x 1.15 through Black-Scholes and
charges a measured spread; nobody has checked that against what the contracts
actually cost.  ORATS keeps the near-close chain (quotes stamped ~15:46 ET,
inside the bot's entry window) for every listed name back to 2007, so for each
backtest trade this script

  1. pulls the entry-day chain (/hist/strikes, 14-45 DTE), picks the expiry
     closest to 30 DTE and the listed strike nearest 0.97 x the stock price
     (the live contract rule), and takes the real bid / ask / IV;
  2. pulls that contract's daily history (/hist/strikes/options) and takes the
     bid / ask on the backtest's exit date (its exits are stock-based, so the
     date does not depend on the option price);
  3. recomputes the P&L buying at the ask and selling at the bid (and mid to
     mid), and compares the profit factor, the entry IV and the entry premium
     with what the model assumed.

    py tests/orats_reprice.py tests/calls_only_trades_<tag>.json [--liquid] [--max N] [--out tests/orats_reprice_<tag>.json]

--liquid applies the scanner's contract filter (OI >= 10, volume >= 10, spread <= 15%) so
trades the live bot would have skipped drop out; without it the nearest strike is priced
whatever its liquidity.  Replies are cached in tests/orats_cache/ (git-ignored).

Two requests per trade (one per ticker-date for the chain); the token lives in
autotrade_config.json as orats_token.  ORATS allows 1,000 requests a minute
and 20,000 a month on the delayed plan.
"""
import hashlib, json, os, sys, time, statistics
from collections import Counter, defaultdict
from datetime import date, datetime
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

API = "https://api.orats.io/datav2"
MIN_DTE, MAX_DTE, DTE_TARGET, ITM = 14, 45, 30, 0.97          # options_bot MIN_DTE / MAX_DTE / dte_target / otm_factor_call
MIN_OI, MIN_VOL, MAX_SPREAD = 10, 10, 15.0                      # the live pick_option filter (MIN_OPEN_INTEREST / MIN_VOLUME / applied max_spread_pct)
CACHE_DIR = os.path.join("tests", "orats_cache")              # every reply kept on disk: a re-run or a variant costs no requests


def pf_of(p):
    w = sum(x for x in p if x > 0)
    l = -sum(x for x in p if x < 0)
    return round(w / l, 2) if l > 0 else (float("inf") if w > 0 else 0.0)


def stats(p):
    if not p:
        return {"n": 0}
    return {"n": len(p), "net": round(statistics.mean(p), 2), "win": round(100 * sum(1 for x in p if x > 0) / len(p), 1), "pf": pf_of(p)}


class Orats:
    def __init__(self, token):
        self.token, self.s, self.n, self.cache, self.hits = token, requests.Session(), 0, {}, 0
        os.makedirs(CACHE_DIR, exist_ok=True)

    def get(self, path, **q):
        key = hashlib.sha1((path + "?" + "&".join(f"{k}={q[k]}" for k in sorted(q))).encode()).hexdigest()
        fp = os.path.join(CACHE_DIR, key + ".json")
        if os.path.exists(fp):
            with open(fp, encoding="utf-8") as f:
                self.hits += 1
                return json.load(f)
        data = self._fetch(path, **q)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data

    def _fetch(self, path, **q):
        q["token"] = self.token
        for attempt in range(4):
            r = self.s.get(API + path, params=q, timeout=60)
            self.n += 1
            if r.status_code == 429:
                time.sleep(5 + 5 * attempt)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:200]}")
            return (r.json() or {}).get("data") or []
        raise RuntimeError(f"{path}: retries exhausted")

    def strikes(self, ticker, trade_date):
        key = (ticker, trade_date)
        if key not in self.cache:
            self.cache[key] = self.get("/hist/strikes", ticker=ticker, tradeDate=trade_date, dte=f"{MIN_DTE},{MAX_DTE}")
        return self.cache[key]

    def option_history(self, ticker, expir, strike):
        return self.get("/hist/strikes/options", ticker=ticker, expirDate=expir, strike=strike)


def pick(rows, side="call", liquid=False):
    """The live contract rule on an ORATS entry-day chain: expiry nearest 30 DTE in [14, 45], strike nearest 0.97 x price.
    liquid=True applies pick_option's filter too: OI >= 10 and volume >= 10 first, then the nearest strike, then
    reject (None) when its spread exceeds 15% - exactly what the scanner does, so a trade can drop out."""
    exps = {}
    for r in rows:
        d = r.get("dte")
        if d is None or not (MIN_DTE <= d <= MAX_DTE):
            continue
        exps.setdefault(r["expirDate"], d)
    if not exps:
        return None
    best = None
    for e, d in sorted(exps.items()):
        if best is None or abs(d - DTE_TARGET) < abs(exps[best] - DTE_TARGET):
            best = e
    cand = [r for r in rows if r["expirDate"] == best]
    S = float(cand[0].get("stockPrice") or cand[0].get("spotPrice") or 0)
    if S <= 0:
        return None
    factor = ITM if side == "call" else 1 / ITM
    if liquid:
        cand = [r for r in cand if (r.get(f"{side}OpenInterest") or 0) >= MIN_OI and (r.get(f"{side}Volume") or 0) >= MIN_VOL]
        if not cand:
            return None
    row = min(cand, key=lambda r: abs(float(r["strike"]) - S * factor))
    if liquid:
        bid, ask = float(row.get(f"{side}BidPrice") or 0), float(row.get(f"{side}AskPrice") or 0)
        if bid <= 0 or ask <= bid or (ask - bid) / ((ask + bid) / 2) * 100 > MAX_SPREAD:
            return None
    return row, S, exps[best]


def fields(row, side):
    p = side
    bid, ask = float(row.get(f"{p}BidPrice") or 0), float(row.get(f"{p}AskPrice") or 0)
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else None
    return {"bid": bid, "ask": ask, "mid": mid, "iv": row.get(f"{p}MidIv"), "value": row.get(f"{p}Value"),
            "oi": row.get(f"{p}OpenInterest"), "volume": row.get(f"{p}Volume"), "smv_vol": row.get("smvVol"),
            "delta": row.get("delta"), "quote_time": row.get("quoteDate")}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = args[0] if args else os.path.join("tests", "calls_only_trades_refit1_6y.json")
    maxn = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else None
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join("tests", "orats_reprice_" + os.path.basename(src).replace("calls_only_trades_", "").replace(".json", "") + ".json")
    side = "put" if "--puts" in sys.argv else "call"
    liquid = "--liquid" in sys.argv
    if liquid and "--out" not in sys.argv:
        out = out.replace(".json", "_liquid.json")
    cfg = json.load(open("autotrade_config.json", encoding="utf-8"))
    token = cfg.get("orats_token")
    if not token:
        print("orats_token missing from autotrade_config.json"); sys.exit(2)
    api = Orats(token)
    d = json.load(open(src, encoding="utf-8"))
    trades = [t for t in d["trades"] if t["strategy"] == ("buy_call" if side == "call" else "buy_put")]
    if maxn:
        trades = trades[:maxn]
    print(f"{len(trades)} {side} trades from {src}")
    t0, rows, skipped = time.time(), [], defaultdict(int)
    for i, t in enumerate(trades, 1):
        tk, ed, xd = t["ticker"], t["entry_date"], t["exit_date"]
        try:
            chain = api.strikes(tk, ed)
            if not chain:
                skipped["no entry chain"] += 1
                continue
            pk = pick(chain, side, liquid)
            if not pk:
                skipped["no liquid contract (live filter)" if liquid else "no expiry in window"] += 1
                continue
            row, S, dte = pk
            ent = fields(row, side)
            if not ent["mid"] or ent["ask"] <= 0:
                skipped["no two-sided entry quote"] += 1
                continue
            K, exp = float(row["strike"]), row["expirDate"]
            hist = api.option_history(tk, exp, K)
            upto = [h for h in hist if h["tradeDate"] <= xd]
            if not upto:
                skipped["no exit history"] += 1
                continue
            xrow = upto[-1]
            ex = fields(xrow, side)
            if ex["bid"] <= 0 and not ex["mid"]:
                skipped["no exit quote"] += 1
                continue
            exit_bid = ex["bid"] if ex["bid"] > 0 else 0.0
            exit_mid = ex["mid"] if ex["mid"] else exit_bid
            real_pct = (exit_bid - ent["ask"]) / ent["ask"] * 100
            mid_pct = (exit_mid - ent["mid"]) / ent["mid"] * 100
            rows.append({**{k: t.get(k) for k in ("ticker", "strategy", "entry_date", "exit_date", "entry_price", "strike", "entry_opt", "exit_opt", "pnl_pct", "cost_pct", "exit_reason", "iv_est")},
                         "orats_price": S, "orats_strike": K, "orats_exp": exp, "orats_dte": dte,
                         "entry_bid": ent["bid"], "entry_ask": ent["ask"], "entry_mid": ent["mid"], "entry_iv": ent["iv"], "entry_smv": ent["smv_vol"],
                         "entry_oi": ent["oi"], "entry_volume": ent["volume"], "entry_delta": ent["delta"], "entry_quote_time": ent["quote_time"],
                         "exit_date_used": xrow["tradeDate"], "exit_bid": exit_bid, "exit_ask": ex["ask"], "exit_mid": exit_mid,
                         "spread_pct_entry": round((ent["ask"] - ent["bid"]) / ent["mid"] * 100, 2),
                         "spread_pct_exit": round((ex["ask"] - ex["bid"]) / ex["mid"] * 100, 2) if ex["mid"] and ex["bid"] > 0 else None,
                         "pnl_real_pct": round(real_pct, 2), "pnl_mid_pct": round(mid_pct, 2),
                         "flags": [f for f, ok in (("exit date missing", xrow["tradeDate"] != xd), ("expired", xd > exp), ("thin OI", (ent["oi"] or 0) < 10)) if ok]})
            if i % 25 == 0 or i == len(trades):
                print(f"  {i}/{len(trades)}  {api.n} requests  {time.time()-t0:.0f}s   model PF {pf_of([r['pnl_pct'] for r in rows])}  real PF {pf_of([r['pnl_real_pct'] for r in rows])}")
        except Exception as e:
            skipped[f"error: {str(e)[:60]}"] += 1
        time.sleep(0.05)

    with open(out, "w", encoding="utf-8") as f:                 # partial save first: the rows cost requests
        json.dump({"summary": {"partial": True, "n_priced": len(rows)}, "trades": rows}, f)
    model = [r["pnl_pct"] for r in rows]
    real = [r["pnl_real_pct"] for r in rows]
    mid = [r["pnl_mid_pct"] for r in rows]
    by = defaultdict(lambda: {"model": [], "real": []})
    for r in rows:
        by[r["entry_date"][:4]]["model"].append(r["pnl_pct"]); by[r["entry_date"][:4]]["real"].append(r["pnl_real_pct"])
    iv_ratio = [r["entry_iv"] * 100 / r["iv_est"] for r in rows if r.get("entry_iv") and r.get("iv_est")]
    prem_ratio = [r["entry_mid"] / r["entry_opt"] for r in rows if r.get("entry_opt")]
    summary = {
        "source": src, "when": datetime.now().isoformat(timespec="minutes"), "requests": api.n, "cache_hits": api.hits,
        "selection": "live filter: OI>=10, volume>=10, spread<=15%" if liquid else "nearest strike, no liquidity filter",
        "n_input": len(trades), "n_priced": len(rows),
        "skipped": dict(skipped),
        "model": stats(model), "real_ask_to_bid": stats(real), "mid_to_mid": stats(mid),
        "by_year": {y: {"n": len(v["model"]), "model_pf": pf_of(v["model"]), "real_pf": pf_of(v["real"])} for y, v in sorted(by.items())},
        "entry_spread_pct_median": round(statistics.median(r["spread_pct_entry"] for r in rows), 2) if rows else None,
        "exit_spread_pct_median": round(statistics.median(r["spread_pct_exit"] for r in rows if r["spread_pct_exit"] is not None), 2) if rows else None,
        "model_cost_pct_median": round(statistics.median(r["cost_pct"] for r in rows), 2) if rows else None,
        "entry_iv_over_model_median": round(statistics.median(iv_ratio), 3) if iv_ratio else None,
        "entry_premium_over_model_median": round(statistics.median(prem_ratio), 3) if prem_ratio else None,
        "flags": dict(Counter(f for r in rows for f in r["flags"])),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "trades": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
