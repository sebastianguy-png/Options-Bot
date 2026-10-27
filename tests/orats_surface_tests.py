"""
Two surface tests on the repriced trades, with real ORATS data (27 Oct 2026).

A. Does the vol surface at entry predict the trade?  Per trade, from ORATS
   /hist/summaries and /hist/ivrank on the entry date: 30-day skew (25-delta
   put IV minus 25-delta call IV), term structure (30d minus 90d IV, and ORATS'
   contango), 1-year IV rank / percentile, IV30 over ORATS' own realized vol,
   ORATS' "skewing".  For each metric: quartile table of the real (ask-to-bid)
   P&L, correlation, the extreme quartiles by year, and a walk-forward cut
   (threshold and direction from 2022-23 trades, applied to 2025-26).

B. Bull call spreads on real quotes: keep the repriced long 0.97 call, sell the
   1.03 / 1.05 / 1.10 call of the same expiry.  Entry debit = long ask minus short
   bid; exit = long bid minus short ask on the backtest's exit date (one exit-day
   chain request per trade, cached).  Profit factor and mean per trade vs the
   outright call, mid-to-mid as well, and how often the short strike capped the
   winner.

    py tests/orats_surface_tests.py [tests/orats_reprice_honest6y.json] [tests/orats_reprice_honest6y_liquid.json]

Replies are cached in tests/orats_cache/ (see orats_reprice.py); about three
requests per trade the first time, none after.
"""
import json, math, os, sys, time, statistics
from collections import defaultdict
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)
from orats_reprice import Orats, pf_of, stats          # noqa: E402

SHORT_FACTORS = (1.03, 1.05, 1.10)


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))]


def corr(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den else 0.0


def quartile_table(rows, key, pnl="pnl_real_pct"):
    vals = [r[key] for r in rows]
    cuts = [q(vals, 0.25), q(vals, 0.5), q(vals, 0.75)]
    out = []
    for i, (lo, hi) in enumerate([(-1e18, cuts[0]), (cuts[0], cuts[1]), (cuts[1], cuts[2]), (cuts[2], 1e18)]):
        b = [r for r in rows if lo <= r[key] < hi] if i < 3 else [r for r in rows if r[key] >= lo]
        if i == 0:
            b = [r for r in rows if r[key] < cuts[0]]
        p = [r[pnl] for r in b]
        out.append({"q": i + 1, "lo": round(lo, 3) if i else None, "hi": round(hi, 3) if i < 3 else None, "n": len(p),
                    "pf": pf_of(p) if p else None, "mid_pf": pf_of([r["pnl_mid_pct"] for r in b]) if b else None,
                    "mean": round(statistics.mean(p), 1) if p else None, "win": round(100 * sum(1 for x in p if x > 0) / len(p), 0) if p else None})
    return cuts, out


def by_year(rows, key, lo, hi):
    """Real PF per year for the trades inside [lo, hi) vs the rest."""
    d = defaultdict(lambda: {"in": [], "out": []})
    for r in rows:
        d[r["entry_date"][:4]]["in" if lo <= r[key] < hi else "out"].append(r["pnl_real_pct"])
    return {y: {"in_n": len(v["in"]), "in_pf": pf_of(v["in"]) if v["in"] else None, "out_n": len(v["out"]), "out_pf": pf_of(v["out"]) if v["out"] else None} for y, v in sorted(d.items())}


def walk_forward(rows, key):
    """Direction and threshold from 2022-23 (drop the worse-performing extreme quartile), applied to 2025-26."""
    train = [r for r in rows if r["entry_date"] < "2024-01-01"]
    test = [r for r in rows if r["entry_date"] >= "2024-01-01"]
    if len(train) < 20 or len(test) < 10:
        return None
    vals = [r[key] for r in train]
    lo, hi = q(vals, 0.25), q(vals, 0.75)
    pf_lo = pf_of([r["pnl_real_pct"] for r in train if r[key] < lo])
    pf_hi = pf_of([r["pnl_real_pct"] for r in train if r[key] >= hi])
    if pf_lo <= pf_hi:
        rule, keep = f"skip {key} < {lo:.3f}", lambda r: r[key] >= lo
    else:
        rule, keep = f"skip {key} >= {hi:.3f}", lambda r: r[key] < hi
    kept = [r["pnl_real_pct"] for r in test if keep(r)]
    skipped = [r["pnl_real_pct"] for r in test if not keep(r)]
    return {"rule": rule, "train_q1_pf": pf_lo, "train_q4_pf": pf_hi,
            "test_all": stats([r["pnl_real_pct"] for r in test]), "test_kept": stats(kept), "test_skipped": stats(skipped)}


def part_a(api, rows, label):
    print(f"\n===== A. surface at entry → outcome  ({label}, {len(rows)} trades)")
    t0 = time.time()
    feats = []
    for i, r in enumerate(rows, 1):
        try:
            s = api.get("/hist/summaries", ticker=r["ticker"], tradeDate=r["entry_date"])
            v = api.get("/hist/ivrank", ticker=r["ticker"], tradeDate=r["entry_date"])
        except Exception as e:
            print(f"  {r['ticker']} {r['entry_date']}: {e}")
            continue
        s = s[0] if s else {}
        v = v[0] if v else {}
        f = dict(r)
        try:
            f["skew30"] = (s["dlt75Iv30d"] - s["dlt25Iv30d"]) * 100            # 25-delta put IV − 25-delta call IV, vol points
            f["skew30_rel"] = (s["dlt75Iv30d"] - s["dlt25Iv30d"]) / s["iv30d"]
            f["term30_90"] = (s["iv30d"] - s["iv90d"]) * 100                      # > 0: front end above the back (fear now)
            f["contango"] = s.get("contango")
            f["iv30_rvol30"] = s["iv30d"] / s["rVol30"] if s.get("rVol30") else None
            f["skewing"] = s.get("skewing")
            f["iv_rank_1y"] = v.get("ivRank1y")
            f["iv_pct_1y"] = v.get("ivPct1y")
            f["iv30d"] = s["iv30d"] * 100
        except (KeyError, TypeError, ZeroDivisionError):
            continue
        feats.append(f)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}  {api.n} requests ({api.hits} cache hits)  {time.time()-t0:.0f}s")
    report = {}
    metrics = [("skew30", "25d put IV − 25d call IV (30d), vol pts"), ("term30_90", "IV30 − IV90, vol pts (inverted > 0)"),
               ("contango", "ORATS contango (term slope)"), ("iv_rank_1y", "IV rank, 1 year"), ("iv_pct_1y", "IV percentile, 1 year"),
               ("iv30_rvol30", "IV30 / realized 30 (ORATS)"), ("skewing", "ORATS skewing")]
    for key, desc in metrics:
        rs = [f for f in feats if f.get(key) is not None]
        if len(rs) < 40:
            continue
        cuts, table = quartile_table(rs, key)
        c = corr([f[key] for f in rs], [f["pnl_real_pct"] for f in rs])
        lo_q, hi_q = cuts[0], cuts[2]
        yr_lo = by_year(rs, key, -1e18, lo_q)
        yr_hi = by_year(rs, key, hi_q, 1e18)
        wf = walk_forward(rs, key)
        report[key] = {"desc": desc, "n": len(rs), "cuts": [round(x, 3) for x in cuts], "quartiles": table, "corr": round(c, 3),
                       "bottom_quartile_by_year": yr_lo, "top_quartile_by_year": yr_hi, "walk_forward": wf}
        print(f"\n  {key}: {desc}   (n={len(rs)}, corr with P&L {c:+.2f})")
        print("    %-4s %9s %9s %4s %8s %8s %7s %5s" % ("Q", "from", "to", "n", "real PF", "mid PF", "mean%", "win%"))
        for t in table:
            print("    Q%-3d %9s %9s %4d %8s %8s %+7.1f %5.0f" % (t["q"], "" if t["lo"] is None else f"{t['lo']:.3f}", "" if t["hi"] is None else f"{t['hi']:.3f}", t["n"], t["pf"], t["mid_pf"], t["mean"], t["win"]))
        print("    bottom quartile by year: " + "  ".join(f"{y}: {v['in_n']}@{v['in_pf']} vs rest {v['out_n']}@{v['out_pf']}" for y, v in yr_lo.items()))
        print("    top quartile by year:    " + "  ".join(f"{y}: {v['in_n']}@{v['in_pf']} vs rest {v['out_n']}@{v['out_pf']}" for y, v in yr_hi.items()))
        if wf:
            print(f"    walk-forward ({wf['rule']}; train Q1 PF {wf['train_q1_pf']}, Q4 PF {wf['train_q4_pf']}): 2025-26 all {wf['test_all']} → kept {wf['test_kept']}, skipped {wf['test_skipped']}")
    return report


def part_b(api, rows, label):
    print(f"\n===== B. bull call spreads on real quotes  ({label}, {len(rows)} trades)")
    t0 = time.time()
    out = {k: [] for k in SHORT_FACTORS}
    outright, capped, skipped = [], defaultdict(int), defaultdict(int)
    debit_ratio = defaultdict(list)
    for i, r in enumerate(rows, 1):
        try:
            entry = [x for x in api.strikes(r["ticker"], r["entry_date"]) if x["expirDate"] == r["orats_exp"]]
            exitc = [x for x in api.get("/hist/strikes", ticker=r["ticker"], tradeDate=r["exit_date_used"], dte="0,90") if x["expirDate"] == r["orats_exp"]]
        except Exception as e:
            skipped[f"error {str(e)[:40]}"] += 1
            continue
        if not entry or not exitc:
            skipped["no chain"] += 1
            continue
        S = r["orats_price"]
        K = r["orats_strike"]
        le = next((x for x in entry if float(x["strike"]) == K), None)
        lx = next((x for x in exitc if float(x["strike"]) == K), None)
        if not le or not lx:
            skipped["long leg missing"] += 1
            continue
        l_ask, l_bid, lx_bid, lx_ask = float(le["callAskPrice"] or 0), float(le["callBidPrice"] or 0), float(lx["callBidPrice"] or 0), float(lx["callAskPrice"] or 0)
        if l_ask <= 0:
            skipped["no long ask"] += 1
            continue
        outright.append((lx_bid - l_ask) / l_ask * 100)
        S_exit = float(lx.get("stockPrice") or 0)
        for k in SHORT_FACTORS:
            se = min((x for x in entry if float(x["strike"]) > K and float(x.get("callBidPrice") or 0) > 0), key=lambda x: abs(float(x["strike"]) - S * k), default=None)
            if se is None:
                skipped[f"no short strike {k}"] += 1
                continue
            sK = float(se["strike"])
            sx = next((x for x in exitc if float(x["strike"]) == sK), None)
            if sx is None:
                skipped[f"short exit missing {k}"] += 1
                continue
            s_bid, s_ask = float(se["callBidPrice"] or 0), float(se["callAskPrice"] or 0)
            sx_ask, sx_bid = float(sx["callAskPrice"] or 0), float(sx["callBidPrice"] or 0)
            debit = l_ask - s_bid
            if debit <= 0.05:
                skipped[f"debit too small {k}"] += 1
                continue
            value = max(lx_bid - sx_ask, 0.0)
            mid_in = (l_ask + l_bid) / 2 - (s_bid + s_ask) / 2
            mid_out = (lx_bid + lx_ask) / 2 - (sx_bid + sx_ask) / 2
            out[k].append({"ticker": r["ticker"], "entry_date": r["entry_date"], "exit_date": r["exit_date_used"], "long": K, "short": sK, "width": sK - K,
                           "debit": round(debit, 3), "exit_value": round(value, 3), "pnl_pct": round((value - debit) / debit * 100, 2),
                           "pnl_mid_pct": round((mid_out - mid_in) / mid_in * 100, 2) if mid_in > 0 else None,
                           "call_pnl_pct": round((lx_bid - l_ask) / l_ask * 100, 2), "capped": S_exit >= sK, "debit_over_call": round(debit / l_ask, 3)})
            debit_ratio[k].append(debit / l_ask)
            if S_exit >= sK:
                capped[k] += 1
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}  {api.n} requests ({api.hits} cache hits)  {time.time()-t0:.0f}s")
    rep = {"outright_call_same_chains": stats(outright), "skipped": dict(skipped), "variants": {}}
    print("\n    %-22s %4s %8s %8s %7s %5s %10s %9s" % ("leg", "n", "real PF", "mid PF", "mean%", "win%", "debit/call", "capped%"))
    print("    %-22s %4d %8s %8s %+7.1f %5.0f" % ("outright 0.97 call", len(outright), pf_of(outright), "", statistics.mean(outright), 100 * sum(1 for x in outright if x > 0) / len(outright)))
    for k in SHORT_FACTORS:
        v = out[k]
        if not v:
            continue
        p = [x["pnl_pct"] for x in v]
        m = [x["pnl_mid_pct"] for x in v if x["pnl_mid_pct"] is not None]
        same = [x["call_pnl_pct"] for x in v]
        rep["variants"][str(k)] = {"n": len(v), "spread": stats(p), "spread_mid": stats(m), "call_same_trades": stats(same),
                                   "debit_over_call_median": round(statistics.median(debit_ratio[k]), 3), "capped": capped[k],
                                   "by_year": {y: {"n": len(g), "spread_pf": pf_of([x["pnl_pct"] for x in g]), "call_pf": pf_of([x["call_pnl_pct"] for x in g])}
                                               for y, g in sorted(_group(v).items())}}
        print("    %-22s %4d %8s %8s %+7.1f %5.0f %10.2f %8.0f   (call on same trades: PF %s, mean %+.1f)" % (
            f"spread short {k:.2f}×S", len(v), pf_of(p), pf_of(m) if m else "-", statistics.mean(p), 100 * sum(1 for x in p if x > 0) / len(p),
            statistics.median(debit_ratio[k]), 100 * capped[k] / len(v), pf_of(same), statistics.mean(same)))
        print("      by year: " + "  ".join(f"{y}: n={g['n']} spread {g['spread_pf']} / call {g['call_pf']}" for y, g in rep["variants"][str(k)]["by_year"].items()))
    rep["trades"] = {str(k): v for k, v in out.items()}
    return rep


def _group(v):
    d = defaultdict(list)
    for x in v:
        d[x["entry_date"][:4]].append(x)
    return d


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--")] or [os.path.join("tests", "orats_reprice_honest6y.json"), os.path.join("tests", "orats_reprice_honest6y_liquid.json")]
    cfg = json.load(open("autotrade_config.json", encoding="utf-8"))
    api = Orats(cfg["orats_token"])
    report = {"when": datetime.now().isoformat(timespec="minutes"), "sets": {}}
    for fn in files:
        rows = json.load(open(fn, encoding="utf-8"))["trades"]
        label = os.path.basename(fn).replace("orats_reprice_", "").replace(".json", "")
        rep = {"n": len(rows)}
        rep["A"] = part_a(api, rows, label) if fn == files[0] else "see first set (subset)"
        rep["B"] = part_b(api, rows, label)
        report["sets"][label] = rep
    report["requests"] = api.n
    report["cache_hits"] = api.hits
    out = os.path.join("tests", "orats_surface_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\n{api.n} requests, {api.hits} cache hits → {out}")


if __name__ == "__main__":
    main()
