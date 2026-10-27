"""
Out-of-sample check of the one surface "lead" (handoff 7o) on the small/mid names.

ORATS `skewing` turned out to be a realized-vol construct, (rVol30 − rVol2y·(1−1/√45))·√45,
not option skew.  On the focus list its bottom quartile (< 0.506) lagged every year and the
walk-forward rule "skip skewing < 0.496" (fitted on 2022-23) improved 2025-26.  Here the same
fixed thresholds are applied, unchanged, to trades the rule has never seen: the 41 small/mid
names' reversion calls from the combined six-year backtest, repriced on real ORATS quotes.

    py tests/calls_only_check.py honestlive6y 6y nobreak live   (the bot's 140-name universe; trades of the 41 small/mid names are the OOS set)
    py tests/orats_oos_check.py                          (reprices the small/mid calls, pulls their surface, applies the rule)

One summaries request per trade plus the repricer's two; everything cached.
"""
import json, os, subprocess, sys, statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)
from orats_reprice import Orats, pf_of, stats          # noqa: E402

RULE_CUT = 0.496                                        # from the focus-list walk-forward (train 2022-23)
FOCUS_QUARTILES = [0.506, 0.785, 1.137]                 # the focus list's skewing quartile cuts (203 trades)
RVOL_QUARTILES = [0.295, 0.373, 0.481]                  # the focus list's rVol30 quartile cuts
RVOL2Y_QUARTILES = [0.257, 0.3, 0.379]                 # the focus list's rVol2y quartile cuts
RVOL30_CUT, RVOL2Y_CUT = 0.296, 0.258                   # the matching walk-forward cuts (train 2022-23): plain "skip the lowest-vol names"


def main():
    import options_bot as ob
    rep = json.load(open(os.path.join("tests", "calls_only_trades_honestlive6y.json"), encoding="utf-8"))
    small = set(ob.SMALLMID_LIST)
    calls = [t for t in rep["trades"] if t["strategy"] == "buy_call"]
    oos = [t for t in calls if t["ticker"] in small]
    print(f"{len(oos)} small/mid reversion calls out of {len(calls)} in the live universe (model PF {pf_of([t['pnl_pct'] for t in oos])}; "
          f"focus part {len(calls) - len(oos)} trades PF {pf_of([t['pnl_pct'] for t in calls if t['ticker'] not in small])})")
    src = os.path.join("tests", "calls_only_trades_smallmid6y.json")
    with open(src, "w", encoding="utf-8") as f:
        json.dump({"tag": "smallmid6y", "when": rep.get("when"), "calls": stats([t["pnl_pct"] for t in oos]), "trades": oos}, f)
    out = os.path.join("tests", "orats_reprice_smallmid6y.json")
    if "--no-reprice" not in sys.argv:
        r = subprocess.run([sys.executable, os.path.join("tests", "orats_reprice.py"), src, "--out", out], capture_output=True, text=True)
        print("\n".join(l for l in r.stdout.splitlines() if "/" in l and "requests" in l)[-400:])
        if r.returncode:
            print(r.stderr[-800:]); sys.exit(1)
    rows = json.load(open(out, encoding="utf-8"))["trades"]
    api = Orats(json.load(open("autotrade_config.json", encoding="utf-8"))["orats_token"])
    feats = []
    for r in rows:
        try:
            s = api.get("/hist/summaries", ticker=r["ticker"], tradeDate=r["entry_date"])
        except Exception as e:
            print(f"  {r['ticker']} {r['entry_date']}: {e}"); continue
        if not s:
            continue
        s = s[0]
        feats.append({**r, "skewing": s.get("skewing"), "rvol30": s.get("rVol30"), "rvol2y": s.get("rVol2y"), "iv30d": s.get("iv30d")})
    feats = [f for f in feats if f["skewing"] is not None]
    real = [f["pnl_real_pct"] for f in feats]
    print(f"\n{len(feats)} trades with surface data; real ask-to-bid: {stats(real)}; model: {stats([f['pnl_pct'] for f in feats])}; mid: {stats([f['pnl_mid_pct'] for f in feats])}")

    def table(key, cuts, label):
        print(f"\n  {label} — fixed focus-list quartile cuts {cuts}")
        edges = [-1e18] + cuts + [1e18]
        for i in range(4):
            b = [f for f in feats if edges[i] <= f[key] < edges[i + 1]]
            p = [f["pnl_real_pct"] for f in b]
            print("    Q%d %-16s n=%3d  real PF %-6s mid PF %-6s mean %+6.1f  win %3.0f%%" % (i + 1, f"[{edges[i] if i else '':.3f}" if i else "[-inf", len(p), pf_of(p) if p else "-", pf_of([f["pnl_mid_pct"] for f in b]) if b else "-", statistics.mean(p) if p else 0, 100 * sum(1 for x in p if x > 0) / len(p) if p else 0))
    table("skewing", FOCUS_QUARTILES, "ORATS skewing")
    table("rvol30", RVOL_QUARTILES, "rVol30 level")
    table("rvol2y", RVOL2Y_QUARTILES, "rVol2y level (the name's normal vol)")
    for key, cut in (("rvol30", RVOL30_CUT), ("rvol2y", RVOL2Y_CUT)):
        k2 = [f["pnl_real_pct"] for f in feats if f[key] >= cut]; s2 = [f["pnl_real_pct"] for f in feats if f[key] < cut]
        print(f"  fixed rule 'skip {key} < {cut}': kept {stats(k2)}, skipped {stats(s2)}")
    kept = [f["pnl_real_pct"] for f in feats if f["skewing"] >= RULE_CUT]
    skip = [f["pnl_real_pct"] for f in feats if f["skewing"] < RULE_CUT]
    print(f"\n  FIXED RULE 'skip skewing < {RULE_CUT}' out of sample: all {stats(real)} → kept {stats(kept)}, skipped {stats(skip)}")
    by = defaultdict(lambda: {"kept": [], "skipped": []})
    for f in feats:
        by[f["entry_date"][:4]]["kept" if f["skewing"] >= RULE_CUT else "skipped"].append(f["pnl_real_pct"])
    print("  by year: " + "  ".join(f"{y}: kept {len(v['kept'])}@{pf_of(v['kept']) if v['kept'] else '-'} / skipped {len(v['skipped'])}@{pf_of(v['skipped']) if v['skipped'] else '-'}" for y, v in sorted(by.items())))
    result = {"n": len(feats), "all": stats(real), "kept": stats(kept), "skipped": stats(skip), "rule": f"skip skewing < {RULE_CUT}",
              "by_year": {y: {"kept": stats(v["kept"]), "skipped": stats(v["skipped"])} for y, v in by.items()}, "requests": api.n, "cache_hits": api.hits}
    with open(os.path.join("tests", "orats_oos_skewing.json"), "w", encoding="utf-8") as f:
        json.dump({"result": result, "trades": feats}, f, indent=1)
    print(f"\n{api.n} requests, {api.hits} cache hits → tests/orats_oos_skewing.json")


if __name__ == "__main__":
    main()
