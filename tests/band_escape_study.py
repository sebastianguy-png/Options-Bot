"""
Band-escape study: when a close ESCAPES the Bollinger bands, does it snap back
(the bot's mean-reversion thesis) or keep going (a selloff / breakout)?  And
which features, visible on the escape day, tell the two apart?

For every FIRST-day escape (previous close inside the band) on the focus list
over the backtest frames, record day-of features and forward outcomes, then
compare the two option plays net of the spread cost:
  reversion  = lower escape → buy call,  upper escape → buy put   (the bot today)
  momentum   = lower escape → buy put,   upper escape → buy call
both ATM ~30 DTE, held HOLD bars with no stop (the simplest honest comparison).

    py tests/band_escape_study.py        → tests/band_escape_report.json + console table
"""
import itertools, json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import options_bot as ob                              # noqa: E402

HOLD = 10
FWD  = (1, 3, 5, 10, 21)


def f(x):
    try:
        v = float(x); return None if v != v else v
    except (TypeError, ValueError):
        return None


def collect(precomp):
    rows = []
    T0 = ob.OPTION_DTE_TARGET / 365.0
    for sym, fr in precomp.items():
        n = fr["n"]; px = fr["price"]; op = fr["open"]; bb_u, bb_m, bb_l = fr["bb_u"], fr["bb_m"], fr["bb_l"]
        for i in range(ob.BT_WARMUP_BARS, n - 22):
            c, u, m, l = px[i], bb_u[i], bb_m[i], bb_l[i]
            if any(v != v for v in (c, u, m, l)) or u <= m:
                continue
            side = "lower" if c < l else "upper" if c > u else None
            if side is None:
                continue
            pc = px[i - 1]
            if not (bb_l[i - 1] <= pc <= bb_u[i - 1]):                     # first day outside only
                continue
            ctx = ob.ctx_at(fr, i)
            z = (c - m) / (u - m)                                         # −1 = at lower band, +1 = at upper
            width = (u - l) / m * 100
            width_20 = ((bb_u[i - 20] - bb_l[i - 20]) / bb_m[i - 20] * 100) if i >= 20 and bb_m[i - 20] == bb_m[i - 20] else None
            avg_v, cur_v = f(fr["avg_vol"][i]), f(fr["cur_vol"][i])
            adx, ma200, rsi = f(fr["adx"][i]), f(fr["ma200"][i]), f(fr["rsi"][i])
            gap = (op[i] - pc) / pc * 100 if pc and op[i] == op[i] else 0.0
            # forward stock returns and the two option plays, net of spread
            fwd = {k: round((px[i + k] - c) / c * 100, 2) for k in FWD}
            back_in_5 = any(bb_l[i + k] <= px[i + k] <= bb_u[i + k] for k in range(1, 6))
            worst10 = round((min(px[i + 1:i + 11]) - c) / c * 100, 2)
            best10  = round((max(px[i + 1:i + 11]) - c) / c * 100, 2)
            iv = float(fr["iv_est"][i]); K = ob._bt_strike(c); S1 = px[i + HOLD]
            T1 = max(T0 - HOLD / 252.0, 0.5 / 365.0)
            plays = {}
            for name, otype in (("reversion", "call" if side == "lower" else "put"), ("momentum", "put" if side == "lower" else "call")):
                e = ob._bt_option_price(c, K, T0, iv, otype)
                if e <= 0.01:
                    plays[name] = None; continue
                x = ob._bt_option_price(S1, K, T1, iv, otype)
                plays[name] = round((x - e) / e * 100 - ob._spread_cost_pct(sym, e), 1)
            rows.append({
                "ticker": sym, "date": fr["dates"][i], "side": side, "z": round(z, 2), "beyond_pct": round((c - (l if side == "lower" else u)) / c * 100, 2),
                "gap_pct": round(gap, 2), "vol_ratio": round(cur_v / avg_v, 2) if (avg_v and cur_v) else None,
                "above_ma200": bool(ma200 is not None and c > ma200), "adx": adx, "rsi": rsi,
                "hv_rank": f(fr["hv_rank"][i]), "garch_expand": bool(fr["garch_expand"][i]),
                "rs_weak": bool(fr["rs_weak"][i]), "rs_strong": bool(fr["rs_strong"][i]),
                "width_pct": round(width, 2), "squeeze_release": bool(width_20 is not None and width > width_20 * 1.25),
                "near_52w_low": bool(c <= f(fr["low_52w"][i]) * 1.05) if f(fr["low_52w"][i]) else False,
                "near_52w_high": bool(c >= f(fr["high_52w"][i]) * 0.95) if f(fr["high_52w"][i]) else False,
                "vix": ctx.get("vix"), "regime": ctx.get("regime"), "market_trend": ctx.get("market_trend"),
                "fwd": fwd, "back_in_band_5d": back_in_5, "worst10": worst10, "best10": best10,
                "reversion_pnl": plays["reversion"], "momentum_pnl": plays["momentum"],
            })
    return rows


def stats(rows):
    rv = [r["reversion_pnl"] for r in rows if r["reversion_pnl"] is not None]
    mo = [r["momentum_pnl"] for r in rows if r["momentum_pnl"] is not None]
    if len(rows) < 30:
        return None
    return {"n": len(rows), "fwd10": round(float(np.mean([r["fwd"][10] for r in rows])), 2),
            "fwd21": round(float(np.mean([r["fwd"][21] for r in rows])), 2),
            "back_in_5d": round(100 * np.mean([r["back_in_band_5d"] for r in rows]), 1),
            "reversion_net": round(float(np.mean(rv)), 1) if rv else None, "reversion_win": round(100 * np.mean([x > 0 for x in rv]), 1) if rv else None,
            "momentum_net": round(float(np.mean(mo)), 1) if mo else None, "momentum_win": round(100 * np.mean([x > 0 for x in mo]), 1) if mo else None}


FEATURES = {
    "big volume (≥2× avg)":   lambda r: r["vol_ratio"] is not None and r["vol_ratio"] >= 2,
    "quiet volume (<1× avg)": lambda r: r["vol_ratio"] is not None and r["vol_ratio"] < 1,
    "gap ≥3% in the move's direction": lambda r: (r["gap_pct"] <= -3) if r["side"] == "lower" else (r["gap_pct"] >= 3),
    "no gap (|gap|<1%)":      lambda r: abs(r["gap_pct"]) < 1,
    "below MA200":            lambda r: not r["above_ma200"],
    "above MA200":            lambda r: r["above_ma200"],
    "ADX ≥30 (trending)":     lambda r: r["adx"] is not None and r["adx"] >= 30,
    "ADX <20 (ranging)":      lambda r: r["adx"] is not None and r["adx"] < 20,
    "far beyond band (≥3%)":  lambda r: abs(r["beyond_pct"]) >= 3,
    "just beyond (<1%)":      lambda r: abs(r["beyond_pct"]) < 1,
    "GARCH expects expansion": lambda r: r["garch_expand"],
    "GARCH no expansion":     lambda r: not r["garch_expand"],
    "RS weak vs SPY":         lambda r: r["rs_weak"],
    "RS strong vs SPY":       lambda r: r["rs_strong"],
    "squeeze release":        lambda r: r["squeeze_release"],
    "VIX high regime":        lambda r: r["regime"] == "high",
    "VIX low regime":         lambda r: r["regime"] == "low",
    "SPY bear trend":         lambda r: str(r["market_trend"]).startswith("bear"),
    "SPY bull_strong":        lambda r: r["market_trend"] == "bull_strong",
    "near 52w low":           lambda r: r["near_52w_low"],
    "near 52w high":          lambda r: r["near_52w_high"],
    "RSI extreme (<25 / >75)": lambda r: r["rsi"] is not None and (r["rsi"] < 25 if r["side"] == "lower" else r["rsi"] > 75),
}


def main():
    t0 = time.time()
    ob._load_garch_cache()
    precomp, split_map, split_date = ob._wfo_prepare(list(dict.fromkeys(ob.FOCUS_LIST)), label="Bands")
    rows = collect(precomp)
    print(f"{len(rows)} first-day band escapes on {len(precomp)} tickers ({time.time()-t0:.0f}s)")
    report = {"n": len(rows), "hold_bars": HOLD, "split_date": str(split_date)[:10], "by_side": {}, "features": {}, "pairs": {}}

    def show(label, st):
        if st:
            print(f"  {label:<34} n={st['n']:5d}  fwd10={st['fwd10']:+6.2f}% fwd21={st['fwd21']:+6.2f}%  back-in-band≤5d {st['back_in_5d']:5.1f}%  "
                  f"reversion net {st['reversion_net']:+6.1f}% (win {st['reversion_win']}%)  momentum net {st['momentum_net']:+6.1f}% (win {st['momentum_win']}%)")

    for side in ("lower", "upper"):
        sub = [r for r in rows if r["side"] == side]
        report["by_side"][side] = {"all": stats(sub), "features": {}}
        print(f"\n=== {side.upper()} band escapes (reversion = {'buy call' if side == 'lower' else 'buy put'}, momentum = the opposite; {HOLD}-bar hold, net of spread) ===")
        show("ALL", stats(sub))
        feats = []
        for name, fn in FEATURES.items():
            st = stats([r for r in sub if fn(r)])
            if st:
                feats.append((name, st)); report["by_side"][side]["features"][name] = st
        feats.sort(key=lambda x: (x[1]["momentum_net"] or -99) - (x[1]["reversion_net"] or -99), reverse=True)
        print("  -- most momentum-favouring features first --")
        for name, st in feats:
            show(name, st)
        # feature pairs: where does momentum pay best, where does reversion?
        pairs = []
        names = list(FEATURES)
        for a, b in itertools.combinations(names, 2):
            st = stats([r for r in sub if FEATURES[a](r) and FEATURES[b](r)])
            if st and st["n"] >= 80:
                pairs.append((f"{a} + {b}", st))
        report["by_side"][side]["pairs_momentum"] = [(k, v) for k, v in sorted(pairs, key=lambda x: -(x[1]["momentum_net"] or -99))[:8]]
        report["by_side"][side]["pairs_reversion"] = [(k, v) for k, v in sorted(pairs, key=lambda x: -(x[1]["reversion_net"] or -99))[:8]]
        print("  -- best MOMENTUM pairs (n ≥ 80) --")
        for k, v in report["by_side"][side]["pairs_momentum"][:5]:
            show(k[:34], v)
        print("  -- best REVERSION pairs (n ≥ 80) --")
        for k, v in report["by_side"][side]["pairs_reversion"][:5]:
            show(k[:34], v)

    out = os.path.join(ROOT, "tests", "band_escape_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(ob._json_clean(report), fh, indent=2)
    print(f"\nreport → {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
