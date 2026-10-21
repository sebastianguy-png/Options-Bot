"""
Engine unit tests on SYNTHETIC price series — no network.  Three worlds: a
trending stock, a mean-reverting one, a gapping one.  Checks the invariants a
rule change can silently break and the parity test (last bar, real data) does
not cover.

    py tests/engine_unit.py
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import options_bot as ob                              # noqa: E402

N = 700
FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def frame(closes, gaps=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=len(closes))
    closes = np.asarray(closes, dtype=float)
    opens = closes * (1 + rng.normal(0, 0.002, len(closes)))
    if gaps:
        for i, g in gaps.items():
            opens[i] = closes[i - 1] * (1 + g)
    highs = np.maximum(opens, closes) * (1 + abs(rng.normal(0, 0.004, len(closes))))
    lows  = np.minimum(opens, closes) * (1 - abs(rng.normal(0, 0.004, len(closes))))
    vol   = rng.integers(500_000, 1_500_000, len(closes)).astype(float)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vol}, index=idx)


def ctx_hist(idx, regime="normal", trend="neutral", vix=18.0):
    return pd.DataFrame({"vix": vix, "regime": regime, "market_trend": trend, "vix_ts": "flat", "yield_inverted": False,
                         "yield_steep": False, "pre_fed": False, "post_fed": False,
                         **{f"macro_{k}": True for k in ("buy_call", "buy_put", "sell_put", "covered_call", "iron_condor")}}, index=idx)


def main():
    rng = np.random.default_rng(1)
    trend = 50 * np.exp(np.cumsum(rng.normal(0.0012, 0.012, N)))                       # drifts up
    ou = [50.0]
    for _ in range(N - 1):
        ou.append(ou[-1] + 0.15 * (50 - ou[-1]) + rng.normal(0, 0.9))                   # mean-reverting around 50
    gappy = 50 * np.exp(np.cumsum(rng.normal(0.0, 0.01, N)))
    gap_days = {400: -0.08, 520: 0.07}
    for i, g in gap_days.items():
        gappy[i:] *= (1 + g)                                                             # a permanent gap
    worlds = {"trending": frame(trend, seed=2), "mean_reverting": frame(ou, seed=3), "gapping": frame(gappy, gaps=gap_days, seed=4)}

    for name, h in worlds.items():
        print(f"\n== {name} ==")
        ch = ctx_hist(pd.DatetimeIndex(h.index))
        fr = ob._precompute_indicators(h, sym=name.upper(), ctx_hist=ch, spy_closes=None, garch="all")
        check(fr["n"] == N and all(k in fr for k in ob.IND_FIELDS), "indicator frame has every field")
        check(np.isnan(fr["rsi"][:13]).all() and not np.isnan(fr["rsi"][ob.BT_WARMUP_BARS:]).any(), "RSI NaN only in warm-up")
        check(np.nanmin(fr["rsi"]) >= 0 and np.nanmax(fr["rsi"]) <= 100, "RSI within 0–100")
        check(np.all(fr["bb_u"][30:] >= fr["bb_m"][30:]) and np.all(fr["bb_m"][30:] >= fr["bb_l"][30:]), "bands ordered")
        c = ob.ctx_at(fr, ob.BT_WARMUP_BARS)
        check(c["regime"] == "normal" and c["macro"]["buy_call"] is True, "context row read back")
        for mode in ("reversion", "escape"):
            for exits in ({**ob.BT_DEFAULT_EXITS}, {**ob.BT_DEFAULT_EXITS, "stop_loss_pct": 0, "profit_target_pct": 0, "hold_days": 10, "stock_stop_pct": 4}):
                tr = ob.run_backtest_ticker(name.upper(), params={**ob.BT_DEFAULT_PARAMS, "entry_mode": mode}, exits=exits, precomp=fr)
                cap = min(exits["hold_days"], ob.OPTION_DTE_BARS)
                first = min((fr["dates"].index(t["entry_date"]) for t in tr), default=10**9)
                check(all(1 <= t["bars_held"] <= cap for t in tr), f"{mode}/hold{exits['hold_days']}: {len(tr)} trades, bars held within 1..{cap}")
                check(first >= ob.BT_WARMUP_BARS, f"{mode}/hold{exits['hold_days']}: first entry after warm-up")
                check(all(abs(t["pnl_pct"] - round(t["pnl_gross_pct"] - t["cost_pct"], 1)) < 0.11 and t["cost_pct"] > 0 for t in tr),
                      f"{mode}/hold{exits['hold_days']}: net = gross − cost on every trade")
                check(all(t["exit_reason"] in ("take_profit", "stop_loss", "stock_stop", "trailing_stop", "stagnation", "end_of_hold",
                                               "rsi_target", "bb_target", "bb_breach", "expired", "assignment_risk") for t in tr),
                      f"{mode}/hold{exits['hold_days']}: exit reasons valid")
                if exits["stock_stop_pct"]:
                    check(all(t["exit_reason"] != "stop_loss" for t in tr), f"{mode}: premium stop off → never fires")
        tr_rev = ob.run_backtest_ticker(name.upper(), params=dict(ob.BT_DEFAULT_PARAMS), precomp=fr)
        if name == "mean_reverting":
            check(len(tr_rev) > 20, f"reversion rules fire on a mean-reverting series ({len(tr_rev)} trades)")
        if name == "gapping":
            tr_esc = ob.run_backtest_ticker("GAPPING", params={**ob.BT_DEFAULT_PARAMS, "entry_mode": "escape"}, precomp=fr)
            gap_dates = {fr["dates"][i] for i in gap_days}
            check(not any(t["entry_date"] in gap_dates for t in tr_esc), "escape mode never enters on a ≥3% gap day")

    # the live exit core with explicit parameters
    ex = dict(ob.BT_DEFAULT_EXITS)
    check(ob._price_exit_core(-15.0, None, 1, ex=ex)[3] == "stop_loss", "exit core: premium stop")
    check(ob._price_exit_core(-40.0, None, 1, ex={**ex, "stop_loss_pct": 0})[3] is None, "exit core: stop off")
    check(ob._price_exit_core(-5.0, None, 1, ex={**ex, "stock_stop_pct": 3}, stock_move_pct=3.5)[3] == "stock_stop", "exit core: stock stop")
    check(ob._price_exit_core(+1.0, None, 10, ex={**ex, "stagnant_days": 0, "hold_days": 10})[3] == "end_of_hold", "exit core: max hold")
    check(ob._adverse_stock_move("buy_put", 100, 104) == 4.0 and ob._adverse_stock_move("buy_call", 100, 104) == -4.0, "adverse move sign by strategy")

    print(f"\n{'ENGINE UNIT OK' if not FAILS else 'ENGINE UNIT FAILED: ' + str(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
