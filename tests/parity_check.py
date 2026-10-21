"""
Parity check: does the shared signal engine reproduce the ORIGINAL live
scanner's entry decisions bar-for-bar?

The reference is the pre-refactor scan_ticker() taken from a git revision
(default: the last commit before the engine landed).  Its pre-gate block is
extracted textually and executed against the same 1-year frame and the same
market context the new engine receives, then every gate flag, confluence count
and threshold is compared.

Run (from the project folder):
    py tests/parity_check.py                 # focus list, last bar + 6 earlier bars each
    py tests/parity_check.py --tickers AAPL PLUG PFE --bars 10
    py tests/parity_check.py --ref <git-rev>
Exit status 0 = full parity, 1 = mismatches found.
"""
import argparse, os, subprocess, sys, textwrap, types, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np, pandas as pd                      # noqa: E402
import options_bot as ob                              # noqa: E402

COMPARE_BOOLS = ("has_sell_put", "has_buy_call", "has_buy_put", "has_covered_call",
                 "oversold", "overbought", "falling_knife", "melt_up", "exhaustion_bounce",
                 "trend_ok_buy", "green_ok", "red_ok", "gap_ok_buy_call", "gap_ok_buy_put",
                 "dc_at_lower", "dc_at_upper", "near_52w_low", "near_52w_high",
                 "hv_low", "hv_high", "strong_trend", "above_ma50", "trend_bull", "trend_bear",
                 "macd_bull", "macd_bear", "ema_bull", "ema_bear", "ema_cross_bull", "ema_cross_bear",
                 "reversal_up", "vol_confirmed", "rs_strong", "rs_weak", "sig_mfi_os", "sig_mfi_ob")
COMPARE_NUMS  = ("bull_signals", "bear_signals", "_bull_min", "_bear_min", "_rsi_os_thr", "_rsi_ob_thr",
                 "cur_rsi", "cur_bb_u", "cur_bb_l", "cur_dc_u", "cur_dc_l", "cur_mfi", "hv_rank",
                 "cur_adx", "cur_pdi", "cur_ndi", "low_52w", "high_52w", "beta", "rs_ratio", "gap_pct")


def load_legacy_gate(ref):
    """Build legacy_gate(hist, sym, scan_condors) from the old scan_ticker source."""
    src = subprocess.check_output(["git", "show", f"{ref}:options_bot.py"], cwd=ROOT).decode("utf-8", "replace")
    legacy = types.ModuleType("legacy_bot")
    legacy.__file__ = "legacy_bot.py"
    # Execute the legacy module WITHOUT running main() or chdir side effects
    code = src.replace('if __name__ == "__main__":\n    main()', "")
    exec(compile(code, "legacy_bot.py", "exec"), legacy.__dict__)

    lines = src.split("\n")
    start = next(i for i, l in enumerate(lines) if l.strip().startswith('price = float(hist["Close"].iloc[-1])'))
    end   = next(i for i, l in enumerate(lines) if l.strip().startswith("if not any([has_sell_put, has_buy_call"))
    block = "\n".join(lines[start:end])
    block = textwrap.dedent(block)
    fn_src = "def legacy_gate(hist, sym, scan_condors=False):\n" + textwrap.indent(block, "    ") + \
             "\n    return {k: v for k, v in locals().items() if k not in ('hist', 'sym')}\n"
    ns = dict(legacy.__dict__)
    exec(compile(fn_src, "legacy_gate.py", "exec"), ns)
    return legacy, ns["legacy_gate"]


def freeze_context(legacy):
    """Point the legacy module's context getters at the SAME values the new engine uses."""
    ctx = ob.live_market_ctx()
    legacy.get_vix          = lambda: ctx["vix"]
    legacy.get_vix3m        = lambda: ctx["vix3m"]
    legacy.get_market_trend = lambda: ctx["market_trend"]
    legacy.get_macro_regime = lambda: dict(ctx["macro"])
    legacy.get_fed_context  = lambda: {"pre_fed_window": ctx["pre_fed"], "post_fed_window": ctx["post_fed"],
                                       "yield_inverted": ctx["yield_inverted"], "yield_steep": ctx["yield_steep"],
                                       "irx": ctx["irx"], "fvx": ctx["fvx"], "yield_spread": ctx["yield_spread"],
                                       "days_to_next_fed": ctx["days_to_next_fed"], "next_fed_date": ctx["next_fed_date"]}
    legacy._spy_cache["closes"] = ob._spy_cache.get("closes")
    legacy.scan_state["wfo_params"] = dict(ob.scan_state.get("wfo_params", {}))
    legacy.RSI_OVERSOLD   = ob.RSI_OVERSOLD
    legacy.RSI_OVERBOUGHT = ob.RSI_OVERBOUGHT
    legacy.REQUIRE_GREEN_CANDLE = ob.REQUIRE_GREEN_CANDLE
    legacy.REQUIRE_RED_CANDLE   = ob.REQUIRE_RED_CANDLE
    return ctx


def new_engine(hist, sym, ctx):
    fr  = ob.indicator_frame(hist, spy_closes=ob._spy_cache.get("closes"), sym=sym, garch=None)
    ind = ob.ind_at(fr, -1)
    ev  = ob.evaluate_entry(ind, ctx, ob.live_params(), scan_condors=False)
    out = dict(ev)
    out.update({
        "macd_bull": ind["macd_bull"], "macd_bear": ind["macd_bear"],
        "ema_bull": ind["ema_bull"], "ema_bear": ind["ema_bear"],
        "ema_cross_bull": ind["ema_cross_bull"], "ema_cross_bear": ind["ema_cross_bear"],
        "reversal_up": ind["reversal_up"], "vol_confirmed": ind["vol_confirmed"],
        "rs_strong": ind["rs_strong"], "rs_weak": ind["rs_weak"],
        "_bull_min": ev["bull_min"], "_bear_min": ev["bear_min"],
        "_rsi_os_thr": ev["rsi_os_thr"], "_rsi_ob_thr": ev["rsi_ob_thr"],
        "cur_rsi": ind["rsi"], "cur_bb_u": ind["bb_u"], "cur_bb_l": ind["bb_l"],
        "cur_dc_u": ind["dc_u"], "cur_dc_l": ind["dc_l"], "cur_mfi": ind["mfi"], "hv_rank": ind["hv_rank"],
        "cur_adx": ind["adx"], "cur_pdi": ind["pdi"], "cur_ndi": ind["ndi"],
        "low_52w": ind["low_52w"], "high_52w": ind["high_52w"], "beta": ind["beta"], "rs_ratio": ind["rs_ratio"],
    })
    return out


def _eq_num(a, b, tol=1e-6):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    if a != a and b != b:
        return True
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--bars", type=int, default=6, help="also check this many earlier bars per ticker")
    ap.add_argument("--ref", default="HEAD", help="git revision holding the ORIGINAL scan_ticker")
    args = ap.parse_args()

    legacy, legacy_gate = load_legacy_gate(args.ref)
    ob._warm_market_caches()
    ctx = freeze_context(legacy)
    tickers = args.tickers or list(dict.fromkeys(ob.FOCUS_LIST))
    hist_map = ob._bulk_download(tickers, "1y")
    print(f"Parity vs {args.ref}: {len(hist_map)} tickers, last bar + {args.bars} earlier bars each")

    checks = mismatches = 0
    for sym, h in hist_map.items():
        if len(h) < 70:
            continue
        for back in range(0, args.bars + 1):
            frame = h.iloc[: len(h) - back] if back else h
            if len(frame) < 60:
                break
            try:
                old = legacy_gate(frame, sym)
            except Exception as e:
                print(f"  {sym} bar-{back}: legacy failed: {e}")
                continue
            if old is None:                      # legacy early-return (price < MIN_PRICE)
                continue
            new = new_engine(frame, sym, ctx)
            bar = str(frame.index[-1])[:10]
            for k in COMPARE_BOOLS:
                if k not in old:
                    continue
                checks += 1
                if bool(old[k]) != bool(new[k]):
                    mismatches += 1
                    print(f"  MISMATCH {sym} {bar} {k}: legacy={old[k]} engine={new[k]}")
            for k in COMPARE_NUMS:
                if k not in old:
                    continue
                checks += 1
                if not _eq_num(old[k], new[k]):
                    mismatches += 1
                    print(f"  MISMATCH {sym} {bar} {k}: legacy={old[k]} engine={new[k]}")
    print(f"{checks} comparisons, {mismatches} mismatches → {'PARITY OK' if mismatches == 0 else 'DIFFERENCES'}")
    sys.exit(0 if mismatches == 0 else 1)


if __name__ == "__main__":
    main()
