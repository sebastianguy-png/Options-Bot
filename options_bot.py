#!/usr/bin/env python3
"""
Options Trading Bot v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategies analysed on every scan:
  1. SELL PUTS      – Oversold stocks with overpriced puts (cash-secured)
  2. BUY CALLS      – Oversold stocks showing reversal / bounce signals
  3. BUY PUTS       – Overbought stocks primed for a pullback
  4. COVERED CALLS  – High-IV stocks where selling the call is attractive

Safety guarantees:
  • The bot executes nothing on its own unless the 🤖 Autotrade tab is enabled
    (autotrader.py: local paper ledger by default, Alpaca paper with keys, real
    money only with OPTIONS_BOT_ALLOW_LIVE=1 — see that file's header)
  • Manual trades: "Add to Queue" → "Confirm & Place Order"; over $1,000 you
    also type CONFIRM; the SoFi window lands on Review and YOU click Submit
    (or enable AUTO_SUBMIT only if you're comfortable)

Run:
    py options_bot.py              → opens http://localhost:5000 in your browser
    py options_bot.py --headless   → no browser (also OPTIONS_BOT_HEADLESS=1);
                                     run_bot_headless.bat restarts it if it exits,
                                     install_autostart.ps1 starts it at logon
    Console output is also written to logs\bot_<date>.log.
"""

# ──────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────
import os, sys, io, re, json, uuid, threading, webbrowser, time, warnings, pickle, socket
from datetime import datetime, timedelta, date
from math import log, sqrt, exp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

# Silence library chatter but keep deprecation/future warnings visible so
# pandas/numpy migration problems don't get hidden.
warnings.filterwarnings("ignore")
warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)
warnings.filterwarnings("default", category=FutureWarning,      module=__name__)

# The log output uses box-drawing / emoji characters.  When the bot is started
# without a console (Task Scheduler, a launcher, redirected output) Windows
# gives it a cp1252 stdout and the very first banner line raised
# UnicodeEncodeError and killed the process.  Force UTF-8, never crash on print.
# line_buffering: with output redirected to a file/pipe, Python block-buffers
# stdout and the scan log only appears in chunks (or is lost on a crash).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

# All data files (JSON state, pickle cache) are referenced by bare filename.
# Anchor the working directory to the script's folder so launching from any
# other directory (Task Scheduler, a shortcut, another terminal) still finds
# the same tracked positions / performance / calibration files.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

import autotrader        # noqa: E402  broker-driven autotrading — idle until enabled in its dashboard tab

# ── Yahoo Finance rate-limit guard ────────────────────────────
# All yfinance calls that hit Yahoo's API (options chain, calendar,
# expiry lists, history) go through this semaphore so we never fire
# more than _YF_MAX_CONCURRENT requests regardless of thread count.
_YF_MAX_CONCURRENT = 8
_YF_SEM            = threading.Semaphore(_YF_MAX_CONCURRENT)
_YF_REFRESH_LOCK   = threading.Lock()
_yf_last_refresh   = 0.0

try:
    from yfinance.exceptions import YFRateLimitError as _YFRateLimitError
except Exception:                       # older/newer yfinance without the class
    class _YFRateLimitError(Exception):
        pass


def _yf_refresh_session():
    """
    Force yfinance to re-fetch its cookie and crumb from Yahoo Finance.
    Uses the YfData singleton's own _get_cookie_and_crumb() method —
    the correct way to re-authenticate in yfinance 1.x.
    Debounced: when several threads hit a 401 at once only one refresh runs.
    """
    global _yf_last_refresh
    with _YF_REFRESH_LOCK:
        if time.time() - _yf_last_refresh < 5:
            return
        try:
            import yfinance.data as _yfd
            yd = _yfd.YfData()          # returns the singleton instance
            yd._crumb  = None           # clear stale crumb so it re-fetches
            yd._cookie = None           # clear stale cookie
            yd._get_cookie_and_crumb()  # re-authenticate properly
            print("  [Auth] yfinance crumb refreshed successfully")
        except Exception as e:
            print(f"  [Auth] crumb refresh failed ({e}) — will retry request anyway")
        _yf_last_refresh = time.time()


def _yf_call(fn, *args, retries=4, base_delay=6, **kwargs):
    """
    Call any yfinance method with rate-limit retry + semaphore.
    Automatically handles Yahoo Finance 401 crumb errors by refreshing
    the yfinance session and retrying.  The semaphore slot is released
    while sleeping so one throttled call doesn't stall the others.
    Usage:  chain = _yf_call(tkr.option_chain, exp)
    """
    for attempt in range(retries + 1):
        try:
            with _YF_SEM:
                return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            is_rate = isinstance(e, _YFRateLimitError) or any(
                x in msg for x in ("too many requests", "rate limit", "429", "rate limited"))
            is_crumb = any(x in msg for x in ("401", "unauthorized", "invalid crumb", "crumb"))
            if is_crumb and attempt < retries:
                print(f"  [Auth] 401 crumb error — refreshing session (attempt {attempt+1}/{retries})")
                _yf_refresh_session()
                time.sleep(2)
            elif is_rate and attempt < retries:
                wait = base_delay * (2 ** attempt)
                print(f"  [Rate limit] retrying in {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise


# ── History sanitising ────────────────────────────────────────
# yfinance frequently appends a row for the current session whose
# Open/High/Low/Close are NaN (only Volume is filled).  Every indicator
# that looks at .iloc[-1] then comes back NaN, which silently disables
# every signal.  dropna(how="all") does NOT remove that row because
# Volume/Dividends/Stock Splits are populated — hence subset=["Close"].
def _clean_hist(df):
    """Return a copy of an OHLCV frame with NaN-Close rows removed (or None)."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):       # single-ticker yf.download
        df = df.droplevel(0, axis=1) if df.columns.nlevels > 1 else df
    if "Close" not in df.columns:
        return None
    out = df.dropna(subset=["Close"])
    out = out[out["Close"] > 0]                          # a 0.0 close (delisted / bad row) poisons every return-based window
    out = out[~out.index.duplicated(keep="last")]        # a duplicated date breaks the rolling windows and the SPY alignment
    return out if not out.empty else None


def _fetch_history(sym, period="1y", auto_adjust=False):
    """
    Fetch one ticker's OHLCV history through the rate-limit guard and
    sanitise it.  Returns (Ticker, DataFrame-or-None).
    auto_adjust=False matches the bulk yf.download path so indicators use
    the same (unadjusted) price basis regardless of which path fetched them.
    """
    tkr  = yf.Ticker(sym)
    hist = _yf_call(tkr.history, period=period, auto_adjust=auto_adjust)
    return tkr, _clean_hist(hist)


def _bulk_download(tickers, period="1y", min_rows=1):
    """
    One yf.download call for many tickers → {sym: clean DataFrame}.
    Handles the yfinance 1.x MultiIndex columns for 1 or N tickers,
    drops NaN-Close rows and tickers with fewer than `min_rows` bars.
    Never raises — returns {} on failure (callers fall back per-ticker).
    """
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return {}
    out = {}
    try:
        with _YF_SEM:
            raw = yf.download(tickers, period=period, group_by="ticker",
                              auto_adjust=False, threads=True, progress=False)
        if raw is None or raw.empty:
            return {}
        for sym in tickers:
            try:
                df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = _clean_hist(df)
                if df is not None and len(df) >= min_rows:
                    out[sym] = df
            except Exception:
                pass
    except Exception as e:
        print(f"  [Bulk download] Failed ({e})")
    return out


# ── Robust JSON persistence ───────────────────────────────────
# All state files were written with open(path, "w") which truncates first:
# a crash mid-write leaves a corrupt file, the loader then returns [] and
# the next save wipes the user's data.  Write to a temp file and os.replace.
def _atomic_json_write(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        # _json_clean: NaN → null.  The old files contained bare NaN tokens,
        # which Python reads back but every strict JSON parser rejects.
        json.dump(_json_clean(obj), f, indent=2, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_json(path, default, label=""):
    """Load a JSON file; on corruption keep a .corrupt backup and log it."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8-sig") as f:     # -sig: tolerate a BOM from editors
            return json.load(f)
    except Exception as e:
        bad = f"{path}.corrupt"
        try:
            os.replace(path, bad)
        except Exception:
            pass
        print(f"  [{label or 'State'}] Could not read {path}: {e} — moved to {bad}")
        return default


def _json_default(o):
    """json.dump fallback: numpy scalars/arrays, pandas timestamps, NaN."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _json_clean(o):
    """Recursively replace NaN/inf with None and numpy scalars with Python ones."""
    if isinstance(o, dict):
        return {k: _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, (np.integer, np.floating, np.bool_, np.ndarray)):
        return _json_clean(_json_default(o))
    return o


def _as_float(v, default=0.0):
    """float() that treats None/NaN/'' as `default` (NaN is truthy, so `or 0` misses it)."""
    try:
        f = float(v)
        return default if f != f else f
    except (TypeError, ValueError):
        return default


def _as_int(v, default=0):
    f = _as_float(v, None)
    return default if f is None else int(f)


# Flask — install with: pip install flask
try:
    from flask import Flask, jsonify, request, render_template_string
    from flask.json.provider import DefaultJSONProvider as _FlaskJSONProvider
except ImportError:
    print("Flask not found. Run:  pip install flask")
    sys.exit(1)


class _SafeJSONProvider(_FlaskJSONProvider):
    """
    jsonify() with numpy scalars converted and NaN/inf emitted as null.
    Browsers reject bare NaN in JSON.parse, which used to break the
    Tracked tab whenever an option quote came back without a price.
    """
    def dumps(self, obj, **kwargs):
        kwargs.setdefault("default", _json_default)
        kwargs.setdefault("allow_nan", False)
        return super().dumps(_json_clean(obj), **kwargs)

# ──────────────────────────────────────────────────────────────
#  CONFIGURATION  ← all user-tunable knobs
# ──────────────────────────────────────────────────────────────
MAX_PRICE_SELL       = 10.00   # Upper price filter for Sell Puts & Covered Calls
#                               # Buy Calls and Buy Puts have NO upper price cap
MIN_PRICE            = 1.50    # Lower stock-price filter (skip sub-penny)
RSI_OVERSOLD         = 35      # RSI below this → oversold
RSI_OVERBOUGHT       = 65      # RSI above this → overbought
IV_RANK_THRESHOLD    = 40      # IV Rank % considered "elevated"
PREMIUM_PCT_MIN      = 3.0     # Put/call mid ÷ stock price × 100 threshold
BB_PERIOD            = 20      # Bollinger Band window
BB_STD_DEV           = 2       # Bollinger Band std-devs
RSI_PERIOD           = 14      # RSI lookback
DC_PERIOD            = 20      # Donchian Channel window (highest high / lowest low)
MIN_DTE              = 14      # Min days to expiry
MAX_DTE              = 45      # Max days to expiry
MIN_OPEN_INTEREST    = 10      # Min OI on selected option
MIN_VOLUME           = 10      # Min daily volume on selected option (avoid getting stuck in illiquid contracts)
MAX_SPREAD_PCT       = 100     # Reject a contract whose (ask − bid) / mid exceeds this % (100 = off); the optimizer sets it
SPREAD_MODEL_FILE    = "spread_model.json"   # today's measured spreads (tests/measure_spreads.py) → the backtest's cost model
MAX_WORKERS          = 8       # Parallel scan threads (I/O-bound; Yahoo latency is the bottleneck)
WFO_CPU_WORKERS      = (os.cpu_count() or 8)   # Parallel WFO threads (CPU-bound; use all logical cores)
PREFETCH_CACHE_FILE      = "prefetch_cache.pkl"   # Disk cache for tkr.info / tkr.calendar
PREFETCH_CACHE_TTL_HOURS = 24                     # Info/calendar barely change day-to-day
HOLD_SWEEP_INTERVALS = [1, 2, 3, 5, 7, 10, 14, 21]  # Trading days for hold-horizon analysis
SERVER_PORT          = 5000    # Web dashboard port
LIVE_REFRESH_MINS    = 5       # How often to re-check current signal tickers (minutes)
AUTO_RESCAN_HOURS    = 2       # How often to re-run the full universe scan (hours)

# ── Scan modes ────────────────────────────────────────────────
# "focus"  → ~80 high-liquidity names with active options markets
# "full"   → entire S&P 500 universe (~503 tickers, slower)
SCAN_MODE            = "focus"   # default; overridden by UI toggle

# ── Green candle confirmation (buy calls only) ─────────────────
# When True: buy call signals require the current bar to close ABOVE
# the previous close — confirming buyers showed up before entering.
# Reduces early entries on falling knives at the cost of some signals.
# Toggle live via the UI button without restarting.
REQUIRE_GREEN_CANDLE = False

# ── Red candle confirmation (buy puts only) ──────────────────────
# When True: buy put signals require the current bar to close BELOW
# the previous close — confirming sellers showed up before entering.
# Avoids entering puts on overbought stocks that are still grinding up.
# Toggle live via the UI button without restarting.
REQUIRE_RED_CANDLE = False

# ── Position sizing — risk budget per trade ──────────────────────
# Max dollar loss you're willing to accept per trade at the stop-loss %.
# 0 = disabled (no recommendation shown).
# Set live via the UI without restarting.
RISK_BUDGET_PER_TRADE = 100   # max dollar loss per trade at the stop-loss %

FOCUS_LIST = [
    # Mega-cap tech
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","ORCL","AMD",
    "INTC","QCOM","MU","AMAT","ARM","PLTR","CRM","SNOW","ADBE","NOW",
    # Financials
    "JPM","BAC","GS","MS","C","WFC","AXP","COF","SCHW","BLK",
    # Healthcare / Biotech
    "UNH","LLY","ABBV","MRK","PFE","TMO","AMGN","GILD","VRTX","MRNA",
    # Energy
    "XOM","CVX","COP","SLB","OXY","HAL","MPC","VLO",
    # Consumer / Retail
    "WMT","COST","TGT","HD","MCD","SBUX","NKE","BKNG","CMG",
    # Communication / Media
    "NFLX","DIS","CMCSA","T","VZ","TMUS","SNAP","SPOT","RBLX","TTD",
    # Industrial / Defence
    "GE","HON","CAT","DE","BA","RTX","LMT","UPS","FDX",
    # EV / Clean energy
    "RIVN","NIO","PLUG","FCEL","BLNK","CHPT",
    # Fintech / Crypto
    "PYPL","XYZ","COIN","HOOD","SOFI","AFRM","UPST",   # XYZ = Block (formerly SQ)
    # ETFs (very liquid options)
    "SPY","QQQ","IWM","XLF","XLE","XLK","ARKK","GLD","SLV","TLT",
]

# ── Monte Carlo ───────────────────────────────────────────────
N_MC_SIMS            = 10_000  # Simulations per signal (higher = slower but more accurate)
RISK_FREE_RATE       = 0.044   # Fallback: ~current 6-month T-bill. Auto-updated on startup via FRED.

# ── Guardrail ─────────────────────────────────────────────────
GUARDRAIL_LIMIT      = 1000.00  # $ threshold requiring explicit CONFIRM
# "Cost" definition per strategy:
#   BUY  trades  → debit paid       = qty × 100 × ask
#   SELL trades  → capital reserved = qty × 100 × strike  (cash-secured)
#   COVERED CALL → $0 additional capital

# ── Execution ─────────────────────────────────────────────────
# AUTO_SUBMIT = False  → bot fills the SoFi order form and stops at Review page;
#                        YOU click "Place Order" yourself in the browser.
# AUTO_SUBMIT = True   → bot clicks "Place Order" after dashboard confirmation.
#                        Only enable this once you trust the automation fully.
AUTO_SUBMIT          = False

# ── Persistence ───────────────────────────────────────────────
QUEUE_FILE           = "trade_queue.json"
HISTORY_FILE         = "trade_history.json"
TRACK_FILE           = "tracked_positions.json"
PERFORMANCE_FILE     = "performance.json"
WFO_FILE             = "wfo_results.json"     # persists last WFO run across restarts
BT_FILE              = "backtest_results.json"  # persists last backtest run across restarts
HS_RESULTS_FILE      = "hold_horizon_results.json"  # persists last Hold Horizon Analysis
SIGNAL_LOG_FILE      = "signal_log.json"      # unbiased log of every bot recommendation

# ── Tracked position sell-signal thresholds ───────────────────
# A sell signal fires when ANY of these conditions are met:
SELL_PROFIT_TARGET   = 30    # % gain on the option premium  → take profit (was 50)
SELL_STOP_LOSS       = 15    # % loss on the option premium  → cut loss (2:1 reward:risk, was 25)
# Trailing stop — once option P&L hits TRAIL_ACTIVATE_PCT, the effective stop
# becomes (peak_pnl − TRAIL_LOCK_PCT). Prevents giving back gains on reversal.
TRAIL_ACTIVATE_PCT   = 15    # Trailing stop activates once option P&L >= this
TRAIL_LOCK_PCT       = 8     # Lock minimum (peak − this) once trailing is active
# Stagnation exit — close positions that haven't moved meaningfully in N days.
# Theta bleeds long options; sitting flat is a slow loss.
STAGNANT_DAYS        = 3     # Trading days held before stagnation check applies
STAGNANT_RANGE       = 5     # If |option P&L| within ±this after STAGNANT_DAYS → exit
# Max hold — close a position after this many trading days whatever its P&L.
# The optimizer's hold_days lands here (via _apply_wfo_params); 21 ≈ the
# ~30-DTE contract's own life, so the default changes nothing by itself.
HOLD_DAYS            = 21
# Stock-based stop — close when the STOCK has moved this % against the position
# since entry (bullish trades: down, buy puts: up).  0 = off.  Swept by the optimizer.
STOCK_STOP_PCT       = 0
# Stock-based trailing exit — close when the STOCK has given back this % from its
# best close since entry (in the trade's direction).  0 = off.  Swept by the optimizer.
STOCK_TRAIL_PCT      = 0
# RSI reversal thresholds — exit when mean reversion is complete:
SELL_RSI_BULL_TARGET = 52    # Buy Call: RSI back above this → bounce done, take profit (was 55)
SELL_RSI_BEAR_TARGET = 48    # Buy Put:  RSI back below this → pullback done, take profit (was 45)
TRACK_CHECK_MINS         = 360  # How often (minutes) to re-run full sell-signal check (6 hours)
TRACK_PRICE_REFRESH_MINS = 3    # How often (minutes) to refresh just the option prices

# ── FOMC meeting decision dates (day of announcement) ─────────
# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
# Update each December when the Fed publishes the following year's schedule.
FOMC_DATES = [
    # 2023 (needed so the backtest's 2-3 year window has Fed context)
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (decision day = second day of each two-day meeting)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

# ──────────────────────────────────────────────────────────────
#  FLASK APP
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.json = _SafeJSONProvider(app)

# Shared scan state (written by background thread, read by API)
scan_state = {
    "running":      False,
    "progress":     0,
    "total":        0,
    "last_scan":    None,
    "last_refresh": None,
    "results":      {"sell_puts": [], "buy_calls": [], "buy_puts": [], "covered_calls": [], "iron_condors": []},
}
scan_lock = threading.Lock()

# Live refresh state
_live_refresh_stop  = threading.Event()   # set to stop the loop
_live_refresh_mins  = LIVE_REFRESH_MINS   # mutable at runtime via UI

# Auto-rescan state
_auto_rescan_hours  = AUTO_RESCAN_HOURS   # mutable at runtime via UI
_next_rescan_at     = None                # datetime of next scheduled full scan

# Tracked positions state
track_lock = threading.Lock()

# News cache  { sym: (timestamp, [articles]) }
_news_cache     = {}
NEWS_CACHE_MINS = 30       # re-fetch news every 30 minutes per ticker

# VADER sentiment analyser singleton (lazy-loaded)
_sia = None
def _get_sia():
    global _sia
    if _sia is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _sia = SentimentIntensityAnalyzer()
        except ImportError:
            pass          # fallback to keywords below
    return _sia


# ══════════════════════════════════════════════════════════════
#  TECHNICAL-ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════

def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)
    # No down-closes in the window → RS is infinite → RSI is 100, not NaN.
    return rsi.mask((loss == 0) & (gain > 0), 100.0)


def calc_bb(closes, period=20, nstd=2):
    sma = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    return sma + nstd * std, sma, sma - nstd * std


def calc_donchian(hist, period=20):
    """
    Donchian Channels: upper = highest high, lower = lowest low over N periods.
    Middle = (upper + lower) / 2.
    Returns (upper, middle, lower) as pandas Series.
    Unlike Bollinger Bands (which measure volatility), Donchian Channels measure
    pure price range extremes — useful for confirming breakouts and breakdowns.
    """
    highs  = hist["High"]  if "High"  in hist.columns else hist["Close"]
    lows   = hist["Low"]   if "Low"   in hist.columns else hist["Close"]
    upper  = highs.rolling(period).max()
    lower  = lows.rolling(period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower


def calc_garch_vol(closes, fallback_window=20):
    """
    Estimate annualised forward volatility using GJR-GARCH(1,1,1) with
    Student-t innovations — a jump-aware volatility model.

    GJR-GARCH adds an asymmetric leverage term (γ) to standard GARCH:
        σ²_t = ω + α·ε²_(t-1) + γ·ε²_(t-1)·I(ε_(t-1)<0) + β·σ²_(t-1)
    The γ term fires only on negative shocks, capturing the well-documented
    asymmetry where crashes spike vol more than rallies of equal magnitude.

    Student-t innovations replace the normal distribution assumption with
    fat tails, capturing jump-like extreme return behaviour without requiring
    an explicit (and fragile) Poisson jump process.

    Falls back in order:
      1. Standard GARCH(1,1) with normal innovations   (if GJR-t fails)
      2. 20-day rolling historical vol                  (if arch unavailable)
      3. 20% flat default                               (last resort)

    Returns annualised vol as a decimal (e.g. 0.32 = 32%).

    NOTE: until Oct 2026 the `arch` package was not installed in the bot's
    Python, so this silently returned the 20-day HV fallback every time —
    every "GJR-GARCH" figure the dashboard showed was plain historical vol.
    The fit now uses the last GARCH_WINDOW returns, the same window the
    backtest's rolling forecast uses, so the two are directly comparable.
    """
    log_ret = (np.log(closes / closes.shift(1)).dropna() * 100).iloc[-GARCH_WINDOW:]   # % returns
    v = _garch_fit_forecast(log_ret)
    if v is not None:
        return v
    return _hv_fallback(log_ret / 100.0, fallback_window)


GARCH_WINDOW      = 240      # returns used per fit — a 1y download has ~249, so live and backtest fit the same window
GARCH_REFIT_BARS  = 5        # backtest: refit every N bars, hold the forecast in between
GARCH_CACHE_FILE  = "garch_cache.pkl"
_garch_cache      = {}       # (sym, window, date_str) → annualised forecast (float)
_garch_cache_lock = threading.Lock()
_garch_available  = None     # None = not checked yet


def _hv_fallback(log_ret, window=20):
    """Trailing realised vol (annualised, decimal) with the same clamps as the GARCH path."""
    try:
        hv = float(pd.Series(log_ret).rolling(window).std().iloc[-1]) * (252 ** 0.5)
        if hv != hv:
            return 0.20
        return float(np.clip(hv, 0.05, 2.0))
    except Exception:
        return 0.20


def _garch_fit_forecast(log_ret_pct):
    """
    One-step-ahead annualised vol forecast from GJR-GARCH(1,1,1)-t, falling
    back to GARCH(1,1)-normal; None when neither fit is financially sensible
    (diverged, non-stationary, nu < 4) or `arch` is unavailable.
    log_ret_pct: pandas Series of log returns in PERCENT (≥ 60 values).
    """
    global _garch_available
    if len(log_ret_pct) < 60:
        return None
    if _garch_available is False:
        return None
    try:
        from arch import arch_model
        _garch_available = True
    except ImportError:
        if _garch_available is None:
            print("  [GARCH] `arch` package not installed — run: pip install arch   (using HV fallback)")
        _garch_available = False
        return None

    def _fit_and_validate(am, max_daily_vol=0.08):
        res = am.fit(disp="off", show_warning=False, options={"maxiter": 300})
        fc       = res.forecast(horizon=1, reindex=False)
        var_pct2 = float(fc.variance.iloc[-1, 0])
        if not np.isfinite(var_pct2) or var_pct2 <= 0:
            return None
        daily_vol  = (var_pct2 ** 0.5) / 100
        if daily_vol > max_daily_vol:                      # model diverged
            return None
        p = res.params
        if "gamma[1]" in p.index:                          # GJR stationarity α + γ/2 + β < 1
            if p.get("alpha[1]", 0) + p.get("gamma[1]", 0) / 2 + p.get("beta[1]", 0) >= 1.0:
                return None
        if "nu" in p.index and float(p["nu"]) < 4:         # fat-tail df too low → variance unreliable
            return None
        return float(np.clip(daily_vol * (252 ** 0.5), 0.05, 2.0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kwargs in (dict(vol="Garch", p=1, o=1, q=1, dist="t"),
                       dict(vol="Garch", p=1, q=1, dist="normal")):
            try:
                v = _fit_and_validate(arch_model(log_ret_pct, rescale=False, **kwargs))
                if v is not None:
                    return v
            except Exception:
                continue
    return None


def _load_garch_cache():
    if not os.path.exists(GARCH_CACHE_FILE):
        return
    try:
        with open(GARCH_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        with _garch_cache_lock:
            _garch_cache.update(data)
        print(f"  [GARCH] Loaded {len(data):,} cached forecasts from {GARCH_CACHE_FILE}")
    except Exception as e:
        print(f"  [GARCH] Could not load {GARCH_CACHE_FILE}: {e}")


_garch_save_lock = threading.Lock()


def _save_garch_cache():
    """Merge this process's fits into the on-disk cache — one writer at a time, unique tmp name."""
    try:
        with _garch_save_lock:
            with _garch_cache_lock:
                snapshot = dict(_garch_cache)
            if os.path.exists(GARCH_CACHE_FILE):
                try:
                    with open(GARCH_CACHE_FILE, "rb") as f:
                        on_disk = pickle.load(f)
                    if isinstance(on_disk, dict):           # a process that never loaded must not shrink the file
                        on_disk.update(snapshot)
                        snapshot = on_disk
                except Exception:
                    pass                                    # unreadable → overwrite it
            tmp = f"{GARCH_CACHE_FILE}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, GARCH_CACHE_FILE)
    except Exception as e:
        print(f"  [GARCH] Could not save cache: {e}")


def rolling_garch_forecast(sym, closes, window=GARCH_WINDOW, refit_every=GARCH_REFIT_BARS, cache=None):
    """
    Per-bar GJR-GARCH forecast for the backtest, using ONLY returns up to and
    including each bar (no look-ahead).  The model is refit every `refit_every`
    bars on the trailing `window` returns and the forecast is held in between;
    the last bar is always a fresh fit so it equals calc_garch_vol() on the same
    history.  NaN for the first 60 bars.  Fits are cached by (sym, number of
    returns fitted, date) — a past date's forecast from the same returns never
    changes — so repeat runs cost nothing.  The final bar is never cached: it
    may be a partial session while the market is open.
    Returns (np.ndarray aligned to closes, new_cache_entries dict).
    """
    cache = _garch_cache if cache is None else cache
    c     = pd.Series(np.asarray(closes, dtype=float))
    lr    = (np.log(c / c.shift(1)) * 100).values            # lr[i] = return INTO bar i (NaN at 0)
    n     = len(c)
    dates = [str(d)[:10] for d in closes.index] if hasattr(closes, "index") else [str(i) for i in range(n)]
    out   = np.full(n, np.nan)
    new   = {}
    last  = None
    for i in range(60, n):
        last_bar = (i == n - 1)
        refit = ((i - 60) % refit_every == 0) or last_bar
        if refit:
            seg = pd.Series(lr[max(1, i - window + 1): i + 1])      # last `window` returns ending at bar i
            key = (sym, len(seg), dates[i])                        # keyed by the returns actually fitted
            v   = None if last_bar else cache.get(key)
            if v is None:
                v = _garch_fit_forecast(seg)
                if v is None:
                    v = _hv_fallback(seg / 100.0)
                if not last_bar:
                    new[key] = v
            last = v
        out[i] = last if last is not None else np.nan
    if new:
        with _garch_cache_lock:
            cache.update(new)
    return out, new


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes option pricing model.
    S     = current stock price
    K     = strike price
    T     = time to expiry in years
    r     = risk-free rate (annual)
    sigma = implied volatility (as decimal, e.g. 0.30 for 30%)
    Returns theoretical fair value, delta, and mispricing info.
    """
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return None

        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)

        return {
            "fair_value": round(price, 3),
            "delta":      round(delta, 3),
        }
    except Exception:
        return None


def bs_mispricing(fair_value, market_price):
    """
    Returns mispricing % and label.
    Positive = overpriced (market > fair), Negative = underpriced (market < fair).
    """
    if not fair_value or not market_price or market_price <= 0:
        return None
    pct = (market_price - fair_value) / fair_value * 100
    if pct > 5:
        label = "Overpriced"
        color = "red"
    elif pct < -5:
        label = "Underpriced"
        color = "green"
    else:
        label = "Fairly Priced"
        color = "yellow"
    return {"pct": round(pct, 1), "label": label, "color": color}


def calc_iv_rank(hist_closes, current_iv):
    """IV Rank proxy using 52-wk rolling 21-day HV range."""
    try:
        ret  = np.log(hist_closes / hist_closes.shift(1)).dropna()
        hvol = ret.rolling(21).std().dropna() * np.sqrt(252) * 100
        if len(hvol) < 10:
            return None
        lo, hi = hvol.min(), hvol.max()
        if hi <= lo:
            return 50.0
        return round(float(np.clip((current_iv - lo) / (hi - lo) * 100, 0, 100)), 1)
    except Exception:
        return None


# ── 6-month T-bill rate (FRED DTB6, refreshed every 6 hours) ─
_rfr_cache = {"value": None, "ts": None}

def fetch_risk_free_rate():
    """
    Fetch the current 6-month US T-bill rate from FRED (series DTB6).
    Updates the global RISK_FREE_RATE. Cached for 6 hours.
    Falls back to the hardcoded value if fetch fails.
    FRED URL: https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB6
    """
    import urllib.request
    global RISK_FREE_RATE
    now = time.time()
    if _rfr_cache["value"] is not None and (now - _rfr_cache["ts"]) < 21600:
        return _rfr_cache["value"]
    try:
        # cosd= limits the download to the last ~2 weeks instead of the full
        # series back to 1958 (~17k lines) just to read one value.
        start = (date.today() - timedelta(days=14)).isoformat()
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB6&cosd={start}"
        with urllib.request.urlopen(url, timeout=6) as resp:
            lines = resp.read().decode().strip().split('\n')
        # CSV: DATE,DTB6 — find last non-missing value
        for line in reversed(lines[1:]):
            parts = line.strip().split(',')
            if len(parts) == 2 and parts[1].strip() not in ('.', ''):
                rate = round(float(parts[1].strip()) / 100.0, 5)
                _rfr_cache.update(value=rate, ts=now)
                RISK_FREE_RATE = rate
                print(f"  [RFR] 6-month T-bill rate updated: {rate*100:.3f}%")
                return rate
    except Exception as e:
        print(f"  [RFR] FRED fetch failed ({e}) — using fallback {RISK_FREE_RATE*100:.2f}%")
    return RISK_FREE_RATE


# ── VIX regime cache (refreshed every 30 min) ────────────────
_vix_cache = {"value": None, "ts": None}

def get_vix():
    """Return current VIX level, cached for 30 min."""
    now = time.time()
    if _vix_cache["value"] is not None and (now - _vix_cache["ts"]) < 1800:
        return _vix_cache["value"]
    try:
        _, v = _fetch_history("^VIX", "5d")
        if v is not None:
            val = round(float(v["Close"].iloc[-1]), 2)
            _vix_cache.update(value=val, ts=now)
            return val
    except Exception:
        pass
    return _vix_cache["value"]   # stale value better than None


def vix_regime(vix):
    """
    low   VIX < 15  → buy options (cheap premium, favor calls/puts)
    normal 15-25    → all strategies valid
    high  VIX > 25  → sell premium (inflated IV, favor sell puts / covered calls)
    """
    if vix is None:
        return "normal"
    if vix < 15:
        return "low"
    if vix > 25:
        return "high"
    return "normal"


# ── VIX3M term structure (refreshed every 30 min) ────────────
_vix3m_cache = {"value": None, "ts": None}

def get_vix3m():
    """
    Return current VIX3M (90-day implied vol index), cached 30 min.
    VIX term structure:
      VIX > VIX3M × 1.05 (backwardation) → acute panic, reversals more likely → favour buying
      VIX < VIX3M × 0.95 (contango)      → calm market → favour selling premium
    """
    now = time.time()
    if _vix3m_cache["value"] is not None and (now - _vix3m_cache["ts"]) < 1800:
        return _vix3m_cache["value"]
    try:
        _, v = _fetch_history("^VIX3M", "5d")
        if v is not None:
            val = round(float(v["Close"].iloc[-1]), 2)
            _vix3m_cache.update(value=val, ts=now)
            return val
    except Exception:
        pass
    return _vix3m_cache["value"]


def vix_term_structure(vix, vix3m):
    """
    Returns 'backwardation', 'contango', or 'flat'.
    Backwardation (VIX > VIX3M by >5%): panic / acute fear → reversals more probable.
    Contango      (VIX < VIX3M by >5%): calm, normal vol  → sell premium setups better.
    """
    if vix is None or vix3m is None or vix3m == 0:
        return "flat"
    ratio = vix / vix3m
    if ratio > 1.05:
        return "backwardation"
    if ratio < 0.95:
        return "contango"
    return "flat"


# ── Fed meeting proximity + yield curve sentiment ─────────────
_fed_cache = {"value": None, "ts": None}

def get_fed_context():
    """
    Returns a dict describing proximity to the next/last FOMC meeting and
    yield-curve-derived rate sentiment.  Cached for 6 hours.

    Keys:
      days_to_next_fed    int  – calendar days until next FOMC decision date
      days_since_last_fed int  – calendar days since last FOMC decision date
      next_fed_date       str  – ISO date of next meeting (YYYY-MM-DD)
      last_fed_date       str  – ISO date of last meeting
      pre_fed_window      bool – within 5 calendar days BEFORE meeting (~3 trading days)
      post_fed_window     bool – within 2 calendar days AFTER meeting
      irx                 float – 3-month T-bill yield (^IRX, annualised %)
      fvx                 float – 5-year Treasury yield (^FVX, annualised %)
      yield_spread        float – fvx - irx (positive = normal; negative = inverted)
      yield_inverted      bool  – irx > fvx → recession / rate-cut expectations
      yield_steep         bool  – fvx - irx > 1.5 → growth / rate-hike expectations

    Strategy implications wired into scoring:
      pre_fed_window  → penalise buy_call / buy_put (IV inflated, will crush post-meeting)
                         bonus for sell_put (collect elevated premium)
      post_fed_window → penalise sell_put (IV already crushed, thin premium)
      yield_inverted  → penalise buy_call (macro headwind); mild bonus for buy_put
      yield_steep     → mild bonus for buy_call (macro tailwind)
    """
    now = time.time()
    if _fed_cache["value"] is not None and (now - _fed_cache["ts"]) < 21600:  # 6 h
        return _fed_cache["value"]

    from datetime import date as _date
    today = _date.today()
    fomc  = sorted([_date.fromisoformat(d) for d in FOMC_DATES])

    past   = [d for d in fomc if d <= today]
    future = [d for d in fomc if d >  today]

    last_fed   = past[-1]   if past   else None
    next_fed   = future[0]  if future else None
    if next_fed is None:
        print("  [Fed] FOMC_DATES has no future meetings — update the list from federalreserve.gov")

    days_to    = (next_fed - today).days if next_fed else None
    days_since = (today - last_fed).days if last_fed else None

    pre_fed  = days_to is not None and 0 < days_to <= 5          # ~3 trading days ahead
    # Decision is announced at 2pm ET on the meeting day; treat the day itself as "pre".
    post_fed = days_since is not None and 0 < days_since <= 2    # 1-2 days after

    # Yield curve from yfinance — ^IRX = 3-month, ^FVX = 5-year
    irx = fvx = None
    try:
        irx_hist = _yf_call(yf.Ticker("^IRX").history, period="5d")
        if not irx_hist.empty:
            irx = round(float(irx_hist["Close"].iloc[-1]), 3)
    except Exception:
        pass
    try:
        fvx_hist = _yf_call(yf.Ticker("^FVX").history, period="5d")
        if not fvx_hist.empty:
            fvx = round(float(fvx_hist["Close"].iloc[-1]), 3)
    except Exception:
        pass

    spread         = round(fvx - irx, 3) if (irx is not None and fvx is not None) else None
    yield_inverted = (spread is not None and spread < 0)
    yield_steep    = (spread is not None and spread > 1.5)

    result = {
        "days_to_next_fed":    days_to,
        "days_since_last_fed": days_since,
        "next_fed_date":       next_fed.isoformat() if next_fed else None,
        "last_fed_date":       last_fed.isoformat() if last_fed else None,
        "pre_fed_window":      pre_fed,
        "post_fed_window":     post_fed,
        "irx":                 irx,
        "fvx":                 fvx,
        "yield_spread":        spread,
        "yield_inverted":      yield_inverted,
        "yield_steep":         yield_steep,
    }
    _fed_cache.update(value=result, ts=now)
    return result


# ── SPY market regime cache (refreshed every 30 min) ─────────
_spy_cache   = {"trend": None, "ts": None}
_macro_cache = {"value": None, "ts": 0.0}


def _warm_market_caches():
    """
    Fetch the shared market-context series once, before a scan's worker
    pool starts.  Otherwise every worker misses the cold caches at the same
    moment and all of them fetch ^VIX, ^VIX3M, ^IRX, ^FVX and SPY at once —
    and the first tickers scanned see beta=1 / RS=0 because the SPY series
    (filled by get_market_trend) isn't there yet.
    """
    for fn in (get_vix, get_vix3m, get_fed_context, get_market_trend, get_macro_regime):
        try:
            fn()
        except Exception as e:
            print(f"  [Warmup] {fn.__name__} failed: {e}")

def get_macro_regime():
    """
    Decide which strategies should be ENABLED based on current macro regime.
    Returns dict: {strategy_name: bool, "reason": str, "label": str}.

    Logic (most-restrictive wins):
      VIX high + backwardation -> defensive only (disable Buy Call)
      VIX > 30                 -> panic — disable all long premium (Buy Call, Buy Put)
      Yield curve inverted     -> disable Buy Call (recession risk on long calls)
      bear_strong market trend -> disable Buy Call (don't catch falling knives)
      bull_strong + VIX low    -> disable Buy Put (no edge in shorting strength)
      Otherwise                -> all strategies enabled

    This is the top-level "should the bot even consider this strategy today"
    gate. Individual signal confluence still has to pass for any qualified
    strategy. Designed to protect portfolio on systemic bad days.
    """
    # Cached for 5 minutes: scan_ticker calls this once per ticker, and the
    # inputs (VIX, VIX3M, Fed context, SPY trend) are themselves cached.
    now = time.time()
    if _macro_cache["value"] is not None and now - _macro_cache["ts"] < 300:
        return dict(_macro_cache["value"])

    try:
        vix      = get_vix() or 18.0
        regime   = vix_regime(vix)        # "low" / "normal" / "high"
        term     = vix_term_structure(vix, get_vix3m())   # "contango" / "backwardation" / "flat"
        fed_ctx  = get_fed_context() or {}
        trend    = get_market_trend()      # "bull_strong" / "bull_weak" / ...
        yi       = fed_ctx.get("yield_inverted", False)
        # The yield-curve rule is a swept parameter now (yield_block_calls): the
        # engine applies it from ctx["yield_inverted"], so the gate itself leaves it out.
        enabled  = macro_rules(vix, regime, term, trend, yi, yield_block_calls=False)
    except Exception as _e:
        enabled = macro_rules(None, "normal", "flat", "neutral", False)
        enabled["reason"] = f"regime lookup error ({_e}) — all strategies enabled by default"

    _macro_cache.update(value=dict(enabled), ts=now)
    return enabled


def macro_rules(vix, regime, term, trend, yield_inverted, yield_block_calls=True):
    """
    The macro gate as a pure function of its inputs — shared by the live
    scanner (get_macro_regime) and the backtest (market_context_history), so
    both apply exactly the same rule at every date.
    yield_block_calls=False leaves the inverted-curve rule to the caller
    (evaluate_entry applies it from its `yield_block_calls` parameter).
    """
    enabled = {"buy_call": True, "buy_put": True, "sell_put": True,
               "covered_call": True, "iron_condor": True, "yield_inverted": bool(yield_inverted)}
    reasons, label_parts = [], []
    vix = vix if vix is not None else 18.0

    # --- Panic regime: VIX > 30 ---
    if vix > 30:
        enabled["buy_call"] = False
        enabled["buy_put"]  = False
        reasons.append(f"VIX {vix:.1f} > 30 — panic regime, no long premium")
        label_parts.append("panic")
    # --- High VIX + backwardation: defensive ---
    elif regime == "high" and term == "backwardation":
        enabled["buy_call"] = False
        reasons.append("High VIX in backwardation — disable Buy Call (downside risk)")
        label_parts.append("def")

    # --- Yield curve inverted: recession risk (optional — see yield_block_calls) ---
    if yield_block_calls and yield_inverted and enabled["buy_call"]:
        enabled["buy_call"] = False
        reasons.append("Yield curve inverted — disable Buy Call (recession lean)")
        label_parts.append("inv")

    # --- Bear-strong trend: don't fight downtrend with calls ---
    if trend == "bear_strong" and enabled["buy_call"]:
        enabled["buy_call"] = False
        reasons.append("Bear-strong trend — disable Buy Call")
        label_parts.append("bear")

    # --- Bull-strong + low VIX: no edge buying puts ---
    if trend == "bull_strong" and regime == "low" and enabled["buy_put"]:
        enabled["buy_put"] = False
        reasons.append("Bull-strong + low VIX — disable Buy Put (no shorting edge)")
        label_parts.append("bull")

    enabled["reason"] = "; ".join(reasons) if reasons else "Normal regime — all strategies enabled"
    enabled["label"]  = "+".join(label_parts) if label_parts else "normal"
    return enabled


def _fed_windows(d, fomc_dates):
    """(pre_fed, post_fed, yield-independent) for calendar date d — same rule as get_fed_context."""
    past   = [x for x in fomc_dates if x <= d]
    future = [x for x in fomc_dates if x >  d]
    days_to    = (future[0] - d).days if future else None
    days_since = (d - past[-1]).days  if past   else None
    pre_fed  = days_to    is not None and 0 < days_to    <= 5
    post_fed = days_since is not None and 0 < days_since <= 2
    return pre_fed, post_fed


def _naive_daily_index(s):
    """Return a Series re-indexed to tz-naive, normalised dates with duplicates dropped."""
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out = pd.Series(np.asarray(s.values, dtype=float), index=idx.normalize())
    return out[~out.index.duplicated(keep="last")]


_ctx_hist_cache = {"df": None, "ts": 0.0, "spy_closes": None}


def market_context_history(period=None, force=False):
    """
    Per-DATE market context for the backtest — the historical counterpart of
    get_vix / get_vix3m / get_fed_context / get_market_trend / get_macro_regime,
    built from the same yfinance series and the same pure rules.

    Returns a DataFrame indexed by trading date with columns:
      vix, vix3m, regime, vix_ts, irx, fvx, yield_inverted, yield_steep,
      spy_close, market_trend, pre_fed, post_fed,
      macro_buy_call, macro_buy_put, macro_sell_put, macro_covered_call, macro_iron_condor
    Cached for an hour.  Also stores the SPY close series for beta / RS.
    """
    period = period or BT_HISTORY_PERIOD                   # always covers the backtest frames (a longer run sets the global)
    now = time.time()
    if not force and _ctx_hist_cache["df"] is not None and now - _ctx_hist_cache["ts"] < 3600 \
            and _ctx_hist_cache.get("period") == period:
        return _ctx_hist_cache["df"]

    _, spy = _fetch_history("SPY", period)
    if spy is None or len(spy) < 220:
        raise RuntimeError("SPY history unavailable — cannot build market context")
    spy_idx = pd.DatetimeIndex(spy.index)
    if spy_idx.tz is not None:
        spy_idx = spy_idx.tz_localize(None)
    spy = spy.copy()
    spy.index = spy_idx.normalize()
    spy = spy[~spy.index.duplicated(keep="last")]

    def _closes(sym):
        _, h = _fetch_history(sym, period)
        return _naive_daily_index(h["Close"]) if h is not None else pd.Series(dtype=float)

    series = {s: _closes(s) for s in ("^VIX", "^VIX3M", "^IRX", "^FVX")}
    empty  = [s for s, v in series.items() if v.empty]
    if empty:            # never build an all-neutral context silently
        raise RuntimeError(f"market-context series unavailable: {', '.join(empty)}")
    vix   = series["^VIX"].reindex(spy.index).ffill()
    vix3m = series["^VIX3M"].reindex(spy.index).ffill()
    irx   = series["^IRX"].reindex(spy.index).ffill()
    fvx   = series["^FVX"].reindex(spy.index).ffill()
    gaps  = {s: int(v.isna().mean() * 100) for s, v in (("^VIX", vix), ("^VIX3M", vix3m), ("^IRX", irx), ("^FVX", fvx))}
    if any(g > 10 for g in gaps.values()):
        print(f"  [Context] WARNING — context series have gaps (% of SPY dates without a value): {gaps}")

    # SPY trend: price vs 200-day MA, ADX > 25 → "strong" (same as get_market_trend)
    ma200 = spy["Close"].rolling(200).mean()
    adx_s, _, _ = calc_adx(spy)
    adx_s = adx_s if adx_s is not None else pd.Series(0.0, index=spy.index)

    fomc = sorted(date.fromisoformat(x) for x in FOMC_DATES)
    rows = []
    for d in spy.index:
        v, v3 = float(vix.get(d, np.nan)), float(vix3m.get(d, np.nan))
        v  = None if v != v else round(v, 2)            # the live getters round VIX / VIX3M to 2 dp …
        v3 = None if v3 != v3 else round(v3, 2)
        regime = vix_regime(v)
        term   = vix_term_structure(v, v3)
        i, f = float(irx.get(d, np.nan)), float(fvx.get(d, np.nan))
        i = None if i != i else round(i, 3)             # … and the yields / spread to 3 dp
        f = None if f != f else round(f, 3)
        spread = round(f - i, 3) if (i is not None and f is not None) else None
        yi = spread is not None and spread < 0
        ys = spread is not None and spread > 1.5
        m200 = float(ma200.get(d, np.nan))
        px   = float(spy["Close"].get(d))
        a    = float(adx_s.get(d, 0.0) or 0.0)
        if m200 != m200:
            trend = "neutral"
        else:
            above, strong = px > m200, a > 25
            trend = ("bull_strong" if above and strong else "bull_weak" if above
                     else "bear_strong" if strong else "bear_weak")
        pre, post = _fed_windows(d.date(), fomc)
        macro = macro_rules(v, regime, term, trend, yi, yield_block_calls=False)   # evaluate_entry applies the yield rule
        rows.append({
            "date": d, "vix": v, "vix3m": v3, "regime": regime, "vix_ts": term,
            "irx": i, "fvx": f,
            "yield_inverted": yi, "yield_steep": ys,
            "spy_close": px, "market_trend": trend, "pre_fed": pre, "post_fed": post,
            **{f"macro_{k}": macro[k] for k in ("buy_call", "buy_put", "sell_put", "covered_call", "iron_condor")},
        })
    df = pd.DataFrame(rows).set_index("date")
    _ctx_hist_cache.update(df=df, ts=now, spy_closes=spy["Close"], period=period)
    print(f"  [Context] Market-context history built: {len(df)} dates ({df.index.min().date()} → {df.index.max().date()})")
    return df


def get_market_trend():
    """
    Return broad market trend based on SPY vs its 200-day MA + ADX.

    Returns one of:
      "bull_strong"  → SPY above 200 MA with ADX > 25 (confirmed uptrend)
      "bull_weak"    → SPY above 200 MA but trend not strong
      "bear_strong"  → SPY below 200 MA with ADX > 25 (confirmed downtrend)
      "bear_weak"    → SPY below 200 MA but trend not strong
      "neutral"      → unable to determine

    Cached for 30 minutes to avoid repeated fetches.
    """
    now = time.time()
    if _spy_cache["trend"] is not None and (now - _spy_cache["ts"]) < 1800:
        return _spy_cache["trend"]
    try:
        _, hist = _fetch_history("SPY", "12mo")
        if hist is None or len(hist) < 200:
            _spy_cache["ts"] = now            # don't re-fetch on every call when SPY is short
            return _spy_cache["trend"] or "neutral"
        closes = hist["Close"]
        price  = float(closes.iloc[-1])
        ma200  = float(closes.rolling(200).mean().iloc[-1])
        above  = price > ma200
        # ADX for trend strength
        adx_ser, pdi_ser, ndi_ser = calc_adx(hist)
        adx = float(adx_ser.iloc[-1]) if adx_ser is not None else 0
        strong = adx > 25
        if above and strong:
            trend = "bull_strong"
        elif above:
            trend = "bull_weak"
        elif not above and strong:
            trend = "bear_strong"
        else:
            trend = "bear_weak"
        _spy_cache.update(trend=trend, ts=now, closes=closes)
        print(f"  [Market] SPY trend = {trend}  (price={price:.2f} ma200={ma200:.2f} adx={adx:.1f})")
        return trend
    except Exception as e:
        print(f"  [Market] SPY trend fetch failed: {e}")
        return _spy_cache["trend"] or "neutral"


# Beta / relative strength vs SPY are computed per bar inside indicator_frame()
# (the shared signal engine) — the previous last-bar-only helpers were removed
# so there is exactly one implementation.


def calc_mfi(hist, period=14):
    """
    Money Flow Index — RSI weighted by volume.
    Combines price direction AND volume participation into one 0-100 oscillator.
    MFI < 20 = oversold on volume (strong buy confirmation)
    MFI > 80 = overbought on volume (strong sell confirmation)
    Returns a pandas Series aligned to hist.index.
    """
    try:
        high   = hist["High"].astype(float)
        low    = hist["Low"].astype(float)
        close  = hist["Close"].astype(float)
        volume = hist["Volume"].astype(float)

        typical_price = (high + low + close) / 3.0
        raw_money_flow = typical_price * volume

        # Positive money flow: typical price rose vs prior bar
        tp_diff = typical_price.diff()
        pos_mf  = raw_money_flow.where(tp_diff > 0, 0.0)
        neg_mf  = raw_money_flow.where(tp_diff < 0, 0.0)

        pos_sum = pos_mf.rolling(period).sum()
        neg_sum = neg_mf.rolling(period).sum()

        # Avoid division by zero
        mfr = pos_sum / neg_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfr))
        return mfi.fillna(50.0)
    except Exception:
        return pd.Series(np.full(len(hist), 50.0), index=hist.index)


def volume_confirmation(hist, lookback=20, multiplier=1.2):
    """
    True if today's volume is at least `multiplier` × 20-day avg.
    Confirms that price moves have participation behind them.
    """
    try:
        vol = hist["Volume"].dropna()
        if len(vol) < lookback + 1:
            return False, None, None
        avg_vol  = float(vol.iloc[-lookback-1:-1].mean())
        cur_vol  = float(vol.iloc[-1])
        confirmed = cur_vol >= avg_vol * multiplier
        return confirmed, round(cur_vol, 0), round(avg_vol, 0)
    except Exception:
        return False, None, None


def get_short_interest(tkr_obj, info=None):
    """
    Return (short_pct_float, short_ratio) from yfinance info.
    short_pct_float: e.g. 0.15 = 15% of float is short.
    short_ratio: days-to-cover.
    Pass info= to reuse an already-fetched info dict and avoid a second API call.
    """
    try:
        if info is None:
            info = _yf_call(lambda: tkr_obj.info)
        pct  = info.get("shortPercentOfFloat")   # 0.0–1.0
        ratio = info.get("shortRatio")
        return (round(float(pct) * 100, 1) if pct   is not None else None,
                round(float(ratio), 1)     if ratio is not None else None)
    except Exception:
        return None, None


def rsi_turning_up(rsi_series, lookback=5):
    """True if RSI had a recent trough and is now rising."""
    tail = rsi_series.dropna().tail(lookback)
    if len(tail) < 3:
        return False
    mid_min = tail.iloc[:-1].min()
    return float(tail.iloc[-1]) > float(mid_min) + 1.5


def analyze_sentiment(text):
    """
    Score a news headline.
    Uses VADER if installed; falls back to keyword counting.
    Returns { compound: float, label: str, color: str }
    """
    sia = _get_sia()
    if sia:
        scores   = sia.polarity_scores(text)
        compound = round(scores["compound"], 3)
    else:
        # Simple keyword fallback
        # Whole-word matches only ("gain" must not match "against", "cut" not "execute")
        words = set(re.findall(r"[a-z]+", text.lower()))
        bull = sum(1 for w in ["beat","upgrade","growth","record","raised","surge",
                                "gain","profit","strong","buy","bullish","positive"] if w in words)
        bear = sum(1 for w in ["miss","downgrade","cut","loss","recall","investigation",
                                "drop","fall","weak","sell","bearish","negative","layoff"] if w in words)
        compound = round((bull - bear) * 0.2, 3)
        compound = max(-1.0, min(1.0, compound))

    if compound >=  0.05: label, color = "Positive", "green"
    elif compound <= -0.05: label, color = "Negative", "red"
    else:                   label, color = "Neutral",  "yellow"
    return {"compound": compound, "label": label, "color": color}


def fetch_ticker_news(sym, max_articles=5):
    """
    Fetch recent news via Yahoo Finance RSS feed (no auth required).
    Results are cached for NEWS_CACHE_MINS minutes.
    Returns list of { title, link, publisher, published, sentiment }.
    """
    now = time.time()
    if sym in _news_cache:
        ts, cached = _news_cache[sym]
        if now - ts < NEWS_CACHE_MINS * 60:
            return cached
    try:
        import requests, xml.etree.ElementTree as ET
        from urllib.parse import quote
        url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(sym)}&region=US&lang=en-US"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            _news_cache[sym] = (now, [])
            return []

        from email.utils import parsedate_to_datetime
        root  = ET.fromstring(resp.text)
        items = root.findall(".//item")
        articles = []
        for item in items[:max_articles]:
            title    = (item.findtext("title")   or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            source   = (item.findtext("source")  or "Yahoo Finance").strip()
            if not title:
                continue
            link = (item.findtext("link") or "").strip() or "#"
            # Only allow web links into the dashboard (RSS data goes into an href)
            if not link.lower().startswith(("http://", "https://")):
                link = "#"

            # Parse pub date to unix timestamp
            pub_ts = 0
            try:
                pub_ts = int(parsedate_to_datetime(pub_date).timestamp())
            except Exception:
                pass

            articles.append({
                "title":     title,
                "link":      link,
                "publisher": source,
                "published": pub_ts,
                "sentiment": analyze_sentiment(title),
            })
        _news_cache[sym] = (now, articles)
        return articles
    except Exception:
        _news_cache[sym] = (now, [])
        return []


def news_signal_conflict(news, strategy):
    """
    Returns a warning string if the overall news sentiment
    contradicts the trade signal, otherwise None.
    """
    if not news:
        return None
    scores    = [a["sentiment"]["compound"] for a in news]
    avg_score = sum(scores) / len(scores)
    bull_strats = {"buy_call", "sell_put"}   # want stock to go up / stay flat
    bear_strats = {"buy_put"}                # want stock to go down

    if strategy in bull_strats and avg_score <= -0.15:
        return f"⚠ News is mostly negative (avg {avg_score:+.2f}) — may suppress bounce"
    if strategy in bear_strats and avg_score >= 0.15:
        return f"⚠ News is mostly positive (avg {avg_score:+.2f}) — may suppress decline"
    return None


def calc_adx(hist, period=14):
    """
    Average Directional Index — measures trend STRENGTH (not direction).
    Returns (adx_series, plus_di_series, minus_di_series).
      ADX > 25  → strong trending move (treat reversals with caution)
      ADX < 20  → weak / sideways market (mean-reversion plays safer)
    +DI > -DI  → uptrend, -DI > +DI → downtrend.
    """
    try:
        high  = hist["High"]
        low   = hist["Low"]
        close = hist["Close"]
        pc    = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - pc).abs(),
            (low  - pc).abs(),
        ], axis=1).max(axis=1)

        up_move   = (high - high.shift(1)).clip(lower=0)
        down_move = (low.shift(1) - low).clip(lower=0)
        # Only the dominant move counts; on a tie both are zero (Wilder).
        # Both masks are taken from the raw moves — zeroing pdm first and then
        # comparing ndm against the zeroed pdm biased -DI upward.
        pdm = up_move.where(up_move > down_move, 0.0)
        ndm = down_move.where(down_move > up_move, 0.0)

        a   = 1.0 / period
        atr = tr.ewm(alpha=a,  adjust=False).mean()
        pdi = 100 * pdm.ewm(alpha=a, adjust=False).mean() / atr
        ndi = 100 * ndm.ewm(alpha=a, adjust=False).mean() / atr

        dx  = (100 * (pdi - ndi).abs() / (pdi + ndi)).replace([np.inf, -np.inf], np.nan).fillna(0)
        adx = dx.ewm(alpha=a, adjust=False).mean()
        return adx, pdi, ndi
    except Exception:
        return None, None, None


def calc_macd(closes, fast=12, slow=26, signal_period=9):
    """Return (macd_line, signal_line, histogram) as pandas Series."""
    ema_fast    = closes.ewm(span=fast,   adjust=False).mean()
    ema_slow    = closes.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def macd_crossover(macd_line, signal_line, lookback=3):
    """
    Scan the last `lookback` bars for a MACD crossover.
    Returns 'bull' (MACD crossed above signal), 'bear' (crossed below), or None.
    """
    m = macd_line.dropna()
    s = signal_line.dropna()
    if len(m) < lookback + 1 or len(s) < lookback + 1:
        return None
    # Newest bar first: if both a bull and a bear cross fall inside the
    # lookback, the most recent one is the live signal.
    for i in range(-1, -lookback - 1, -1):
        prev = float(m.iloc[i - 1]) - float(s.iloc[i - 1])
        curr = float(m.iloc[i])     - float(s.iloc[i])
        if prev < 0 and curr >= 0:
            return "bull"
        if prev > 0 and curr <= 0:
            return "bear"
    return None


def calc_put_call_ratio(tkr_obj, chain=None, exp=None):
    """
    Put/Call volume ratio for the nearest eligible expiry.
    > 1.2  → bearish sentiment (fear / hedging)
    < 0.7  → bullish sentiment (greed / speculation)
    Pass chain=, exp= to reuse pre-fetched data and avoid extra API calls.
    """
    try:
        if exp is None:
            exp, _ = _best_exp(tkr_obj)
        if not exp:
            return None
        if chain is None:
            chain = _option_chain(tkr_obj, exp)
        put_vol  = float(chain.puts["volume"].fillna(0).sum())
        call_vol = float(chain.calls["volume"].fillna(0).sum())
        if call_vol <= 0:
            return None
        return round(put_vol / call_vol, 2)
    except Exception:
        return None


def _parse_earnings_dte(cal):
    """
    Parse a pre-fetched calendar dict/DataFrame and return days until next earnings.
    Extracted so _prefetch_ticker can call this without needing the tkr object.
    """
    from datetime import date as _date, datetime as _dt
    try:
        if cal is None:
            return None
        ed = None
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if isinstance(ed, list):
                ed = ed[0] if ed else None
        elif hasattr(cal, "loc"):
            if "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"].iloc[0]
            elif "Earnings Date" in (cal.columns if hasattr(cal, "columns") else []):
                ed = cal["Earnings Date"].iloc[0]
        if ed is None:
            return None
        if hasattr(ed, "date"):
            ed = ed.date()
        elif isinstance(ed, str):
            ed = _dt.strptime(ed[:10], "%Y-%m-%d").date()
        days = (ed - _date.today()).days
        return int(days) if days >= 0 else None
    except Exception:
        return None


def get_next_earnings_dte(tkr_obj, cal=None):
    """
    Return days until the next earnings date, or None if unavailable.
    Works with both dict and DataFrame variants of yfinance calendar.
    Pass cal= to reuse a pre-fetched calendar and avoid an extra API call.
    """
    try:
        if cal is None:
            cal = _yf_call(lambda: tkr_obj.calendar)
        return _parse_earnings_dte(cal)
    except Exception:
        return None


def safe_list(series, n=60):
    return [round(float(v), 4) if not (v != v) else None for v in series.iloc[-n:]]


# ══════════════════════════════════════════════════════════════
#  OPTIONS-CHAIN HELPERS
# ══════════════════════════════════════════════════════════════

# ── Option data source: Alpaca when configured, Yahoo otherwise ──────────────
_opt_data = {"obj": None, "ts": 0.0}


def implied_vol(price, S, K, T, r, option_type):
    """Black-Scholes implied volatility (decimal) by bisection; None outside the no-arbitrage bounds."""
    try:
        price, S, K, T = float(price), float(S), float(K), float(T)
    except (TypeError, ValueError):
        return None
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None

    def f(sig):
        bs = black_scholes(S, K, T, r, sig, option_type)
        return (bs["fair_value"] if bs else 0.0) - price
    lo, hi = 0.01, 5.0
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-4:
            break
    return round((lo + hi) / 2, 4)


def _alpaca_data():
    """
    The Alpaca option/stock feed (alpaca_data.AlpacaData) when API keys are in
    autotrade_config.json and a probe call succeeds; None otherwise (Yahoo is
    used).  A failed probe is retried every 30 minutes.
    """
    if _opt_data["obj"] is not None:
        return _opt_data["obj"]
    if time.time() - _opt_data["ts"] < 1800:
        return None
    _opt_data["ts"] = time.time()
    try:
        cfg = autotrader.load_config()
        if not (cfg.get("alpaca_key_id") and cfg.get("alpaca_secret")):
            return None
        import alpaca_data
        ad = alpaca_data.AlpacaData(cfg["alpaca_key_id"], cfg["alpaca_secret"], paper=bool(cfg.get("paper", True)),
                                    iv_solver=implied_vol, risk_free=lambda: RISK_FREE_RATE, max_dte=MAX_DTE + 15)
        ad.probe()
        _opt_data["obj"] = ad
        print("  [Data] Alpaca feed active — option chains keep their closing quotes off-hours, IEX pre-market prices")
        return ad
    except Exception as e:
        print(f"  [Data] Alpaca feed unavailable ({e}) — using Yahoo option chains")
        return None


def _option_expirations(tkr_obj, sym=None):
    """Listed expiries for a ticker: Alpaca when the feed is up, else Yahoo."""
    sym = sym or getattr(tkr_obj, "ticker", None)
    ad = _alpaca_data()
    if ad is not None and sym:
        try:
            return ad.expirations(sym)
        except Exception as e:
            print(f"  [Data] {sym} Alpaca expirations failed ({e}) — Yahoo")
    return _yf_call(lambda: tkr_obj.options)


def _option_chain(tkr_obj, exp, sym=None):
    """One expiry's option chain (.calls / .puts): Alpaca when the feed is up, else Yahoo."""
    sym = sym or getattr(tkr_obj, "ticker", None)
    ad = _alpaca_data()
    if ad is not None and sym:
        try:
            return ad.chain(sym, exp)
        except Exception as e:
            print(f"  [Data] {sym} Alpaca chain failed ({e}) — Yahoo")
    return _yf_call(tkr_obj.option_chain, exp)


def _premarket_overlay(sym, hist):
    """
    Before the open (04:00–09:30 ET, Alpaca feed only): append today's latest
    pre-market trade as a provisional daily bar so the gate sees the gap.
    Returns (hist, premarket_price or None).
    """
    ad = _alpaca_data()
    if ad is None or hist is None or len(hist) == 0:
        return hist, None
    try:
        from broker import _eastern_now
        now = _eastern_now()
        minutes = now.hour * 60 + now.minute
        if now.weekday() >= 5 or not (4 * 60 <= minutes < 9 * 60 + 30):
            return hist, None
        last_bar = pd.Timestamp(hist.index[-1])
        last_day = (last_bar.tz_localize(None) if last_bar.tzinfo is None else last_bar.tz_convert(None)).date()
        if last_day >= now.date():
            return hist, None
        px, ts = ad.stock_last(sym)
        if not px or ts is None:
            return hist, None
        ts_et = ts.astimezone(now.tzinfo) if ts.tzinfo else ts
        if ts_et.date() != now.date():
            return hist, None                                  # no print yet today
        idx = pd.Timestamp(now.date())
        if hist.index.tz is not None:
            idx = idx.tz_localize(hist.index.tz)
        row = pd.DataFrame({c: [px if c != "Volume" else 0.0] for c in hist.columns}, index=[idx])
        return pd.concat([hist, row]), float(px)
    except Exception as e:
        print(f"  [Data] {sym} pre-market overlay skipped: {e}")
        return hist, None


def _best_exp(tkr_obj, target=None):
    """Return the expiry closest to `target` DTE (the live dte_target parameter, 30 by default) within [MIN_DTE, MAX_DTE]."""
    exps  = _option_expirations(tkr_obj)
    if not exps:
        return None, None
    target = int(target or live_params().get("dte_target") or OPTION_DTE_TARGET)
    hi     = max(MAX_DTE, target + 15)                    # a longer target widens the window; 30 keeps the classic one
    today = date.today()
    best_exp, best_dte = None, None
    for exp_str in exps:
        # Whole calendar days; subtracting datetime.now() truncated a 30-day expiry to 29.
        dte = (datetime.strptime(exp_str, "%Y-%m-%d").date() - today).days
        if MIN_DTE <= dte <= hi:
            if best_dte is None or abs(dte - target) < abs(best_dte - target):
                best_exp, best_dte = exp_str, dte
    return best_exp, best_dte


def pick_option(tkr_obj, stock_price, option_type="put",
                otm_factor=1.0, require_oi=True, chain=None, exp=None, dte=None):
    """
    Find ATM (or slightly OTM) option for the best expiry.
    otm_factor > 1.0 → OTM call; < 1.0 → OTM put.
    Pass chain=, exp=, dte= to reuse pre-fetched data and avoid extra API calls.
    """
    if exp is None:
        exp, dte = _best_exp(tkr_obj)
    if not exp:
        return None

    try:
        if chain is None:
            chain = _option_chain(tkr_obj, exp)
        opts  = chain.puts if option_type == "put" else chain.calls
        if opts is None or opts.empty:
            return None

        if require_oi:
            opts = opts[(opts["openInterest"].fillna(0) >= MIN_OPEN_INTEREST) &
                        (opts["volume"].fillna(0) >= MIN_VOLUME)]
        if opts.empty:
            return None

        # Nearest strike to target — computed as a local Series so the shared
        # prefetched chain is never mutated.
        target_strike = stock_price * otm_factor
        dist = (opts["strike"] - target_strike).abs()
        row  = opts.loc[dist.idxmin()]

        # yfinance leaves bid/ask/IV/volume as NaN for untraded contracts;
        # NaN is truthy so `x or 0` doesn't catch it — use the NaN-aware helpers.
        bid = _as_float(row.get("bid"))
        ask = _as_float(row.get("ask"))
        mid = _option_mid(row)
        iv  = _as_float(row.get("impliedVolatility")) * 100

        if not (mid >= 0.01) or not (iv >= 1):
            return None
        # Liquidity rule (the backtest applies the same cap to each ticker's measured spread)
        if MAX_SPREAD_PCT < 100 and bid > 0 and ask > bid and (ask - bid) / mid * 100 > MAX_SPREAD_PCT:
            return None

        vol = _as_int(row.get("volume"))
        oi  = _as_int(row.get("openInterest"))
        # Unusual Options Activity (UOA): high volume vs open interest suggests
        # informed/institutional positioning. Ratio > 1.0 = today's volume
        # exceeds existing positions; > 2.0 = strongly unusual.
        vol_oi_ratio = round(vol / max(oi, 1), 2)
        return {
            "type":          option_type,
            "expiration":    exp,
            "dte":           dte,
            "strike":        round(float(row["strike"]), 2),
            "bid":           round(bid, 3),
            "ask":           round(ask, 3),
            "mid":           round(mid, 3),
            "iv":            round(iv, 1),
            "volume":        vol,
            "open_interest": oi,
            "vol_oi_ratio":  vol_oi_ratio,
            "premium_pct":   round(mid / stock_price * 100, 2),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
#  UNIVERSE LOADER
# ══════════════════════════════════════════════════════════════

def get_universe(force_mode=None):
    """
    Fetch ticker universe from multiple sources, falling back gracefully.
    Priority: GitHub CSV → Wikipedia → built-in expanded list
    Respects the global SCAN_MODE — 'focus' returns FOCUS_LIST directly.
    Pass force_mode='full' to override SCAN_MODE (used by WFO when the
    user explicitly requested the full universe).
    """
    import requests
    effective_mode = force_mode or SCAN_MODE
    if effective_mode == "focus":
        tickers = list(dict.fromkeys(FOCUS_LIST))
        print(f"  → Focus mode: {len(tickers)} tickers")
        return tickers
    tickers = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/124.0.0.0 Safari/537.36"}

    # ── Source 1: GitHub CSV (maintained S&P 500 list, very reliable) ──
    try:
        url = ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
               "/main/data/constituents.csv")
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            lines = resp.text.strip().split("\n")[1:]   # skip header
            t = [l.split(",")[0].strip().replace(".", "-")
                 for l in lines if l.strip()]
            t = [x for x in t if 1 <= len(x) <= 6]      # single-letter tickers (C, F, T…) are valid
            tickers.extend(t)
            print(f"  ✓ S&P 500 (GitHub CSV): {len(t)} tickers")
    except Exception as e:
        print(f"  ✗ GitHub CSV: {e}")

    # ── Source 2: Wikipedia (original approach) ──────────────────────
    if not tickers:
        for label, url, col in [
            ("S&P 500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
            ("S&P 400", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol"),
        ]:
            try:
                html = requests.get(url, headers=headers, timeout=15).text
                # pandas 3 treats a bare string as a file path — wrap it
                t = pd.read_html(io.StringIO(html))[0][col].tolist()
                t = [str(x).replace(".", "-") for x in t if isinstance(x, str) and 1 <= len(str(x)) <= 6]
                tickers.extend(t)
                print(f"  ✓ {label} (Wikipedia): {len(t)} tickers")
            except Exception as e:
                print(f"  ✗ {label}: {e}")

    # ── Source 3: Expanded built-in list (500+ tickers) ──────────────
    if not tickers:
        print("  ⚠ All online sources failed — using built-in ticker list")
        tickers = [
            # Mega-cap tech
            "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","TSLA","AVGO","ORCL",
            "AMD","INTC","QCOM","TXN","MU","AMAT","LRCX","KLAC","MRVL","ADI",
            # Financials
            "JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","COF","USB","PNC",
            "TFC","BK","STT","ALLY","SYF","FITB","RF","HBAN","KEY","CFG",
            # Healthcare
            "UNH","JNJ","LLY","ABBV","MRK","PFE","TMO","ABT","DHR","BMY","AMGN",
            "GILD","VRTX","REGN","ISRG","HUM","CI","CVS","MCK","CAH",
            # Energy
            "XOM","CVX","COP","SLB","EOG","MPC","VLO","PSX","OXY","HAL",
            "DVN","BKR","FANG","APA","EQT","AR","RRC",
            # Consumer
            "WMT","COST","TGT","HD","LOW","MCD","SBUX","NKE","YUM","CMG","DRI",
            "AMZN","BKNG","MAR","HLT","MGM","LVS","WYNN","CZR","DKNG","PENN",
            # Communication
            "META","GOOGL","NFLX","DIS","CMCSA","T","VZ","TMUS","WBD",
            "SNAP","PINS","SPOT","RBLX","TTD","ZM","MTCH",
            # Industrial
            "GE","HON","CAT","DE","BA","RTX","LMT","NOC","GD","MMM",
            "UPS","FDX","DAL","UAL","AAL","LUV","ALK","JBLU","NCLH","RCL","CCL",
            # Materials
            "FCX","NEM","AA","CLF","MT","NUE","STLD","CMC","HCC",
            # EV / Clean energy
            "TSLA","RIVN","LCID","NIO","XPEV","LI","PLUG","FCEL","BLNK","CHPT",
            # Fintech / Crypto
            "PYPL","XYZ","COIN","HOOD","SOFI","AFRM","UPST","LC","NU","MQ",   # XYZ = Block (formerly SQ)
            # Retail / E-comm
            "SHOP","ETSY","EBAY","W","CVNA","KSS","M","GAP","ANF",
            # Biotech
            "MRNA","BNTX","BIIB","ALNY","RARE","IONS","ACAD","SAGE","INCY",
            # REITs / Utilities
            "AMT","PLD","EQIX","CCI","SPG","O","VTR","WELL","EXR","AVB",
            "NEE","DUK","SO","D","AEP","XEL","PCG","EIX","ES","AWK",
            # Small/mid popular
            "PLTR","HOOD","SOFI","MARA","RIOT","SPCE","VALE","BB","NOK","SIRI",
            "F","GM","LCID","RIVN","CCL","AAL","NCLH","AMC","GME",
        ]

    # Normalise + dedupe while preserving order (a set here shuffled the
    # scan order on every run, which made progress logs hard to compare).
    tickers = list(dict.fromkeys(
        str(t).replace(".", "-") for t in tickers
        if isinstance(t, str) and 1 <= len(t) <= 6))
    print(f"  → Universe: {len(tickers)} tickers total")
    return tickers


# ══════════════════════════════════════════════════════════════
#  PER-TICKER SCAN  (runs four strategies simultaneously)
# ══════════════════════════════════════════════════════════════

# Prefetch cache: info / calendar with 24h TTL.
# These two yfinance calls barely change day-to-day but they're 2 of the 4
# network calls per ticker per scan. Caching them skips ~50% of repeat
# prefetch round-trips. Pickle is used because tkr.calendar may be a
# pandas DataFrame in some yfinance versions.
# Only the handful of `info` keys the scanner reads are stored — a full
# tkr.info dict is ~150 keys and was making the cache file several MB.
_prefetch_cache      = {}
_prefetch_cache_lock = threading.Lock()
_INFO_KEYS_USED      = ("shortName", "sector", "shortPercentOfFloat", "shortRatio")


def _load_prefetch_cache():
    """Best-effort load of cache from disk. Called once at startup."""
    if not os.path.exists(PREFETCH_CACHE_FILE):
        return
    try:
        with open(PREFETCH_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        with _prefetch_cache_lock:
            _prefetch_cache.update(data)
        print(f"  [Cache] Loaded {len(data)} prefetch entries from {PREFETCH_CACHE_FILE}")
    except Exception as e:
        print(f"  [Cache] Could not load {PREFETCH_CACHE_FILE}: {e}")


def _save_prefetch_cache():
    """Atomic snapshot save (write to .tmp, then os.replace)."""
    try:
        with _prefetch_cache_lock:
            snapshot = dict(_prefetch_cache)
        tmp = PREFETCH_CACHE_FILE + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, PREFETCH_CACHE_FILE)
    except Exception as e:
        print(f"  [Cache] Could not save: {e}")


def _cache_get_field(sym, field):
    """Return cached value for sym[field] if that field is fresh; None otherwise."""
    with _prefetch_cache_lock:
        entry = _prefetch_cache.get(sym)
        if not entry or field not in entry:
            return None
        # Per-field timestamps (a stale `info` must not ride on a fresh `calendar`).
        fetched_at = (entry.get("fetched_at_by_field") or {}).get(field) or entry.get("fetched_at")
        value = entry[field]
    if not fetched_at:
        return None
    try:
        age_h = (datetime.now() - datetime.fromisoformat(fetched_at)).total_seconds() / 3600
        if age_h < PREFETCH_CACHE_TTL_HOURS:
            return value
    except Exception:
        pass
    return None


def _cache_set_field(sym, field, value):
    """Store value for sym[field] with a per-field timestamp. Safe to call concurrently."""
    if value is None:
        return
    if field == "info" and isinstance(value, dict):
        value = {k: value.get(k) for k in _INFO_KEYS_USED if k in value}
    now = datetime.now().isoformat()
    with _prefetch_cache_lock:
        entry = dict(_prefetch_cache.get(sym, {}))       # copy: never mutate a dict a reader may hold
        entry[field] = value
        stamps = dict(entry.get("fetched_at_by_field") or {})
        stamps[field] = now
        entry["fetched_at_by_field"] = stamps
        entry["fetched_at"] = now
        _prefetch_cache[sym] = entry


def _prefetch_ticker(sym, tkr=None):
    """
    Fetch all per-ticker network data for one symbol in one shot:
    info + calendar (24h disk cache), best expiry and that expiry's option chain.
    Returns dict: {tkr, info, calendar, exp, dte, chain} — fields may be None on error.
    """
    result = {"tkr": tkr, "info": None, "calendar": None, "exp": None, "dte": None, "chain": None}
    try:
        tkr = tkr or yf.Ticker(sym)
        result["tkr"] = tkr
        info = _cache_get_field(sym, "info")
        if info is None:
            try:
                info = _yf_call(lambda: tkr.info) or {}
                _cache_set_field(sym, "info", info)
            except Exception as e:
                print(f"  [Prefetch] {sym} info failed: {e}")
        result["info"] = info
        cal = _cache_get_field(sym, "calendar")
        if cal is None:
            try:
                cal = _yf_call(lambda: tkr.calendar)
                _cache_set_field(sym, "calendar", cal)
            except Exception as e:
                print(f"  [Prefetch] {sym} calendar failed: {e}")
        result["calendar"] = cal
        # Best expiry + chain
        exp, dte = _best_exp(tkr)
        if exp:
            result["exp"] = exp
            result["dte"] = dte
            try:
                result["chain"] = _option_chain(tkr, exp, sym)
            except Exception as e:
                print(f"  [Prefetch] {sym} option chain failed: {e}")
    except Exception as e:
        print(f"  [Prefetch] {sym} failed: {e}")
    return result


# ══════════════════════════════════════════════════════════════
#  SHARED SIGNAL ENGINE
#  One set of pure functions decides entries and exits for BOTH the live
#  scanner and the backtest / optimizer:
#    indicator_frame()  per-bar arrays for every indicator the rules read
#    ind_at()           the scalars for one bar (with the live rounding rules)
#    evaluate_entry()   the entry rules  (gate flags, confluence, thresholds)
#    _price_exit_core() / _strategy_exit_check()   the exit rules
#  The live scanner evaluates the LAST bar of a LIVE_HISTORY_PERIOD frame; the
#  backtest evaluates every bar from BT_WARMUP_BARS on of a BT_HISTORY_PERIOD
#  frame plus a per-date market-context table.  Every ranking window (HV rank,
#  IV range, 52-week range, MA200) is full on both sides, so bar i of a backtest
#  frame equals a live scan on that date.  The only things that differ are
#  inputs the past doesn't have: option quotes (P/C ratio, premium, liquidity),
#  short interest and earnings dates.
# ══════════════════════════════════════════════════════════════

GARCH_EXPAND_RATIO   = 1.15   # forecast > trailing HV30 × this → volatility expansion expected
GARCH_CONTRACT_RATIO = 0.85   # forecast < trailing HV30 × this → volatility contraction expected
GARCH_MODES          = ("off", "confluence", "gate")
OPTION_DTE_TARGET    = 30     # the scanner picks the expiry closest to 30 DTE
OPTION_DTE_BARS      = 21     # ≈ 30 calendar days in trading bars (backtest option lifetime)
LIVE_HISTORY_PERIOD  = "2y"   # live frame: ≥ 500 bars, so every window below is full at the last bar
BT_HISTORY_PERIOD    = "3y"   # backtest frame; bars before BT_WARMUP_BARS are warm-up only
BT_WARMUP_BARS       = 282    # first evaluated bar: HV rank (252 HV30 values = 282 closes), IV range, 52w, MA200 all full

IND_FIELDS = (
    # price / candles
    "price", "open", "prev_close",
    # oscillators & bands
    "rsi", "bb_u", "bb_m", "bb_l", "dc_u", "dc_m", "dc_l", "mfi",
    # trend
    "ma50", "ma200", "adx", "pdi", "ndi", "ema9", "ema21",
    "macd", "macd_sig", "macd_hist", "macd_bull", "macd_bear",
    "ema_bull", "ema_bear", "ema_cross_bull", "ema_cross_bear",
    "reversal_up", "vol_confirmed", "cur_vol", "avg_vol",
    # ranges / vol
    "hv_rank", "hv30", "iv_rank_lo", "iv_rank_hi", "low_52w", "high_52w",
    "garch_vol", "garch_expand", "garch_contract",
    # relative strength
    "rs_ratio", "rs_strong", "rs_weak", "beta",
)


def live_params():
    """
    The rule parameters the live scanner runs with: the WFO-applied values
    (scan_state["wfo_params"]) plus the UI toggles.  The backtest receives the
    same dict shape, so a parameter can't exist in one place and not the other.
    """
    w = scan_state.get("wfo_params", {}) or {}
    return {
        "rsi_oversold":         RSI_OVERSOLD,
        "rsi_overbought":       RSI_OVERBOUGHT,
        "bull_min":             w.get("bull_min", 1),
        "bear_min":             w.get("bear_min", 2),
        "adx_threshold":        w.get("adx_threshold", 30),
        "garch_mode":           w.get("garch_mode", "off"),
        "put_trend_gate":       w.get("put_trend_gate", "off"),         # off / not_strong_bull / bear_only
        "yield_block_calls":    bool(w.get("yield_block_calls", False)),  # the old inverted-curve block on calls (off by default)
        "max_spread_pct":       MAX_SPREAD_PCT,                            # liquidity rule (pick_option live / per-ticker in the backtest)
        "entry_mode":           w.get("entry_mode", "reversion"),          # reversion (the classic rules) / escape (band-escape classifier)
        "escape_normal":        w.get("escape_normal", "skip"),            # escape mode, normal-VIX lower escape: skip / call
        "escape_breakout_vol":  w.get("escape_breakout_vol", 2.0),         # escape mode: volume ratio that makes an upper escape a breakout
        "escape_breakout_trend": w.get("escape_breakout_trend", "any"),    # any / bull_only: breakout calls only when SPY is in an uptrend
        "escape_require_uptrend": bool(w.get("escape_require_uptrend", False)),   # whole escape strategy off while SPY < MA200
        "breadth_chg5_min":     w.get("breadth_chg5_min"),                 # escape mode: no entry while breadth (share of the universe above MA50) fell more than this in 5 sessions (None = off)
        "otm_factor_call":      w.get("otm_factor_call", 1.0),             # contract selection: strike / stock price for calls (0.97 = slightly ITM)
        "otm_factor_put":       w.get("otm_factor_put", 1.0),              # … for puts (1.03 = slightly ITM put)
        "dte_target":           w.get("dte_target", OPTION_DTE_TARGET),    # expiry closest to this many days
        "require_green_candle": REQUIRE_GREEN_CANDLE,
        "require_red_candle":   REQUIRE_RED_CANDLE,
    }


# ── Transaction costs: bid/ask spread as % of the option premium ────────────
# No historical option quotes exist.  tests/measure_spreads.py measures today's
# round-trip spread ((ask − bid) / mid) on the ~30-DTE ATM contracts of every
# focus ticker and writes SPREAD_MODEL_FILE; the backtest charges each trade its
# ticker's measured spread (premium-bucket medians when a ticker is missing).
SPREAD_FALLBACK = [(0.5, 16.0), (1.0, 10.0), (3.0, 6.0), (10.0, 4.0), (1e9, 2.5)]   # (premium <, round-trip %)
_spread_model = None


def _get_spread_model():
    """Lazy per-process load of SPREAD_MODEL_FILE (worker processes read it themselves)."""
    global _spread_model
    if _spread_model is None:
        m = _load_json(SPREAD_MODEL_FILE, {}, "Spread") if os.path.exists(SPREAD_MODEL_FILE) else {}
        _spread_model = m if isinstance(m, dict) else {}
    return _spread_model


def _ticker_spread_pct(ticker):
    """Measured median round-trip spread (% of premium) for a ticker's ATM options, or None."""
    t = (_get_spread_model().get("by_ticker") or {}).get(ticker)
    return float(t["spread_pct"]) if t and t.get("n", 0) >= 3 else None


def _spread_cost_pct(ticker, premium):
    """Round-trip cost of one trade as % of its entry premium: the ticker's measurement, else by premium level."""
    t = _ticker_spread_pct(ticker)
    if t is not None:
        return t
    buckets = _get_spread_model().get("buckets") or SPREAD_FALLBACK
    for lim, pct in buckets:
        if premium < lim:
            return float(pct)
    return float(buckets[-1][1])


EXIT_KEYS = ("stop_loss_pct", "profit_target_pct", "trail_activate_pct", "trail_lock_pct",
             "stagnant_days", "stagnant_range", "hold_days", "stock_stop_pct", "stock_trail_pct")
# The exit rules as they shipped (the backtest's "default rules" baseline); a 0 switches a rule off.
BT_DEFAULT_EXITS = {"stop_loss_pct": 15, "profit_target_pct": 30, "trail_activate_pct": 15, "trail_lock_pct": 8,
                    "stagnant_days": 3, "stagnant_range": 5, "hold_days": 21, "stock_stop_pct": 0, "stock_trail_pct": 0}


def live_exit_params():
    """The exit rules the live tracker runs with: the module constants, which the optimizer updates."""
    return {"stop_loss_pct": SELL_STOP_LOSS, "profit_target_pct": SELL_PROFIT_TARGET,
            "trail_activate_pct": TRAIL_ACTIVATE_PCT, "trail_lock_pct": TRAIL_LOCK_PCT,
            "stagnant_days": STAGNANT_DAYS, "stagnant_range": STAGNANT_RANGE,
            "hold_days": HOLD_DAYS, "stock_stop_pct": STOCK_STOP_PCT, "stock_trail_pct": STOCK_TRAIL_PCT}


def _stock_best(strategy, prev_best, cur):
    """Best close since entry in the trade's direction (highest for bullish trades, lowest for buy puts)."""
    cur, prev = _as_float(cur), _as_float(prev_best)
    if cur <= 0:
        return prev if prev > 0 else None
    if prev <= 0:
        return cur
    return min(prev, cur) if strategy == "buy_put" else max(prev, cur)


def _stock_giveback(strategy, best, cur):
    """% the stock has given back from its best close since entry (None when unknown)."""
    best, cur = _as_float(best), _as_float(cur)
    if best <= 0 or cur <= 0:
        return None
    return round(((cur - best) / best * 100) if strategy == "buy_put" else ((best - cur) / best * 100), 2)


_live_breadth = {"breadth50": None, "breadth50_chg5": None, "n": 0, "ts": None}


def breadth_series(hist_map, min_tickers=20):
    """
    Market breadth from a dict of OHLCV histories: per date, the share of tickers
    closing above their 50-day MA (breadth50) and its change over five sessions
    (breadth50_chg5).  Dates with fewer than `min_tickers` values are NaN.
    """
    cols = {}
    for sym, h in (hist_map or {}).items():
        if h is None or len(h) < 55:
            continue
        c = _naive_daily_index(h["Close"])
        ma = c.rolling(50).mean()
        cols[sym] = (c > ma).where(ma.notna())
    if not cols:
        return pd.DataFrame(columns=["breadth50", "breadth50_chg5"])
    m = pd.DataFrame(cols).sort_index()
    m = m[~m.index.duplicated(keep="last")]
    b = m.mean(axis=1).where(m.notna().sum(axis=1) >= min_tickers)
    return pd.DataFrame({"breadth50": b, "breadth50_chg5": b - b.shift(5)})


def _attach_breadth(hist_map, label="Context"):
    """Backtest: add the universe's breadth columns to the cached market-context history (same dates)."""
    ctx = market_context_history()
    b = breadth_series(hist_map)
    if b.empty:
        return
    ctx["breadth50"] = b["breadth50"].reindex(ctx.index)
    ctx["breadth50_chg5"] = b["breadth50_chg5"].reindex(ctx.index)
    ok = ctx["breadth50_chg5"].notna().mean() * 100
    print(f"  [{label}] Breadth attached from {len(hist_map)} tickers ({ok:.0f}% of dates have a 5-day change)")


def _set_live_breadth(hist_map):
    """Live scan: today's breadth from the bulk download, read by live_market_ctx()."""
    b = breadth_series(hist_map)
    if b.empty or b["breadth50"].dropna().empty:
        _live_breadth.update(breadth50=None, breadth50_chg5=None, n=0, ts=datetime.now().isoformat())
        return
    last = b.dropna(subset=["breadth50"]).iloc[-1]
    chg = last["breadth50_chg5"]
    _live_breadth.update(breadth50=round(float(last["breadth50"]), 3),
                         breadth50_chg5=None if chg != chg else round(float(chg), 3),
                         n=len(hist_map), ts=datetime.now().isoformat())
    chg_txt = "n/a" if _live_breadth["breadth50_chg5"] is None else f"{_live_breadth['breadth50_chg5'] * 100:+.0f} pts"
    print(f"  [Breadth] {_live_breadth['breadth50'] * 100:.0f}% of {len(hist_map)} tickers above their 50-day MA; 5-day change {chg_txt}")


def live_market_ctx():
    """Market context for the live scanner, same keys as ctx_at() builds for a past bar."""
    vix   = get_vix()
    vix3m = get_vix3m()
    fed   = get_fed_context() or {}
    return {
        "breadth50":      _live_breadth.get("breadth50"),
        "breadth50_chg5": _live_breadth.get("breadth50_chg5"),
        "vix":            vix,
        "vix3m":          vix3m,
        "regime":         vix_regime(vix),
        "vix_ts":         vix_term_structure(vix, vix3m),
        "market_trend":   get_market_trend(),
        "pre_fed":        bool(fed.get("pre_fed_window")),
        "post_fed":       bool(fed.get("post_fed_window")),
        "yield_inverted": bool(fed.get("yield_inverted")),
        "yield_steep":    bool(fed.get("yield_steep")),
        "irx":            fed.get("irx"),
        "fvx":            fed.get("fvx"),
        "yield_spread":   fed.get("yield_spread"),
        "days_to_next_fed": fed.get("days_to_next_fed"),
        "next_fed_date":  fed.get("next_fed_date"),
        "macro":          get_macro_regime(),
    }


def _macd_cross_flags(diff):
    """
    Vectorised macd_crossover(lookback=3): for each bar look at the three newest
    transitions, newest first; the first bull (prev<0, curr>=0) or bear
    (prev>0, curr<=0) cross wins.  Bars 0-2 are False (fewer than 4 values).
    """
    n = len(diff)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    for i in range(3, n):
        for k in range(0, 3):
            p, c = diff[i - k - 1], diff[i - k]
            if p < 0 and c >= 0:
                bull[i] = True
                break
            if p > 0 and c <= 0:
                bear[i] = True
                break
    return bull, bear


def _rsi_turning_up_flags(rsi):
    """Vectorised rsi_turning_up(lookback=5): last of the 5 newest non-NaN RSI values > min(previous 4) + 1.5."""
    n   = len(rsi)
    out = np.zeros(n, dtype=bool)
    tail = []                               # the newest ≤5 non-NaN values up to bar i
    for i in range(n):
        v = rsi[i]
        if v == v:
            tail.append(float(v))
            if len(tail) > 5:
                tail.pop(0)
        if len(tail) >= 3:
            out[i] = tail[-1] > min(tail[:-1]) + 1.5
    return out


def _hv_rank_array(hv30, window=252, min_values=60):
    """Fraction (×100) of the trailing `window` HV values strictly below the current one — the live formula."""
    n   = len(hv30)
    out = np.full(n, 50.0)
    valid = np.where(~np.isnan(hv30))[0]
    if len(valid) == 0:
        return out
    first = valid[0]
    for i in range(first, n):
        seg = hv30[max(first, i - window + 1): i + 1]
        if len(seg) >= min_values and hv30[i] == hv30[i]:
            out[i] = float((seg < hv30[i]).mean() * 100)
    return out


def indicator_frame(hist, spy_closes=None, sym=None, garch="last"):
    """
    Every per-bar indicator the entry/exit rules read, as numpy arrays aligned
    to hist.index (tz-naive, normalised dates).  Computed exactly as the live
    scanner's helpers did at the last bar — rounding, windows and warm-up
    behaviour included — so evaluating bar i here equals a live scan on a frame
    ending at bar i.

    spy_closes : SPY Close series (for beta / relative strength); None → RS neutral
    garch      : "last" = GJR-GARCH fit at the final bar only (live scan),
                 "all"  = rolling refit every GARCH_REFIT_BARS (backtest, cached),
                 None   = skip (garch_vol = trailing HV fallback)
    Window-length precondition: HV rank ranks the trailing 252 HV30 values (282
    closes), the IV range and 52-week range use 252 bars, MA200 needs 200.  Bar i
    equals a live scan only once all of those are full — i ≥ BT_WARMUP_BARS on a
    backtest frame, always true at the last bar of a LIVE_HISTORY_PERIOD frame.
    EWM-based series (ADX, MACD, EMA) are seeded from the frame start; after ~200
    bars the seed effect is < 1e-6.
    """
    closes  = hist["Close"].astype(float)
    highs   = hist["High"].astype(float) if "High" in hist.columns else closes
    lows    = hist["Low"].astype(float)  if "Low"  in hist.columns else closes
    opens   = hist["Open"].astype(float) if "Open" in hist.columns else closes
    volumes = hist["Volume"].astype(float) if "Volume" in hist.columns else pd.Series(np.nan, index=hist.index)
    n = len(closes)

    idx = pd.DatetimeIndex(hist.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()

    rsi_s = calc_rsi(closes, RSI_PERIOD)
    bb_u, bb_m, bb_l = calc_bb(closes, BB_PERIOD, BB_STD_DEV)
    dc_u, dc_m, dc_l = calc_donchian(hist, DC_PERIOD)
    mfi_s = calc_mfi(hist, period=14)
    ma50  = closes.rolling(50).mean()
    ma200 = closes.rolling(200).mean()
    adx_s, pdi_s, ndi_s = calc_adx(hist)
    nan_s = pd.Series(np.nan, index=closes.index)
    if adx_s is None:
        adx_s, pdi_s, ndi_s = nan_s, nan_s, nan_s
    macd_line, macd_sig, macd_hist = calc_macd(closes)
    ema9  = closes.ewm(span=9,  adjust=False).mean()
    ema21 = closes.ewm(span=21, adjust=False).mean()

    rets   = closes.pct_change()
    hv30_pct = rets.rolling(30).std() * np.sqrt(252) * 100           # annualised %, like the live code
    hv_rank  = _hv_rank_array(hv30_pct.values)
    # calc_iv_rank's range: min/max of 21-day realised vol over the trailing year
    lret   = np.log(closes / closes.shift(1))
    hv21   = lret.rolling(21).std() * np.sqrt(252) * 100
    iv_lo  = hv21.rolling(252, min_periods=10).min()
    iv_hi  = hv21.rolling(252, min_periods=10).max()
    low_52w  = closes.rolling(252, min_periods=1).min()
    high_52w = closes.rolling(252, min_periods=1).max()
    bb_width = (bb_u - bb_l) / bb_m * 100                             # band width % (squeeze release = width > 1.25× its value 20 bars ago)

    macd_bull, macd_bear = _macd_cross_flags((macd_line - macd_sig).values)
    ema_diff = (ema9 - ema21).values
    ema_bull = np.zeros(n, dtype=bool); ema_bear = np.zeros(n, dtype=bool)
    ema_cross_bull = np.zeros(n, dtype=bool); ema_cross_bear = np.zeros(n, dtype=bool)
    if n >= 21:
        ema_bull[20:] = (ema9.values > ema21.values)[20:]
        ema_bear[20:] = (ema9.values < ema21.values)[20:]
    if n >= 3:
        ema_cross_bull[2:] = (ema_diff[:-2] <= 0) & (ema_diff[2:] > 0)
        ema_cross_bear[2:] = (ema_diff[:-2] >= 0) & (ema_diff[2:] < 0)

    reversal = _rsi_turning_up_flags(rsi_s.values)

    # volume_confirmation(): today's volume ≥ 1.2 × mean of the PRIOR 20 bars (today
    # excluded), on the NaN-dropped series exactly as the helper did — a NaN volume
    # inside the window is skipped, a NaN volume today repeats the last bar's verdict.
    vol_vals = volumes.values
    v_valid  = ~np.isnan(vol_vals)
    v_ok_s   = pd.Series(vol_vals[v_valid])
    avg_s    = v_ok_s.rolling(20).mean().shift(1)
    ok_s     = ((v_ok_s >= avg_s * 1.2) & avg_s.notna()).values
    v_pos    = np.cumsum(v_valid) - 1                     # position in v_ok_s of the last non-NaN volume ≤ bar i
    if len(ok_s):
        vol_ok  = np.where(v_pos >= 0, ok_s[np.clip(v_pos, 0, None)], False)
        avg_vol = np.where(v_pos >= 0, avg_s.values[np.clip(v_pos, 0, None)], np.nan)
    else:
        vol_ok  = np.zeros(n, dtype=bool)
        avg_vol = np.full(n, np.nan)

    # Relative strength vs SPY (calc_rs_vs_spy / calc_beta on common dates)
    rs_ratio  = np.zeros(n); rs_strong = np.zeros(n, dtype=bool)
    rs_weak   = np.zeros(n, dtype=bool); beta_arr = np.ones(n)
    if spy_closes is not None and len(spy_closes) > 0:
        try:
            stk = pd.Series(closes.values, index=idx)
            stk = stk[~stk.index.duplicated(keep="last")]   # same de-duplication as the SPY side
            spy = _naive_daily_index(spy_closes)
            common = stk.index.intersection(spy.index)
            if len(common) >= 21:
                s = stk.loc[common]; p = spy.loc[common]
                sr, mr = s.pct_change(), p.pct_change()
                cov = sr.rolling(60).cov(mr); var = mr.rolling(60).var()
                beta = (cov / var).clip(-3.0, 3.0).round(2)
                beta = beta.where(~(var == 0), 1.0).fillna(1.0)
                stk_ret = s / s.shift(20) - 1
                spy_ret = p / p.shift(20) - 1
                resid   = stk_ret - beta * spy_ret
                al = pd.DataFrame({"resid": resid, "beta": beta}).reindex(stk.index).ffill().reindex(idx)
                r = al["resid"].values
                rs_ratio  = np.where(np.isnan(r), 0.0, np.round(r * 100, 2))
                rs_strong = np.where(np.isnan(r), False, r >  0.03)
                rs_weak   = np.where(np.isnan(r), False, r < -0.03)
                beta_arr  = al["beta"].fillna(1.0).values
        except Exception as e:
            print(f"  [RS] {sym or ''} relative-strength calc failed: {e}")

    # GARCH forecast vs trailing realised vol
    hv30_dec = (hv30_pct / 100.0).values
    garch_new = {}
    try:
        if garch == "all":
            garch_arr, garch_new = rolling_garch_forecast(sym or "?", pd.Series(closes.values, index=idx))
        elif garch == "last" and n >= 61:
            garch_arr = np.full(n, np.nan)
            garch_arr[-1] = calc_garch_vol(closes)
        else:
            garch_arr = np.full(n, np.nan)
    except Exception as e:                       # a GARCH failure must never kill a scan
        print(f"  [GARCH] {sym or ''} forecast failed: {e}")
        garch_arr = np.full(n, np.nan)
    with np.errstate(invalid="ignore"):
        g_exp = (garch_arr > hv30_dec * GARCH_EXPAND_RATIO)
        g_con = (garch_arr < hv30_dec * GARCH_CONTRACT_RATIO)

    return {
        "index": idx, "n": n, "sym": sym,
        "price": closes.values, "open": opens.values,
        "prev_close": np.concatenate([[np.nan], closes.values[:-1]]),
        "rsi": rsi_s.values, "bb_u": bb_u.values, "bb_m": bb_m.values, "bb_l": bb_l.values,
        "prev_bb_u": bb_u.shift(1).values, "prev_bb_l": bb_l.shift(1).values,
        "bb_width": bb_width.values, "bb_width_20": bb_width.shift(20).values,
        "dc_u": dc_u.values, "dc_m": dc_m.values, "dc_l": dc_l.values, "mfi": mfi_s.values,
        "ma50": ma50.values, "ma200": ma200.values,
        "adx": np.round(adx_s.values, 1), "pdi": np.round(pdi_s.values, 1), "ndi": np.round(ndi_s.values, 1),
        "ema9": ema9.values, "ema21": ema21.values,
        "macd": macd_line.values, "macd_sig": macd_sig.values, "macd_hist": macd_hist.values,
        "macd_bull": macd_bull, "macd_bear": macd_bear,
        "ema_bull": ema_bull, "ema_bear": ema_bear,
        "ema_cross_bull": ema_cross_bull, "ema_cross_bear": ema_cross_bear,
        "reversal_up": reversal, "vol_confirmed": vol_ok,
        "cur_vol": volumes.values, "avg_vol": avg_vol,
        "hv_rank": hv_rank, "hv30": hv30_dec, "iv_rank_lo": iv_lo.values, "iv_rank_hi": iv_hi.values,
        "low_52w": low_52w.values, "high_52w": high_52w.values,
        "garch_vol": garch_arr, "garch_expand": g_exp, "garch_contract": g_con,
        "rs_ratio": rs_ratio, "rs_strong": rs_strong, "rs_weak": rs_weak, "beta": beta_arr,
        "volume": volumes.values,
        "_garch_new": garch_new,          # fresh GARCH fits (worker processes hand these back to the main cache)
    }


def _f(x):
    """numpy scalar → float, NaN stays NaN (comparisons on NaN are False, like the live code)."""
    return float(x)


def _f_or_none(x):
    v = float(x)
    return None if v != v else v


def ind_at(fr, i):
    """The indicator scalars for bar i of an indicator_frame, with the live rounding/None rules."""
    i = i if i >= 0 else fr["n"] + i
    g = lambda k: fr[k][i]
    mfi = _f(g("mfi"))
    hv  = _f(g("hv_rank"))
    return {
        "i": i, "date": fr["index"][i],
        "price": _f(g("price")), "open": _f(g("open")), "prev_close": _f(g("prev_close")),
        "rsi": _f(g("rsi")), "bb_u": _f(g("bb_u")), "bb_m": _f(g("bb_m")), "bb_l": _f(g("bb_l")),
        "prev_bb_u": _f(g("prev_bb_u")), "prev_bb_l": _f(g("prev_bb_l")),
        "bb_width": _f_or_none(g("bb_width")), "bb_width_20": _f_or_none(g("bb_width_20")),
        "dc_u": _f(g("dc_u")), "dc_m": _f(g("dc_m")), "dc_l": _f(g("dc_l")),
        "mfi": 50.0 if mfi != mfi else mfi,
        "ma50": _f_or_none(g("ma50")), "ma200": _f_or_none(g("ma200")),
        "adx": _f_or_none(g("adx")), "pdi": _f_or_none(g("pdi")), "ndi": _f_or_none(g("ndi")),
        "macd": _f(g("macd")), "macd_sig": _f(g("macd_sig")), "macd_hist": _f(g("macd_hist")),
        "macd_bull": bool(g("macd_bull")), "macd_bear": bool(g("macd_bear")),
        "ema_bull": bool(g("ema_bull")), "ema_bear": bool(g("ema_bear")),
        "ema_cross_bull": bool(g("ema_cross_bull")), "ema_cross_bear": bool(g("ema_cross_bear")),
        "reversal_up": bool(g("reversal_up")), "vol_confirmed": bool(g("vol_confirmed")),
        "cur_vol": _f_or_none(g("cur_vol")), "avg_vol": _f_or_none(g("avg_vol")),
        "hv_rank": 50.0 if hv != hv else hv, "hv30": _f_or_none(g("hv30")),
        "iv_rank_lo": _f_or_none(g("iv_rank_lo")), "iv_rank_hi": _f_or_none(g("iv_rank_hi")),
        "low_52w": _f(g("low_52w")), "high_52w": _f(g("high_52w")),
        "garch_vol": _f_or_none(g("garch_vol")),
        "garch_expand": bool(g("garch_expand")), "garch_contract": bool(g("garch_contract")),
        "rs_ratio": _f(g("rs_ratio")), "rs_strong": bool(g("rs_strong")), "rs_weak": bool(g("rs_weak")),
        "beta": _f(g("beta")),
    }


def evaluate_entry(ind, ctx, params, scan_condors=False):
    """
    THE entry rules.  Pure function of one bar's indicators (`ind`), the market
    context for that date (`ctx`) and the rule parameters (`params`).
    Returns every intermediate boolean the dashboard shows plus the five
    has_* gate flags.  pc_bullish / pc_bearish are NOT inputs: the live scanner
    only knows the put/call ratio after the gate, so it never affects whether a
    strategy fires (it is added to the confluence counts for scoring afterwards).
    """
    p        = params
    price    = ind["price"]
    rsi      = ind["rsi"]
    regime   = ctx.get("regime", "normal")
    macro    = ctx.get("macro") or {}
    rsi_os_base = p.get("rsi_oversold", RSI_OVERSOLD)
    rsi_ob_base = p.get("rsi_overbought", RSI_OVERBOUGHT)

    # VIX-regime adjusted RSI thresholds
    if regime == "high":
        rsi_os_thr = max(rsi_os_base - 5, 20)
        rsi_ob_thr = min(rsi_ob_base + 5, 80)
    elif regime == "low":
        rsi_os_thr = rsi_os_base + 2
        rsi_ob_thr = rsi_ob_base - 2
    else:
        rsi_os_thr = rsi_os_base
        rsi_ob_thr = rsi_ob_base

    dc_at_lower  = price <= ind["dc_l"] * 1.01
    dc_at_upper  = price >= ind["dc_u"] * 0.99
    dc_range_pct = (ind["dc_u"] - ind["dc_l"]) / price * 100 if price > 0 else 0
    dc_squeeze   = dc_range_pct < 8

    oversold   = rsi < rsi_os_thr or price < ind["bb_l"] or dc_at_lower
    overbought = rsi > rsi_ob_thr or price > ind["bb_u"] or dc_at_upper

    ma50, ma200 = ind["ma50"], ind["ma200"]
    above_ma50 = ma50  is not None and price > ma50
    trend_bull = ma200 is not None and price > ma200
    trend_bear = ma200 is not None and price < ma200

    adx, pdi, ndi = ind["adx"], ind["pdi"], ind["ndi"]
    adx_thr      = float(p.get("adx_threshold", 30))
    strong_trend = adx is not None and adx > adx_thr
    adx_mode     = ("ranging" if (adx or 0) < 20 else "trending" if (adx or 0) > 30 else "normal")

    near_52w_low  = price <= ind["low_52w"]  * 1.05
    near_52w_high = price >= ind["high_52w"] * 0.95

    reversal_up       = ind["reversal_up"]
    _raw_knife        = trend_bear and strong_trend and (ndi or 0) > (pdi or 0)
    exhaustion_bounce = rsi < 30 and reversal_up
    falling_knife     = _raw_knife and not exhaustion_bounce
    melt_up           = trend_bull and strong_trend and (pdi or 0) > (ndi or 0)

    hv_rank = ind["hv_rank"]
    hv_high = hv_rank >= 60
    hv_low  = hv_rank <= 40
    sig_mfi_os = ind["mfi"] < 20
    sig_mfi_ob = ind["mfi"] > 80

    regime_favors_buying = regime == "low"
    vix_backw    = ctx.get("vix_ts") == "backwardation"
    vix_contango = ctx.get("vix_ts") == "contango"

    garch_mode = p.get("garch_mode", "off")
    g_exp = ind["garch_expand"] if garch_mode != "off" else False
    g_con = ind["garch_contract"] if garch_mode != "off" else False

    _rng = adx_mode == "ranging"
    _trd = adx_mode == "trending"
    bull_signals = sum([
        rsi < 30,
        rsi < 30 and _rng,
        price < ind["bb_l"],
        price < ind["bb_l"] and _rng,
        dc_at_lower,
        reversal_up,
        ind["macd_bull"],
        ind["macd_bull"] and _trd,
        ind["vol_confirmed"],
        regime_favors_buying,
        False,                              # pc_bullish — unknown at gate time (see docstring)
        ind["rs_strong"],
        hv_low,
        near_52w_low,
        vix_backw,
        sig_mfi_os,
        ind["ema_bull"],
        ind["ema_cross_bull"],
        g_exp and garch_mode == "confluence",   # vol expansion expected → long options favoured
    ])
    bear_signals = sum([
        rsi > rsi_ob_thr,
        rsi > rsi_ob_thr and _rng,
        price > ind["bb_u"],
        price > ind["bb_u"] and _rng,
        dc_at_upper,
        ind["macd_bear"],
        ind["macd_bear"] and _trd,
        ind["vol_confirmed"],
        False,                              # pc_bearish — unknown at gate time
        ind["rs_weak"],
        near_52w_high,
        sig_mfi_ob,
        ind["ema_bear"],
        ind["ema_cross_bear"],
        g_exp and garch_mode == "confluence",
    ])

    bull_min = min(int(p.get("bull_min", 1)), 3)
    bear_min = min(int(p.get("bear_min", 2)), 3)
    market_trend = ctx.get("market_trend", "neutral")
    breadth_chg5 = ctx.get("breadth50_chg5")                 # None until a scan / backtest has computed the universe's breadth
    if market_trend == "bull_strong":
        bear_min = min(bear_min + 1, 4)
    elif market_trend == "bear_strong":
        bear_min = max(1, bear_min - 1)
    # Buy-put trend gate (swept): puts only when the broad market is not strongly bullish / only in a bear trend
    put_gate     = p.get("put_trend_gate", "off")
    put_trend_ok = (put_gate == "bear_only" and str(market_trend).startswith("bear")) or \
                   (put_gate == "not_strong_bull" and market_trend != "bull_strong") or \
                   put_gate not in ("bear_only", "not_strong_bull")
    # Inverted yield curve blocks buy calls only when the parameter says so (it was always-on before)
    yield_blocked = bool(p.get("yield_block_calls", False)) and bool(ctx.get("yield_inverted", False))

    prev_close   = ind["prev_close"]
    have_prev    = prev_close == prev_close
    green_candle = price > prev_close if have_prev else True
    red_candle   = price < prev_close if have_prev else True
    green_ok     = (not p.get("require_green_candle", False)) or green_candle or exhaustion_bounce
    red_ok       = (not p.get("require_red_candle", False)) or red_candle
    if have_prev and prev_close != 0:
        gap_pct = (ind["open"] - prev_close) / prev_close * 100   # a NaN Open gives NaN: both gap filters fail, as the original scanner did
    else:
        gap_pct = 0.0
    gap_ok_buy_call = gap_pct >= -3.0
    gap_ok_buy_put  = gap_pct <=  3.0
    trend_ok_buy    = above_ma50 or rsi < 25

    has_sell_put     = bool(macro.get("sell_put", True))     and (oversold or ind["macd_bear"]) and price <= MAX_PRICE_SELL and not falling_knife and gap_ok_buy_call
    has_buy_call     = bool(macro.get("buy_call", True))     and not yield_blocked and oversold and bull_signals >= bull_min and not falling_knife and green_ok and trend_ok_buy and gap_ok_buy_call
    has_buy_put      = bool(macro.get("buy_put", True))      and put_trend_ok and (overbought or ind["macd_bear"]) and bear_signals >= bear_min and not melt_up and red_ok and gap_ok_buy_put
    has_covered_call = bool(macro.get("covered_call", True)) and price <= MAX_PRICE_SELL
    has_iron_condor  = bool(macro.get("iron_condor", True))  and (not oversold) and (not overbought) and (35 <= rsi <= 65) and bool(scan_condors)

    # ── Entry mode "escape": classify a FIRST-DAY Bollinger-band escape (tests/band_escape_study.py) ──
    # Lower escape in a scared market (high-VIX regime / SPY bear trend) snaps back → buy call,
    # sell put.  Lower escape in a calm market (low VIX) with the vol model expecting expansion
    # or price under its MA200 keeps falling → buy put.  A ≥3% gap-down is news → no trade.
    # Upper escape: only a big-volume breakout continues → buy call; a breakout is never shorted.
    # The macro gate is NOT applied in this mode (its "VIX > 30: no long premium" rule is the
    # opposite of what the escapes showed).  The GARCH gate below still applies.
    vol_ratio = (ind["cur_vol"] / ind["avg_vol"]) if (ind.get("cur_vol") and ind.get("avg_vol")) else None
    bbw, bbw20 = ind.get("bb_width"), ind.get("bb_width_20")
    squeeze_release = bool(bbw is not None and bbw20 is not None and bbw > bbw20 * 1.25)
    escape_case = None
    if p.get("entry_mode", "reversion") == "escape":
        pbl, pbu = ind.get("prev_bb_l"), ind.get("prev_bb_u")
        prev_inside  = have_prev and pbl == pbl and pbu == pbu and pbl <= prev_close <= pbu
        lower_escape = prev_inside and price < ind["bb_l"]
        upper_escape = prev_inside and price > ind["bb_u"]
        scared = regime == "high" or str(market_trend).startswith("bear")
        calm   = regime == "low"
        if lower_escape and gap_pct > -3.0:
            if scared:
                escape_case = "reversion"
            elif calm and (ind["garch_expand"] or not trend_bull):
                escape_case = "momentum"
            elif not calm and p.get("escape_normal", "skip") == "call":
                escape_case = "reversion"
        elif upper_escape and vol_ratio is not None and vol_ratio >= float(p.get("escape_breakout_vol", 2.0)) \
                and (p.get("escape_breakout_trend", "any") != "bull_only" or str(market_trend).startswith("bull")):
            escape_case = "breakout"                      # bull_only: no breakout calls while SPY is under its 200-day MA (2022 lesson)
        if escape_case and p.get("escape_require_uptrend", False) and not str(market_trend).startswith("bull"):
            escape_case = None                        # regime switch: no long premium while SPY is under its 200-day MA (2022)
        bmin = p.get("breadth_chg5_min")
        if escape_case and bmin is not None and breadth_chg5 is not None and breadth_chg5 < float(bmin):
            escape_case = None                        # breadth collapsing = a market-wide selloff in progress: the escape is a crash, not a dip
        has_buy_call     = escape_case in ("reversion", "breakout") and gap_ok_buy_call
        has_buy_put      = escape_case == "momentum" and gap_ok_buy_put
        has_sell_put     = escape_case == "reversion" and price <= MAX_PRICE_SELL and gap_ok_buy_call
        has_covered_call = False
        has_iron_condor  = False
    if garch_mode == "gate":
        # Long premium only when the model expects vol to expand; short premium only when it expects contraction.
        has_buy_call     = has_buy_call     and g_exp
        has_buy_put      = has_buy_put      and g_exp
        has_sell_put     = has_sell_put     and g_con
        has_covered_call = has_covered_call and g_con

    return {
        "rsi_os_thr": rsi_os_thr, "rsi_ob_thr": rsi_ob_thr,
        "dc_at_lower": dc_at_lower, "dc_at_upper": dc_at_upper,
        "dc_range_pct": dc_range_pct, "dc_squeeze": dc_squeeze,
        "oversold": oversold, "overbought": overbought,
        "above_ma50": above_ma50, "trend_bull": trend_bull, "trend_bear": trend_bear,
        "adx_thr": adx_thr, "strong_trend": strong_trend, "adx_mode": adx_mode,
        "near_52w_low": near_52w_low, "near_52w_high": near_52w_high,
        "exhaustion_bounce": exhaustion_bounce, "falling_knife": falling_knife, "melt_up": melt_up,
        "hv_high": hv_high, "hv_low": hv_low, "sig_mfi_os": sig_mfi_os, "sig_mfi_ob": sig_mfi_ob,
        "regime_favors_buying": regime_favors_buying, "vix_backw": vix_backw, "vix_contango": vix_contango,
        "garch_mode": garch_mode, "garch_expand": g_exp, "garch_contract": g_con,
        "bull_signals": bull_signals, "bear_signals": bear_signals,
        "bull_min": bull_min, "bear_min": bear_min,
        "green_candle": green_candle, "red_candle": red_candle, "green_ok": green_ok, "red_ok": red_ok,
        "gap_pct": gap_pct, "gap_ok_buy_call": gap_ok_buy_call, "gap_ok_buy_put": gap_ok_buy_put,
        "trend_ok_buy": trend_ok_buy, "put_trend_ok": put_trend_ok, "yield_blocked": yield_blocked,
        "market_trend": market_trend, "entry_mode": p.get("entry_mode", "reversion"), "escape_case": escape_case,
        "breadth50_chg5": breadth_chg5,
        "vol_ratio": vol_ratio, "squeeze_release": squeeze_release,
        "has_sell_put": has_sell_put, "has_buy_call": has_buy_call, "has_buy_put": has_buy_put,
        "has_covered_call": has_covered_call, "has_iron_condor": has_iron_condor,
        "any": any([has_sell_put, has_buy_call, has_buy_put, has_covered_call, has_iron_condor]),
        # Individual flags recorded on signal-log entries and backtest trades (calibration inputs)
        "flags": {
            "sig_rsi_os": rsi < rsi_os_thr, "sig_bb_low": price < ind["bb_l"], "sig_dc_low": dc_at_lower,
            "sig_reversal": reversal_up, "sig_macd_bull": ind["macd_bull"], "sig_vol_ok": ind["vol_confirmed"],
            "sig_rsi_ob": rsi > rsi_ob_thr, "sig_bb_high": price > ind["bb_u"], "sig_dc_high": dc_at_upper,
            "sig_macd_bear": ind["macd_bear"], "sig_above_ma50": above_ma50,
            "sig_hv_high": hv_high, "sig_hv_low": hv_low,
            "sig_rs_strong": ind["rs_strong"], "sig_rs_weak": ind["rs_weak"],
            "sig_near_52w_low": near_52w_low, "sig_near_52w_high": near_52w_high,
            "sig_mfi_os": sig_mfi_os, "sig_mfi_ob": sig_mfi_ob,
            "sig_ema_bull": ind["ema_bull"], "sig_ema_bear": ind["ema_bear"],
            "sig_ema_cross_bull": ind["ema_cross_bull"], "sig_ema_cross_bear": ind["ema_cross_bear"],
            "sig_vix_backw": vix_backw, "sig_vix_contango": vix_contango,
            "sig_regime_low": regime_favors_buying,
            "sig_garch_expand": ind["garch_expand"], "sig_garch_contract": ind["garch_contract"],
            "sig_exhaustion": exhaustion_bounce,
            "sig_squeeze_release": squeeze_release,
            "sig_escape_reversion": escape_case == "reversion", "sig_escape_momentum": escape_case == "momentum",
            "sig_escape_breakout": escape_case == "breakout",
        },
    }


def entry_blockers(ev, ind):
    """Human-readable reason each strategy did NOT fire (for the debug endpoint)."""
    price = ind["price"]
    out = {}
    out["sell_put"] = (
        f"price > ${MAX_PRICE_SELL:.0f} limit" if price > MAX_PRICE_SELL
        else "falling_knife" if ev["falling_knife"]
        else "not oversold or no macd_bear" if not (ev["oversold"] or ind["macd_bear"])
        else "gap down > 3%" if not ev["gap_ok_buy_call"]
        else "garch gate (no contraction)" if ev["garch_mode"] == "gate" and not ev["garch_contract"]
        else None if ev["has_sell_put"] else "macro regime disabled")
    out["buy_call"] = (
        "falling_knife (not exhaustion_bounce)" if ev["falling_knife"]
        else "not oversold" if not ev["oversold"]
        else f"bull_signals {ev['bull_signals']} < min {ev['bull_min']}" if ev["bull_signals"] < ev["bull_min"]
        else "green_candle filter ON — needs green day or exhaustion bounce" if not ev["green_ok"]
        else "below MA50 and RSI ≥ 25 (trend_ok_buy)" if not ev["trend_ok_buy"]
        else "gap down > 3%" if not ev["gap_ok_buy_call"]
        else "garch gate (no expansion)" if ev["garch_mode"] == "gate" and not ev["garch_expand"]
        else "yield curve inverted (yield_block_calls on)" if ev.get("yield_blocked")
        else None if ev["has_buy_call"] else "macro regime disabled")
    out["buy_put"] = (
        "melt_up" if ev["melt_up"]
        else "not overbought or no macd_bear" if not (ev["overbought"] or ind["macd_bear"])
        else f"bear_signals {ev['bear_signals']} < min {ev['bear_min']}" if ev["bear_signals"] < ev["bear_min"]
        else "red_candle filter" if not ev["red_ok"]
        else "gap up > 3%" if not ev["gap_ok_buy_put"]
        else "garch gate (no expansion)" if ev["garch_mode"] == "gate" and not ev["garch_expand"]
        else f"put_trend_gate: market trend is {ev.get('market_trend')}" if not ev.get("put_trend_ok", True)
        else None if ev["has_buy_put"] else "macro regime disabled")
    return out


# ── Exit rules (shared by the live tracker and the backtest walk) ──────────

def _adverse_stock_move(strategy, entry_stock, cur_stock):
    """% the STOCK has moved against a position since entry (bullish trades: down, buy puts: up); None when unknown."""
    e, c = _as_float(entry_stock), _as_float(cur_stock)
    if e <= 0 or c <= 0:
        return None
    if strategy in ("buy_call", "sell_put", "covered_call"):
        return round((e - c) / e * 100, 2)
    if strategy == "buy_put":
        return round((c - e) / e * 100, 2)
    return None


def _price_exit_core(pct_change, prev_peak, days_held, ex=None, stock_move_pct=None, stock_giveback_pct=None):
    """
    Price-based exits in the live tracker's order:
      profit target → premium stop → stock stop → stock trail → trailing stop → stagnation → max hold.
    ex                 : the exit parameters (live_exit_params() live, the combo's exits in the
                         backtest); a 0 switches that rule off
    days_held          : trading days held (np.busday_count live, bars in the backtest)
    stock_move_pct     : % the stock has moved AGAINST the position since entry (None = unknown)
    stock_giveback_pct : % the stock has given back from its best close since entry (None = unknown)
    Returns (triggered, reasons, peak_pct, code).
    """
    ex = ex or live_exit_params()
    tp = float(ex.get("profit_target_pct") or 0)
    sl = float(ex.get("stop_loss_pct") or 0)
    ss = float(ex.get("stock_stop_pct") or 0)
    st = float(ex.get("stock_trail_pct") or 0)
    ta = float(ex.get("trail_activate_pct") or 0)
    tl = float(ex.get("trail_lock_pct") or 0)
    sd = int(ex.get("stagnant_days") or 0)
    sr = float(ex.get("stagnant_range") or 0)
    mh = int(ex.get("hold_days") or 0)
    reasons, triggered, code = [], False, None
    if tp > 0 and pct_change >= tp:
        triggered, code = True, "take_profit"
        reasons.append(f"🎯 Profit target hit: +{pct_change:.1f}% (target +{tp:g}%)")
    elif sl > 0 and pct_change <= -sl:
        triggered, code = True, "stop_loss"
        reasons.append(f"🛑 Stop loss hit: {pct_change:.1f}% (limit -{sl:g}%)")
    if not triggered and ss > 0 and stock_move_pct is not None and stock_move_pct >= ss:
        triggered, code = True, "stock_stop"
        reasons.append(f"🧯 Stock stop: underlying moved {stock_move_pct:.1f}% against the position (limit {ss:g}%)")
    if not triggered and st > 0 and stock_giveback_pct is not None and stock_giveback_pct >= st:
        triggered, code = True, "stock_trail"
        reasons.append(f"📉 Stock trail: underlying gave back {stock_giveback_pct:.1f}% from its best close (limit {st:g}%)")

    peak_pct = max(_as_float(prev_peak, -999.0), pct_change)
    if ta > 0 and peak_pct >= ta and not triggered:
        trailing_stop = peak_pct - tl
        if pct_change <= trailing_stop:
            triggered, code = True, "trailing_stop"
            reasons.append(f"📉 Trailing stop: dropped to {pct_change:+.1f}% from peak +{peak_pct:.1f}% "
                           f"(trail lock at {trailing_stop:+.1f}%)")

    if days_held is not None and not triggered:
        if sd > 0 and days_held >= sd and abs(pct_change) <= sr:
            triggered, code = True, "stagnation"
            reasons.append(f"⏳ Stagnation exit: held {days_held} trading days with P&L {pct_change:+.1f}% "
                           f"(within ±{sr:g}%) — theta bleed, exit and redeploy")
        elif mh > 0 and days_held >= mh:
            triggered, code = True, "end_of_hold"
            reasons.append(f"⌛ Max hold reached: {days_held} trading days (limit {mh}) — close and redeploy")
    return triggered, reasons, round(peak_pct, 2), code


def _strategy_exit_check(strategy, price, rsi, bb_u, bb_l, strike=None, short_put=None, short_call=None):
    """
    The per-strategy RSI / Bollinger / strike exits from the live tracker.
    Returns (triggered, reasons, signal_still_valid, note, code).
    """
    reasons, triggered, code, note = [], False, None, None
    still_valid = True
    if strategy == "buy_call":
        if rsi >= SELL_RSI_BULL_TARGET:
            triggered, code = True, "rsi_target"
            reasons.append(f"📈 RSI recovered to {rsi:.1f} (target ≥{SELL_RSI_BULL_TARGET}) — mean reversion complete, take profit")
        if price >= bb_u:
            triggered, code = True, code or "bb_target"
            reasons.append(f"📈 Price ${price:.2f} hit upper Bollinger Band ${bb_u:.2f} — extended, take profit")
        still_valid = rsi < SELL_RSI_BULL_TARGET
    elif strategy == "buy_put":
        if rsi <= SELL_RSI_BEAR_TARGET:
            triggered, code = True, "rsi_target"
            reasons.append(f"📉 RSI pulled back to {rsi:.1f} (target ≤{SELL_RSI_BEAR_TARGET}) — mean reversion complete, take profit")
        if price <= bb_l:
            triggered, code = True, code or "bb_target"
            reasons.append(f"📉 Price ${price:.2f} hit lower Bollinger Band ${bb_l:.2f} — extended, take profit")
        still_valid = rsi > SELL_RSI_BEAR_TARGET
    elif strategy == "sell_put":
        if price < bb_l:
            triggered, code = True, "bb_breach"
            reasons.append(f"⚠️ Stock ${price:.2f} broke below BB lower ${bb_l:.2f} — short put at risk, consider closing")
        if rsi > 55:
            note = f"✅ RSI {rsi:.1f} — stock recovered, put likely near max profit"
        still_valid = price >= bb_l
    elif strategy == "covered_call":
        k = _as_float(strike)
        if price > k * 1.05:
            triggered, code = True, "assignment_risk"
            reasons.append(f"⚠️ Stock ${price:.2f} running past strike ${strike} — call may be assigned")
        still_valid = True
    elif strategy == "iron_condor":
        sp, sc = _as_float(short_put), _as_float(short_call)
        if sp and sc:
            if price <= sp:
                triggered, code = True, "short_put_breach"
                reasons.append(f"⚠️ Stock ${price:.2f} broke below short put ${sp:.2f}")
            elif price >= sc:
                triggered, code = True, "short_call_breach"
                reasons.append(f"⚠️ Stock ${price:.2f} broke above short call ${sc:.2f}")
        still_valid = (sp < price < sc) if (sp and sc) else True
    return triggered, reasons, still_valid, note, code


def _atm_iv(chain, price):
    """(call IV %, put IV %) of the strikes nearest the stock price, from a chain object; (None, None) when unknown."""
    out = []
    for df in ((getattr(chain, "calls", None), getattr(chain, "puts", None)) if chain is not None else (None, None)):
        iv = None
        try:
            if df is not None and not df.empty:
                d = df[df["impliedVolatility"].fillna(0) > 0]
                if not d.empty:
                    row = d.loc[(d["strike"] - price).abs().idxmin()]
                    iv = round(float(row["impliedVolatility"]) * 100, 1)
        except Exception:
            iv = None
        out.append(iv)
    return out[0], out[1]


def scan_ticker(sym, pre_hist=None, scan_condors=False, pre_fetch=None, prefetch_fn=None):
    """
    Evaluate one ticker for all strategies.
    Returns {"_signals": {strategy: signal_dict, ...}, "_log": str} when at
    least one strategy fires, or None (no signal / insufficient data).
    pre_hist:      optional pre-fetched OHLCV DataFrame (from the bulk download)
    scan_condors:  if True, also evaluate iron condor setups (slow — on demand only)
    pre_fetch:     optional dict from _prefetch_ticker (info/calendar/chain already fetched)
    prefetch_fn:   optional callable(sym, tkr) → prefetch dict, invoked only AFTER the
                   indicator gate passes — so network calls happen just for candidates.
    """
    try:
        pf  = pre_fetch or {}
        tkr = pf.get("tkr")
        pre_hist = _clean_hist(pre_hist)
        if pre_hist is not None and len(pre_hist) >= 60:
            hist = pre_hist
            tkr  = tkr or yf.Ticker(sym)
        else:
            tkr, hist = _fetch_history(sym, LIVE_HISTORY_PERIOD)

        if hist is None or len(hist) < 60:
            return None
        hist, premarket_px = _premarket_overlay(sym, hist)    # before the open: today's pre-market print as a provisional bar

        price = float(hist["Close"].iloc[-1])
        # Minimum price filter applies to all strategies (skip sub-penny stocks)
        if price < MIN_PRICE:
            return None
        # NOTE: MAX_PRICE_SELL is enforced per-strategy below —
        #   Sell Puts & Covered Calls: price ≤ MAX_PRICE_SELL
        #   Buy Calls & Buy Puts: no upper price cap

        # ── Indicators + entry rules: the SAME engine the backtest runs ──
        # indicator_frame() computes every per-bar series; ind_at() takes the
        # last bar; live_market_ctx() is today's market context; evaluate_entry()
        # applies the rules.  Nothing below this point decides whether a strategy
        # fires — that is all inside evaluate_entry (shared with the backtest).
        closes = hist["Close"]
        prm    = live_params()
        ctx    = live_market_ctx()                  # first: get_market_trend() fills the SPY cache the frame reads
        need_garch = prm["garch_mode"] != "off" or prm.get("entry_mode") == "escape"   # the GJR-GARCH fit is the costly part — pre-gate only when a rule reads it
        fr     = indicator_frame(hist, spy_closes=_spy_cache.get("closes"), sym=sym,
                                 garch="last" if need_garch else None)
        ind    = ind_at(fr, -1)
        ev     = evaluate_entry(ind, ctx, prm, scan_condors=scan_condors)
        if not ev["any"]:
            return None
        if not need_garch:                          # gated ticker only: fill the display / signal-log fields
            try:
                gv = calc_garch_vol(closes)
            except Exception:
                gv = None
            hv = ind["hv30"]
            ind["garch_vol"]      = gv
            ind["garch_expand"]   = bool(gv is not None and hv is not None and gv > hv * GARCH_EXPAND_RATIO)
            ind["garch_contract"] = bool(gv is not None and hv is not None and gv < hv * GARCH_CONTRACT_RATIO)
            ev["flags"]["sig_garch_expand"]   = ind["garch_expand"]
            ev["flags"]["sig_garch_contract"] = ind["garch_contract"]

        # ── Unpack for the annotation / scoring code below (names kept) ──
        cur_rsi  = ind["rsi"];   cur_bb_u = ind["bb_u"]; cur_bb_m = ind["bb_m"]; cur_bb_l = ind["bb_l"]
        cur_dc_u = ind["dc_u"];  cur_dc_m = ind["dc_m"]; cur_dc_l = ind["dc_l"]
        cur_mfi  = ind["mfi"];   sig_mfi_os = ev["sig_mfi_os"]; sig_mfi_ob = ev["sig_mfi_ob"]
        dc_at_lower, dc_at_upper = ev["dc_at_lower"], ev["dc_at_upper"]
        dc_range_pct, dc_squeeze = ev["dc_range_pct"], ev["dc_squeeze"]
        oversold, overbought     = ev["oversold"], ev["overbought"]
        reversal_up              = ind["reversal_up"]
        ma50, ma200              = ind["ma50"], ind["ma200"]
        above_ma50, trend_bull, trend_bear = ev["above_ma50"], ev["trend_bull"], ev["trend_bear"]
        rs_ratio, rs_strong, rs_weak, beta = ind["rs_ratio"], ind["rs_strong"], ind["rs_weak"], ind["beta"]
        hv_rank, hv_high, hv_low = ind["hv_rank"], ev["hv_high"], ev["hv_low"]
        cur_adx, cur_pdi, cur_ndi = ind["adx"], ind["pdi"], ind["ndi"]
        strong_trend, adx_mode   = ev["strong_trend"], ev["adx_mode"]
        low_52w, high_52w        = ind["low_52w"], ind["high_52w"]
        near_52w_low, near_52w_high = ev["near_52w_low"], ev["near_52w_high"]
        exhaustion_bounce, falling_knife, melt_up = ev["exhaustion_bounce"], ev["falling_knife"], ev["melt_up"]
        macd_bull, macd_bear     = ind["macd_bull"], ind["macd_bear"]
        cur_macd, cur_macd_sig, cur_macd_h = round(ind["macd"], 4), round(ind["macd_sig"], 4), round(ind["macd_hist"], 4)
        cur_vix, cur_vix3m, regime, vix_ts = ctx["vix"], ctx["vix3m"], ctx["regime"], ctx["vix_ts"]
        vix_backw, vix_contango  = ev["vix_backw"], ev["vix_contango"]
        pre_fed, post_fed        = ctx["pre_fed"], ctx["post_fed"]
        yield_inv, yield_steep   = ctx["yield_inverted"], ctx["yield_steep"]
        _rsi_os_thr, _rsi_ob_thr = ev["rsi_os_thr"], ev["rsi_ob_thr"]
        vol_confirmed, cur_vol, avg_vol = ind["vol_confirmed"], ind["cur_vol"], ind["avg_vol"]
        bull_signals, bear_signals = ev["bull_signals"], ev["bear_signals"]
        market_trend             = ctx["market_trend"]
        green_candle, red_candle = ev["green_candle"], ev["red_candle"]
        has_sell_put, has_buy_call, has_buy_put = ev["has_sell_put"], ev["has_buy_call"], ev["has_buy_put"]
        has_covered_call, has_iron_condor       = ev["has_covered_call"], ev["has_iron_condor"]
        garch_vol = ind["garch_vol"]
        # Series for the chart block
        bb_u = pd.Series(fr["bb_u"], index=closes.index); bb_m = pd.Series(fr["bb_m"], index=closes.index)
        bb_l = pd.Series(fr["bb_l"], index=closes.index); rsi  = pd.Series(fr["rsi"],  index=closes.index)
        macd_line = pd.Series(fr["macd"], index=closes.index); macd_sig_line = pd.Series(fr["macd_sig"], index=closes.index)
        macd_hist_ser = pd.Series(fr["macd_hist"], index=closes.index)

        # ── Network data (only fetched when a signal is possible) ─
        # In the main scan nothing was prefetched: the gate above ran on bulk
        # OHLCV alone, so info/calendar/option chain are fetched here, only for
        # the small set of tickers that can actually produce a signal.
        if not pf and prefetch_fn is not None:
            pf = prefetch_fn(sym, tkr) or {}
        try:
            info = pf.get("info")
            if info is None:
                info = _yf_call(lambda: tkr.info) or {}
            name   = info.get("shortName") or sym
            sector = info.get("sector") or "N/A"
        except Exception:
            info, name, sector = {}, sym, "N/A"

        # ── Short interest ────────────────────────────────────
        short_pct, short_ratio = get_short_interest(tkr, info=info)
        # Squeeze setup: high short interest + oversold = potential short squeeze
        squeeze_setup = (short_pct is not None and short_pct >= 10 and oversold)

        # ── Put/Call ratio for nearest expiry ─────────────────
        pf_chain = pf.get("chain")
        pf_exp   = pf.get("exp")
        # Real implied vol at the money (Alpaca / Yahoo chain): skew and the IV-vs-forecast gap the
        # backtest can't see — recorded on the card and in the signal log for later analysis.
        iv_call_atm, iv_put_atm = _atm_iv(pf_chain, price)
        iv_skew     = round(iv_put_atm / iv_call_atm, 3) if (iv_call_atm and iv_put_atm) else None
        iv_vs_garch = (round(((iv_call_atm + iv_put_atm) / 2) / (ind["garch_vol"] * 100), 3)
                       if (iv_call_atm and iv_put_atm and ind.get("garch_vol")) else None)
        pc_ratio = calc_put_call_ratio(tkr, chain=pf_chain, exp=pf_exp)
        pc_bearish = pc_ratio is not None and pc_ratio > 1.2
        pc_bullish = pc_ratio is not None and pc_ratio < 0.7
        # The P/C ratio was listed in the confluence sums above but was always
        # False there (not yet fetched); count it now so it reaches the scores.
        bull_signals += int(pc_bullish)
        bear_signals += int(pc_bearish)

        # ── Next earnings date ────────────────────────────────
        # "Soon" = the company reports before the option we'd hold expires.
        earnings_dte  = get_next_earnings_dte(tkr, cal=pf.get("calendar"))
        earnings_soon = (earnings_dte is not None and
                         0 <= earnings_dte <= (pf.get("dte") or MAX_DTE))

        # ── Chart data (last 60 days) ─────────────────────────
        n = 60
        chart = {
            "dates":    [d.strftime("%m/%d") for d in hist.index[-n:]],
            "price":    safe_list(closes),
            "bb_upper": safe_list(bb_u),
            "bb_mid":   safe_list(bb_m),
            "bb_lower": safe_list(bb_l),
            "rsi":      safe_list(rsi),
            "macd":     safe_list(macd_line),
            "macd_sig": safe_list(macd_sig_line),
            "macd_hist":safe_list(macd_hist_ser),
        }

        base = {
            "ticker": sym, "name": name, "sector": sector,
            "price": round(price, 3),
            "bar_date": str(fr["index"][-1].date()),   # the bar whose close is `price` (the signal log resolves from it)
            "premarket_price": premarket_px,            # set when `price` is today's pre-market print (Alpaca feed, before the open)
            "iv_call_atm": iv_call_atm, "iv_put_atm": iv_put_atm, "iv_skew": iv_skew, "iv_vs_garch": iv_vs_garch,
            "rsi":   round(cur_rsi, 1),
            "bb_upper": round(cur_bb_u, 3),
            "bb_mid":   round(cur_bb_m, 3),
            "bb_lower": round(cur_bb_l, 3),
            "dc_upper":    round(cur_dc_u, 3),
            "dc_mid":      round(cur_dc_m, 3),
            "dc_lower":    round(cur_dc_l, 3),
            "dc_at_lower": dc_at_lower,
            "dc_at_upper": dc_at_upper,
            "dc_squeeze":  dc_squeeze,
            "dc_range_pct": round(dc_range_pct, 1),
            # MACD
            "macd_bull":    macd_bull,
            "macd_bear":    macd_bear,
            "macd_val":     cur_macd,
            "macd_sig_val": cur_macd_sig,
            "macd_hist_val":cur_macd_h,
            # Put/Call ratio
            "pc_ratio":   pc_ratio,
            "pc_bearish": pc_bearish,
            "pc_bullish": pc_bullish,
            # Earnings
            "earnings_dte":  earnings_dte,
            "earnings_soon": earnings_soon,
            # Trend & strength
            "ma200":         round(ma200, 3) if ma200 else None,
            "trend_bull":    trend_bull,
            "trend_bear":    trend_bear,
            "adx":           cur_adx,
            "adx_pdi":       cur_pdi,
            "adx_ndi":       cur_ndi,
            "strong_trend":  strong_trend,
            "falling_knife":    falling_knife,
            "exhaustion_bounce":exhaustion_bounce,
            "melt_up":          melt_up,
            "bull_signals":  bull_signals,
            "bear_signals":  bear_signals,
            # VIX regime + dynamic thresholds
            "vix":           cur_vix,
            "vix_regime":    regime,
            "rsi_os_thr":    _rsi_os_thr,
            "rsi_ob_thr":    _rsi_ob_thr,
            "cal_active":    bool(_cal_weights),
            # Broad market trend (SPY)
            "market_trend":  market_trend,
            # MA50 trend filter
            "ma50":          round(ma50, 3) if ma50 else None,
            "above_ma50":    above_ma50,
            # Relative strength vs SPY
            "rs_ratio":      rs_ratio,
            "beta":          beta,
            "rs_strong":     rs_strong,
            "rs_weak":       rs_weak,
            # HV Rank (options cheapness indicator)
            "hv_rank":       round(hv_rank, 1),
            "hv_high":       hv_high,
            "hv_low":        hv_low,
            # ADX mode
            "adx_mode":      adx_mode,
            # 52-week proximity
            "low_52w":       round(low_52w, 3),
            "high_52w":      round(high_52w, 3),
            "near_52w_low":  near_52w_low,
            "near_52w_high": near_52w_high,
            # VIX term structure
            "vix3m":         cur_vix3m,
            "vix_ts":        vix_ts,
            "vix_backw":     vix_backw,
            "vix_contango":  vix_contango,
            # Fed meeting proximity + yield curve
            "days_to_next_fed":    ctx["days_to_next_fed"],
            "next_fed_date":       ctx["next_fed_date"],
            "pre_fed_window":      pre_fed,
            "post_fed_window":     post_fed,
            "irx":                 ctx["irx"],
            "fvx":                 ctx["fvx"],
            "yield_spread":        ctx["yield_spread"],
            "yield_inverted":      yield_inv,
            "yield_steep":         yield_steep,
            # GJR-GARCH forecast vs trailing realised vol (same arrays the backtest uses)
            "garch_vol":      round(garch_vol * 100, 1) if garch_vol is not None else None,   # as % e.g. 32.4
            "hv30":           round(ind["hv30"] * 100, 1) if ind["hv30"] is not None else None,
            "garch_expand":   ind["garch_expand"],
            "garch_contract": ind["garch_contract"],
            "garch_mode":     ev["garch_mode"],
            # Every individual flag the rules looked at (signal log + calibration read these)
            "flags":          ev["flags"],
            # MFI
            "mfi":           round(cur_mfi, 1),
            "sig_mfi_os":    sig_mfi_os,
            "sig_mfi_ob":    sig_mfi_ob,
            # Volume confirmation
            "vol_confirmed": vol_confirmed,
            "green_candle":  green_candle,
            "red_candle":    red_candle,
            "cur_vol":       int(cur_vol) if cur_vol else None,
            "avg_vol":       int(avg_vol) if avg_vol else None,
            # Short interest
            "short_pct":     short_pct,
            "short_ratio":   short_ratio,
            "squeeze_setup": squeeze_setup,
            "chart": chart,
        }

        results = {}

        # The has_* flags below are the same booleans the early-exit gate used,
        # so the macro-regime and gap filters can't be skipped by a block that
        # re-states only part of the condition.
        # ── 1. SELL PUT ───────────────────────────────────────
        if has_sell_put:
            opt = pick_option(tkr, price, "put", otm_factor=float(prm.get("otm_factor_put", 1.0) or 1.0),
                              chain=pf_chain, exp=pf_exp, dte=pf.get("dte"))
            if opt and opt["bid"] >= 0.01:
                iv_rank = calc_iv_rank(closes, opt["iv"])
                iv_high  = iv_rank is not None and iv_rank >= IV_RANK_THRESHOLD
                prem_high = opt["premium_pct"] >= PREMIUM_PCT_MIN
                if iv_high or prem_high:
                    score = _score_sell_put(cur_rsi, price, cur_bb_l, iv_rank, opt["premium_pct"],
                                            hv_high=hv_high, vix_contango=vix_contango,
                                            adx_mode=adx_mode, pre_fed=pre_fed, post_fed=post_fed)
                    mc = run_monte_carlo(price, opt["strike"], opt["iv"], opt["dte"], "sell_put", opt["mid"])
                    _bs = black_scholes(price, opt["strike"], opt["dte"]/365.0, RISK_FREE_RATE, opt["iv"]/100, "put")
                    _bs_mp = bs_mispricing(_bs["fair_value"] if _bs else None, opt["mid"])
                    results["sell_put"] = {**base, "option": opt,
                                           "iv_rank": iv_rank, "iv_high": iv_high,
                                           "prem_high": prem_high,
                                           "rsi_oversold": cur_rsi < RSI_OVERSOLD,
                                           "bb_oversold": price < cur_bb_l,
                                           "score": score,
                                           "strategy": "sell_put",
                                           "strategy_label": "Sell Put",
                                           "mc": mc, "bs": _bs, "bs_misprice": _bs_mp}

        # ── 2. BUY CALL ───────────────────────────────────────
        if has_buy_call:
            opt = pick_option(tkr, price, "call", otm_factor=float(prm.get("otm_factor_call", 1.0) or 1.0),
                              chain=pf_chain, exp=pf_exp, dte=pf.get("dte"))
            if opt and opt["ask"] >= 0.01:
                score = _score_buy_call(cur_rsi, price, cur_bb_l, reversal_up,
                                        above_ma50=above_ma50, hv_low=hv_low,
                                        rs_strong=rs_strong, squeeze=squeeze_setup,
                                        near_52w_low=near_52w_low, vix_backw=vix_backw,
                                        adx_mode=adx_mode, confluence=bull_signals,
                                        pre_fed=pre_fed, yield_inverted=yield_inv,
                                        yield_steep=yield_steep)
                mc = run_monte_carlo(price, opt["strike"], opt["iv"], opt["dte"], "buy_call", opt["mid"])
                _bs = black_scholes(price, opt["strike"], opt["dte"]/365.0, RISK_FREE_RATE, opt["iv"]/100, "call")
                _bs_mp = bs_mispricing(_bs["fair_value"] if _bs else None, opt["mid"])
                results["buy_call"] = {**base, "option": opt,
                                        "reversal": reversal_up,
                                        "rsi_oversold": cur_rsi < RSI_OVERSOLD,
                                        "bb_oversold": price < cur_bb_l,
                                        "score": score,
                                        "strategy": "buy_call",
                                        "strategy_label": "Buy Call",
                                        "mc": mc, "bs": _bs, "bs_misprice": _bs_mp}

        # ── 3. BUY PUT ────────────────────────────────────────
        if has_buy_put:
            opt = pick_option(tkr, price, "put", otm_factor=float(prm.get("otm_factor_put", 1.0) or 1.0),
                              chain=pf_chain, exp=pf_exp, dte=pf.get("dte"))
            if opt and opt["ask"] >= 0.01:
                score = _score_buy_put(cur_rsi, price, cur_bb_u,
                                       rs_weak=rs_weak, earnings_soon=earnings_soon,
                                       near_52w_high=near_52w_high, adx_mode=adx_mode,
                                       confluence=bear_signals, pre_fed=pre_fed,
                                       yield_inverted=yield_inv)
                mc = run_monte_carlo(price, opt["strike"], opt["iv"], opt["dte"], "buy_put", opt["mid"])
                _bs = black_scholes(price, opt["strike"], opt["dte"]/365.0, RISK_FREE_RATE, opt["iv"]/100, "put")
                _bs_mp = bs_mispricing(_bs["fair_value"] if _bs else None, opt["mid"])
                results["buy_put"] = {**base, "option": opt,
                                       "rsi_overbought": cur_rsi > RSI_OVERBOUGHT,
                                       "bb_overbought": price > cur_bb_u,
                                       "score": score,
                                       "strategy": "buy_put",
                                       "strategy_label": "Buy Put",
                                       "mc": mc, "bs": _bs, "bs_misprice": _bs_mp}

        # ── 4. COVERED CALL ───────────────────────────────────
        # Only for cheaper stocks — covered calls require owning 100 shares
        if has_covered_call:
            opt_cc = pick_option(tkr, price, "call", otm_factor=1.05,
                                 chain=pf_chain, exp=pf_exp, dte=pf.get("dte"))
        else:
            opt_cc = None
        if opt_cc:
            iv_rank_cc = calc_iv_rank(closes, opt_cc["iv"])
            iv_high_cc = iv_rank_cc is not None and iv_rank_cc >= IV_RANK_THRESHOLD
            prem_high_cc = opt_cc["premium_pct"] >= PREMIUM_PCT_MIN
            if iv_high_cc or prem_high_cc:
                score = _score_covered_call(iv_rank_cc, opt_cc["premium_pct"])
                mc = run_monte_carlo(price, opt_cc["strike"], opt_cc["iv"], opt_cc["dte"], "covered_call", opt_cc["mid"])
                _bs_cc = black_scholes(price, opt_cc["strike"], opt_cc["dte"]/365.0, RISK_FREE_RATE, opt_cc["iv"]/100, "call")
                _bs_mp_cc = bs_mispricing(_bs_cc["fair_value"] if _bs_cc else None, opt_cc["mid"])
                results["covered_call"] = {**base, "option": opt_cc,
                                            "iv_rank": iv_rank_cc, "iv_high": iv_high_cc,
                                            "prem_high": prem_high_cc,
                                            "score": score,
                                            "strategy": "covered_call",
                                            "strategy_label": "Covered Call",
                                            "note": "Requires 100 shares of " + sym,
                                            "mc": mc, "bs": _bs_cc, "bs_misprice": _bs_mp_cc}

        # ── 5. IRON CONDOR ────────────────────────────────────
        # Signal: stock is range-bound, IV is elevated.  Only on the dedicated
        # condor scan (has_iron_condor includes scan_condors) — this block used
        # to run on every main-scan ticker and cost 5 extra chain downloads each.
        iv_rank_ic = None
        if has_iron_condor:
            _ic = dict(chain=pf_chain, exp=pf_exp, dte=pf.get("dte"))   # one chain, four legs
            opt_sp = pick_option(tkr, price, "put",  otm_factor=0.95, **_ic)  # short put
            opt_lp = pick_option(tkr, price, "put",  otm_factor=0.90, **_ic)  # long put
            opt_sc = pick_option(tkr, price, "call", otm_factor=1.05, **_ic)  # short call
            opt_lc = pick_option(tkr, price, "call", otm_factor=1.10, **_ic)  # long call

            if opt_sp and opt_lp and opt_sc and opt_lc:
                net_credit = (opt_sp["bid"] + opt_sc["bid"]) - (opt_lp["ask"] + opt_lc["ask"])
                put_width  = round(opt_sp["strike"] - opt_lp["strike"], 2)
                call_width = round(opt_lc["strike"] - opt_sc["strike"], 2)
                avg_iv     = (opt_sp["iv"] + opt_sc["iv"]) / 2
                iv_rank_ic = calc_iv_rank(closes, avg_iv)
                iv_high_ic = iv_rank_ic is not None and iv_rank_ic >= IV_RANK_THRESHOLD

                if net_credit >= 0.10:
                    score_ic = _score_iron_condor(cur_rsi, iv_rank_ic, net_credit, price)
                    mc_ic    = run_monte_carlo(
                        price, opt_sp["strike"], avg_iv, opt_sp["dte"],
                        "iron_condor", net_credit,
                        extra={
                            "short_call_strike": opt_sc["strike"],
                            "long_put_strike":   opt_lp["strike"],
                            "long_call_strike":  opt_lc["strike"],
                            "put_width":  put_width,
                            "call_width": call_width,
                        }
                    )
                    # summary option dict for queue/history compatibility
                    opt_summary = {
                        "type":          "condor",
                        "expiration":    opt_sp["expiration"],
                        "dte":           opt_sp["dte"],
                        "strike":        opt_sp["strike"],
                        "net_credit":    round(net_credit, 3),
                        "bid":           round(net_credit, 3),
                        "ask":           round(net_credit, 3),
                        "mid":           round(net_credit, 3),
                        "iv":            round(avg_iv, 1),
                        "volume":        min(opt_sp["volume"], opt_sc["volume"]),
                        "open_interest": min(opt_sp["open_interest"], opt_sc["open_interest"]),
                        "premium_pct":   round(net_credit / price * 100, 2),
                    }
                    results["iron_condor"] = {
                        **base,
                        "option":   opt_summary,
                        "legs": {
                            "short_put":  opt_sp,
                            "long_put":   opt_lp,
                            "short_call": opt_sc,
                            "long_call":  opt_lc,
                        },
                        "net_credit":    round(net_credit, 3),
                        "put_width":     put_width,
                        "call_width":    call_width,
                        "iv_rank":       iv_rank_ic,
                        "iv_high":       iv_high_ic,
                        "rsi_neutral":   True,
                        "score":         score_ic,
                        "strategy":      "iron_condor",
                        "strategy_label":"Iron Condor",
                        "mc":            mc_ic,
                    }

        if results:
            found = ", ".join(results.keys())
            return {"_signals": results, "_log": f"  [Scan] {sym:6s} ✓  {found}"}
        return None

    except Exception as e:
        return {"_signals": {}, "_log": f"  [Scan] {sym:6s} ✗  error: {e}"}


# ── Scoring helpers ───────────────────────────────────────────

def _score_sell_put(rsi, price, bb_l, iv_rank, prem_pct,
                    hv_high=False, vix_contango=False, adx_mode="normal",
                    pre_fed=False, post_fed=False):
    w     = _cal_weights.get("sell_put", {})
    rsi_w = w.get("sig_rsi_os",    1.0)
    bb_w  = w.get("sig_bb_low",    1.0)
    hv_w  = w.get("sig_hv_high",   1.0)
    s = 0
    if rsi < RSI_OVERSOLD: s += int(max(0, 35 + (RSI_OVERSOLD - rsi)) * rsi_w)
    if price < bb_l:        s += int(25 * bb_w)
    if iv_rank and iv_rank >= IV_RANK_THRESHOLD:
        s += min(int(iv_rank - IV_RANK_THRESHOLD), 25)
    if prem_pct >= PREMIUM_PCT_MIN:
        s += min(int((prem_pct - PREMIUM_PCT_MIN) * 3), 15)
    # High HV Rank → options expensive, more premium to collect
    if hv_high:      s += int(12 * hv_w)
    # VIX contango → calm market, sell premium setups more reliable
    if vix_contango: s += 10
    # Ranging market → mean-reversion (stock stays near support) more reliable
    if adx_mode == "ranging": s += 8
    # Pre-Fed window: IV is inflated → more premium to collect; favour selling
    if pre_fed:  s += 10
    # Post-Fed window: IV has already crushed → premium is thin; penalise selling
    if post_fed: s -= 12
    return min(max(round(s), 0), 100)

def _score_buy_call(rsi, price, bb_l, reversal,
                    above_ma50=False, hv_low=False, rs_strong=False, squeeze=False,
                    near_52w_low=False, vix_backw=False, adx_mode="normal",
                    confluence=0, pre_fed=False, yield_inverted=False,
                    yield_steep=False):
    w     = _cal_weights.get("buy_call", {})
    rsi_w = w.get("sig_rsi_os",     1.0)
    bb_w  = w.get("sig_bb_low",     1.0)
    rev_w = w.get("sig_reversal",   1.0)
    ma_w  = w.get("sig_above_ma50", 1.0)
    hv_w  = w.get("sig_hv_low",     1.0)
    rs_w  = w.get("sig_rs_strong",  1.0)
    s = 0
    if rsi < 30:     s += int(40 * rsi_w)
    elif rsi < 35:   s += int(25 * rsi_w)
    if price < bb_l: s += int(25 * bb_w)
    if reversal:     s += int(20 * rev_w)
    # Trend alignment: above 50-day MA → bounce more likely to stick
    if above_ma50:     s += int(15 * ma_w)
    # Cheap options: low HV rank → better risk/reward on long options
    if hv_low:         s += int(10 * hv_w)
    # Relative strength vs SPY: stock holding up in sell-off = leadership
    if rs_strong:      s += int(10 * rs_w)
    # Short squeeze potential
    if squeeze:        s += 10
    # Near 52-week low: hitting annual floor = strongest support level
    if near_52w_low:   s += 15
    # VIX backwardation: acute panic → reversal probability higher
    if vix_backw:      s += 12
    # ADX mode: ranging market → mean-reversion (RSI/BB) more reliable
    if adx_mode == "ranging": s += 10
    # ADX mode: strong trend → counter-trend buy calls are dangerous
    if adx_mode == "trending" and not above_ma50: s -= 15
    # Multi-signal confluence bonus: 4+ signals firing = high conviction
    if confluence >= 4: s += 15
    elif confluence >= 3: s += 8
    # Pre-Fed window: IV inflated, will crush after announcement → penalise long options
    if pre_fed:        s -= 12
    # Yield curve inverted (3m > 5y): recession / rate-cut expectations → macro headwind
    if yield_inverted: s -= 8
    # Yield curve steep (5y - 3m > 1.5%): growth / normalisation expectations → tailwind
    if yield_steep:    s += 6
    return min(max(round(s), 0), 100)

def _score_buy_put(rsi, price, bb_u, rs_weak=False, earnings_soon=False,
                   near_52w_high=False, adx_mode="normal", confluence=0,
                   pre_fed=False, yield_inverted=False):
    w     = _cal_weights.get("buy_put", {})
    rsi_w = w.get("sig_rsi_ob",    1.0)
    bb_w  = w.get("sig_bb_high",   1.0)
    rs_w  = w.get("sig_rs_weak",   1.0)      # was reading the bull-side key by mistake
    s = 0
    if rsi > 75:     s += int(40 * rsi_w)
    elif rsi > 65:   s += int(25 * rsi_w)
    if price > bb_u: s += int(30 * bb_w)
    # Relative weakness vs SPY → downside more likely
    if rs_weak:        s += int(10 * rs_w)
    # Earnings approaching → directional uncertainty
    if earnings_soon:  s += 8
    # Near 52-week high: hitting annual ceiling = strongest resistance
    if near_52w_high:  s += 15
    # ADX mode: ranging market → overbought mean-reversion more reliable
    if adx_mode == "ranging": s += 10
    # Multi-signal confluence bonus
    if confluence >= 4: s += 15
    elif confluence >= 3: s += 8
    # Pre-Fed window: IV inflated, will crush after announcement → penalise long options
    if pre_fed:        s -= 12
    # Yield curve inverted: recession / rate-cut pricing → mild macro tailwind for puts
    if yield_inverted: s += 5
    return min(max(round(s), 0), 100)

def _score_iron_condor(rsi, iv_rank, net_credit, price):
    s = 0
    # Reward neutral RSI (45–55 is ideal)
    dist_from_50 = abs(rsi - 50)
    if dist_from_50 <= 5:   s += 35
    elif dist_from_50 <= 10: s += 20
    elif dist_from_50 <= 15: s += 10
    # Reward high IV (more premium collected)
    if iv_rank and iv_rank >= IV_RANK_THRESHOLD: s += min(int(iv_rank - IV_RANK_THRESHOLD), 35)
    # Reward attractive credit
    credit_pct = net_credit / price * 100
    if credit_pct >= 2.0: s += 20
    elif credit_pct >= 1.0: s += 10
    return min(max(round(s), 0), 100)

def _score_covered_call(iv_rank, prem_pct):
    s = 0
    if iv_rank and iv_rank >= IV_RANK_THRESHOLD: s += min(int(iv_rank - IV_RANK_THRESHOLD), 40)
    if prem_pct >= PREMIUM_PCT_MIN: s += min(int((prem_pct - PREMIUM_PCT_MIN) * 5), 40)
    return min(max(round(s), 0), 100)


# ══════════════════════════════════════════════════════════════
#  MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════

def run_monte_carlo(price, strike, iv_pct, dte, strategy, premium, n_sims=None, extra=None):
    """
    Simulate n_sims price paths using Geometric Brownian Motion and
    compute trade statistics for each strategy.

    Returns dict with:
      pop       – probability of profit (%)
      ev        – expected value per contract ($)
      breakeven – break-even stock price at expiry
      max_profit – max profit per contract (None = unlimited)
      max_loss   – max loss per contract ($, negative)
      dist       – histogram buckets for the P&L distribution chart
    """
    if n_sims is None:
        n_sims = N_MC_SIMS

    T     = max(dte, 1) / 365.0
    sigma = iv_pct / 100.0
    r     = RISK_FREE_RATE

    rng   = np.random.default_rng(seed=42)
    Z     = rng.standard_normal(n_sims)
    S_T   = price * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # ── P&L per share at expiry ───────────────────────────────
    if strategy == "sell_put":
        # Collect premium; pay out if assigned
        pnl       = np.where(S_T >= strike, premium, premium - (strike - S_T))
        max_profit = premium
        max_loss   = -(strike - premium)   # stock goes to 0
        breakeven  = strike - premium

    elif strategy == "buy_call":
        pnl        = np.where(S_T > strike, S_T - strike - premium, -premium)
        max_profit = None                   # unlimited upside
        max_loss   = -premium
        breakeven  = strike + premium

    elif strategy == "buy_put":
        pnl        = np.where(S_T < strike, strike - S_T - premium, -premium)
        max_profit = strike - premium       # if stock goes to 0
        max_loss   = -premium
        breakeven  = strike - premium

    elif strategy == "covered_call":
        # Stock bought at current price + call sold at 5% OTM strike
        stock_pnl  = S_T - price
        call_pnl   = np.where(S_T >= strike,
                               premium - (S_T - strike),
                               premium)
        pnl        = stock_pnl + call_pnl
        max_profit = (strike - price) + premium
        max_loss   = -(price - premium)    # stock goes to 0
        breakeven  = price - premium

    elif strategy == "iron_condor":
        # strike      = short_put_strike
        # extra keys  = short_call_strike, long_put_strike, long_call_strike
        ex            = extra or {}
        short_put_k   = strike
        long_put_k    = float(ex.get("long_put_strike",   strike * 0.90))
        short_call_k  = float(ex.get("short_call_strike", strike * 1.10))
        long_call_k   = float(ex.get("long_call_strike",  strike * 1.15))
        put_width     = float(ex.get("put_width",  short_put_k  - long_put_k))
        call_width    = float(ex.get("call_width", long_call_k  - short_call_k))

        # P&L per share: premium collected minus losses on either side
        put_loss  = np.where(S_T >= short_put_k,  0,
                    np.where(S_T <= long_put_k,   -put_width,
                             -(short_put_k - S_T)))
        call_loss = np.where(S_T <= short_call_k, 0,
                    np.where(S_T >= long_call_k,  -call_width,
                             -(S_T - short_call_k)))
        pnl       = premium + put_loss + call_loss

        max_profit = premium
        max_loss   = -(max(put_width, call_width) - premium)
        breakeven  = (short_put_k - premium + short_call_k + premium) / 2  # midpoint

    else:
        return None

    # ── Per-contract P&L (×100 shares) ───────────────────────
    pnl_contract = pnl * 100

    pop = float(np.mean(pnl_contract > 0) * 100)
    ev  = float(np.mean(pnl_contract))

    # ── Distribution histogram (30 bins) ─────────────────────
    counts, edges = np.histogram(pnl_contract, bins=30)
    dist = {
        "counts": counts.tolist(),
        "edges":  [round(float(e), 2) for e in edges.tolist()],
    }

    return {
        "pop":        round(pop, 1),
        "ev":         round(ev, 2),
        "breakeven":  round(float(breakeven), 3),
        "max_profit": round(float(max_profit) * 100, 2) if max_profit is not None else None,
        "max_loss":   round(float(max_loss) * 100, 2),
        "n_sims":     n_sims,
        "dist":       dist,
    }


# ══════════════════════════════════════════════════════════════
#  TRACKED POSITIONS  (load / save / sell-signal logic)
# ══════════════════════════════════════════════════════════════

def _option_mid(row_series):
    """
    Compute a reliable mid price from an option chain row.

    Priority:
      1. (bid + ask) / 2           when both quotes look fresh (bid >= ask * 0.5)
      2. ask * 0.95                when bid is stale/zero but ask is present
                                   (ask is always the current market offer — reliable)
      3. lastPrice                 only when no ask is available (last resort —
                                   lastPrice can be days old for illiquid options)

    Never use lastPrice when ask > 0: ask is current, lastPrice can be stale.
    Also returns data_quality flag for the tracker UI.
    """
    bid        = _as_float(row_series.get("bid"))
    ask        = _as_float(row_series.get("ask"))
    last_price = _as_float(row_series.get("lastPrice"))

    if ask > 0 and bid > 0 and bid >= ask * 0.5:
        # Both quotes fresh — true mid
        return (bid + ask) / 2
    elif ask > 0 and bid > 0:
        # Bid suspiciously low (stale) — ask-weighted estimate
        return ask * 0.9 + bid * 0.1
    elif ask > 0:
        # No valid bid — ask is current; lastPrice may be stale so ignore it
        return ask * 0.95
    elif last_price > 0:
        # No live quotes at all — last resort
        return last_price
    return 0.0


def _option_price_detail(row_series):
    """Return (mid, bid, ask, last_price, quality_label) for tracker display."""
    bid        = _as_float(row_series.get("bid"))
    ask        = _as_float(row_series.get("ask"))
    last_price = _as_float(row_series.get("lastPrice"))
    mid        = _option_mid(row_series)

    if ask > 0 and bid > 0 and bid >= ask * 0.5:
        quality = "live"          # fresh two-sided quote
    elif ask > 0 and bid > 0:
        quality = "stale_bid"     # bid looks old, ask is current
    elif ask > 0:
        quality = "ask_only"      # no bid yet (common at market open)
    elif last_price > 0:
        quality = "last_only"     # no live quote — using last traded price
    else:
        quality = "unavailable"

    return mid, bid, ask, last_price, quality


# ── Position P&L helpers (shared by tracker, refresh loop and API) ─────
# Premium SELLERS profit when the option gets cheaper.  Every P&L site used
# to compute (current - entry) regardless of strategy, so a short put that
# decayed from 1.00 → 0.85 (a win) showed -15% and fired the stop loss.
SHORT_PREMIUM_STRATS = {"sell_put", "covered_call", "iron_condor"}
PRICE_EXIT_MARKERS   = ("🎯", "🛑", "🧯", "📉 Trailing", "📉 Stock trail", "⏳", "⌛ Max hold")   # reasons produced by _price_exit_check


def _position_pnl(strategy, entry_price, current_price, contracts=1):
    """Return (pct_change, dollar_change) for a position, or (None, None)."""
    entry = _as_float(entry_price)
    cur   = _as_float(current_price)
    if entry <= 0 or cur <= 0:
        return None, None
    sign = -1.0 if strategy in SHORT_PREMIUM_STRATS else 1.0
    n    = max(1, _as_int(contracts, 1))
    pct  = sign * (cur - entry) / entry * 100
    usd  = sign * (cur - entry) * 100 * n
    return round(pct, 1), round(usd, 2)


def _trading_days_since(iso_ts):
    """Whole NYSE trading days (Mon-Fri) between an ISO timestamp and today."""
    try:
        start = datetime.fromisoformat(iso_ts).date()
        return int(np.busday_count(start, date.today()))
    except Exception:
        return 0


def _fetch_option_quote(sym, exp, strike, otype, tkr=None, chain=None):
    """
    Live quote for one contract → (mid, bid, ask, quality, chain).
    quality is one of live / stale_bid / ask_only / last_only / unavailable.
    Iron condors ("condor") have no single contract to quote — the net
    credit can only be entered manually, so they come back unavailable.
    Rows with IV > 300% are rejected as misaligned yfinance data.
    """
    if not (sym and exp and strike) or otype not in ("call", "put"):
        return 0.0, 0.0, 0.0, "unavailable", chain
    try:
        if chain is None:
            tkr   = tkr or yf.Ticker(sym)
            chain = _option_chain(tkr, exp, sym)
        opts = chain.calls if otype == "call" else chain.puts
        row  = opts[(opts["strike"] - float(strike)).abs() < 0.01]
        if row.empty:
            return 0.0, 0.0, 0.0, "unavailable", chain
        r = row.iloc[0]
        if _as_float(r.get("impliedVolatility")) > 3.0:
            print(f"[PriceCheck] {sym} {exp} {strike}{otype[0].upper()} rejected (IV > 300%, misaligned row)")
            return 0.0, 0.0, 0.0, "unavailable", chain
        mid, bid, ask, _, quality = _option_price_detail(r)
        return mid, bid, ask, quality, chain
    except Exception as e:
        print(f"[PriceCheck] {sym} {exp} {strike} quote failed: {e}")
        return 0.0, 0.0, 0.0, "unavailable", chain


def _price_exit_check(pct_change, prev_peak, tracked_at, stock_move_pct=None, stock_giveback_pct=None):
    """
    Live wrapper around the shared price-exit rules (_price_exit_core, defined
    with the signal engine): converts the position's tracked_at timestamp into
    trading days held and applies the live exit parameters.
    Returns (triggered, reasons, peak_pct).
    """
    days_held = _trading_days_since(tracked_at) if tracked_at else None
    triggered, reasons, peak, _ = _price_exit_core(pct_change, prev_peak, days_held, ex=live_exit_params(),
                                                   stock_move_pct=stock_move_pct, stock_giveback_pct=stock_giveback_pct)
    return triggered, reasons, peak


def _apply_price_update(position, sig, mid, bid, ask, quality):
    """
    Merge a fresh option quote into a position's last_signal dict IN PLACE:
    updates quote fields, P&L, peak and the price-based exit reasons while
    leaving the RSI/BB reasons from the last full check untouched.
    """
    sig["current_option_mid"] = round(mid, 3) if mid > 0 else None
    sig["current_bid"]        = round(bid, 3) if bid > 0 else None
    sig["current_ask"]        = round(ask, 3) if ask > 0 else None
    sig["price_quality"]      = quality
    entry = position.get("entry_price") or (position.get("option") or {}).get("mid")
    pct, usd = _position_pnl(position.get("strategy", ""), entry, mid, position.get("contracts", 1))
    if pct is None:
        return sig
    sig["pct_change"]    = pct
    sig["dollar_change"] = usd
    strat_ = position.get("strategy", "")
    stock_move = _adverse_stock_move(strat_, position.get("price"), sig.get("current_price"))
    sig["stock_best"] = _stock_best(strat_, sig.get("stock_best"), sig.get("current_price"))
    triggered, reasons, peak = _price_exit_check(pct, sig.get("peak_pct_change"), position.get("tracked_at"), stock_move,
                                                 _stock_giveback(strat_, sig.get("stock_best"), sig.get("current_price")))
    sig["peak_pct_change"] = peak
    kept = [r for r in (sig.get("reasons") or []) if not r.startswith(PRICE_EXIT_MARKERS)]
    sig["reasons"]   = kept + reasons
    sig["triggered"] = bool(kept) and bool(sig.get("triggered")) or triggered
    return sig


def get_portfolio_greeks():
    """
    Compute net portfolio delta across all tracked positions.
    Returns dict: {net_delta, by_position, n_positions, warning}.
    Delta is in 'shares-equivalent' terms (1 contract = 100 shares).

    Positive net delta = portfolio gains when SPY rises (long market exposure).
    Negative net delta = portfolio gains when SPY falls (short market exposure).

    On a 'bad day' (sharp SPY drop), a portfolio with high positive net delta
    loses big. Tracking this helps spot concentration before it bites.
    """
    with track_lock:
        positions = load_tracked()
    if not positions:
        return {"net_delta": 0.0, "by_position": [], "n_positions": 0, "warning": None}

    today        = date.today()
    by_pos       = []
    net_delta    = 0.0
    for p in positions:
        try:
            opt        = p.get("option") or {}
            strat      = p.get("strategy", "")
            contracts  = max(1, _as_int(p.get("contracts"), 1))
            exp_str    = opt.get("expiration") or ""
            iv         = _as_float(opt.get("iv")) / 100.0  # iv stored as % (e.g., 33.1 -> 0.331)
            if iv < 0.05:
                iv = 0.30  # fallback for missing IV
            cur_price  = _as_float((p.get("last_signal") or {}).get("current_price")) or _as_float(p.get("price"))
            if not (exp_str and cur_price > 0):
                continue
            exp_dt = datetime.strptime(exp_str[:10], "%Y-%m-%d").date()
            # Expiry is at the close of the expiration day → count it as a full day
            T      = max((exp_dt - today).days + 1, 0.5) / 365.0

            def _delta(strike, otype):
                bs = black_scholes(cur_price, strike, T, RISK_FREE_RATE, iv, otype)
                return float(bs.get("delta") or 0) if bs else None

            if strat == "iron_condor":
                legs = p.get("legs") or {}
                ds = [(_delta(_as_float(legs.get("short_put",  {}).get("strike")), "put"),  -1),
                      (_delta(_as_float(legs.get("long_put",   {}).get("strike")), "put"),  +1),
                      (_delta(_as_float(legs.get("short_call", {}).get("strike")), "call"), -1),
                      (_delta(_as_float(legs.get("long_call",  {}).get("strike")), "call"), +1)]
                if any(d is None for d, _ in ds):
                    continue
                delta = sum(d * s for d, s in ds)
            else:
                strike   = _as_float(opt.get("strike"))
                opt_type = (opt.get("type") or "").lower()
                if not (strike > 0 and opt_type in ("call", "put")):
                    continue
                delta = _delta(strike, opt_type)
                if delta is None:
                    continue
                if strat == "sell_put":
                    delta = -delta                       # short the put
                elif strat == "covered_call":
                    delta = 1.0 - delta                  # 100 long shares + short call
            # 1 contract = 100 shares of underlying exposure
            shares_eq  = delta * 100 * contracts
            net_delta += shares_eq
            by_pos.append({
                "id":         p.get("id"),
                "ticker":     p.get("ticker"),
                "strategy":   strat,
                "delta":      round(delta, 3),
                "shares_eq":  round(shares_eq, 1),
            })
        except Exception:
            continue

    warning = None
    if abs(net_delta) > 500:
        side = "long" if net_delta > 0 else "short"
        warning = f"Net delta {net_delta:+.0f} ({side}-biased) — heavy directional exposure"

    return {
        "net_delta":   round(net_delta, 1),
        "by_position": by_pos,
        "n_positions": len(by_pos),
        "warning":     warning,
    }


def get_performance_stats():
    """
    Closed-trade analytics. Skips Sharpe (poor fit for asymmetric option returns)
    and uses better metrics: expectancy, profit factor, Sortino, max drawdown.

    Returns dict with:
      n_trades, n_wins, n_losses, win_rate_pct
      total_pnl, avg_win, avg_loss
      expectancy: avg $ made per trade (the killer metric — if negative, you lose money on average)
      profit_factor: total_wins / abs(total_losses)  (>1.5 = decent; >2.5 = strong edge)
      sortino: mean_return / downside_stdev  (Sharpe but downside-only — proper for options)
      max_drawdown: worst peak-to-trough $ loss across the equity curve
      max_drawdown_pct: same as % of peak equity
    """
    with _perf_lock:
        perf = load_performance()

    if not perf:
        return {"n_trades": 0, "note": "No closed trades yet"}

    pnls   = [float(t.get("pnl_total") or 0) for t in perf]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n      = len(pnls)
    n_w    = len(wins)
    n_l    = len(losses)

    total_wins   = sum(wins)
    total_losses = sum(losses)
    total_pnl    = total_wins + total_losses
    avg_win      = total_wins / n_w if n_w else 0.0
    avg_loss     = total_losses / n_l if n_l else 0.0   # negative
    win_rate     = n_w / n if n else 0.0

    # Expectancy per trade ($)
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Profit factor (ratio of wins to losses by magnitude)
    profit_factor = total_wins / abs(total_losses) if total_losses < 0 else float("inf") if total_wins > 0 else 0.0

    # Sortino — mean / downside deviation, where downside deviation is
    # sqrt(mean(min(p, 0)^2)) over ALL trades (not just the losers).
    mean_pnl = total_pnl / n if n else 0.0
    if n >= 2:
        d_std   = (sum(min(p, 0.0) ** 2 for p in pnls) / n) ** 0.5
        sortino = mean_pnl / d_std if d_std > 0 else 0.0
    else:
        sortino = 0.0

    # Max drawdown — walk the equity curve, track peak and max distance from peak
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    max_dd_pct = 0.0
    for p in pnls:
        eq += p
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = (dd / peak * 100) if peak > 0 else 0.0

    return {
        "n_trades":          n,
        "n_wins":            n_w,
        "n_losses":          n_l,
        "win_rate_pct":      round(win_rate * 100, 1),
        "total_pnl":         round(total_pnl, 2),
        "avg_win":           round(avg_win, 2),
        "avg_loss":          round(avg_loss, 2),
        "expectancy":        round(expectancy, 2),
        "profit_factor":     round(profit_factor, 2) if profit_factor != float("inf") else None,
        "sortino":           round(sortino, 3),
        "max_drawdown":      round(max_dd, 2),
        "max_drawdown_pct":  round(max_dd_pct, 1),
    }


def get_portfolio_risk():
    """
    Open-position risk metrics. Tells you 'how bad could it get today?'

    Returns dict with:
      n_positions
      total_max_loss: $ lost if every long-premium position went to $0
      stop_loss_exposure: $ lost if every position hit -15% (SELL_STOP_LOSS)
      largest_position_$ / largest_position_ticker: concentration check
      largest_concentration_pct: % of total exposure in the largest single position
      pnl_stdev: dispersion of current P&L% across positions (high = mixed bag)
      pnl_min / pnl_max: worst and best current P&L%
    """
    with track_lock:
        positions = load_tracked()
    if not positions:
        return {"n_positions": 0}

    total_exposure = 0.0
    by_ticker     = []
    pnl_pcts      = []
    for p in positions:
        entry      = _as_float(p.get("entry_price"))
        contracts  = max(1, _as_int(p.get("contracts"), 1))
        strategy   = p.get("strategy", "")
        opt        = p.get("option") or {}
        strike     = _as_float(opt.get("strike"))
        # Long-premium positions: max loss = premium paid
        # Cash-secured put:        max loss = strike × 100 (stock to zero) − premium received
        # Covered call:            max loss = the shares going to zero (strike as proxy) − premium
        # Iron condor:             max loss = wider wing width − net credit
        if strategy in ("buy_call", "buy_put"):
            max_loss = entry * 100 * contracts
        elif strategy == "sell_put":
            max_loss = max(strike - entry, 0) * 100 * contracts
        elif strategy == "covered_call":
            cur_px   = _as_float((p.get("last_signal") or {}).get("current_price")) or _as_float(p.get("price")) or strike
            max_loss = max(cur_px - entry, 0) * 100 * contracts
        elif strategy == "iron_condor":
            width    = max(_as_float(p.get("put_width")), _as_float(p.get("call_width")))
            max_loss = max(width - entry, 0) * 100 * contracts if width > 0 else entry * 100 * contracts
        else:
            max_loss = entry * 100 * contracts

        total_exposure += max_loss
        by_ticker.append({
            "ticker":   p.get("ticker", "?"),
            "strategy": strategy,
            "max_loss": round(max_loss, 2),
        })
        sig = p.get("last_signal") or {}
        pct = sig.get("pct_change")
        if pct is not None:
            pnl_pcts.append(float(pct))

    largest = max(by_ticker, key=lambda x: x["max_loss"]) if by_ticker else None

    # Stdev of current P&L%
    if len(pnl_pcts) >= 2:
        mean_pnl = sum(pnl_pcts) / len(pnl_pcts)
        var      = sum((x - mean_pnl) ** 2 for x in pnl_pcts) / len(pnl_pcts)
        pnl_std  = var ** 0.5
    else:
        pnl_std  = 0.0

    # Stop-loss exposure: SELL_STOP_LOSS is a % of the option premium, so apply
    # it to premium at risk (entry × 100 × contracts), not to strike collateral.
    premium_at_risk = sum(_as_float(p.get("entry_price")) * 100 * max(1, _as_int(p.get("contracts"), 1))
                          for p in positions)

    return {
        "n_positions":              len(positions),
        "total_max_loss":           round(total_exposure, 2),
        "stop_loss_exposure":       round(premium_at_risk * (SELL_STOP_LOSS / 100.0), 2),
        "largest_position_ticker":  largest["ticker"]   if largest else None,
        "largest_position":         largest["max_loss"] if largest else 0,
        "largest_concentration_pct": round(largest["max_loss"] / total_exposure * 100, 1) if largest and total_exposure > 0 else 0,
        "pnl_stdev":                round(pnl_std, 2),
        "pnl_min":                  round(min(pnl_pcts), 2) if pnl_pcts else 0,
        "pnl_max":                  round(max(pnl_pcts), 2) if pnl_pcts else 0,
        "by_ticker":                by_ticker,
    }


def load_tracked():
    """Callers hold track_lock."""
    data = _load_json(TRACK_FILE, [], "Track")
    return data if isinstance(data, list) else []


def save_tracked(positions):
    """Callers hold track_lock."""
    _atomic_json_write(TRACK_FILE, positions)


# ══════════════════════════════════════════════════════════════
#  PERFORMANCE TRACKER  (realized P&L history)
# ══════════════════════════════════════════════════════════════

_perf_lock = threading.Lock()

def load_performance():
    """Callers hold _perf_lock."""
    data = _load_json(PERFORMANCE_FILE, [], "Perf")
    return data if isinstance(data, list) else []

def save_performance(trades):
    """Callers hold _perf_lock."""
    _atomic_json_write(PERFORMANCE_FILE, trades)

def calc_performance_summary(trades):
    """Compute aggregate stats across all closed trades."""
    if not trades:
        return {
            "total_trades": 0, "winners": 0, "losers": 0,
            "win_rate": 0, "total_pnl": 0,
            "avg_win_pct": 0, "avg_loss_pct": 0,
            "best_trade": 0, "worst_trade": 0,
            "avg_pnl_pct": 0,
        }
    pnls     = [_as_float(t.get("pnl_total")) for t in trades]
    pnl_pcts = [_as_float(t.get("pnl_pct"))   for t in trades]
    winners  = [p for p in pnl_pcts if p > 0]
    losers   = [p for p in pnl_pcts if p <= 0]
    # Total P&L % = total dollar gain / total capital deployed
    total_invested = sum(
        float(t.get("entry_price", 0)) * int(t.get("contracts", 1)) * 100
        for t in trades
    )
    total_pnl_pct = round(sum(pnls) / total_invested * 100, 1) if total_invested > 0 else 0
    return {
        "total_trades":   len(trades),
        "winners":        len(winners),
        "losers":         len(losers),
        "win_rate":       round(len(winners) / len(trades) * 100, 1),
        "total_pnl":      round(sum(pnls), 2),
        "total_pnl_pct":  total_pnl_pct,
        "total_invested": round(total_invested, 2),
        "avg_win_pct":    round(sum(winners) / len(winners), 1) if winners else 0,
        "avg_loss_pct":   round(sum(losers)  / len(losers),  1) if losers  else 0,
        "best_trade":     round(max(pnl_pcts), 1),
        "worst_trade":    round(min(pnl_pcts), 1),
        "avg_pnl_pct":    round(sum(pnl_pcts) / len(pnl_pcts), 1),
    }


def get_sell_signal(position: dict) -> dict:
    """
    Re-scan the ticker for a tracked position and check whether
    a sell signal has triggered.

    Returns:
        {
          "triggered": bool,
          "reasons":   [str],       # list of reasons why sell was triggered
          "current_rsi":  float,
          "current_price": float,
          "signal_still_valid": bool,
        }
    """
    sym      = position.get("ticker", "")
    strategy = position.get("strategy", "")
    entry    = position.get("option") or {}
    entry_price = _as_float(position.get("entry_price")) or _as_float(entry.get("mid")) or _as_float(entry.get("ask"))

    # Live price override — user-supplied price when yfinance data is stale
    _live_override   = _as_float(position.get("live_price_override"))
    _override_active = _live_override > 0

    result = {
        "triggered":          False,
        "reasons":            [],
        "current_rsi":        None,
        "current_price":      None,
        "signal_still_valid": True,
    }

    try:
        tkr, hist = _fetch_history(sym, LIVE_HISTORY_PERIOD)   # same frame as the scanner → identical RSI / BB
        if hist is None or len(hist) < 30:
            return result

        closes  = hist["Close"]
        price   = float(closes.iloc[-1])
        rsi_ser = calc_rsi(closes, RSI_PERIOD)
        cur_rsi = float(rsi_ser.iloc[-1])
        bb_u, bb_m, bb_l = calc_bb(closes, BB_PERIOD, BB_STD_DEV)
        cur_bb_u = float(bb_u.iloc[-1])
        cur_bb_l = float(bb_l.iloc[-1])

        result["current_rsi"]   = round(cur_rsi, 1)
        result["current_price"] = round(price, 3)

        # ── Expired contract ───────────────────────────────────
        # There is nothing left to manage once the option has expired: flag
        # it so the card says so instead of showing an unpriceable position.
        _exp = str(entry.get("expiration") or "")[:10]
        try:
            if _exp and date.fromisoformat(_exp) < date.today():
                result["expired"]   = True
                result["triggered"] = True
                result["reasons"].append(f"⌛ Option expired {_exp} — close & log this position")
        except ValueError:
            pass

        # ── RSI momentum / trajectory ──────────────────────────
        # slope = average RSI change per day over last 3 sessions
        rsi_vals = rsi_ser.dropna()
        if len(rsi_vals) >= 4:
            rsi_slope = float((rsi_vals.iloc[-1] - rsi_vals.iloc[-4]) / 3)
        elif len(rsi_vals) >= 2:
            rsi_slope = float(rsi_vals.iloc[-1] - rsi_vals.iloc[-2])
        else:
            rsi_slope = 0.0
        rsi_slope = round(rsi_slope, 2)

        # Distance and ETA to the relevant exit threshold
        if strategy in ("buy_call",):
            rsi_distance = SELL_RSI_BULL_TARGET - cur_rsi   # positive = still has room
            direction_ok = rsi_slope > 0                    # want RSI rising
        elif strategy in ("buy_put",):
            rsi_distance = cur_rsi - SELL_RSI_BEAR_TARGET   # positive = still has room
            direction_ok = rsi_slope < 0                    # want RSI falling
        else:
            rsi_distance = 0.0
            direction_ok = True

        # ETA in trading days (None if slope is wrong direction or flat)
        if abs(rsi_slope) >= 0.1 and direction_ok and rsi_distance > 0:
            eta_days = round(rsi_distance / abs(rsi_slope))
        else:
            eta_days = None

        # Momentum label
        # Key distinction: "Reversing" only applies when the bounce/pullback has
        # STARTED and then turned the wrong way. If RSI is still in oversold/overbought
        # territory (same zone that triggered entry), a wrong-direction slope just means
        # the stock is still in the setup — not that the trade has failed.
        abs_slope = abs(rsi_slope)
        if strategy == "buy_call":
            if cur_rsi <= RSI_OVERSOLD:
                # Still in oversold zone — slope doesn't determine reversal
                if rsi_slope > 0.3:
                    momentum_label = "turning_up"   # bounce just starting
                elif abs_slope < 0.3:
                    momentum_label = "basing"        # consolidating, waiting for turn
                else:
                    momentum_label = "basing"        # still falling into oversold
            else:
                # Bounce has started (RSI > 35) — now slope direction matters
                if rsi_slope < -0.5:
                    momentum_label = "reversing"
                elif abs_slope < 0.3:
                    momentum_label = "stalling"
                elif rsi_slope >= 2.0:
                    momentum_label = "accelerating"
                else:
                    momentum_label = "on_track"
        elif strategy == "buy_put":
            if cur_rsi >= RSI_OVERBOUGHT:
                # Still in overbought zone — slope doesn't determine reversal
                if rsi_slope < -0.3:
                    momentum_label = "turning_down"  # pullback just starting
                elif abs_slope < 0.3:
                    momentum_label = "basing"
                else:
                    momentum_label = "basing"
            else:
                # Pullback has started (RSI < 65) — now slope direction matters
                if rsi_slope > 0.5:
                    momentum_label = "reversing"
                elif abs_slope < 0.3:
                    momentum_label = "stalling"
                elif rsi_slope <= -2.0:
                    momentum_label = "accelerating"
                else:
                    momentum_label = "on_track"
        else:
            # sell_put, covered_call etc — simple slope check
            if not direction_ok and abs_slope >= 0.5:
                momentum_label = "reversing"
            elif abs_slope < 0.3:
                momentum_label = "stalling"
            elif direction_ok and abs_slope >= 2.0:
                momentum_label = "accelerating"
            elif direction_ok:
                momentum_label = "on_track"
            else:
                momentum_label = "stalling"

        result["rsi_slope"]       = rsi_slope
        result["rsi_distance"]    = round(rsi_distance, 1) if rsi_distance else 0
        result["eta_days"]        = eta_days
        result["momentum_label"]  = momentum_label

        # ── Fetch current option price ─────────────────────────
        current_mid, current_bid_val, current_ask_val, price_quality, _ = _fetch_option_quote(
            sym, entry.get("expiration"), entry.get("strike"), (entry.get("type") or "call"), tkr=tkr)

        # ── Apply live price override when yfinance is stale ───
        # If the user manually entered a price AND yfinance didn't return a
        # fresh two-sided quote, use the override. Clear it once yfinance
        # comes back with a live (bid+ask) quote.
        if _override_active:
            if price_quality == "live":
                # yfinance now has a real quote — discard the manual override
                # (write-back happens at the caller level via track_lock)
                result["_clear_live_override"] = True
            else:
                # Use manual price in place of stale yfinance data
                current_mid   = _live_override
                price_quality = "manual"

        # ── Profit / loss + price-based exits (strategy-direction aware) ──
        result["current_option_mid"] = round(current_mid, 3) if current_mid > 0 else None
        result["current_bid"]        = round(current_bid_val, 3) if current_bid_val > 0 else None
        result["current_ask"]        = round(current_ask_val, 3) if current_ask_val > 0 else None
        result["price_quality"]      = price_quality
        prev_sig = position.get("last_signal") or {}
        pct_change, dollar_change = _position_pnl(strategy, entry_price, current_mid, position.get("contracts", 1))
        if pct_change is not None:
            result["pct_change"]    = pct_change
            result["dollar_change"] = dollar_change
            result["stock_best"] = _stock_best(strategy, prev_sig.get("stock_best"), price)
            trig, reasons, peak = _price_exit_check(pct_change, prev_sig.get("peak_pct_change"),
                                                     position.get("tracked_at"),
                                                     _adverse_stock_move(strategy, position.get("price"), price),
                                                     _stock_giveback(strategy, result["stock_best"], price))
            result["peak_pct_change"] = peak
            result["reasons"].extend(reasons)
            result["triggered"] = result["triggered"] or trig
        else:
            # No quote this round — carry the previous P&L/peak forward so the
            # trailing stop isn't reset by one failed chain fetch.
            for k in ("pct_change", "dollar_change", "peak_pct_change"):
                if prev_sig.get(k) is not None:
                    result[k] = prev_sig[k]

        # ── RSI / BB / strike reversal checks — shared with the backtest ──
        legs = position.get("legs") or {}
        trig, reasons, still_valid, note, _ = _strategy_exit_check(
            strategy, price, cur_rsi, cur_bb_u, cur_bb_l,
            strike=entry.get("strike"),
            short_put=(legs.get("short_put") or {}).get("strike"),
            short_call=(legs.get("short_call") or {}).get("strike"))
        result["triggered"] = result["triggered"] or trig
        result["reasons"].extend(reasons)
        result["signal_still_valid"] = still_valid
        if note:
            result["note"] = note

    except Exception as e:
        print(f"[Track] {sym} sell-signal check failed: {e}")
        result["error"] = str(e)

    return result


def _run_tracked_check():
    """Background thread — re-check sell signals on all tracked positions.

    Loads a snapshot of position IDs at the start, computes each sell signal
    (network calls, outside the lock), then re-loads from disk before writing
    back — so any edits the user makes during the check (entry price, strike,
    etc.) are never overwritten by stale data.
    """
    with track_lock:
        snapshot = load_tracked()
    if not snapshot:
        return
    print(f"\n[Track] Checking {len(snapshot)} tracked position(s)…")
    try:
        greeks = get_portfolio_greeks()
        if greeks.get("n_positions"):
            warn = f"  ⚠ {greeks['warning']}" if greeks.get("warning") else ""
            print(f"[Portfolio] Net delta: {greeks['net_delta']:+.1f} shares-equivalent across {greeks['n_positions']} positions{warn}")
    except Exception:
        pass
    for pos in snapshot:
        try:
            sig = get_sell_signal(pos)
            clear_override = sig.pop("_clear_live_override", False)
            # Re-load fresh copy from disk so any user edits are preserved.
            # CRITICAL: only write last_signal back if the option (strike/exp/type)
            # hasn't been changed since we loaded the snapshot — if it has, the
            # _refresh_after_edit thread will handle the fresh signal for the new option.
            snap_opt = pos.get("option") or {}
            with track_lock:
                positions = load_tracked()
                for p in positions:
                    if p["id"] == pos["id"]:
                        disk_opt = p.get("option") or {}
                        option_unchanged = (
                            disk_opt.get("strike")     == snap_opt.get("strike") and
                            disk_opt.get("expiration") == snap_opt.get("expiration") and
                            disk_opt.get("type")       == snap_opt.get("type")
                        )
                        if not option_unchanged:
                            # Option was edited during our network call — skip this write-back
                            print(f"[Track] {p.get('ticker')} option changed during check, skipping stale signal write.")
                            break
                        p["last_checked"] = datetime.now().isoformat()
                        if clear_override:
                            p.pop("live_price_override",    None)
                            p.pop("live_price_override_at", None)
                            print(f"[Track] {p.get('ticker')} — live quote restored, clearing manual price override")
                        p["last_signal"] = sig
                        break
                save_tracked(positions)
        except Exception as e:
            print(f"[Track] {pos.get('ticker')} check failed: {e}")
        time.sleep(2)   # space out option chain fetches to avoid rate limits
    print(f"[Track] Done checking tracked positions.")


def _schedule_track_checker():
    """Re-run the sell signal checker every TRACK_CHECK_MINS minutes."""
    try:
        _run_tracked_check()
    except Exception as e:
        print(f"[Track] Checker crashed: {e}")
    finally:
        # Always re-arm — an exception above used to stop the scheduler for good.
        t = threading.Timer(TRACK_CHECK_MINS * 60, _schedule_track_checker)
        t.daemon = True
        t.start()


def _refresh_tracked_prices():
    """
    Lightweight price-only refresh for tracked positions.
    Runs every TRACK_PRICE_REFRESH_MINS minutes.
    Only updates bid/ask/mid/price_quality in last_signal — does NOT
    re-run the full RSI/BB sell-signal analysis (that stays on the 6h cycle).
    Uses the same safe per-position save pattern to avoid race conditions.
    """
    while True:
        time.sleep(TRACK_PRICE_REFRESH_MINS * 60)
        try:
            with track_lock:
                positions = load_tracked()
            if not positions:
                continue

            chains = {}   # (ticker, expiration) → chain, so N positions on one expiry cost one call
            for pos in positions:
                opt    = pos.get("option") or {}
                exp    = opt.get("expiration")
                strike = opt.get("strike")
                otype  = opt.get("type", "call")
                sym    = pos.get("ticker", "")
                if not (exp and strike and sym) or otype not in ("call", "put"):
                    continue

                # Skip if user has an active manual price override
                if _as_float(pos.get("live_price_override")) > 0:
                    continue

                mid, bid, ask, quality, chain = _fetch_option_quote(
                    sym, exp, strike, otype, chain=chains.get((sym, exp)))
                if chain is not None:
                    chains[(sym, exp)] = chain
                if quality == "unavailable":
                    continue

                # Save only if option hasn't changed since we fetched
                with track_lock:
                    positions_now = load_tracked()
                    for p in positions_now:
                        if p["id"] != pos["id"]:
                            continue
                        disk_opt = p.get("option") or {}
                        if (disk_opt.get("strike")     != strike or
                            disk_opt.get("expiration") != exp    or
                            disk_opt.get("type")       != otype):
                            break   # option changed during fetch — skip
                        # Quote + P&L + price-based exits (target / stop / trail /
                        # stagnation) are evaluated here too, so a stop isn't missed
                        # for up to 6 hours between full sell-signal checks.
                        p["last_signal"]  = _apply_price_update(p, p.get("last_signal") or {}, mid, bid, ask, quality)
                        p["last_checked"] = datetime.now().isoformat()
                        break
                    save_tracked(positions_now)
                time.sleep(0.5)

        except Exception as e:
            print(f"  [PriceRefresh] Error: {e}")


# ══════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════

_bt_state = {"running": False, "progress": 0, "total": 0, "results": None, "ticker": None}
_bt_lock  = threading.Lock()


def _bt_option_price(S, K, T, sigma, option_type):
    """Black-Scholes option price. T in years. Returns 0 on error."""
    try:
        result = black_scholes(S, K, T, RISK_FREE_RATE, sigma, option_type)
        if result is None:
            return 0.0
        return max(0.0, result["fair_value"])
    except Exception:
        return 0.0


# Default rule parameters for a backtest run — the module LITERALS, never the
# live-mutated globals, so a "default" backtest is reproducible regardless of
# what optimizer result happens to be applied in the running process.
BT_DEFAULT_PARAMS = {
    "rsi_oversold": 35, "rsi_overbought": 65, "bull_min": 1, "bear_min": 2,
    "adx_threshold": 30, "garch_mode": "off", "put_trend_gate": "off", "yield_block_calls": False,
    "max_spread_pct": 100, "entry_mode": "reversion", "escape_normal": "skip", "escape_breakout_vol": 2.0,
    "otm_factor_call": 1.0, "otm_factor_put": 1.0, "dte_target": 30, "escape_breakout_trend": "any", "escape_require_uptrend": False,
    "breadth_chg5_min": None,
    "require_green_candle": False, "require_red_candle": False,
}
_CTX_COLS = ("vix", "vix3m", "regime", "vix_ts", "market_trend", "pre_fed", "post_fed",
             "yield_inverted", "yield_steep",
             "macro_buy_call", "macro_buy_put", "macro_sell_put", "macro_covered_call", "macro_iron_condor",
             "breadth50", "breadth50_chg5")


def _precompute_indicators(hist, sym=None, ctx_hist=None, spy_closes=None, garch="all"):
    """
    The backtest's per-ticker data: the shared indicator_frame() (every
    indicator, every bar) plus the market context of each bar's date and the
    option-model volatility.  Context comes from market_context_history()
    (same yfinance series + same pure rules as the live getters).
    """
    if ctx_hist is None:
        ctx_hist = market_context_history()
    if spy_closes is None:
        spy_closes = _ctx_hist_cache.get("spy_closes")
    fr = indicator_frame(hist, spy_closes=spy_closes, sym=sym, garch=garch)
    ctx = ctx_hist.reindex(fr["index"], method="ffill")
    fr["ctx"]    = {c: ctx[c].values for c in _CTX_COLS if c in ctx.columns}
    fr["dates"]  = [d.strftime("%Y-%m-%d") for d in fr["index"]]
    fr["closes"] = fr["price"]
    # Option-model volatility: trailing 30-day realised vol × 1.15 — a stand-in
    # for market implied vol (which usually sits above realised).  GARCH stays a
    # SIGNAL (garch_expand / garch_contract), not the pricing vol, so a correct
    # vol forecast can actually show up as P&L instead of being priced in.
    hv = pd.Series(fr["hv30"]) * 1.15
    rets = pd.Series(fr["price"]).pct_change()
    hv = hv.fillna(rets.rolling(20, min_periods=5).std() * np.sqrt(252) * 1.15).fillna(0.30)
    fr["iv_est"] = hv.clip(0.05, 2.0).values
    return fr


_PC_CTX, _PC_SPY = None, None


def _precompute_worker_init(ctx_hist, spy_closes, garch_cache):
    global _PC_CTX, _PC_SPY
    _PC_CTX, _PC_SPY = ctx_hist, spy_closes
    _garch_cache.update(garch_cache)


def _precompute_worker(item):
    sym, hist = item
    try:
        fr  = _precompute_indicators(hist, sym=sym, ctx_hist=_PC_CTX, spy_closes=_PC_SPY)
        new = fr.pop("_garch_new", {}) or {}
        return sym, fr, new, None
    except Exception as e:
        return sym, None, {}, str(e)


def _precompute_many(hist_map, label="Precompute"):
    """
    _precompute_indicators for many tickers, in parallel processes when there
    are enough of them (the rolling GARCH refits are the cost: ~90 fits per
    ticker on a 2y frame).  Fresh GARCH fits come back from the workers and are
    merged into the on-disk cache, so the second run of any job is near-instant.
    """
    ctx = market_context_history()
    spy = _ctx_hist_cache.get("spy_closes")
    items = list(hist_map.items())
    out, errors = {}, 0
    if len(items) >= 6 and (os.cpu_count() or 1) > 2:
        with _garch_cache_lock:
            seed = dict(_garch_cache)
        with ProcessPoolExecutor(max_workers=min(WFO_CPU_WORKERS, len(items)),
                                 initializer=_precompute_worker_init, initargs=(ctx, spy, seed)) as pool:
            for sym, fr, new, err in pool.map(_precompute_worker, items, chunksize=2):
                if fr is None:
                    errors += 1
                    print(f"  [{label}] {sym} failed: {err}")
                    continue
                out[sym] = fr
                if new:
                    with _garch_cache_lock:
                        _garch_cache.update(new)
    else:
        for sym, h in items:
            try:
                fr = _precompute_indicators(h, sym=sym, ctx_hist=ctx, spy_closes=spy)
                fr.pop("_garch_new", None)
                out[sym] = fr
            except Exception as e:
                errors += 1
                print(f"  [{label}] {sym} failed: {e}")
    _save_garch_cache()
    print(f"  [{label}] {len(out)} tickers ready" + (f", {errors} failed" if errors else ""))
    return out


def ctx_at(fr, i):
    """Market context for bar i of a backtest frame (same keys as live_market_ctx)."""
    c = fr["ctx"]
    def g(k, d=None):
        if k not in c:
            return d
        v = c[k][i]
        return d if (v is None or v != v) else v        # a NaN cell (bar before the first context row) = missing
    def _b(x):
        return bool(x) if x == x else False
    vix = g("vix"); vix3m = g("vix3m")
    return {
        "vix":            None if vix is None or vix != vix else float(vix),
        "vix3m":          None if vix3m is None or vix3m != vix3m else float(vix3m),
        "regime":         g("regime", "normal") or "normal",
        "vix_ts":         g("vix_ts", "flat") or "flat",
        "market_trend":   g("market_trend", "neutral") or "neutral",
        "pre_fed":        _b(g("pre_fed", False)),
        "post_fed":       _b(g("post_fed", False)),
        "yield_inverted": _b(g("yield_inverted", False)),
        "yield_steep":    _b(g("yield_steep", False)),
        "macro": {k: _b(g(f"macro_{k}", True)) for k in ("buy_call", "buy_put", "sell_put", "covered_call", "iron_condor")},
        "breadth50":      None if g("breadth50") is None else float(g("breadth50")),
        "breadth50_chg5": None if g("breadth50_chg5") is None else float(g("breadth50_chg5")),
    }


def _bt_entry_at(fr, idx, params):
    """Evaluate the shared entry rules at bar idx. Returns (ind, ctx, ev) or None."""
    ind = ind_at(fr, idx)
    if ind["price"] != ind["price"] or ind["price"] < MIN_PRICE:
        return None
    ctx = ctx_at(fr, idx)
    ev  = evaluate_entry(ind, ctx, params)
    if not (ev["has_buy_call"] or ev["has_buy_put"] or ev["has_sell_put"]):
        return None
    return ind, ctx, ev


def _bt_walk_exit(fr, idx, K, iv_est, exits, strategy, entry_opt):
    """
    Walk bar by bar after entry and close at the FIRST exit the live tracker
    would flag, using the same shared rules in the same order:
      price exits  (_price_exit_core):  target → premium stop → stock stop → trailing →
                                        stagnation → max hold — all from `exits`
      strategy exits (_strategy_exit_check): RSI/Bollinger reversal, BB breach
      expiry: the modelled option is the scanner's ~30-DTE contract — it expires
              after OPTION_DTE_BARS bars and settles at intrinsic value.
    exits: an exit-parameter dict (BT_DEFAULT_EXITS keys; a 0 switches a rule off).
    P&L is rounded to 1 dp before the threshold tests, as the live tracker does.
    Returns dict: pnl_pct (close-level), pnl_trigger_pct (the stop/target level if
    such an order is assumed to fill at its trigger), exit_reason, exit_offset,
    exit_price, exit_opt — or None when the walk runs off the end of the data.
    """
    ex       = {**BT_DEFAULT_EXITS, **(exits or {})}
    opt_type = "call" if strategy in ("buy_call", "covered_call") else "put"
    is_short = strategy in SHORT_PREMIUM_STRATS
    closes, rsi, bb_u, bb_l = fr["price"], fr["rsi"], fr["bb_u"], fr["bb_l"]
    n = fr["n"]
    dte_days = float(ex.get("dte_days") or OPTION_DTE_TARGET)          # the contract's life (contract-selection parameter)
    dte_bars = int(ex.get("dte_bars") or OPTION_DTE_BARS)
    T_entry = dte_days / 365.0
    cap  = min(int(ex.get("hold_days") or dte_bars), dte_bars)
    ex_w = {**ex, "hold_days": cap if cap < dte_bars else 0}          # at the contract's own expiry the label is "expired"
    entry_stock = float(closes[idx])

    def _pnl(curr_opt):
        return ((entry_opt - curr_opt) if is_short else (curr_opt - entry_opt)) / entry_opt * 100.0

    peak, best = -999.0, entry_stock
    for i in range(1, cap + 1):
        j = idx + i
        if j >= n:
            return None                                   # right-censored: outcome unknown
        close = float(closes[j])
        if i >= dte_bars:
            cur_opt = max(0.0, (close - K) if opt_type == "call" else (K - close))   # expiry: intrinsic
        else:
            T_rem   = max(T_entry - i / 252.0, 0.5 / 365.0)
            cur_opt = _bt_option_price(close, K, T_rem, iv_est, opt_type)
        pnl   = _pnl(cur_opt)
        pnl_r = round(pnl, 1)
        best  = _stock_best(strategy, best, close)
        trig, _, peak, code = _price_exit_core(pnl_r, peak, i, ex=ex_w,
                                               stock_move_pct=_adverse_stock_move(strategy, entry_stock, close),
                                               stock_giveback_pct=_stock_giveback(strategy, best, close))
        if not trig:
            trig, _, _, _, code = _strategy_exit_check(strategy, close, float(rsi[j]), float(bb_u[j]), float(bb_l[j]), strike=K)
        if not trig and i >= dte_bars:
            trig, code = True, "expired"
        if trig:
            if code == "stop_loss":
                pnl_trig = -float(ex["stop_loss_pct"])
            elif code == "take_profit":
                pnl_trig = float(ex["profit_target_pct"])
            else:
                pnl_trig = pnl_r
            return {"pnl_pct": pnl_r, "pnl_trigger_pct": round(pnl_trig, 1), "exit_reason": code,
                    "exit_offset": i, "exit_price": round(close, 2), "exit_opt": round(cur_opt, 3)}
    return None


def _bt_strike(price):
    """
    Nearest listed strike for a simulated ATM option.  round(price) put a
    $2.49 stock at K=2 (20% OTM) and a $2.51 stock at K=3 (20% ITM) — exactly
    the sub-$10 names Sell Put trades.  Exchanges list $0.50 strikes under
    $25, $1 strikes to $200 and $5 strikes above.
    """
    if price < 25:
        return round(price * 2) / 2
    if price < 200:
        return float(round(price))
    return float(round(price / 5) * 5)


def _bt_iv_rank_proxy(ind, iv_pct):
    """calc_iv_rank() with the modelled IV: position of iv_pct inside the trailing-year 21-day HV range."""
    lo, hi = ind.get("iv_rank_lo"), ind.get("iv_rank_hi")
    if lo is None or hi is None:
        return None
    if hi <= lo:
        return 50.0
    return round(float(np.clip((iv_pct - lo) / (hi - lo) * 100, 0, 100)), 1)


def run_backtest_ticker(ticker, hold_days=21, params=None, hist=None,
                        idx_start=60, idx_end=None, precomp=None,
                        record_path=False, use_exits=True, exits=None):
    """
    Walk history for `ticker` applying the SHARED entry rules (evaluate_entry)
    at every bar and the SHARED exit rules (_bt_walk_exit) to every trade.
    exits       : exit parameters (EXIT_KEYS — the same dict live_exit_params() returns);
                  missing keys take BT_DEFAULT_EXITS
    hold_days   : max bars held when `exits` carries no hold_days (the live tracker's
                  HOLD_DAYS rule; the modelled ~30-DTE option expires after OPTION_DTE_BARS anyway)
    params      : rule parameters (same keys as live_params()); missing keys take BT_DEFAULT_PARAMS
    hist        : pre-downloaded BT_HISTORY_PERIOD DataFrame (fetched if None)
    idx_start / idx_end : bar window (walk-forward splits); never below BT_WARMUP_BARS
    precomp     : dict from _precompute_indicators (built here if None — slow per ticker, use
                  _precompute_many for batches)
    record_path : also record unmanaged hold-to-day-d option P&L at HOLD_SWEEP_INTERVALS (Hold Horizon tool)
    What the backtest cannot know (and treats as the live gate would at gate time):
    put/call ratio (0), short interest / earnings (score-only live), real quotes
    and liquidity — the sell_put gate uses the modelled premium only (see below).
    Costs: every trade is charged its ticker's measured round-trip bid/ask spread
    (_spread_cost_pct; half when the option expires), so pnl_pct is NET and
    pnl_gross_pct is what the price model alone gave.  params["max_spread_pct"]
    drops tickers whose measured spread exceeds it (the live pick_option rule).
    """
    p  = {**BT_DEFAULT_PARAMS, **(params or {})}
    xp = {**BT_DEFAULT_EXITS, **(exits or {})}
    if not exits or "hold_days" not in exits:
        xp["hold_days"] = hold_days
    t_spread = _ticker_spread_pct(ticker)
    if float(p.get("max_spread_pct", 100)) < 100 and t_spread is not None and t_spread > float(p["max_spread_pct"]):
        return []                                        # illiquid name: the live scanner would reject its contracts
    if hist is None and precomp is None:
        try:
            _, hist = _fetch_history(ticker, BT_HISTORY_PERIOD)
            if hist is None or len(hist) <= BT_WARMUP_BARS:
                return []
        except Exception:
            return []
    if precomp is None:
        precomp = _precompute_indicators(hist, sym=ticker)

    trades  = []
    fr      = precomp
    n_bars  = fr["n"]
    # Contract selection (entry parameters): expiry closest to dte_target days, strike at otm_factor × price
    dte_days = int(p.get("dte_target", OPTION_DTE_TARGET) or OPTION_DTE_TARGET)
    dte_bars = max(2, int(round(dte_days * 252 / 365)))
    xp["dte_days"], xp["dte_bars"] = dte_days, dte_bars
    otm = {"buy_call": float(p.get("otm_factor_call", 1.0) or 1.0), "buy_put": float(p.get("otm_factor_put", 1.0) or 1.0),
           "sell_put": float(p.get("otm_factor_put", 1.0) or 1.0)}
    cap     = min(int(xp["hold_days"] or dte_bars), dte_bars)
    start   = max(BT_WARMUP_BARS, idx_start)
    # Entries stop `cap` bars before the window end so every trade resolves inside
    # the window: nothing is dropped for running off the data (which kept only the
    # fast exits), and a walk-forward training trade can never read validation bars.
    end     = (idx_end or n_bars) - cap
    T_entry = dte_days / 365.0
    # One open position per strategy at a time (the tracker rejects a duplicate
    # ticker+strategy while one is open).
    busy_until = {"buy_call": -1, "buy_put": -1, "sell_put": -1}

    for idx in range(start, end):
        hit = _bt_entry_at(fr, idx, p)
        if hit is None:
            continue
        ind, ctx, ev = hit
        price   = ind["price"]
        iv_est  = float(fr["iv_est"][idx])
        iv_rank = _bt_iv_rank_proxy(ind, iv_est * 100)

        for strategy, flag in (("buy_call", ev["has_buy_call"]),
                               ("buy_put",  ev["has_buy_put"]),
                               ("sell_put", ev["has_sell_put"])):
            if not flag or idx < busy_until[strategy]:
                continue
            K         = _bt_strike(price * otm[strategy])
            opt_type  = "call" if strategy == "buy_call" else "put"
            entry_opt = _bt_option_price(price, K, T_entry, iv_est, opt_type)
            if entry_opt <= 0.01:                          # live: mid >= 0.01 / ask >= 0.01
                continue
            premium_pct = entry_opt / price * 100
            proxy_gate  = None
            if strategy == "sell_put":
                # live: (calc_iv_rank >= IV_RANK_THRESHOLD) or (premium_pct >= PREMIUM_PCT_MIN).
                # No historical IV exists, and the modelled IV rank (1.15 × HV inside its
                # own HV range) would wave almost everything through — so only the premium
                # leg gates here: stricter than live, never looser.  iv_rank stays on the
                # trade for information.
                prem_high  = premium_pct >= PREMIUM_PCT_MIN
                proxy_gate = "premium" if prem_high else "blocked"
                if not prem_high:
                    continue

            ex = _bt_walk_exit(fr, idx, K, iv_est, xp, strategy, entry_opt)
            if ex is None:
                continue                                   # not enough bars after entry to resolve
            busy_until[strategy] = idx + ex["exit_offset"]

            # ── Hold-path recording (Hold Horizon Analysis): unmanaged P&L if closed on day d ──
            pnl_path = {}
            if record_path:
                for d in HOLD_SWEEP_INTERVALS:
                    j = idx + d
                    if j >= n_bars:
                        break
                    ep    = float(fr["price"][j])
                    T_rem = max(T_entry - d / 252.0, 0.5 / 365.0)
                    ex_val = (max(0.0, (ep - K) if opt_type == "call" else (K - ep)) if d >= dte_bars
                              else _bt_option_price(ep, K, T_rem, iv_est, opt_type))
                    pnl_path[d] = round(((entry_opt - ex_val) if strategy == "sell_put" else (ex_val - entry_opt))
                                        / entry_opt * 100, 1)

            # Transaction cost: the round-trip spread on this ticker's options; an
            # expiry settlement has no closing trade, so only the entry half applies.
            cost = _spread_cost_pct(ticker, float(entry_opt))
            if ex["exit_reason"] == "expired":
                cost *= 0.5
            pnl_net  = round(float(ex["pnl_pct"]) - cost, 1)
            trig_net = round(float(ex["pnl_trigger_pct"]) - cost, 1)
            trades.append({
                "ticker":        ticker,
                "strategy":      strategy,
                "entry_date":    fr["dates"][idx],
                "exit_date":     fr["dates"][idx + ex["exit_offset"]],
                "entry_price":   round(price, 2),
                "exit_price":    ex["exit_price"],
                "strike":        K,
                "entry_opt":     round(float(entry_opt), 3),
                "exit_opt":      round(float(ex["exit_opt"]), 3),
                "premium_pct":   round(premium_pct, 2),
                "proxy_gate":    proxy_gate,
                "pnl_gross_pct": float(ex["pnl_pct"]),
                "cost_pct":      round(cost, 2),
                "pnl_pct":       pnl_net,
                "pnl_trigger_pct": trig_net,
                "win":           bool(pnl_net > 0),
                "exit_reason":   ex["exit_reason"],
                "bars_held":     ex["exit_offset"],
                "iv_est":        round(iv_est * 100, 1),
                "iv_rank":       iv_rank,
                "garch_vol":     round(ind["garch_vol"] * 100, 1) if ind["garch_vol"] is not None else None,
                "hv30":          round(ind["hv30"] * 100, 1) if ind["hv30"] is not None else None,
                "regime":        ctx["regime"],
                "market_trend":  ctx["market_trend"],
                "bull_signals":  ev["bull_signals"],
                "bear_signals":  ev["bear_signals"],
                "pnl_path":      pnl_path,
                # Individual signal flags — the engine's definitions (same keys as the live signal log)
                **{k: bool(v) for k, v in ev["flags"].items()},
            })
    return trades


def _bt_metrics(trades):
    """
    Full risk/return metric suite for a list of trade dicts.
    All P&L figures are in PERCENT (simulated trades have no $ size).
      expectancy    — mean P&L% per trade (positive = edge)
      profit_factor — gross winning % / gross losing % (>1.5 is healthy)
      sortino       — mean P&L% / downside deviation (penalises losses only)
      max_drawdown  — deepest peak-to-trough dip on the cumulative %-curve
    """
    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    pnls   = [t["pnl_pct"] for t in trades]
    n      = len(trades)
    win_rate = round(len(wins) / n * 100, 1)
    avg_pnl  = round(sum(pnls) / n, 1)
    avg_win  = round(sum(t["pnl_pct"] for t in wins)   / len(wins),   1) if wins   else 0
    avg_loss = round(sum(t["pnl_pct"] for t in losses) / len(losses), 1) if losses else 0
    # Sharpe-like: mean / std of all P&L
    std    = float(np.std(pnls)) if n > 1 else 0.0
    sharpe = round(avg_pnl / std, 2) if std > 0 else 0
    # Sortino: mean / downside deviation (std of sub-zero returns only)
    downside = [p for p in pnls if p < 0]
    # Downside deviation = sqrt(mean(min(p,0)^2)) over ALL trades.  np.std of
    # just the losers measured how *similar* the losses were (every stop-loss
    # records exactly -15%, so that was ~0 and Sortino exploded).
    dd_std   = float(np.sqrt(np.mean(np.square(np.minimum(pnls, 0))))) if n > 1 else 0.0
    sortino  = round(avg_pnl / dd_std, 2) if dd_std > 0 else 0
    # Profit factor: gross winning % / gross losing %
    gross_win  = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))
    if gross_loss > 0:
        profit_factor = round(gross_win / gross_loss, 2)
    else:
        profit_factor = 99.0 if gross_win > 0 else 0.0
    # Max drawdown — walk the cumulative %-curve in CHRONOLOGICAL order
    # (sorting matters: trades are appended ticker-by-ticker, which would
    # otherwise interleave per-ticker histories instead of a real time-series).
    # Reported three ways for interpretability:
    #   max_drawdown      — raw cum-% dip (the "absolute" number, big units)
    #   max_drawdown_pct  — drawdown as a % of the peak at the time (digestible:
    #                       "20% off peak" instead of raw "-7253%")
    #   recovery_factor   — total gain / |max drawdown| (>3 healthy, >5 strong)
    pnls_chrono  = [t["pnl_pct"] for t in sorted(trades, key=lambda t: str(t.get("entry_date", "")))]
    equity = peak = max_dd = peak_at_dd = 0.0
    for p in pnls_chrono:
        equity += p
        if equity > peak:
            peak = equity
        dd = equity - peak
        if dd < max_dd:
            max_dd     = dd
            peak_at_dd = peak
    total_pnl_v     = sum(pnls)
    mdd_pct_of_peak = round(abs(max_dd) / peak_at_dd * 100, 1) if peak_at_dd > 0 else 0.0
    recovery_factor = round(total_pnl_v / abs(max_dd), 2) if max_dd < 0 else 0.0
    trig = [t.get("pnl_trigger_pct", t["pnl_pct"]) for t in trades]
    exit_mix = {}
    for t in trades:
        exit_mix[t.get("exit_reason", "?")] = exit_mix.get(t.get("exit_reason", "?"), 0) + 1
    return {
        "count":            n,
        "total_trades":     n,
        "win_rate":         win_rate,
        "avg_pnl":          avg_pnl,
        "avg_pnl_trigger_fill": round(sum(trig) / n, 1),   # if stop/target orders fill at the trigger level
        "avg_bars_held":    round(sum(t.get("bars_held", 0) for t in trades) / n, 1),
        "exit_mix":         exit_mix,
        "expectancy":       avg_pnl,
        "avg_win":          avg_win,
        "avg_loss":         avg_loss,
        "sharpe":           sharpe,
        "sortino":          sortino,
        "profit_factor":    profit_factor,
        "max_drawdown":     round(max_dd, 1),
        "max_drawdown_pct": mdd_pct_of_peak,
        "recovery_factor":  recovery_factor,
        "total_pnl":        round(total_pnl_v, 1),
        "best":             max(pnls),
        "worst":            min(pnls),
    }


def _bt_summary(trades):
    """Aggregate stats over a list of trade dicts (overall + per-strategy)."""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "avg_pnl": 0,
            "avg_win": 0, "avg_loss": 0, "sharpe": 0, "sortino": 0,
            "profit_factor": 0, "expectancy": 0, "max_drawdown": 0,
            "max_drawdown_pct": 0, "recovery_factor": 0,
            "total_pnl": 0, "best": 0, "worst": 0, "by_strategy": {},
        }
    overall = _bt_metrics(trades)
    by_strat = {}
    for strat in ("buy_call", "buy_put", "sell_put"):
        st = [t for t in trades if t["strategy"] == strat]
        if st:
            by_strat[strat] = _bt_metrics(st)
    return {**overall, "by_strategy": by_strat}


def _run_backtest_job(tickers, hold_days):
    all_trades = []
    try:
        with _bt_lock:
            _bt_state.update(running=True, progress=0, total=len(tickers),
                             results=None, stage="Downloading history…")

        # ── Bulk download 2y OHLCV for all tickers in one shot ───────
        print(f"  [Backtest] Bulk downloading {BT_HISTORY_PERIOD} history for {len(tickers)} tickers…")
        hist_map = _bulk_download(tickers, BT_HISTORY_PERIOD, min_rows=BT_WARMUP_BARS + OPTION_DTE_BARS + 2)
        print(f"  [Backtest] Got data for {len(hist_map)}/{len(tickers)} tickers")

        with _bt_lock:
            _bt_state["stage"] = f"Pre-computing indicators + GARCH for {len(hist_map)} tickers…"
        precomp_map = _precompute_many(hist_map, label="Backtest")

        with _bt_lock:
            _bt_state["stage"] = "Scanning signals…"

        # The dashboard backtest runs the rules exactly as the live scanner is
        # configured right now (applied optimizer params + UI toggles).
        params = live_params()
        for i, sym in enumerate(tickers):
            try:
                pc = precomp_map.get(sym)
                if pc is None:
                    continue
                trades = run_backtest_ticker(sym, hold_days, params=params, precomp=pc,
                                             exits={**live_exit_params(), "hold_days": hold_days})
                all_trades.extend(trades)
                if trades:
                    print(f"  [Backtest] {sym}: {len(trades)} trades")
            except Exception as e:
                print(f"  [Backtest] {sym} error: {e}")
            with _bt_lock:
                _bt_state["progress"] = i + 1

        summary      = _bt_summary(all_trades)
        sig_quality  = signal_quality_analysis(all_trades)
        all_trades.sort(key=lambda x: x["entry_date"], reverse=True)
        bt_results = {
            "trades":       all_trades[:500],
            "summary":      summary,
            "sig_quality":  sig_quality,
            "tickers_used": len(set(t["ticker"] for t in all_trades)),
            "hold_days":    hold_days,
            "completed_at": datetime.now().isoformat(),
        }
        with _bt_lock:
            _bt_state.update(stage="Complete", results=bt_results)
        # Persist to disk so results survive bot restarts
        try:
            _atomic_json_write(BT_FILE, bt_results)
            print(f"  [Backtest] Results saved to {BT_FILE}")
        except Exception as e:
            print(f"  [Backtest] Could not save results: {e}")
        print(f"  [Backtest] Done — {len(all_trades)} trades, "
              f"win rate {summary.get('win_rate', 'N/A')}%")
    except Exception as e:
        print(f"  [Backtest] FAILED: {e}")
        with _bt_lock:
            _bt_state["stage"] = f"Error: {e}"
    finally:
        with _bt_lock:
            _bt_state["running"] = False


# ══════════════════════════════════════════════════════════════
#  SIGNAL QUALITY ANALYSIS
# ══════════════════════════════════════════════════════════════

# Human labels for every flag evaluate_entry() records (backtest trades and
# signal-log entries carry the same keys, so one table serves both).
SIGNAL_FLAG_LABELS = {
    "sig_rsi_os":         "RSI Oversold",
    "sig_bb_low":         "Below BB Lower",
    "sig_dc_low":         "At Donchian Low",
    "sig_reversal":       "RSI Turning Up",
    "sig_macd_bull":      "MACD Bull Cross",
    "sig_vol_ok":         "Volume Confirmed",
    "sig_rsi_ob":         "RSI Overbought",
    "sig_bb_high":        "Above BB Upper",
    "sig_dc_high":        "At Donchian High",
    "sig_macd_bear":      "MACD Bear Cross",
    "sig_above_ma50":     "Above 50-day MA",
    "sig_hv_high":        "HV Rank High (≥60)",
    "sig_hv_low":         "HV Rank Low (≤40)",
    "sig_rs_strong":      "RS Strong vs SPY",
    "sig_rs_weak":        "RS Weak vs SPY",
    "sig_near_52w_low":   "Near 52-Week Low",
    "sig_near_52w_high":  "Near 52-Week High",
    "sig_mfi_os":         "MFI Oversold (<20)",
    "sig_mfi_ob":         "MFI Overbought (>80)",
    "sig_ema_bull":       "EMA9 > EMA21",
    "sig_ema_bear":       "EMA9 < EMA21",
    "sig_ema_cross_bull": "EMA9/21 Bull Cross",
    "sig_ema_cross_bear": "EMA9/21 Bear Cross",
    "sig_vix_backw":      "VIX Backwardation",
    "sig_vix_contango":   "VIX Contango",
    "sig_regime_low":     "Low VIX Regime (<15)",
    "sig_garch_expand":   "GARCH: Vol Expansion",
    "sig_garch_contract": "GARCH: Vol Contraction",
    "sig_squeeze_release": "BB Squeeze Release",
    "sig_escape_reversion": "Band Escape: Reversion",
    "sig_escape_momentum":  "Band Escape: Momentum",
    "sig_escape_breakout":  "Band Escape: Breakout",
    "sig_exhaustion":     "Exhaustion Bounce",
}


def signal_quality_analysis(trades):
    """
    For each individual signal flag, compute win rate across all trades
    where that signal was active.  Returned as sorted list of dicts.
    """
    SIGNAL_LABELS = SIGNAL_FLAG_LABELS
    rows = []
    for flag, label in SIGNAL_LABELS.items():
        subset = [t for t in trades if t.get(flag)]
        if len(subset) < 5:
            continue
        wins = [t for t in subset if t["win"]]
        pnls = [t["pnl_pct"] for t in subset]
        rows.append({
            "signal":    label,
            "count":     len(subset),
            "win_rate":  round(len(wins) / len(subset) * 100, 1),
            "avg_pnl":   round(sum(pnls) / len(pnls), 1),
            "best":      max(pnls),
            "worst":     min(pnls),
        })
    rows.sort(key=lambda x: x["win_rate"], reverse=True)
    return rows


# ══════════════════════════════════════════════════════════════
#  WALK-FORWARD OPTIMIZER
# ══════════════════════════════════════════════════════════════

# Parameter grid — each combination is tested independently.  Entry keys not
# listed stay at the live defaults; exit keys are the tracker's own rules
# (a 0 switches that rule off).  tests/run_exit_sweep.py runs a bigger,
# staged version of this grid from the command line.
WFO_PARAM_GRID = {
    # entries
    "bear_min":           [2, 3],
    "put_trend_gate":     ["off", "bear_only"],       # buy puts only in a bear trend
    "garch_mode":         ["off", "gate"],            # GJR-GARCH vol-forecast edge: ignored / hard gate
    # exits
    "stop_loss_pct":      [15, 30, 50, 0],            # premium stop; 0 = the premium is the max loss
    "profit_target_pct":  [30, 100],
    "trail_activate_pct": [15, 0],                    # trailing stop on/off (lock stays TRAIL_LOCK_PCT)
    "stagnant_days":      [3, 0],                     # stagnation exit on/off
    "hold_days":          [5, 21],                    # max trading days held (the option expires after 21 anyway)
    "stock_stop_pct":     [0, 4],                     # stock-based stop
}
# → 2·2·2·4·2·2·2·2·2 = 1,024 combinations

_wfo_state = {"running": False, "progress": 0, "total": 0, "results": None, "stage": ""}
_wfo_lock  = threading.Lock()


def _save_wfo_results(results: dict):
    """Persist WFO results + timestamp to disk so they survive restarts."""
    try:
        _atomic_json_write(WFO_FILE, {"saved_at": datetime.now().isoformat(), "results": results})
        print(f"  [WFO] Results saved to {WFO_FILE}")
    except Exception as e:
        print(f"  [WFO] Could not save results: {e}")


def _apply_wfo_params(suggestion: dict) -> dict:
    """
    Push an optimizer suggestion into the live scanner: RSI thresholds become
    the module constants, everything else (bull_min, bear_min, adx_threshold)
    goes into scan_state["wfo_params"] where scan_ticker reads it.
    Shared by /api/optimizer/apply and the startup restore.
    """
    global RSI_OVERSOLD, RSI_OVERBOUGHT
    applied = {}
    if "rsi_oversold" in suggestion:
        RSI_OVERSOLD = suggestion["rsi_oversold"];     applied["RSI_OVERSOLD"] = RSI_OVERSOLD
    if "rsi_overbought" in suggestion:
        RSI_OVERBOUGHT = suggestion["rsi_overbought"]; applied["RSI_OVERBOUGHT"] = RSI_OVERBOUGHT
    # Exit rules → the tracker's constants (the same rules the backtest walked with)
    _exit_const = {"stop_loss_pct": "SELL_STOP_LOSS", "profit_target_pct": "SELL_PROFIT_TARGET",
                   "trail_activate_pct": "TRAIL_ACTIVATE_PCT", "trail_lock_pct": "TRAIL_LOCK_PCT",
                   "stagnant_days": "STAGNANT_DAYS", "stagnant_range": "STAGNANT_RANGE",
                   "stock_stop_pct": "STOCK_STOP_PCT", "stock_trail_pct": "STOCK_TRAIL_PCT"}
    for key, const in _exit_const.items():
        if key in suggestion:
            val = _as_int(suggestion[key], 0) if key == "stagnant_days" else _as_float(suggestion[key], 0.0)
            globals()[const] = max(0, val); applied[const] = globals()[const]
    if "hold_days" in suggestion:                      # the tracker's max-hold rule — the same cap the backtest walked with
        globals()["HOLD_DAYS"] = max(1, min(_as_int(suggestion["hold_days"], 21), OPTION_DTE_BARS)); applied["HOLD_DAYS"] = HOLD_DAYS
    if "max_spread_pct" in suggestion:                 # liquidity rule → pick_option
        globals()["MAX_SPREAD_PCT"] = max(1.0, min(100.0, _as_float(suggestion["max_spread_pct"], 100.0))); applied["MAX_SPREAD_PCT"] = MAX_SPREAD_PCT
    live = {k: v for k, v in suggestion.items()
            if k not in ("rsi_oversold", "rsi_overbought", "max_spread_pct") and k not in EXIT_KEYS}
    with scan_lock:
        scan_state["wfo_params"] = live
    applied.update(live)
    return applied


def _save_hs_results(results: dict):
    """Persist Hold Horizon Analysis results to disk so they survive restarts."""
    try:
        _atomic_json_write(HS_RESULTS_FILE, {"saved_at": datetime.now().isoformat(), "results": results})
        print(f"  [HoldSweep] Results saved to {HS_RESULTS_FILE}")
    except Exception as e:
        print(f"  [HoldSweep] Could not save results: {e}")


def _load_hs_results():
    """Load persisted Hold Horizon results on startup. Restores _hs_state['results']."""
    try:
        payload = _load_json(HS_RESULTS_FILE, {}, "HoldSweep")
        results = payload.get("results") if isinstance(payload, dict) else None
        if not results:
            return
        with _hs_lock:
            _hs_state["results"] = results
            _hs_state["stage"]   = f"Loaded from disk (saved {payload.get('saved_at','')[:10]})"
        n = results.get("n_trades", 0)
        print(f"  [HoldSweep] Restored {n:,} trades from {HS_RESULTS_FILE}")
    except Exception as e:
        print(f"  [HoldSweep] Could not load saved results: {e}")


def _hs_targets_for_position(strategy, days_held):
    """
    Look up Hold Horizon Analysis data for a tracked position.
    Returns dict with optimal_day, optimal_pnl, current_avg_pnl, win_rate_now,
    n_trades, and a short text 'verdict' — or None if no HH data exists.
    """
    with _hs_lock:
        results = _hs_state.get("results")
    if not results:
        return None
    by_strategy = results.get("by_strategy", {})
    sd = by_strategy.get(strategy)
    if not sd:
        return None
    intervals = results.get("intervals") or HOLD_SWEEP_INTERVALS
    by_day = sd.get("by_day", {})
    if not by_day:
        return None
    # Find the interval closest to days_held (snap to nearest checkpoint)
    try:
        days_int = max(0, int(days_held))
    except Exception:
        days_int = 0
    nearest = min(intervals, key=lambda d: abs(d - days_int))
    cur = by_day.get(str(nearest)) or {}
    optimal_day = sd.get("optimal_day")
    opt = by_day.get(str(optimal_day)) or {}
    current_avg = cur.get("avg_pnl")
    optimal_avg = opt.get("avg_pnl")
    # Build a short verdict to surface in the UI
    verdict = ""
    if optimal_day is not None and current_avg is not None and optimal_avg is not None:
        if days_int >= optimal_day:
            verdict = f"At/past optimal hold day ({optimal_day}d). Data suggests considering exit."
        elif optimal_avg > current_avg + 1.0:
            verdict = (f"Avg gains compound: avg at {nearest}d = {current_avg:+.1f}%, "
                       f"avg at optimal {optimal_day}d = {optimal_avg:+.1f}%. Data favours holding longer.")
        elif current_avg > optimal_avg - 0.5:
            verdict = f"Near peak avg P&L for this strategy ({nearest}d ≈ optimal {optimal_day}d)."
        else:
            verdict = f"Current avg ({current_avg:+.1f}% at {nearest}d) below optimal ({optimal_avg:+.1f}% at {optimal_day}d)."
    return {
        "strategy":         strategy,
        "days_held":        days_int,
        "nearest_day":      nearest,
        "current_avg_pnl":  current_avg,
        "current_win_rate": cur.get("win_rate"),
        "current_n":        cur.get("n"),
        "optimal_day":      optimal_day,
        "optimal_avg_pnl":  optimal_avg,
        "optimal_win_rate": opt.get("win_rate"),
        "verdict":          verdict,
    }


def _load_wfo_results():
    """
    Load persisted WFO results on startup.
    If valid, restores _wfo_state and auto-applies the best params to the scanner
    so you don't have to click Apply after every restart.
    """
    try:
        payload = _load_json(WFO_FILE, {}, "WFO")
        if not isinstance(payload, dict):
            return
        saved_at = payload.get("saved_at", "")
        results  = payload.get("results", {})
        if not results:
            return

        with _wfo_lock:
            _wfo_state["results"] = results
            _wfo_state["stage"]   = f"Loaded from disk (saved {saved_at[:10]})"

        # Auto-apply the best suggestion — but only if it actually validated.
        # A run whose best out-of-sample Sharpe is <= 0 found nothing worth
        # trading on; applying it anyway just made the scanner stricter.
        suggestion = results.get("suggestion") or {}
        best       = results.get("best") or {}
        val_sharpe = _as_float(best.get("val_sharpe"), -99.0)
        if suggestion and (val_sharpe > 0 or results.get("force_apply")):
            _apply_wfo_params(suggestion)
            print(f"  [WFO] Restored results from {WFO_FILE} (saved {saved_at[:10]}) — "
                  f"params applied ({'chosen by hand: ' + str(results.get('note', '')) if results.get('force_apply') else f'val Sharpe {val_sharpe:.2f}'})")
        else:
            print(f"  [WFO] Restored results from {WFO_FILE} (saved {saved_at[:10]}) — "
                  f"NOT applied (val Sharpe {val_sharpe:.2f} ≤ 0)")
    except Exception as e:
        print(f"  [WFO] Could not load saved results: {e}")


def _wfo_sharpe(trades):
    """
    Per-trade Sharpe-like ratio: mean P&L / std P&L.
    Returns -99 when there are too few trades to mean anything (fewer than
    10 — three trades used to be enough to "win" the optimizer).
    """
    if len(trades) < 10:
        return -99.0
    pnls = [t["pnl_pct"] for t in trades]
    m = sum(pnls) / len(pnls)
    std = float(np.std(pnls))
    return round(m / std, 3) if std > 0 else 0.0


def _expand_grid(grid):
    """Cartesian product of all param grid values → list of param dicts."""
    import itertools
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    return [dict(zip(keys, c)) for c in combos]


# Process-pool worker globals (set per-worker via initializer).
# Lets workers reuse data maps across all combos without re-pickling them.
_WFO_SPLIT_MAP   = None
_WFO_PRECOMP_MAP = None


def _wfo_worker_init(split_map, precomp_map, risk_free_rate):
    """Initializer for ProcessPoolExecutor - runs once per worker process."""
    global _WFO_SPLIT_MAP, _WFO_PRECOMP_MAP, RISK_FREE_RATE
    _WFO_SPLIT_MAP   = split_map
    _WFO_PRECOMP_MAP = precomp_map
    RISK_FREE_RATE   = risk_free_rate      # the FRED-updated rate only exists in the main process


def _wfo_strategy_stats(trades):
    """Per-strategy win rate / avg P&L / count for a leaderboard row."""
    out = {}
    for s in ("buy_call", "buy_put", "sell_put"):
        st = [t for t in trades if t["strategy"] == s]
        if st:
            out[s] = {"n": len(st),
                      "win_rate": round(100 * sum(1 for t in st if t["win"]) / len(st), 1),
                      "avg_pnl":  round(sum(t["pnl_pct"] for t in st) / len(st), 1)}
    return out


def _split_combo(combo):
    """An optimizer combo → (entry params, exit params)."""
    return ({k: v for k, v in combo.items() if k not in EXIT_KEYS},
            {k: v for k, v in combo.items() if k in EXIT_KEYS})


def _wfo_quick_stats(trades):
    """The few numbers the leaderboard needs, cheap enough to compute per combo."""
    n = len(trades)
    if not n:
        return {"sharpe": -99.0, "n": 0, "win_rate": 0.0, "avg_pnl": 0.0, "pf": 0.0, "avg_pnl_trigger_fill": 0.0}
    pnls  = [t["pnl_pct"] for t in trades]
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = -sum(p for p in pnls if p < 0)
    return {"sharpe": _wfo_sharpe(trades), "n": n,
            "win_rate": round(sum(1 for t in trades if t["win"]) / n * 100, 1),
            "avg_pnl":  round(sum(pnls) / n, 2),
            "pf":       round(gross_w / gross_l, 2) if gross_l > 0 else (99.0 if gross_w > 0 else 0.0),
            "avg_pnl_trigger_fill": round(sum(t["pnl_trigger_pct"] for t in trades) / n, 2)}


def _wfo_eval_train_proc(combo):
    """Process-pool worker: evaluate one param combo on the training window → (stats, combo)."""
    params, exits = _split_combo(combo)
    all_t = []
    for sym, pc in _WFO_PRECOMP_MAP.items():
        all_t.extend(run_backtest_ticker(sym, params=params, exits=exits, idx_end=_WFO_SPLIT_MAP[sym], precomp=pc))
    return _wfo_quick_stats(all_t), combo


def _wfo_eval_val_proc(item):
    """Process-pool worker: evaluate one combo on the validation window → leaderboard row."""
    train, combo = item
    params, exits = _split_combo(combo)
    val_trades = []
    for sym, pc in _WFO_PRECOMP_MAP.items():
        val_trades.extend(run_backtest_ticker(sym, params=params, exits=exits, idx_start=_WFO_SPLIT_MAP[sym], precomp=pc))
    val = _wfo_quick_stats(val_trades)
    bars = [t["bars_held"] for t in val_trades]
    return {
        "params":        combo,
        "train_sharpe":  train["sharpe"],
        "train_trades":  train["n"],
        "train_pf":      train["pf"],
        "train_avg_pnl": train["avg_pnl"],
        "train_win_rate": train["win_rate"],
        "val_sharpe":    val["sharpe"],
        "val_trades":    val["n"],
        "val_pf":        val["pf"],
        "val_win_rate":  val["win_rate"],
        "val_avg_pnl":   val["avg_pnl"],
        "val_avg_pnl_trigger_fill": val["avg_pnl_trigger_fill"],
        "val_avg_bars_held": round(sum(bars) / len(bars), 1) if bars else 0,
        "val_by_strategy": _wfo_strategy_stats(val_trades),
        # profitable in BOTH windows with a real sample — the only kind of result worth applying
        "robust": bool(train["pf"] > 1.0 and val["pf"] > 1.0 and train["n"] >= 200 and val["n"] >= 100),
    }


def _run_wfo_job(tickers):
    """
    Walk-forward optimization across `tickers`.

    Algorithm:
      1. Download 2y history for each ticker once.
      2. Pre-compute all indicators once per ticker (huge speedup for combo sweep).
      3. Split each history: train = first 75%, validation = last 25%.
      4. For each param combo (parallelized), run backtest on ALL tickers × training window.
         Aggregate Sharpe across all resulting trades.
      5. Take top-10 combos by training Sharpe.
      6. Validate each on ALL tickers × validation window.
      7. Rank by validation Sharpe (guards against overfitting).
      8. Return best combo + full leaderboard.

    Progress: total = len(combos) + 10  (486 training + 10 validation = 496)
    """
    try:
        _run_wfo_job_inner(tickers)
    except Exception as e:
        print(f"  [WFO] FAILED: {e}")
        with _wfo_lock:
            _wfo_state.update(stage=f"Error: {e}")
    finally:
        with _wfo_lock:
            _wfo_state["running"] = False      # never leave the optimizer stuck on "running"


def _wfo_prepare(tickers, label="WFO"):
    """
    Download + pre-compute + date split for a walk-forward run.
    Returns (precomp_map, split_map, split_date) or None when no data came back.
    The split is by DATE, not per-ticker row count: a ticker with a short
    history (IPO, data gap) used to have its "first 75%" land in the months that
    were the validation window for everyone else.
    """
    print(f"  [{label}] Downloading {BT_HISTORY_PERIOD} history for {len(tickers)} tickers…")
    hist_map = _bulk_download(tickers, BT_HISTORY_PERIOD, min_rows=BT_WARMUP_BARS + OPTION_DTE_BARS + 2)
    if not hist_map:
        return None
    _attach_breadth(hist_map, label=label)                 # breadth of this universe → ctx columns (the breadth gate)
    print(f"  [{label}] Pre-computing indicators for {len(hist_map)} tickers…")
    precomp_map = _precompute_many(hist_map, label=label)
    hist_map    = {s: h for s, h in hist_map.items() if s in precomp_map}
    all_dates   = sorted(set().union(*[set(h.index) for h in hist_map.values()]))
    split_date  = all_dates[int(len(all_dates) * 0.75)]
    split_map   = {sym: int(h.index.searchsorted(split_date)) for sym, h in hist_map.items()}
    print(f"  [{label}] Train/validation split at {pd.Timestamp(split_date).date()}")
    return precomp_map, split_map, split_date


def _wfo_sweep(combos, precomp_map, split_map, top_n=10, progress=None, label="WFO"):
    """
    Train every combo in a process pool, validate the `top_n` by training
    Sharpe, return the leaderboard sorted by validation Sharpe.
    progress(done, total) is called as work completes.
    """
    total = len(combos) + top_n
    print(f"  [{label}] Training sweep: {len(combos)} combos x {len(precomp_map)} tickers using {WFO_CPU_WORKERS} processes")
    print(f"  [{label}] Spawning workers (first run after startup takes ~10-60s)...")
    train_scores, leaderboard = [], []
    done = 0
    with ProcessPoolExecutor(max_workers=WFO_CPU_WORKERS, initializer=_wfo_worker_init,
                             initargs=(split_map, precomp_map, RISK_FREE_RATE)) as pool:
        futs = {pool.submit(_wfo_eval_train_proc, combo): combo for combo in combos}
        for fut in as_completed(futs):
            try:
                train_scores.append(fut.result())
            except Exception as e:
                print(f"  [{label}] combo failed: {e}")
            done += 1
            if progress:
                progress(done, total)
        train_scores.sort(key=lambda x: x[0]["sharpe"], reverse=True)
        top = train_scores[:top_n]
        for fut in as_completed([pool.submit(_wfo_eval_val_proc, item) for item in top]):
            try:
                leaderboard.append(fut.result())
            except Exception as e:
                print(f"  [{label}] validation failed: {e}")
            done += 1
            if progress:
                progress(done, total)
    leaderboard.sort(key=lambda x: x["val_sharpe"], reverse=True)
    return leaderboard


def _run_wfo_job_inner(tickers):
    combos = _expand_grid(WFO_PARAM_GRID)
    total_steps = len(combos) + 10
    with _wfo_lock:
        _wfo_state.update(running=True, progress=0, total=total_steps,
                          results=None, stage="Downloading history…")
    prep = _wfo_prepare(tickers)
    if prep is None:
        with _wfo_lock:
            _wfo_state.update(stage="No data available")
        return
    precomp_map, split_map, split_date = prep
    hist_map = precomp_map                       # (keys only used below)

    with _wfo_lock:
        _wfo_state["stage"] = f"Training {len(combos)} param combos..."

    def _progress(done, total):
        with _wfo_lock:
            _wfo_state["progress"] = done
            if done == len(combos):
                _wfo_state["stage"] = "Validating top 10 combos..."

    leaderboard = _wfo_sweep(combos, precomp_map, split_map, top_n=10, progress=_progress)
    best = leaderboard[0] if leaderboard else None

    # ── Live config suggestion (shown to the user; Apply is a separate click) ─
    suggestion = dict(best["params"]) if best else None

    wfo_results = {
        "leaderboard":   leaderboard,
        "best":          best,
        "suggestion":    suggestion,
        "split_date":    str(pd.Timestamp(split_date).date()),
        "tickers_used":  list(hist_map.keys()),
        "completed_at":  datetime.now().isoformat(),
    }
    with _wfo_lock:
        _wfo_state.update(stage="Complete" if best else "Complete — no combo produced enough trades",
                          progress=total_steps, results=wfo_results)
    # Persist only a run that actually validated (val Sharpe > 0) — or the very
    # first run — so a quick single-ticker experiment can't overwrite a good
    # full-universe result on disk.  The run is still shown in the UI either way.
    if best and (_as_float(best.get("val_sharpe"), -99.0) > 0 or not os.path.exists(WFO_FILE)):
        _save_wfo_results(wfo_results)
    elif best:
        print("  [WFO] Best combo did not validate (val Sharpe ≤ 0) — results shown but not saved over the previous run")
    print(f"  [WFO] Done — best val Sharpe {best['val_sharpe'] if best else 'N/A'}")


# ══════════════════════════════════════════════════════════════
#  SIGNAL CALIBRATION ENGINE
#  Unlike WFO (which sweeps param combos), calibration measures
#  how reliable each individual signal has been in recent data,
#  then applies those weights to the live scorer so signals that
#  aren't working in the current market regime get down-weighted.
# ══════════════════════════════════════════════════════════════

CALIBRATION_FILE = "calibration_data.json"

_cal_state = {
    "running":  False,
    "progress": 0,
    "total":    0,
    "results":  None,
    "stage":    "",
}
_cal_lock = threading.Lock()

# In-memory calibration weights: strategy → signal_flag → multiplier (0.5–1.3)
# Default empty = all weights 1.0 (no adjustment) until first calibration run.
_cal_weights: dict = {}
_lr_state:    dict = {"running": False, "stage": "Idle", "results": None}
_lr_lock             = threading.Lock()
_hs_state:    dict = {"running": False, "stage": "Idle", "results": None}
_hs_lock             = threading.Lock()


def _cal_wr_to_weight(win_rate_pct: float) -> float:
    """Map a signal win-rate % to a score multiplier."""
    if win_rate_pct >= 70: return 1.3
    if win_rate_pct >= 60: return 1.1
    if win_rate_pct >= 50: return 1.0
    if win_rate_pct >= 40: return 0.75
    return 0.5


def _run_calibration_job(months: int = 12, universe: str = "focus"):
    """
    Signal calibration backtest.

    Steps:
      1. Bulk-download last N months of history for chosen universe.
      2. Run run_backtest_ticker on that window (hold_days=14) — each
         trade carries individual signal flags.
      3. Run signal_quality_analysis() to get per-signal win-rates.
      4. Read signal_log.json for unbiased live recommendation outcomes.
      5. Read performance.json for actual user trades (real outcomes).
      6. Build signal weight dict and save to calibration_data.json.
      7. Apply weights immediately to _cal_weights so live scores update.
    """
    try:
        _run_calibration_job_inner(months, universe)
    except Exception as e:
        print(f"  [CAL] FAILED: {e}")
        with _cal_lock:
            _cal_state["stage"] = f"Error: {e}"
    finally:
        with _cal_lock:
            _cal_state["running"] = False      # a crash used to leave this True → 409 forever


def _calibration_universe(universe):
    if universe == "snp500":
        tickers = get_universe(force_mode="full")
        print(f"  [CAL] Universe: S&P 500 — {len(tickers)} tickers")
    else:
        tickers = list(dict.fromkeys(FOCUS_LIST))
        print(f"  [CAL] Universe: Focus list — {len(tickers)} tickers")
    return tickers


def _run_calibration_job_inner(months, universe):
    global _cal_weights

    tickers = _calibration_universe(universe)

    with _cal_lock:
        _cal_state.update(running=True, progress=0, total=len(tickers),
                          results=None, stage="Downloading history…")

    # ── 1. Bulk-download ──────────────────────────────────────
    # Always download 2y regardless of the signal window — the 200-day MA
    # needs 200 bars of history to be valid.  We restrict which bars we
    # *score* to the last `months` months via idx_start below.
    print(f"  [CAL] Downloading {BT_HISTORY_PERIOD} history for {len(tickers)} tickers "
          f"(signal window: last {months}m)…")
    hist_map = _bulk_download(tickers, BT_HISTORY_PERIOD, min_rows=BT_WARMUP_BARS + OPTION_DTE_BARS + 2)

    if not hist_map:
        with _cal_lock:
            _cal_state.update(stage="Download failed — no data")
        return

    with _cal_lock:
        _cal_state.update(stage=f"Running backtest on {len(hist_map)} tickers…",
                          total=len(hist_map))

    # ── 2. Backtest each ticker in the recent window (live rules, live params) ──
    with _cal_lock:
        _cal_state["stage"] = f"Pre-computing indicators + GARCH for {len(hist_map)} tickers…"
    precomp_map = _precompute_many(hist_map, label="CAL")
    params      = live_params()
    all_trades: list = []
    for i, (sym, pc) in enumerate(precomp_map.items()):
        with _cal_lock:
            _cal_state["progress"] = i + 1
        try:
            cutoff    = pc["index"][-1] - pd.DateOffset(months=months)
            start_idx = max(BT_WARMUP_BARS, int(pc["index"].searchsorted(cutoff)))
            trades = run_backtest_ticker(sym, params=params, exits=live_exit_params(), precomp=pc, idx_start=start_idx)
            all_trades.extend(trades)
        except Exception as e:
            print(f"  [CAL] {sym} error: {e}")

    print(f"  [CAL] Backtest complete — {len(all_trades)} trades from {len(precomp_map)} tickers")

    if not all_trades:
        with _cal_lock:
            _cal_state.update(running=False, stage=f"No trades in last {months}m window")
        return

    with _cal_lock:
        _cal_state["stage"] = "Analysing signal quality…"

    # ── 3. Per-signal win-rates from simulated trades ─────────
    sim_quality = signal_quality_analysis(all_trades)

    # ── 3b. Blend in live signal log (unbiased live recommendations) ──
    sig_log_summary = _signal_log_summary()
    sig_log_quality = sig_log_summary.get("sig_quality", [])

    # ── 4. Actual user trades from performance.json ───────────
    real_trades: list = []
    actual_by_strat: dict = {}
    try:
        if os.path.exists(PERFORMANCE_FILE):
            with open(PERFORMANCE_FILE) as f:
                perf = json.load(f)
            for t in perf:
                if t.get("pnl_pct") is not None:
                    real_trades.append(t)
            for strat in ("buy_call", "buy_put", "sell_put"):
                st = [t for t in real_trades if t.get("strategy") == strat]
                if st:
                    wins = [t for t in st if t.get("pnl_pct", 0) > 0]
                    pnls = [t["pnl_pct"] for t in st]
                    actual_by_strat[strat] = {
                        "count":    len(st),
                        "win_rate": round(len(wins) / len(st) * 100, 1),
                        "avg_pnl":  round(sum(pnls) / len(pnls), 1),
                    }
    except Exception as e:
        print(f"  [CAL] Could not read performance.json: {e}")

    # ── 5. Build signal weights ───────────────────────────────
    # Label → flag for every flag the engine records (one table, see SIGNAL_FLAG_LABELS).
    LABEL_TO_FLAG = {v: k for k, v in SIGNAL_FLAG_LABELS.items()}
    # The flags the live scorers actually read (_score_sell_put / _score_buy_call /
    # _score_buy_put).  The old sets produced keys no scorer looked at and
    # omitted the ones they did, so most calibration weights never took effect.
    BULL_FLAGS = {"sig_rsi_os", "sig_bb_low", "sig_reversal", "sig_above_ma50",
                  "sig_hv_low", "sig_hv_high", "sig_rs_strong"}
    BEAR_FLAGS = {"sig_rsi_ob", "sig_bb_high", "sig_rs_weak"}

    # Blend simulated + live signal log win rates.
    # Signal log (real live recs) is weighted 2× vs simulated backtest.
    log_by_label = {r["signal"]: r for r in sig_log_quality}
    blended_quality = []
    for row in sim_quality:
        log_row = log_by_label.get(row["signal"])
        if log_row and log_row["count"] >= 5:
            sim_n  = row["count"]
            log_n  = log_row["count"] * 2   # 2× weight for live recs
            blended_wr = round(
                (row["win_rate"] * sim_n + log_row["win_rate"] * log_n) / (sim_n + log_n), 1
            )
            blended_quality.append({**row, "win_rate": blended_wr,
                                     "log_count": log_row["count"],
                                     "log_win_rate": log_row["win_rate"]})
        else:
            blended_quality.append(row)
    blended_quality.sort(key=lambda x: x["win_rate"], reverse=True)

    new_weights: dict = {"buy_call": {}, "buy_put": {}, "sell_put": {}}
    for row in blended_quality:
        flag = LABEL_TO_FLAG.get(row["signal"])
        if not flag:
            continue
        w = _cal_wr_to_weight(row["win_rate"])
        if flag in BULL_FLAGS:
            new_weights["buy_call"][flag] = w
            new_weights["sell_put"][flag] = w
        if flag in BEAR_FLAGS:
            new_weights["buy_put"][flag] = w

    _cal_weights = new_weights   # apply immediately to live scorer
    # This full replace used to silently drop the finer-grained LR weights
    # until the next restart re-overlaid them — re-apply them now instead.
    with _lr_lock:
        _lr_w = (_lr_state.get("results") or {}).get("weights") or {}
    if _lr_w:
        _apply_lr_weights(_lr_w)

    # ── 6. Persist to disk ────────────────────────────────────
    cur_vix    = get_vix()
    cur_regime = vix_regime(cur_vix)
    sim_summary = _bt_summary(all_trades)

    results = {
        "months":              months,
        "universe":            universe,
        "tickers_used":        len(hist_map),
        "total_trades":        len(all_trades),
        "sim_summary":         sim_summary,
        "sim_quality":         blended_quality,   # blended sim + live log
        "sig_log_resolved":    sig_log_summary.get("resolved", 0),
        "sig_log_pending":     sig_log_summary.get("pending", 0),
        "sig_log_quality":     sig_log_quality,   # raw live log quality (for display)
        "actual_trades":       actual_by_strat,
        "real_trade_count":    len(real_trades),
        "weights":             new_weights,
        "vix_at_run":          cur_vix,
        "regime_at_run":       cur_regime,
        "completed_at":        datetime.now().isoformat(),
    }
    try:
        _atomic_json_write(CALIBRATION_FILE, results)
        print(f"  [CAL] Saved to {CALIBRATION_FILE}")
    except Exception as e:
        print(f"  [CAL] Save failed: {e}")

    with _cal_lock:
        _cal_state.update(stage="Complete", progress=len(hist_map), results=results)
    print(f"  [CAL] Done — {len(all_trades)} trades, {len(sim_quality)} signals analysed")


LR_CALIBRATION_FILE = "lr_calibration.json"

# ── Signal flags and human labels used by LR calibration ──────────
_LR_SIGNAL_FLAGS = [
    "sig_rsi_os", "sig_bb_low", "sig_dc_low", "sig_reversal", "sig_macd_bull", "sig_vol_ok",
    "sig_rsi_ob", "sig_bb_high", "sig_dc_high", "sig_macd_bear",
    "sig_mfi_os", "sig_mfi_ob", "sig_above_ma50", "sig_hv_high", "sig_hv_low",
    "sig_rs_strong", "sig_rs_weak", "sig_near_52w_low", "sig_near_52w_high",
    "sig_ema_bull", "sig_ema_bear", "sig_ema_cross_bull", "sig_ema_cross_bear",
    "sig_vix_backw", "sig_vix_contango", "sig_regime_low",
    "sig_garch_expand", "sig_garch_contract", "sig_exhaustion",
]
_LR_FLAG_LABELS = dict(SIGNAL_FLAG_LABELS)


def _fit_lr_on_trade_list(trade_list, source_label=""):
    """
    Core logistic regression fitting logic.  Accepts any list of trade dicts
    that each have signal flags (sig_*) and a 'win' boolean or 'outcome' field.

    Returns (strategy_results, new_weights) or raises on fatal error.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        raise RuntimeError("scikit-learn not installed — "
                           "run: pip install scikit-learn --break-system-packages")

    STRATEGIES  = ["buy_call", "sell_put", "buy_put", "covered_call"]
    MIN_SAMPLES = 20

    new_weights      = {}
    strategy_results = {}

    for strat in STRATEGIES:
        strat_entries = [e for e in trade_list if e.get("strategy") == strat]

        if len(strat_entries) < MIN_SAMPLES:
            strategy_results[strat] = {
                "status":  "insufficient_data",
                "count":   len(strat_entries),
                "message": f"Need {MIN_SAMPLES} samples (have {len(strat_entries)})",
            }
            continue

        with _lr_lock:
            _lr_state["stage"] = (
                f"Fitting {strat} ({len(strat_entries)} samples)…"
            )

        # Build binary feature matrix — use all flags present in _LR_SIGNAL_FLAGS
        X = np.array([
            [1.0 if e.get(flag) else 0.0 for flag in _LR_SIGNAL_FLAGS]
            for e in strat_entries
        ])

        # ── Win label strategy ─────────────────────────────────────────
        # First try the natural win/loss split (pnl_pct > 0 or outcome == "win").
        # If all outcomes end up identical (e.g. turbulent market where all
        # oversold bounces won, or all falling-knife buys lost), fall back to a
        # median split: "above-median P&L = win, below = loss".  This always
        # gives a balanced dataset and teaches the model what separates a *better
        # than average* trade from a worse one — which is what we actually want.
        # Signal-log entries carry the STOCK move, not the option P&L.  For a
        # put buyer a rising stock is the losing outcome, so flip the sign for
        # buy_put before the median split (it used to label those as wins).
        _dir = -1.0 if strat == "buy_put" else 1.0
        pnl_vals = [
            e["pnl_pct"] if e.get("pnl_pct") is not None
            else (None if e.get("stock_return_pct") is None else _dir * float(e["stock_return_pct"]))
            for e in strat_entries
        ]
        has_pnl  = all(v is not None for v in pnl_vals)

        y_natural = np.array([
            1 if (e.get("win") is True or e.get("outcome") == "win") else 0
            for e in strat_entries
        ])

        if len(set(y_natural)) >= 2:
            y           = y_natural
            split_method = "natural (pnl > 0)"
        elif has_pnl:
            # Median split fallback
            median_pnl  = float(np.median([v for v in pnl_vals if v is not None]))
            y           = np.array([1 if (v or 0) >= median_pnl else 0 for v in pnl_vals])
            split_method = f"median split (≥{round(median_pnl,1)}%)"
            print(f"  [LR] {strat}: using median split (all natural outcomes identical)")
        else:
            strategy_results[strat] = {
                "status":  "single_class",
                "count":   len(y_natural),
                "message": "All outcomes identical and no P&L data for median split",
            }
            continue

        if len(set(y)) < 2:
            strategy_results[strat] = {
                "status":  "single_class",
                "count":   len(y_natural),
                "message": "All outcomes identical — need both wins and losses",
            }
            continue

        try:
            model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            model.fit(X, y)

            accuracy = round(float(model.score(X, y)) * 100, 1)
            win_rate = round(float(y.mean()) * 100, 1)
            coefs    = model.coef_[0]

            weights      = {}
            coef_details = []
            for i, flag in enumerate(_LR_SIGNAL_FLAGS):
                coef       = float(coefs[i])
                odds_ratio = float(np.exp(coef))
                weight     = float(np.clip(odds_ratio, 0.3, 2.0))
                weights[flag] = round(weight, 3)
                coef_details.append({
                    "flag":       flag,
                    "label":      _LR_FLAG_LABELS.get(flag, flag),
                    "coef":       round(coef, 3),
                    "odds_ratio": round(odds_ratio, 3),
                    "weight":     round(weight, 3),
                })

            new_weights[strat]      = weights
            strategy_results[strat] = {
                "status":       "ok",
                "count":        len(strat_entries),
                "accuracy":     accuracy,
                "win_rate":     win_rate,
                "split_method": split_method,
                "coef_details": coef_details,
            }
            print(f"  [LR] {strat} ({source_label}): n={len(strat_entries)}, "
                  f"acc={accuracy}%, wr={win_rate}%, split={split_method}")
        except Exception as e:
            strategy_results[strat] = {"status": "error", "error": str(e)}
            print(f"  [LR] {strat} fit error: {e}")

    return strategy_results, new_weights


def _apply_lr_weights(new_weights):
    """Merge LR weights into _cal_weights (LR takes precedence)."""
    global _cal_weights
    if not new_weights:
        return
    merged = {k: dict(v) for k, v in _cal_weights.items()}
    for strat, weights in new_weights.items():
        if strat not in merged:
            merged[strat] = {}
        merged[strat].update(weights)
    _cal_weights = merged
    print(f"  [LR] Applied weights for: {list(new_weights.keys())}")


def _run_logistic_calibration(source="signal_log", months=12, universe="focus"):
    """
    Joint signal calibration via logistic regression.

    source="signal_log"  — use the resolved live signal log (fast, ~seconds)
    source="backtest"    — download historical data and run a full backtest
                           (same data pipeline as regular calibration, slower)

    Coefficients are converted to odds-ratios (exp(coef)) and clipped to
    [0.3, 2.0] for use as score multipliers in _cal_weights.
    """
    global _cal_weights

    with _lr_lock:
        _lr_state.update(running=True, stage="Starting…")

    try:
        if source == "backtest":
            # ── Download + backtest (same pipeline as _run_calibration_job) ──
            if universe == "snp500":
                import requests as _req
                _snp = []
                try:
                    resp = _req.get(
                        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
                        "/main/data/constituents.csv",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                    if resp.status_code == 200:
                        lines = resp.text.strip().split("\n")[1:]
                        _snp = [l.split(",")[0].strip().replace(".", "-")
                                for l in lines if l.strip()]
                        _snp = [x for x in _snp if 1 < len(x) <= 6]
                except Exception as e:
                    print(f"  [LR] S&P fetch failed ({e}), falling back to focus list")
                tickers = list(dict.fromkeys(_snp)) if _snp else list(dict.fromkeys(FOCUS_LIST))
            else:
                tickers = list(dict.fromkeys(FOCUS_LIST))

            print(f"  [LR] Downloading {BT_HISTORY_PERIOD} history for {len(tickers)} tickers…")
            with _lr_lock:
                _lr_state["stage"] = f"Downloading history for {len(tickers)} tickers…"

            hist_map = _bulk_download(tickers, BT_HISTORY_PERIOD, min_rows=BT_WARMUP_BARS + OPTION_DTE_BARS + 2)
            if not hist_map:
                with _lr_lock:
                    _lr_state.update(running=False, stage="Download failed — no data")
                return

            with _lr_lock:
                _lr_state["stage"] = f"Running backtest on {len(hist_map)} tickers…"

            with _lr_lock:
                _lr_state["stage"] = f"Pre-computing indicators + GARCH for {len(hist_map)} tickers…"
            precomp_map = _precompute_many(hist_map, label="LR")
            params      = live_params()
            all_trades: list = []
            for i, (sym, pc) in enumerate(precomp_map.items()):
                if i % 50 == 0:
                    with _lr_lock:
                        _lr_state["stage"] = f"Backtesting {i}/{len(precomp_map)} tickers…"
                try:
                    cutoff    = pc["index"][-1] - pd.DateOffset(months=months)
                    start_idx = max(BT_WARMUP_BARS, int(pc["index"].searchsorted(cutoff)))
                    trades    = run_backtest_ticker(sym, params=params, exits=live_exit_params(), precomp=pc, idx_start=start_idx)
                    all_trades.extend(trades)
                except Exception as e:
                    print(f"  [LR] {sym} backtest error: {e}")

            print(f"  [LR] Backtest complete — {len(all_trades)} trades")
            if not all_trades:
                with _lr_lock:
                    _lr_state.update(running=False, stage="No trades produced — try a longer window")
                return

            trade_list   = all_trades
            source_label = f"backtest {months}m {universe}"
            total_count  = len(all_trades)

        else:
            # ── Signal log (live resolved recommendations) ──
            entries  = _load_signal_log()
            resolved = [e for e in entries
                        if e.get("resolved") and e.get("outcome") in ("win", "loss")]
            if not resolved:
                with _lr_lock:
                    _lr_state.update(running=False,
                                     stage="No resolved signal log entries yet — run "
                                           "more scans and wait 7–21 days for outcomes to resolve")
                return
            trade_list   = resolved
            source_label = "signal_log"
            total_count  = len(resolved)

        # ── Fit LR models ──────────────────────────────────────────────
        strategy_results, new_weights = _fit_lr_on_trade_list(trade_list, source_label)
        _apply_lr_weights(new_weights)

        results = {
            "source":        source,
            "months":        months,
            "universe":      universe,
            "strategies":    strategy_results,
            "weights":       new_weights,
            "total_entries": total_count,
            "completed_at":  datetime.now().isoformat(),
        }

        try:
            with open(LR_CALIBRATION_FILE, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"  [LR] Save failed: {e}")

        strats_ok = [s for s, r in strategy_results.items() if r.get("status") == "ok"]
        with _lr_lock:
            _lr_state.update(running=False, stage="Complete", results=results)
        print(f"  [LR] Done — {total_count} samples, models fitted: {strats_ok}")

    except Exception as e:
        with _lr_lock:
            _lr_state.update(running=False, stage=f"Error: {e}")
        print(f"  [LR] Fatal error: {e}")


def _aggregate_hold_sweep(all_trades):
    """
    Given a list of backtest trades that each have a 'pnl_path' dict
    (day → P&L %), aggregate avg P&L, win rate, and trade count for
    every strategy × interval combination.
    """
    results = {}
    for strat in ("buy_call", "buy_put", "sell_put"):
        strat_trades = [t for t in all_trades if t.get("strategy") == strat]
        if not strat_trades:
            continue
        by_day = {}
        for d in HOLD_SWEEP_INTERVALS:
            vals = [t["pnl_path"][d] for t in strat_trades
                    if d in t.get("pnl_path", {})]
            if not vals:
                continue
            by_day[d] = {
                "n":        len(vals),
                "avg_pnl":  round(float(np.mean(vals)), 2),
                "win_rate": round(float(np.mean([1 if v > 0 else 0 for v in vals])) * 100, 1),
                "p25":      round(float(np.percentile(vals, 25)), 1),
                "p75":      round(float(np.percentile(vals, 75)), 1),
            }
        if by_day:
            # Optimal day = highest avg_pnl
            optimal_day = max(by_day, key=lambda d: by_day[d]["avg_pnl"])
            results[strat] = {
                "n_trades":   len(strat_trades),
                "by_day":     {str(k): v for k, v in by_day.items()},
                "optimal_day": optimal_day,
            }
    return results


def _run_hold_sweep_job(months=12, universe="focus"):
    """
    Background job for Hold Horizon Analysis.
    Downloads history for the chosen universe, runs the backtest with
    record_path=True to capture P&L at every HOLD_SWEEP_INTERVALS checkpoint,
    then aggregates into per-strategy curves.
    """
    with _hs_lock:
        _hs_state.update(running=True, stage="Downloading tickers…", results=None)

    try:
        # ── Get ticker list ──────────────────────────────────────────
        if universe == "snp500":
            tickers = get_universe(force_mode="full")
        else:
            tickers = list(dict.fromkeys(FOCUS_LIST))
        if not tickers:
            with _hs_lock:
                _hs_state.update(stage="No tickers found")
            return

        cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=months)

        # One bulk download instead of ~500 individual history() calls
        # (unadjusted prices — same basis as the calibration/LR jobs).
        with _hs_lock:
            _hs_state["stage"] = f"Downloading {BT_HISTORY_PERIOD} history for {len(tickers)} tickers…"
        hist_map = _bulk_download(tickers, BT_HISTORY_PERIOD, min_rows=BT_WARMUP_BARS + OPTION_DTE_BARS + 2)

        # The live scanner's exact parameters (applied optimizer values + UI
        # toggles) so the analysis covers the setups it would actually trade.
        params = live_params()
        print(f"  [HoldSweep] Using live params: {params}")

        with _hs_lock:
            _hs_state["stage"] = f"Pre-computing indicators + GARCH for {len(hist_map)} tickers…"
        precomp_map = _precompute_many(hist_map, label="HoldSweep")
        with _hs_lock:
            _hs_state["stage"] = f"Analysing {len(precomp_map)} tickers…"

        all_trades = []
        done = 0

        def _sweep_one(tkr):
            try:
                pc = precomp_map.get(tkr)
                if pc is None:
                    return []
                # Indicators are on the FULL 2y history (the 200-day MA and
                # 52-week range need it); only bars after the cutoff are scored.
                idx_start = int(pc["index"].searchsorted(pd.Timestamp(cutoff)))
                if pc["n"] - idx_start < 25:
                    return []
                return run_backtest_ticker(
                    tkr, precomp=pc, idx_start=idx_start, params=params,
                    exits={**live_exit_params(), "hold_days": 21}, record_path=True
                )
            except Exception as _e:
                print(f"  [HoldSweep] {tkr} skipped: {_e}")
                return []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(_sweep_one, t): t for t in tickers}
            for fut in as_completed(futs):
                done += 1
                trades = fut.result()
                all_trades.extend(trades)
                if done % 20 == 0 or done == len(tickers):
                    pct = int(done / len(tickers) * 100)
                    with _hs_lock:
                        _hs_state["stage"] = (
                            f"Processed {done}/{len(tickers)} tickers "
                            f"({pct}%)  — {len(all_trades):,} trades so far"
                        )

        if not all_trades:
            with _hs_lock:
                _hs_state.update(stage="No trades produced — try a longer window")
            return

        agg = _aggregate_hold_sweep(all_trades)
        result = {
            "completed_at": datetime.now().isoformat(),
            "months":       months,
            "universe":     universe,
            "n_trades":     len(all_trades),
            "intervals":    HOLD_SWEEP_INTERVALS,
            "by_strategy":  agg,
        }
        with _hs_lock:
            _hs_state.update(stage=f"Done — {len(all_trades):,} trades analysed", results=result)
        _save_hs_results(result)

    except Exception as e:
        print(f"  [HoldSweep] FAILED: {e}")
        with _hs_lock:
            _hs_state.update(stage=f"Error: {e}", results=None)
    finally:
        with _hs_lock:
            _hs_state["running"] = False


def _load_lr_calibration_on_startup():
    """Restore saved LR calibration weights from disk on startup."""
    try:
        data = _load_json(LR_CALIBRATION_FILE, None, "LR")
        if not isinstance(data, dict):
            return
        with _lr_lock:
            _lr_state["results"] = data
        weights = data.get("weights", {})
        if weights:
            _apply_lr_weights(weights)
            print(f"  [LR] Loaded LR weights (run: {data.get('completed_at','?')[:10]})")
    except Exception as e:
        print(f"  [LR] Could not load {LR_CALIBRATION_FILE}: {e}")


def _load_bt_results():
    """Restore last backtest results from disk on startup."""
    try:
        data = _load_json(BT_FILE, None, "Backtest")
        if not isinstance(data, dict):
            return
        with _bt_lock:
            _bt_state["results"] = data
            _bt_state["stage"]   = f"Loaded from disk ({data.get('completed_at','')[:10]})"
        print(f"  [Backtest] Loaded results from {BT_FILE} "
              f"({data.get('tickers_used','?')} tickers, "
              f"{len(data.get('trades',[]))} trades)")
    except Exception as e:
        print(f"  [Backtest] Could not load {BT_FILE}: {e}")


def _load_calibration_on_startup():
    """Restore saved calibration weights from disk on startup."""
    global _cal_weights
    try:
        data = _load_json(CALIBRATION_FILE, None, "CAL")
        if not isinstance(data, dict):
            return
        with _cal_lock:
            _cal_state["results"] = data
            _cal_state["stage"]   = f"Loaded from disk ({data.get('completed_at','')[:10]})"
        weights = data.get("weights", {})
        if weights:
            _cal_weights = weights
            print(f"  [CAL] Loaded calibration weights (run: {data.get('completed_at','?')[:10]})")
    except Exception as e:
        print(f"  [CAL] Could not load calibration_data.json: {e}")


# ══════════════════════════════════════════════════════════════
#  UNBIASED SIGNAL LOG
#  Every recommendation the bot generates is logged automatically,
#  regardless of whether the user trades it.  After a hold period
#  each entry is resolved against the actual stock price so the
#  calibration engine can learn from ALL signals — not just the
#  ones the user chose to enter.
# ══════════════════════════════════════════════════════════════

_signal_log_lock = threading.Lock()

SIGNAL_LOG_MAX = 5000


def _load_signal_log() -> list:
    """Load signal log from disk (callers hold _signal_log_lock)."""
    data = _load_json(SIGNAL_LOG_FILE, [], "SigLog")
    return data if isinstance(data, list) else []


def _save_signal_log(entries: list):
    """
    Write the signal log atomically (callers hold _signal_log_lock).
    Keeps every pending entry plus the most recent resolved ones, up to
    SIGNAL_LOG_MAX total.  (`resolved[-0:]` used to return the whole list, so
    nothing was ever trimmed.)
    """
    try:
        pending  = [e for e in entries if not e.get("resolved")]
        resolved = [e for e in entries if e.get("resolved")]
        keep     = max(0, SIGNAL_LOG_MAX - len(pending))
        trimmed  = pending + (resolved[-keep:] if keep else [])
        _atomic_json_write(SIGNAL_LOG_FILE, trimmed)
    except Exception as e:
        print(f"  [SigLog] Save failed: {e}")


def _log_new_signals(totals: dict):
    """
    Called at the end of every scan.  Appends one entry per new recommendation
    to the signal log.  Entries that already exist for the same ticker+strategy
    within the last 3 days are skipped (avoids duplicates from live-refresh).
    """
    try:
      # Hold the lock across load → append → save so a concurrent resolver
      # pass can't be overwritten with stale (still-pending) entries.
      with _signal_log_lock:
        entries    = _load_signal_log()
        now_str    = datetime.now().isoformat()
        today      = datetime.now().date()
        # One resolution window for every strategy: the tracker's max-hold rule
        # (HOLD_DAYS trading days), so a logged signal resolves exactly as a tracked
        # position would.  check_after is counted in TRADING days — the simulator
        # needs that many bars after the signal before it can walk the whole hold.
        hold_days  = HOLD_DAYS
        check_date = str(np.busday_offset(np.datetime64(today), hold_days + 1, roll="forward"))

        # Dedup: skip a ticker+strategy that was already logged within the last
        # 3 trading days (resolved or not).  Same-day-only dedup logged the same
        # setup every day of a multi-day oversold stretch, which fed the
        # calibration overlapping, autocorrelated "samples" of one trade.
        recent_keys = set()
        for e in entries:
            try:
                age = int(np.busday_count(datetime.fromisoformat(e["signal_date"]).date(), today))
            except Exception:
                continue
            if age <= 3:
                recent_keys.add((e["ticker"], e["strategy"]))

        STRAT_MAP = {
            "sell_puts":     "sell_put",
            "buy_calls":     "buy_call",
            "buy_puts":      "buy_put",
            "covered_calls": "covered_call",
            "iron_condors":  "iron_condor",
        }

        added = 0
        for bucket, strat_key in STRAT_MAP.items():
            for rec in totals.get(bucket, []):
                ticker = rec.get("ticker", "")
                key    = (ticker, strat_key)
                if key in recent_keys:
                    continue
                entry = {
                    "id":           str(uuid.uuid4()),
                    "ticker":       ticker,
                    "strategy":     strat_key,
                    "signal_date":  now_str,
                    "bar_date":     rec.get("bar_date"),        # the bar whose close is price_at_signal
                    "check_after":  check_date,
                    "hold_days":    hold_days,
                    "price_at_signal": rec.get("price"),
                    "rsi_at_signal":   rec.get("rsi"),
                    "score":           rec.get("score"),
                    "vix_regime":      rec.get("vix_regime"),
                    # Which rule set produced the flags: 2 = the shared engine (regime-adjusted
                    # RSI thresholds, recorded here); older entries used literal 30 / 70.
                    "flags_version":   2,
                    "rsi_os_thr":      rec.get("rsi_os_thr"),
                    "rsi_ob_thr":      rec.get("rsi_ob_thr"),
                    # Market IV vs model vol at signal time — the only place the
                    # real IV-vs-GARCH edge can be measured (no historical IV exists)
                    "iv_at_signal":    (rec.get("option") or {}).get("iv"),
                    "iv_call_atm":     rec.get("iv_call_atm"), "iv_put_atm": rec.get("iv_put_atm"),
                    "iv_skew":         rec.get("iv_skew"),     "iv_vs_garch": rec.get("iv_vs_garch"),
                    "escape_case":     next((k[11:] for k in ("sig_escape_reversion", "sig_escape_momentum", "sig_escape_breakout")
                                             if (rec.get("flags") or {}).get(k)), None),
                    "garch_at_signal": rec.get("garch_vol"),
                    "hv30_at_signal":  rec.get("hv30"),
                    "garch_expand":    rec.get("garch_expand"),
                    "garch_contract":  rec.get("garch_contract"),
                    # Individual signal flags — the engine's own definitions (the same
                    # keys the backtest records), not a separate set of thresholds.
                    **(rec.get("flags") or {
                        "sig_rsi_os":    rec.get("rsi", 100) < 30,
                        "sig_bb_low":    rec.get("price", 0) < rec.get("bb_lower", 0),
                        "sig_rsi_ob":    rec.get("rsi", 0) > 70,
                        "sig_bb_high":   rec.get("price", 0) > rec.get("bb_upper", 0),
                        "sig_macd_bull": rec.get("macd_bull", False),
                        "sig_macd_bear": rec.get("macd_bear", False),
                        "sig_reversal":  rec.get("reversal", False),
                        "sig_vol_ok":    rec.get("vol_confirmed", False),
                        "sig_mfi_os":    rec.get("sig_mfi_os", False),
                        "sig_mfi_ob":    rec.get("sig_mfi_ob", False),
                    }),
                    "resolved":      False,
                    "outcome":       None,
                }
                entries.append(entry)
                recent_keys.add(key)
                added += 1

        if added:
            _save_signal_log(entries)
            print(f"  [SigLog] Logged {added} new recommendations")
    except Exception as e:
        print(f"  [SigLog] Logging failed: {e}")


def _simulate_signal_exit(entry, hist):
    """
    Walk daily through the hold period and find the first sell-signal trigger
    based on simulated OPTION premium P&L (matches the live bot's
    SELL_PROFIT_TARGET=+30% and SELL_STOP_LOSS=-15% thresholds).

    Option premium is estimated daily via Black-Scholes using:
      - K = round(price_at_signal)              (ATM at entry)
      - IV = GARCH estimate on history through signal_date
      - T = remaining hold days / 365

    Returns dict with exit_day, exit_reason, exit_option_pct (the option P&L
    that triggered exit), stock_return_pct (actual stock move), exit_price,
    and win; the max-hold rule closes it at hold_days if nothing else fires.
    Returns the string "pending" when the history does not yet reach hold_days
    bars past the signal (the caller leaves the entry open), and None when the
    signal bar itself cannot be simulated.
    """
    strat       = entry.get("strategy", "")
    entry_price = _as_float(entry.get("price_at_signal"))
    signal_date = (entry.get("bar_date") or entry.get("signal_date") or "")[:10]   # the bar whose close is price_at_signal
    hold_days   = int(entry.get("hold_days", HOLD_DAYS) or HOLD_DAYS)

    if entry_price <= 0 or not signal_date or strat not in ("buy_call", "buy_put", "sell_put", "covered_call"):
        return None
    try:
        entry_dt = pd.Timestamp(signal_date)
    except Exception:
        return None

    # Same engine as the backtest: indicator frame on UNADJUSTED daily bars
    # (indices for the RSI/BB exits), the same option model and the same walk.
    fr = indicator_frame(hist, spy_closes=None, sym=entry.get("ticker"), garch=None)
    pos = int(fr["index"].searchsorted(entry_dt, side="right")) - 1   # the last bar ≤ the signal date
    if pos < 0 or pos + 1 >= fr["n"]:
        return None                                   # signal bar not in history / nothing after it
    hv = pd.Series(fr["hv30"]) * 1.15
    rets = pd.Series(fr["price"]).pct_change()
    hv = hv.fillna(rets.rolling(20, min_periods=5).std() * np.sqrt(252) * 1.15).fillna(0.30).clip(0.05, 2.0)
    sigma = float(hv.iloc[pos])

    opt_type  = "call" if strat in ("buy_call", "covered_call") else "put"
    K         = _bt_strike(entry_price * (1.05 if strat == "covered_call" else 1.0))   # the scanner's covered call is ~5% OTM
    entry_opt = _bt_option_price(entry_price, K, OPTION_DTE_TARGET / 365.0, sigma, opt_type)
    if entry_opt < 0.01:
        return None

    ex = _bt_walk_exit(fr, pos, K, sigma, {**live_exit_params(), "hold_days": hold_days}, strat, entry_opt)
    if ex is None:
        return "pending"                              # hold period not complete yet (in trading days)
    j = pos + ex["exit_offset"]
    close = float(fr["price"][j])
    return {
        "exit_day":         fr["dates"][j] if "dates" in fr else str(fr["index"][j])[:10],
        "exit_day_idx":     ex["exit_offset"],
        "exit_reason":      ex["exit_reason"],
        "exit_option_pct":  round(ex["pnl_pct"], 2),
        "stock_return_pct": round((close - entry_price) / entry_price * 100, 2),
        "exit_price":       round(close, 2),
        "win":              bool(ex["pnl_pct"] > 0),
    }


def _resolve_signal_log():
    """
    Background job (runs every 6 hours):
    - ALL pending entries: refresh live stock price → store current_return_pct
      so the UI can show live unrealized P&L immediately after signal date.
    - Entries past check_after: simulate daily sell-signal exit (NOT fixed
      end-of-hold). First take-profit or stop-loss trigger wins.

    Safe pattern: snapshot → network work → re-load + merge by UUID.
    """
    try:
        with _signal_log_lock:
            snapshot = _load_signal_log()

        today    = datetime.now().date()
        pending  = [e for e in snapshot if not e.get("resolved")]

        if not pending:
            return

        # Batch all pending by ticker (one API call per ticker)
        ticker_groups: dict = {}
        for e in pending:
            ticker_groups.setdefault(e["ticker"], []).append(e)

        all_updates: dict = {}   # id → fields to merge
        resolved_count = 0
        live_count     = 0

        for sym, group in ticker_groups.items():
            try:
                # Fetch history from the earliest signal_date in the group through today.
                # Needed so the simulator can walk each hold period day-by-day.
                earliest_date = min(
                    (e.get("signal_date") or "")[:10] for e in group if e.get("signal_date")
                )
                if not earliest_date:
                    continue
                try:
                    earliest_dt = datetime.strptime(earliest_date, "%Y-%m-%d").date()
                except Exception:
                    continue
                # Start well before the earliest entry so the RSI/BB exit indicators
                # are valid on the signal bar; UNADJUSTED prices like every other path
                # (the default auto_adjust=True compared adjusted closes with the
                # unadjusted price_at_signal across ex-dividend dates).
                start_dt = earliest_dt - timedelta(days=120)
                hist = _clean_hist(_yf_call(yf.Ticker(sym).history, start=start_dt.isoformat(), auto_adjust=False))
                if hist is None or hist.empty:
                    continue
                cur_price = float(hist["Close"].iloc[-1])

                for e in group:
                    entry_price = e.get("price_at_signal")
                    past_hold   = (
                        e.get("check_after")
                        and datetime.strptime(e["check_after"][:10], "%Y-%m-%d").date() <= today
                    )

                    if not entry_price or entry_price <= 0:
                        # Can't compute P&L — only mark resolved if hold is done
                        if past_hold:
                            all_updates[e["id"]] = {"resolved": True, "outcome": "no_price"}
                            resolved_count += 1
                        continue

                    stock_return_pct = round((cur_price - entry_price) / entry_price * 100, 2)
                    update = {
                        "current_price":      round(cur_price, 2),
                        "current_return_pct": stock_return_pct,
                        "live_updated_at":    datetime.now().isoformat(),
                    }

                    if past_hold:
                        # Simulate day-by-day sell-signal exit instead of fixed end-of-hold —
                        # the backtest's walk: target / stop / trailing / stagnation / RSI-BB / max hold.
                        sim = _simulate_signal_exit(e, hist)
                        if sim == "pending":
                            live_count += 1           # not enough trading days yet — stays open, re-checked next run
                        elif sim is not None:
                            resolved_count += 1
                            update.update({
                                "resolved":          True,
                                "resolved_at":       datetime.now().isoformat(),
                                "exit_day":          sim["exit_day"],
                                "exit_day_idx":      sim["exit_day_idx"],
                                "exit_reason":       sim["exit_reason"],
                                "option_return_pct": sim["exit_option_pct"],
                                "stock_return_pct":  sim["stock_return_pct"],
                                "price_at_resolve":  sim["exit_price"],
                                "win":               sim["win"],
                                "outcome":           "win" if sim["win"] else "loss",
                            })
                        else:
                            # Simulator couldn't run (no history for the signal bar) — fixed-hold fallback on the stock move
                            resolved_count += 1
                            strat = e.get("strategy", "")
                            if strat == "buy_call":
                                win = stock_return_pct > 3.0
                            elif strat == "buy_put":
                                win = stock_return_pct < -3.0
                            elif strat in ("sell_put", "covered_call"):
                                win = stock_return_pct > -5.0
                            else:
                                win = stock_return_pct > 0
                            update.update({
                                "resolved":         True,
                                "resolved_at":      datetime.now().isoformat(),
                                "exit_reason":      "no_data_fallback",
                                "price_at_resolve": round(cur_price, 2),
                                "stock_return_pct": stock_return_pct,
                                "win":              win,
                                "outcome":          "win" if win else "loss",
                            })
                    else:
                        live_count += 1

                    all_updates[e["id"]] = update

            except Exception as ex:
                print(f"  [SigLog] Price fetch error for {sym}: {ex}")

        if all_updates:
            with _signal_log_lock:
                entries = _load_signal_log()
                for entry in entries:
                    updates = all_updates.get(entry.get("id"))
                    if updates:
                        entry.update(updates)
                _save_signal_log(entries)
            print(f"  [SigLog] Updated {len(all_updates)} entries "
                  f"({resolved_count} resolved, {live_count} live P&L refresh)")
    except Exception as e:
        print(f"  [SigLog] Resolve job failed: {e}")


def _signal_log_summary() -> dict:
    """Return aggregate stats from the signal log for the calibration display."""
    try:
        entries   = _load_signal_log()
        resolved  = [e for e in entries if e.get("resolved") and e.get("outcome") not in (None, "no_price")]
        pending   = [e for e in entries if not e.get("resolved")]
        if not resolved:
            return {"resolved": 0, "pending": len(pending), "by_strategy": {}, "sig_quality": []}

        by_strat: dict = {}
        for strat in ("buy_call", "buy_put", "sell_put", "covered_call"):
            st = [e for e in resolved if e.get("strategy") == strat]
            if st:
                wins = [e for e in st if e.get("win")]
                by_strat[strat] = {
                    "count":    len(st),
                    "win_rate": round(len(wins) / len(st) * 100, 1),
                }

        # Per-signal quality using actual live recommendations
        SIGNAL_LABELS = {
            "sig_rsi_os":    "RSI Oversold",
            "sig_bb_low":    "Below BB Lower",
            "sig_reversal":  "RSI Turning Up",
            "sig_macd_bull": "MACD Bull Cross",
            "sig_vol_ok":    "Volume Confirmed",
            "sig_rsi_ob":    "RSI Overbought",
            "sig_bb_high":   "Above BB Upper",
            "sig_macd_bear": "MACD Bear Cross",
        }
        sig_rows = []
        for flag, label in SIGNAL_LABELS.items():
            subset = [e for e in resolved if e.get(flag)]
            if len(subset) < 3:
                continue
            wins = [e for e in subset if e.get("win")]
            sig_rows.append({
                "signal":   label,
                "count":    len(subset),
                "win_rate": round(len(wins) / len(subset) * 100, 1),
            })
        sig_rows.sort(key=lambda x: x["win_rate"], reverse=True)

        return {
            "resolved":    len(resolved),
            "pending":     len(pending),
            "by_strategy": by_strat,
            "sig_quality": sig_rows,
        }
    except Exception:
        return {"resolved": 0, "pending": 0, "by_strategy": {}, "sig_quality": []}


def _schedule_signal_log_resolver():
    """Daemon thread: resolve pending signal log entries every 6 hours."""
    while True:
        try:
            _resolve_signal_log()   # runs once immediately on startup, then every 6 h
        except Exception as e:
            print(f"  [SigLog] Resolver crashed: {e}")   # keep the thread alive
        time.sleep(6 * 3600)


# ══════════════════════════════════════════════════════════════
#  LIVE REFRESH — re-checks only current signal tickers
# ══════════════════════════════════════════════════════════════

def _live_refresh_loop():
    """
    Runs in a daemon thread after a scan completes.
    Every _live_refresh_mins minutes it re-runs scan_ticker() on the
    tickers currently showing signals and updates the results in-place.
    Stops when _live_refresh_stop is set (e.g. a new full scan starts).
    """
    # Capture THIS loop's stop event.  _start_live_refresh() swaps the global
    # for a fresh Event; a loop still mid-refresh that re-read the global would
    # see the new, unset event and never stop — leaving two loops running.
    stop = _live_refresh_stop

    print(f"  [Live] Refresh loop started — interval: {_live_refresh_mins} min")
    while not stop.wait(timeout=_live_refresh_mins * 60):
        with scan_lock:
            if scan_state["running"]:
                continue   # full scan in progress — skip refresh
            current = scan_state["results"]
            condors = list(current.get("iron_condors", []))

        signal_tickers = sorted({r["ticker"] for k, strat in current.items()
                                 if k != "iron_condors" for r in strat})
        if not signal_tickers:
            print("  [Live] No signal tickers to refresh — waiting")
            continue

        print(f"  [Live] Refreshing {len(signal_tickers)} signal ticker(s): {', '.join(signal_tickers)}")

        # Bulk download for the small signal set (per-ticker fallback inside scan_ticker)
        bulk_hist = _bulk_download(signal_tickers, LIVE_HISTORY_PERIOD)

        new_totals = {"sell_puts": [], "buy_calls": [], "buy_puts": [],
                      "covered_calls": [], "iron_condors": condors}   # condors come from their own scan — keep them
        for sym in signal_tickers:
            if stop.is_set():
                return
            try:
                res_raw = scan_ticker(sym, bulk_hist.get(sym), False, prefetch_fn=_prefetch_ticker)
                if res_raw:
                    res = res_raw.get("_signals", {})
                    if "sell_put"     in res: new_totals["sell_puts"].append(res["sell_put"])
                    if "buy_call"     in res: new_totals["buy_calls"].append(res["buy_call"])
                    if "buy_put"      in res: new_totals["buy_puts"].append(res["buy_put"])
                    if "covered_call" in res: new_totals["covered_calls"].append(res["covered_call"])
            except Exception as e:
                print(f"  [Live] {sym} error: {e}")

        for key in new_totals:
            new_totals[key] = sorted(new_totals[key], key=lambda x: x["score"], reverse=True)

        refreshed_at = datetime.now().isoformat()
        dropped = [sym for sym in signal_tickers
                   if not any(r["ticker"] == sym
                              for strat in new_totals.values() for r in strat)]
        if dropped:
            print(f"  [Live] Dropped (signal gone): {', '.join(dropped)}")

        with scan_lock:
            if stop.is_set() or scan_state["running"]:
                return          # a full scan started meanwhile — don't clobber its results
            scan_state["results"]      = new_totals
            scan_state["last_refresh"] = refreshed_at
        _save_prefetch_cache()

        print(f"  [Live] Refresh complete at {refreshed_at}")

    print("  [Live] Refresh loop stopped.")


_live_refresh_thread = None


def _start_live_refresh():
    """Start (or restart) the live refresh daemon thread."""
    global _live_refresh_stop, _live_refresh_thread
    _live_refresh_stop.set()          # stop any existing loop
    _live_refresh_stop = threading.Event()
    _live_refresh_thread = threading.Thread(target=_live_refresh_loop, daemon=True)
    _live_refresh_thread.start()


def _live_refresh_active():
    t = _live_refresh_thread
    return bool(t and t.is_alive() and not _live_refresh_stop.is_set())


def _auto_rescan_loop():
    """
    Daemon thread: re-runs the full universe scan every _auto_rescan_hours hours.
    Schedules the next run from the moment the PREVIOUS scan FINISHES, so a
    slow scan never causes a backlog.  Skips a cycle if a scan is already running.
    Sleeps in 60 s steps so a change to the interval from the UI takes effect
    without waiting out the old timer.
    """
    global _next_rescan_at
    while True:
        _next_rescan_at = datetime.now() + timedelta(hours=_auto_rescan_hours)
        print(f"  [AutoScan] Next full scan in {_auto_rescan_hours}h "
              f"(~{_next_rescan_at.strftime('%H:%M')})")
        while datetime.now() < _next_rescan_at:
            time.sleep(60)
            _next_rescan_at = min(_next_rescan_at,
                                  datetime.now() + timedelta(hours=_auto_rescan_hours))
        with scan_lock:
            if scan_state.get("running", False):
                already_running = True
            else:
                already_running = False
                scan_state["running"] = True      # claim the slot, same as /api/scan/start
        if already_running:
            print("  [AutoScan] Skipping — scan already in progress")
            continue
        print("  [AutoScan] Triggering scheduled full scan…")
        try:
            _run_scan()   # blocks until scan completes; loop then reschedules
        except Exception as e:
            print(f"  [AutoScan] Scheduled scan failed: {e}")



# ══════════════════════════════════════════════════════════════
#  BACKGROUND SCAN THREAD
# ══════════════════════════════════════════════════════════════

_STRAT_BUCKET = {"sell_put": "sell_puts", "buy_call": "buy_calls", "buy_put": "buy_puts",
                 "covered_call": "covered_calls", "iron_condor": "iron_condors"}


def _empty_totals():
    return {k: [] for k in _STRAT_BUCKET.values()}


def _run_scan(mode=None):
    """
    Full-universe scan.  The caller (/api/scan/start or the auto-rescan loop)
    has already set scan_state["running"] = True; this function owns clearing
    it — in a finally, so a crash can never leave the UI stuck on "scanning".
    """
    global SCAN_MODE
    # Stop any running live refresh before starting a full scan
    _live_refresh_stop.set()
    totals  = _empty_totals()
    t0      = time.time()
    try:
        # Refresh 6-month T-bill rate (cached 6h — near-instant if recent)
        fetch_risk_free_rate()

        if mode in ("focus", "full"):
            SCAN_MODE = mode
        tickers = get_universe()

        with scan_lock:
            scan_state.update(running=True, progress=0, total=len(tickers), results=_empty_totals())

        # Market-context series (VIX, Fed, SPY) once, before the workers start.
        _warm_market_caches()

        # ── Bulk OHLCV download (single API call for all tickers) ────────────
        print(f"  [Bulk download] Fetching {LIVE_HISTORY_PERIOD} OHLCV for {len(tickers)} tickers…")
        bulk_hist = _bulk_download(tickers, LIVE_HISTORY_PERIOD)
        print(f"  [Bulk download] Done — {len(bulk_hist)}/{len(tickers)} tickers have data")
        _set_live_breadth(bulk_hist)                       # today's breadth for the escape gate (live_market_ctx)

        # ── Scan ─────────────────────────────────────────────────────────────
        # The indicator gate runs on the bulk history alone.  Only tickers that
        # pass it fetch info / calendar / option chain (via prefetch_fn), so the
        # ~500 chain downloads the old prefetch phase did up-front are gone.
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(scan_ticker, sym, bulk_hist.get(sym), False, None, _prefetch_ticker): sym
                       for sym in tickers}
            for fut in as_completed(futures):
                done += 1
                sym = futures[fut]
                try:
                    raw = fut.result()
                except Exception as e:
                    raw = {"_signals": {}, "_log": f"  [Scan] {sym:6s} ✗  error: {e}"}
                log = raw.get("_log") if raw else None
                res = raw.get("_signals", {}) if raw else {}
                if log:
                    print(log)
                for strat, bucket in _STRAT_BUCKET.items():
                    if strat in res:
                        totals[bucket].append(res[strat])

                with scan_lock:
                    scan_state["progress"] = done
                    # Stream partial results live
                    scan_state["results"] = {
                        k: sorted(v, key=lambda x: x["score"], reverse=True)
                        for k, v in totals.items()
                    }
        _save_prefetch_cache()

        for key in totals:
            totals[key] = sorted(totals[key], key=lambda x: x["score"], reverse=True)

        with scan_lock:
            scan_state.update(last_scan=datetime.now().isoformat(), results=totals)
        print(f"\n  Scan complete in {time.time() - t0:.0f}s — " +
              "  ".join(f"{k}:{len(v)}" for k, v in totals.items()))

        # Log all recommendations to the unbiased signal log (snapshot — the
        # live refresh may replace scan_state["results"] while this runs).
        threading.Thread(target=_log_new_signals, args=(json.loads(json.dumps(totals, default=_json_default)),),
                         daemon=True).start()
    except Exception as e:
        print(f"  [Scan] FAILED: {e}")
    finally:
        with scan_lock:
            scan_state["running"] = False

    # Also refresh sell signals on all tracked positions after every scan
    try:
        _run_tracked_check()
    except Exception as e:
        print(f"[Track] post-scan check failed: {e}")

    # ── Kick off live refresh loop ───────────────────────────────
    _start_live_refresh()


# ══════════════════════════════════════════════════════════════
#  TRADE QUEUE & GUARDRAIL
# ══════════════════════════════════════════════════════════════

_queue_lock = threading.Lock()   # guards trade_queue.json and trade_history.json


def load_queue():
    data = _load_json(QUEUE_FILE, [], "Queue")
    return data if isinstance(data, list) else []


def save_queue(q):
    _atomic_json_write(QUEUE_FILE, q)


def load_history():
    data = _load_json(HISTORY_FILE, [], "History")
    return data if isinstance(data, list) else []


def save_history(h):
    _atomic_json_write(HISTORY_FILE, h)


def calc_trade_cost(trade):
    """
    Returns the dollar amount that counts against the guardrail.
    • BUY  option  → debit = qty × 100 × ask
    • SELL put     → capital reserved = qty × 100 × strike
    • SELL call (covered) → $0 additional capital
    """
    qty    = max(1, _as_int(trade.get("qty"), 1))
    strat  = _norm_strategy(trade.get("strategy", ""))
    opt    = trade.get("option") or {}
    strike = _as_float(opt.get("strike"))
    ask    = _as_float(opt.get("ask"))

    if strat == "buy_call" or strat == "buy_put":
        return qty * 100 * ask
    elif strat == "sell_put":
        return qty * 100 * strike
    elif strat == "covered_call":
        return 0.0   # no additional capital required
    elif strat == "iron_condor":
        # Max loss = spread_width - net_credit (per contract × qty)
        put_width  = _as_float(trade.get("put_width"))
        call_width = _as_float(trade.get("call_width"))
        net_credit = _as_float(opt.get("net_credit"), ask)
        spread_w   = max(put_width, call_width) if (put_width or call_width) else strike * 0.05
        return qty * 100 * max(spread_w - net_credit, 0)
    return qty * 100 * ask


def _norm_strategy(s):
    """'sell_puts' / 'Sell Put' / 'sell_put' → 'sell_put' (the dashboard sends plural bucket names)."""
    s = str(s or "").strip().lower().replace(" ", "_")
    return {"sell_puts": "sell_put", "buy_calls": "buy_call", "buy_puts": "buy_put",
            "covered_calls": "covered_call", "iron_condors": "iron_condor"}.get(s, s)


def guardrail_check(trade):
    """Returns (passes: bool, cost: float, message: str)."""
    cost = calc_trade_cost(trade)
    if cost > GUARDRAIL_LIMIT:
        return (False, cost,
                f"This trade requires ~${cost:,.2f} in capital, which exceeds your "
                f"${GUARDRAIL_LIMIT:,.2f} guardrail limit. Type CONFIRM to proceed anyway.")
    return (True, cost, "ok")


# ══════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/scan/start", methods=["POST"])
def api_scan_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", SCAN_MODE)
    with scan_lock:
        if scan_state["running"]:
            return jsonify({"ok": False, "message": "Scan already running"}), 409
        # Mark running immediately so the JS poll doesn't miss the start
        scan_state["running"] = True
    t = threading.Thread(target=_run_scan, args=(mode,), daemon=True)
    t.start()
    return jsonify({"ok": True, "mode": mode})


@app.route("/api/scan/status")
def api_scan_status():
    with scan_lock:
        return jsonify({
            "running":         scan_state["running"],
            "progress":        scan_state["progress"],
            "total":           scan_state["total"],
            "last_scan":       scan_state["last_scan"],
            "last_refresh":    scan_state.get("last_refresh"),
            "live_active":     _live_refresh_active(),
            "live_mins":       _live_refresh_mins,
            "next_rescan":     _next_rescan_at.strftime("%H:%M") if _next_rescan_at else None,
            "rescan_hours":    _auto_rescan_hours,
            "counts": {k: len(v) for k, v in scan_state["results"].items()},
        })


@app.route("/api/scan/green-candle", methods=["POST"])
def api_green_candle_toggle():
    """Toggle green candle confirmation on/off. Pass {enabled: true/false}."""
    global REQUIRE_GREEN_CANDLE
    data = request.get_json(silent=True) or {}
    REQUIRE_GREEN_CANDLE = bool(data.get("enabled", not REQUIRE_GREEN_CANDLE))
    print(f"  [Config] REQUIRE_GREEN_CANDLE = {REQUIRE_GREEN_CANDLE}")
    return jsonify({"ok": True, "green_candle": REQUIRE_GREEN_CANDLE})


@app.route("/api/scan/green-candle", methods=["GET"])
def api_green_candle_status():
    return jsonify({"green_candle": REQUIRE_GREEN_CANDLE})


@app.route("/api/scan/red-candle", methods=["POST"])
def api_red_candle_toggle():
    """Toggle red candle confirmation on/off. Pass {enabled: true/false}."""
    global REQUIRE_RED_CANDLE
    data = request.get_json(silent=True) or {}
    REQUIRE_RED_CANDLE = bool(data.get("enabled", not REQUIRE_RED_CANDLE))
    print(f"  [Config] REQUIRE_RED_CANDLE = {REQUIRE_RED_CANDLE}")
    return jsonify({"ok": True, "red_candle": REQUIRE_RED_CANDLE})


@app.route("/api/scan/red-candle", methods=["GET"])
def api_red_candle_status():
    return jsonify({"red_candle": REQUIRE_RED_CANDLE})


@app.route("/api/config/risk-budget", methods=["GET"])
def api_risk_budget_get():
    return jsonify({"risk_budget": RISK_BUDGET_PER_TRADE,
                    "stop_loss_pct": SELL_STOP_LOSS,
                    "profit_target_pct": SELL_PROFIT_TARGET})


@app.route("/api/config/risk-budget", methods=["POST"])
def api_risk_budget_set():
    global RISK_BUDGET_PER_TRADE
    data = request.get_json(silent=True) or {}
    try:
        val = float(data.get("risk_budget", 0))
        if val < 0:
            return jsonify({"ok": False, "message": "Budget must be 0 or greater"})
        RISK_BUDGET_PER_TRADE = round(val, 2)
        print(f"  [Config] RISK_BUDGET_PER_TRADE = ${RISK_BUDGET_PER_TRADE}")
        return jsonify({"ok": True, "risk_budget": RISK_BUDGET_PER_TRADE})
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "Invalid value"})


@app.route("/api/refresh/set", methods=["POST"])
def api_refresh_set():
    """Set or change live refresh interval (minutes). Pass {mins: N} or {stop: true}."""
    global _live_refresh_mins
    data = request.get_json(silent=True) or {}
    if data.get("stop"):
        _live_refresh_stop.set()
        return jsonify({"ok": True, "live_active": False})
    mins = int(data.get("mins", _live_refresh_mins))
    if mins < 1:
        mins = 1
    _live_refresh_mins = mins
    # Restart loop with new interval
    _start_live_refresh()
    return jsonify({"ok": True, "live_active": True, "live_mins": _live_refresh_mins})


# ── Iron condor on-demand scan ────────────────────────────────
_condor_state = {"running": False, "progress": 0, "total": 0}
_condor_lock  = threading.Lock()

def _run_condor_scan():
    """Caller has set _condor_state['running']; cleared here in a finally."""
    condors = []
    done    = 0
    try:
        tickers = get_universe()
        with _condor_lock:
            _condor_state.update(progress=0, total=len(tickers))
        print(f"  [Condor] Scanning {len(tickers)} tickers for iron condors…")
        _warm_market_caches()
        bulk_hist = _bulk_download(tickers, LIVE_HISTORY_PERIOD)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(scan_ticker, sym, bulk_hist.get(sym), True, None, _prefetch_ticker): sym
                       for sym in tickers}
            for fut in as_completed(futures):
                done += 1
                try:
                    raw = fut.result()
                except Exception as e:
                    print(f"  [Condor] {futures[fut]} error: {e}")
                    raw = None
                if raw and "iron_condor" in raw.get("_signals", {}):
                    condors.append(raw["_signals"]["iron_condor"])
                with _condor_lock:
                    _condor_state["progress"] = done
        _save_prefetch_cache()

        condors = sorted(condors, key=lambda x: x["score"], reverse=True)
        with scan_lock:
            scan_state["results"]["iron_condors"] = condors
        print(f"  [Condor] Done — {len(condors)} iron condor(s) found")
    except Exception as e:
        print(f"  [Condor] FAILED: {e}")
    finally:
        with _condor_lock:
            _condor_state.update(running=False, progress=done)


def _start_job(state, lock, target, args=(), name="job"):
    """
    Check-and-claim a background job's `running` flag atomically, then start
    its thread.  Returns a 409 response tuple when already running, else None.
    Setting the flag here (not inside the thread) closes two gaps: a double
    POST starting two jobs, and the UI's first status poll seeing running=False
    while the thread was still doing its setup.
    """
    with lock:
        if state.get("running"):
            return jsonify({"ok": False, "message": f"{name} already running"}), 409
        state["running"] = True
        state["stage"]   = "Starting…"
    threading.Thread(target=target, args=args, daemon=True).start()
    return None


@app.route("/api/condor/scan", methods=["POST"])
def api_condor_scan():
    busy = _start_job(_condor_state, _condor_lock, _run_condor_scan, name="Condor scan")
    return busy or jsonify({"ok": True})


@app.route("/api/condor/status")
def api_condor_status():
    with _condor_lock:
        return jsonify(_condor_state)


@app.route("/api/backtest/run", methods=["POST"])
def api_backtest_run():
    data      = request.get_json(silent=True) or {}
    hold_days = max(1, min(OPTION_DTE_BARS, _as_int(data.get("hold_days"), 21)))   # the contract expires after 21 bars
    ticker    = str(data.get("ticker") or "").strip().upper()
    mode      = data.get("mode", "focus")   # "ticker" | "focus" | "full" | "signals"
    if mode == "ticker" and ticker:
        tickers = [ticker]
    elif mode == "full":
        tickers = get_universe(force_mode="full")   # "Full" must not depend on the scan-mode toggle
    elif mode == "signals":
        # Pull unique tickers from whatever is currently showing signals
        with scan_lock:
            tickers = sorted({r["ticker"]
                              for strat in scan_state["results"].values()
                              for r in strat})
        if not tickers:
            return jsonify({"ok": False, "message": "No signals found — run a scan first"}), 400
    else:
        tickers = list(dict.fromkeys(FOCUS_LIST))
    busy = _start_job(_bt_state, _bt_lock, _run_backtest_job, (tickers, hold_days), name="Backtest")
    return busy or jsonify({"ok": True, "tickers": len(tickers), "hold_days": hold_days})


@app.route("/api/backtest/status")
def api_backtest_status():
    with _bt_lock:
        s = dict(_bt_state)
    # Don't send full results in status poll — just summary + counts
    if s.get("results"):
        s["summary"]      = s["results"].get("summary", {})
        s["trade_count"]  = len(s["results"].get("trades", []))
        del s["results"]
    return jsonify(s)


@app.route("/api/backtest/results")
def api_backtest_results():
    with _bt_lock:
        r = _bt_state.get("results")
    return jsonify(r or {})


@app.route("/api/vix")
def api_vix():
    return jsonify({"vix": get_vix(), "regime": vix_regime(get_vix())})


@app.route("/api/bs-analyze", methods=["POST"])
def api_bs_analyze():
    """
    Full Black-Scholes breakdown for the B-S Model tab.
    Expects: { ticker, stock_price, strike, dte, iv_pct, market_price, option_type }
    Returns: fair_value, greeks, intrinsic, extrinsic, vol comparison, interpretation
    """
    data        = request.get_json(silent=True) or {}
    ticker      = str(data.get("ticker") or "").upper().strip()
    # Blank form fields arrive as null — _as_float turns those into 0 instead of a 500
    S           = _as_float(data.get("stock_price"))
    K           = _as_float(data.get("strike"))
    dte         = _as_float(data.get("dte"))
    iv_pct      = _as_float(data.get("iv_pct"))
    mkt_price   = _as_float(data.get("market_price"))
    opt_type    = "put" if data.get("option_type") == "put" else "call"

    if S <= 0 or K <= 0 or dte <= 0 or iv_pct <= 0:
        return jsonify({"ok": False, "error": "Please fill in all fields with valid values."})

    T     = dte / 365.0
    sigma = iv_pct / 100.0
    r     = RISK_FREE_RATE

    try:
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if opt_type == "call":
            fair_value = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
            delta      = norm.cdf(d1)
            intrinsic  = max(0.0, S - K)
        else:
            fair_value = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta      = -norm.cdf(-d1)
            intrinsic  = max(0.0, K - S)

        extrinsic = max(0.0, fair_value - intrinsic)

        # Greeks
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        vega  = S * norm.pdf(d1) * sqrt(T) / 100          # per 1% IV move
        # Rate term is −rK·e^(−rT)·N(d2) for calls and +rK·e^(−rT)·N(−d2) for puts
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt(T))
                 + (-r * K * exp(-r * T) * norm.cdf(d2) if opt_type == "call"
                    else r * K * exp(-r * T) * norm.cdf(-d2))) / 365
        rho   = (K * T * exp(-r * T) * (norm.cdf(d2) if opt_type=="call" else -norm.cdf(-d2))) / 100

        # GARCH vol comparison
        garch_vol_pct = None
        vol_verdict   = "unavailable"
        vol_diff_pct  = None
        if ticker:
            try:
                _, hist_data = _fetch_history(ticker, "6mo")
                if hist_data is not None and len(hist_data) >= 60:
                    gv = calc_garch_vol(hist_data["Close"])
                    garch_vol_pct = round(gv * 100, 1)
                    vol_diff_pct  = round(iv_pct - garch_vol_pct, 1)
                    if vol_diff_pct > 10:
                        vol_verdict = "overpriced"
                    elif vol_diff_pct < -10:
                        vol_verdict = "underpriced"
                    else:
                        vol_verdict = "fair"
            except Exception:
                pass

        # Overall mispricing vs market price
        mispricing_pct = None
        mispricing_label = None
        if mkt_price > 0:
            mispricing_pct = round((mkt_price - fair_value) / fair_value * 100, 1)
            if mispricing_pct > 5:
                mispricing_label = "overpriced"
            elif mispricing_pct < -5:
                mispricing_label = "underpriced"
            else:
                mispricing_label = "fairly priced"

        # Plain-English interpretation
        lines = []
        if opt_type == "call":
            lines.append(f"This is a <b>{'in-the-money' if S > K else 'out-of-the-money'} call</b>. "
                         f"Intrinsic value is <b>${intrinsic:.2f}</b> — {'the stock is already above the strike, so you are paying for real value plus time.' if intrinsic > 0 else 'the stock has not reached the strike yet, so you are paying entirely for the possibility of a move up.'}")
        else:
            lines.append(f"This is a <b>{'in-the-money' if K > S else 'out-of-the-money'} put</b>. "
                         f"Intrinsic value is <b>${intrinsic:.2f}</b> — {'the strike is above the stock price, so part of your premium is real downside protection.' if intrinsic > 0 else 'the stock is above the strike, so you are paying entirely for the possibility of a move down.'}")

        lines.append(f"Time value (extrinsic) is <b>${extrinsic:.2f}</b> — this decays to zero by expiry regardless of direction. "
                     f"At {dte:.0f} DTE, theta is <b>${theta:.3f}/day</b>, accelerating as expiry approaches.")

        if vol_verdict == "overpriced":
            lines.append(f"⚠️ <b>Vol is overpriced:</b> Market IV is {iv_pct:.1f}% vs GJR-GARCH estimate of {garch_vol_pct:.1f}% (+{vol_diff_pct:.1f}%). "
                         f"The market is pricing in more future movement than our model expects. "
                         f"{'Good for selling this option — you collect inflated premium.' if opt_type in ['put'] else 'Be cautious buying — you may be overpaying for vol that does not materialise.'}")
        elif vol_verdict == "underpriced":
            lines.append(f"✅ <b>Vol is underpriced:</b> Market IV is {iv_pct:.1f}% vs GJR-GARCH estimate of {garch_vol_pct:.1f}% ({vol_diff_pct:.1f}%). "
                         f"The market is underestimating how much this stock moves. "
                         f"{'Good for buying — you are getting cheap vol.' if opt_type in ['call','put'] else ''}")
        elif vol_verdict == "fair":
            lines.append(f"✅ <b>Vol is fairly priced:</b> Market IV ({iv_pct:.1f}%) is close to GARCH estimate ({garch_vol_pct:.1f}%). "
                         f"No edge from a volatility mispricing standpoint — the trade lives or dies on direction.")

        if mispricing_label == "overpriced":
            lines.append(f"📌 The market price (${mkt_price:.2f}) is <b>{mispricing_pct:.1f}% above</b> BS fair value (${fair_value:.2f}). "
                         f"You are paying above theoretical fair value — make sure the directional case is strong.")
        elif mispricing_label == "underpriced":
            lines.append(f"📌 The market price (${mkt_price:.2f}) is <b>{abs(mispricing_pct):.1f}% below</b> BS fair value (${fair_value:.2f}). "
                         f"You are buying below theoretical fair value — positive edge on entry.")

        lines.append(f"Delta of <b>{delta:.3f}</b> means for every $1 the stock moves, this option moves ~${abs(delta):.2f}. "
                     f"Vega of <b>{vega:.3f}</b> means a 1% rise in IV adds ~${vega:.3f} to the option price.")

        return jsonify({
            "ok":             True,
            "fair_value":     round(fair_value, 3),
            "intrinsic":      round(intrinsic, 3),
            "extrinsic":      round(extrinsic, 3),
            "mispricing_pct": mispricing_pct,
            "mispricing_label": mispricing_label,
            "garch_vol_pct":  garch_vol_pct,
            "vol_diff_pct":   vol_diff_pct,
            "vol_verdict":    vol_verdict,
            "greeks": {
                "delta": round(delta, 4),
                "gamma": round(gamma, 5),
                "theta": round(theta, 4),
                "vega":  round(vega,  4),
                "rho":   round(rho,   4),
            },
            "interpretation": " ".join(f"<p>{l}</p>" for l in lines),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ── Walk-forward optimizer routes ─────────────────────────────
@app.route("/api/optimizer/run", methods=["POST"])
def api_optimizer_run():
    data   = request.get_json(silent=True) or {}
    mode   = data.get("mode", "focus")
    ticker = str(data.get("ticker") or "").strip().upper()
    if mode == "ticker" and ticker:
        tickers = [ticker]
    elif mode == "signals":
        with scan_lock:
            tickers = sorted({r["ticker"]
                              for strat in scan_state["results"].values()
                              for r in strat})
        if not tickers:
            return jsonify({"ok": False, "message": "No signals — run a scan first"}), 400
    elif mode == "full":
        tickers = get_universe(force_mode="full")
    else:
        tickers = list(dict.fromkeys(FOCUS_LIST))
    busy = _start_job(_wfo_state, _wfo_lock, _run_wfo_job, (tickers,), name="Optimizer")
    return busy or jsonify({"ok": True, "tickers": len(tickers)})


@app.route("/api/optimizer/status")
def api_optimizer_status():
    with _wfo_lock:
        s = dict(_wfo_state)
    if s.get("results"):
        s["best"]         = s["results"].get("best")
        s["leaderboard"]  = s["results"].get("leaderboard", [])[:5]
        s["completed_at"] = s["results"].get("completed_at")
        del s["results"]
    return jsonify(s)


@app.route("/api/optimizer/results")
def api_optimizer_results():
    with _wfo_lock:
        r = _wfo_state.get("results")
    return jsonify(r or {})


@app.route("/api/optimizer/apply", methods=["POST"])
def api_optimizer_apply():
    """Apply the best found parameters to the live scanner config."""
    with _wfo_lock:
        r = _wfo_state.get("results")
    if not r or not r.get("suggestion"):
        return jsonify({"ok": False, "message": "No optimization results yet"}), 400
    applied = _apply_wfo_params(r["suggestion"])
    print(f"  [WFO] Applied best params: {applied}")
    return jsonify({"ok": True, "applied": applied})


@app.route("/api/calibrate/run", methods=["POST"])
def api_calibrate_run():
    data     = request.get_json(silent=True) or {}
    months   = max(3, min(24, _as_int(data.get("months"), 12)))
    universe = "snp500" if data.get("universe") == "snp500" else "focus"
    busy = _start_job(_cal_state, _cal_lock, _run_calibration_job, (months, universe), name="Calibration")
    return busy or jsonify({"ok": True, "months": months, "universe": universe})


@app.route("/api/calibrate/status")
def api_calibrate_status():
    with _cal_lock:
        s = dict(_cal_state)
    if s.get("results"):
        r = s["results"]
        s["completed_at"]  = r.get("completed_at")
        s["total_trades"]  = r.get("total_trades")
        s["tickers_used"]  = r.get("tickers_used")
        s["months"]        = r.get("months")
        del s["results"]
    return jsonify(s)


@app.route("/api/calibrate/results")
def api_calibrate_results():
    with _cal_lock:
        r = _cal_state.get("results")
    return jsonify(r or {})


@app.route("/api/calibrate/logistic", methods=["POST"])
def api_calibrate_logistic():
    data     = request.get_json(silent=True) or {}
    source   = "backtest" if data.get("source") == "backtest" else "signal_log"
    months   = max(3, min(24, _as_int(data.get("months"), 12)))
    universe = "focus" if data.get("universe") == "focus" else "snp500"
    busy = _start_job(_lr_state, _lr_lock, _run_logistic_calibration, (source, months, universe),
                      name="Logistic calibration")
    return busy or jsonify({"ok": True, "source": source, "months": months, "universe": universe})


@app.route("/api/calibrate/logistic/results")
def api_calibrate_logistic_results():
    with _lr_lock:
        return jsonify({
            "running": _lr_state.get("running", False),
            "stage":   _lr_state.get("stage", "Idle"),
            "results": _lr_state.get("results"),
        })


@app.route("/api/calibrate/hold-sweep", methods=["POST"])
def api_hold_sweep():
    data     = request.get_json(silent=True) or {}
    months   = max(3, min(24, _as_int(data.get("months"), 12)))
    universe = "snp500" if data.get("universe") == "snp500" else "focus"
    busy = _start_job(_hs_state, _hs_lock, _run_hold_sweep_job, (months, universe), name="Hold Horizon analysis")
    return busy or jsonify({"ok": True, "started": True, "months": months, "universe": universe})


@app.route("/api/calibrate/hold-sweep/results")
def api_hold_sweep_results():
    with _hs_lock:
        return jsonify({
            "running": _hs_state.get("running", False),
            "stage":   _hs_state.get("stage", "Idle"),
            "results": _hs_state.get("results"),
        })


@app.route("/api/signal-log/summary")
def api_signal_log_summary():
    return jsonify(_signal_log_summary())


@app.route("/api/signal-log/entries")
def api_signal_log_entries():
    """Return recent signal log entries for display in the Calibrate tab."""
    entries = _load_signal_log()
    # Sort most recent first
    entries.sort(key=lambda e: e.get("signal_date", ""), reverse=True)
    # Return last 300 entries (enough for display, not too heavy)
    out = []
    for e in entries[:300]:
        out.append({
            "ticker":        e.get("ticker"),
            "strategy":      e.get("strategy"),
            "signal_date":   e.get("signal_date", "")[:10],
            "check_after":   e.get("check_after", "")[:10],
            "hold_days":     e.get("hold_days", 7),
            "score":         e.get("score"),
            "price":         e.get("price_at_signal"),
            "vix_regime":    e.get("vix_regime"),
            "resolved":      e.get("resolved", False),
            "outcome":            e.get("outcome"),
            "win":                e.get("win"),
            "stock_return_pct":   e.get("stock_return_pct"),
            "current_return_pct": e.get("current_return_pct"),
            "current_price":      e.get("current_price"),
            "live_updated_at":    (e.get("live_updated_at") or "")[:16],
        })
    return jsonify({"entries": out, "total": len(entries)})


@app.route("/api/vix/regime")
def api_vix_regime():
    vix    = get_vix()
    regime = vix_regime(vix)
    if regime == "high":
        rsi_os  = max(RSI_OVERSOLD   - 5, 20)
        rsi_ob  = min(RSI_OVERBOUGHT + 5, 80)
        desc    = "High volatility — stocks can stay oversold longer; requiring more extreme RSI"
        color   = "#ef4444"
        icon    = "🔴"
    elif regime == "low":
        rsi_os  = RSI_OVERSOLD  + 2
        rsi_ob  = RSI_OVERBOUGHT - 2
        desc    = "Low volatility — options are cheap; thresholds slightly relaxed"
        color   = "#22c55e"
        icon    = "🟢"
    else:
        rsi_os  = RSI_OVERSOLD
        rsi_ob  = RSI_OVERBOUGHT
        desc    = "Normal conditions — standard thresholds active"
        color   = "#f59e0b"
        icon    = "🟡"
    with _cal_lock:
        cal_results = _cal_state.get("results")
    return jsonify({
        "vix":           vix,
        "regime":        regime,
        "icon":          icon,
        "color":         color,
        "description":   desc,
        "rsi_oversold":  rsi_os,
        "rsi_overbought":rsi_ob,
        "base_rsi_os":   RSI_OVERSOLD,
        "base_rsi_ob":   RSI_OVERBOUGHT,
        "cal_active":    bool(_cal_weights),
        "cal_date":      (cal_results or {}).get("completed_at", "")[:10] if cal_results else "",
    })


@app.route("/api/debug/scan/<sym>")
def api_debug_scan(sym):
    """
    Diagnostic endpoint — runs scan_ticker on one symbol and returns a full
    breakdown of every signal condition so you can see exactly what blocked it.
    Usage: GET /api/debug/scan/AAPL
    """
    sym = sym.upper()
    try:
        tkr, hist = _fetch_history(sym, LIVE_HISTORY_PERIOD)
        if hist is None or len(hist) < 60:
            return jsonify({"error": "Not enough history data"}), 400

        # Exactly what the live scanner does — same engine, same parameters —
        # so the breakdown can never drift from the real gate again.
        _warm_market_caches()
        fr  = indicator_frame(hist, spy_closes=_spy_cache.get("closes"), sym=sym, garch="last")
        ind = ind_at(fr, -1)
        ctx = live_market_ctx()
        prm = live_params()
        ev  = evaluate_entry(ind, ctx, prm, scan_condors=False)

        # Check if option chain is available
        opt_available = False
        try:
            exp, dte = _best_exp(tkr)
            opt_available = exp is not None
        except Exception:
            pass

        ind_out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ind.items() if k not in ("i", "date")}
        ev_out  = {k: v for k, v in ev.items() if k != "flags"}
        return jsonify({
            "ticker":         sym,
            "bar_date":       str(ind["date"])[:10],
            "price":          round(ind["price"], 2),
            "params":         prm,
            "context":        {k: v for k, v in ctx.items() if k != "macro"},
            "macro":          ctx["macro"],
            "indicators":     ind_out,
            "evaluation":     ev_out,
            "flags":          ev["flags"],
            "strategy_gates": {
                "has_sell_put":     ev["has_sell_put"],
                "has_buy_call":     ev["has_buy_call"],
                "has_buy_put":      ev["has_buy_put"],
                "has_covered_call": ev["has_covered_call"],
                "price_ok_for_sell": ind["price"] <= MAX_PRICE_SELL,
            },
            "blockers":       entry_blockers(ev, ind),
            "option_chain_available": opt_available,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/results/<strategy>")
def api_results(strategy):
    key_map = {"sell_puts": "sell_puts", "buy_calls": "buy_calls",
               "buy_puts": "buy_puts", "covered_calls": "covered_calls",
               "iron_condors": "iron_condors"}
    key = key_map.get(strategy)
    if not key:
        return jsonify([])
    with scan_lock:
        return jsonify(scan_state["results"].get(key, []))


@app.route("/api/queue", methods=["GET"])
def api_queue_get():
    with _queue_lock:
        return jsonify(load_queue())


@app.route("/api/queue", methods=["POST"])
def api_queue_add():
    """
    Add a trade to the queue.
    Body: { ticker, strategy, option, qty, name, price }
    Returns guardrail info so the UI can warn the user immediately.
    """
    data  = request.get_json(silent=True) or {}
    if not data.get("ticker") or not isinstance(data.get("option"), dict):
        return jsonify({"ok": False, "message": "ticker and option are required"}), 400
    trade = {
        "id":         str(uuid.uuid4())[:8],
        "queued_at":  datetime.now().isoformat(),
        "status":     "pending",    # pending | executing | error  (done trades move to history)
        "ticker":     str(data.get("ticker", "")).upper(),
        "name":       data.get("name", ""),
        "price":      _as_float(data.get("price")),
        "strategy":   _norm_strategy(data.get("strategy", "")),
        "strategy_label": data.get("strategy_label", ""),
        "option":     data.get("option", {}),
        "qty":        max(1, _as_int(data.get("qty"), 1)),
        "put_width":  _as_float(data.get("put_width")),
        "call_width": _as_float(data.get("call_width")),
        "legs":       data.get("legs",       None),
        "entry_price": _as_float(data.get("entry_price")),
    }

    passes, cost, msg = guardrail_check(trade)
    trade["estimated_cost"] = round(cost, 2)
    trade["guardrail_ok"]   = passes

    with _queue_lock:
        q = load_queue()
        q.append(trade)
        save_queue(q)
    return jsonify({"ok": True, "trade": trade, "guardrail": {"passes": passes, "cost": cost, "message": msg}})


@app.route("/api/queue/<trade_id>", methods=["DELETE"])
def api_queue_remove(trade_id):
    with _queue_lock:
        q = [t for t in load_queue() if t["id"] != trade_id]
        save_queue(q)
    return jsonify({"ok": True})


@app.route("/api/execute/<trade_id>", methods=["POST"])
def api_execute(trade_id):
    """
    ──────────────────────────────────────────────────────────
    EXECUTION ENDPOINT — called only when the user clicks
    "Confirm & Place Order" in the dashboard.

    Body (optional):  { "override_confirm": "CONFIRM" }
      → Required if the trade exceeds the $1,000 guardrail.
    ──────────────────────────────────────────────────────────
    """
    body     = request.get_json(silent=True) or {}
    override = str(body.get("override_confirm", "")).strip().upper()

    # Check + claim under the lock so two clicks can't both pass the
    # "pending" test and place the same order twice.
    with _queue_lock:
        q     = load_queue()
        trade = next((t for t in q if t["id"] == trade_id), None)
        if not trade:
            return jsonify({"ok": False, "message": "Trade not found in queue"}), 404
        if trade["status"] == "executing":
            return jsonify({"ok": False, "message": "Already executing"}), 409

        # ── Guardrail check ───────────────────────────────────────
        passes, cost, msg = guardrail_check(trade)
        if not passes and override != "CONFIRM":
            return jsonify({"ok": False, "guardrail_fail": True, "cost": cost, "message": msg}), 403

        trade["status"] = "executing"
        save_queue(q)

    # ── Hand off to SoFi automation (network — outside the lock) ──
    try:
        from sofi_trader import place_order_on_sofi
        result = place_order_on_sofi(trade, auto_submit=AUTO_SUBMIT)
    except ImportError:
        result = {"ok": False, "message":
                  "sofi_trader.py not found. Make sure it is in the same folder."}
    except Exception as e:
        result = {"ok": False, "message": str(e)}

    # ── Update queue + history ────────────────────────────────
    with _queue_lock:
        q     = load_queue()
        trade = next((t for t in q if t["id"] == trade_id), trade)
        trade["status"]      = "done" if result.get("ok") else "error"
        trade["executed_at"] = datetime.now().isoformat()
        trade["result"]      = result.get("message", "")
        if result.get("ok"):
            h = load_history()
            h.append(trade)
            save_history(h)
            q = [t for t in q if t["id"] != trade_id]
        save_queue(q)

    return jsonify({"ok": bool(result.get("ok")), "message": result.get("message", "")})


@app.route("/api/history")
def api_history():
    with _queue_lock:
        return jsonify(load_history())


@app.route("/api/history/perf")
def api_history_perf():
    """
    Fetch current option mid prices for all history entries in parallel.
    Returns { trade_id: { entry_mid, current_mid, pct_change, dollar_change } }
    """
    with _queue_lock:
        history = load_history()
    perf    = {}

    def _fetch(t):
        tid = t.get("id")
        if not tid:
            return
        try:
            opt          = t.get("option") or {}
            entry_price  = _as_float(t.get("entry_price")) or _as_float(opt.get("mid")) or _as_float(opt.get("ask"))
            if entry_price <= 0:
                return
            current_mid, _, _, quality, _ = _fetch_option_quote(
                t.get("ticker"), opt.get("expiration"), opt.get("strike"), opt.get("type", "call"))
            if quality == "unavailable":
                return
            pct, dollar = _position_pnl(_norm_strategy(t.get("strategy")), entry_price, current_mid, t.get("qty", 1))
            if pct is None:
                return
            perf[tid] = {
                "entry_mid":    round(entry_price,  3),
                "current_mid":  round(current_mid,  3),
                "pct_change":   pct,
                "dollar_change": dollar,
            }
        except Exception as e:
            print(f"[History] {t.get('ticker')} price fetch failed: {e}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        pool.map(_fetch, history)

    return jsonify(perf)


@app.route("/api/news/<ticker>")
def api_news(ticker):
    """Return fresh news + sentiment for a single ticker (on-demand)."""
    sym      = ticker.strip().upper()
    articles = fetch_ticker_news(sym)
    return jsonify({"ticker": sym, "articles": articles})


@app.route("/api/news/debug/<ticker>")
def api_news_debug(ticker):
    """Debug endpoint — shows raw RSS response to diagnose news issues."""
    sym = ticker.strip().upper()
    try:
        import requests
        from urllib.parse import quote
        url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(sym)}&region=US&lang=en-US"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        return jsonify({
            "url":    url,
            "status": resp.status_code,
            "length": len(resp.text),
            "preview": resp.text[:800],
        })
    except Exception as e:
        # Try Google News RSS as fallback test
        try:
            import requests
            url2 = f"https://news.google.com/rss/search?q={sym}+stock&hl=en-US&gl=US&ceid=US:en"
            resp2 = requests.get(url2, headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
            return jsonify({
                "yahoo_error": str(e),
                "google_url": url2,
                "google_status": resp2.status_code,
                "google_preview": resp2.text[:800],
            })
        except Exception as e2:
            return jsonify({"yahoo_error": str(e), "google_error": str(e2)})


@app.route("/api/search", methods=["POST"])
def api_search():
    """Run all strategies on a single user-supplied ticker."""
    sym = str((request.get_json(silent=True) or {}).get("ticker") or "").strip().upper()
    if not sym or not re.fullmatch(r"[A-Z0-9.\-^=]{1,10}", sym):
        return jsonify({"ok": False, "message": "No valid ticker provided"}), 400
    _warm_market_caches()                       # SPY / VIX / Fed context before the frame is built (same as a full scan)
    raw     = scan_ticker(sym, prefetch_fn=_prefetch_ticker)
    signals = raw.get("_signals", {}) if raw else {}
    if not signals:
        return jsonify({"ok": False, "message": f"No signals found for {sym} — it may not meet any strategy criteria right now."})
    return jsonify({"ok": True, "ticker": sym, "signals": signals})


@app.route("/api/track", methods=["POST"])
def api_track_add():
    """Add a signal to the tracked positions list."""
    data = request.get_json(silent=True) or {}
    if not data.get("ticker") or not isinstance(data.get("option"), dict):
        return jsonify({"ok": False, "message": "ticker and option are required"}), 400
    position = {
        "id":           str(uuid.uuid4())[:8],
        "tracked_at":   datetime.now().isoformat(),
        "ticker":       str(data.get("ticker", "")).upper(),
        "name":         data.get("name", ""),
        "price":        _as_float(data.get("price")),
        "strategy":     _norm_strategy(data.get("strategy", "")),
        "strategy_label": data.get("strategy_label", ""),
        "option":       data.get("option", {}),
        "legs":         data.get("legs", None),
        "net_credit":   _as_float(data.get("net_credit")),
        "put_width":    _as_float(data.get("put_width")),
        "call_width":   _as_float(data.get("call_width")),
        "contracts":    max(1, _as_int(data.get("contracts") or data.get("qty"), 1)),
        "entry_price":  _as_float(data.get("entry_price")),   # option mid at time of tracking
        "last_signal":  None,
        "last_checked": None,
    }
    with track_lock:
        positions = load_tracked()
        # Avoid duplicates (same ticker+strategy already tracked)
        dup = any(
            p["ticker"] == position["ticker"] and p["strategy"] == position["strategy"]
            for p in positions
        )
        if dup:
            return jsonify({"ok": False, "message": f"{position['ticker']} {position['strategy_label']} is already being tracked."})
        positions.append(position)
        save_tracked(positions)
    return jsonify({"ok": True, "position": position})


@app.route("/api/portfolio/greeks", methods=["GET"])
def api_portfolio_greeks():
    """Return net portfolio delta + per-position breakdown."""
    return jsonify(get_portfolio_greeks())


@app.route("/api/portfolio/risk", methods=["GET"])
def api_portfolio_risk():
    """Risk metrics for OPEN positions: max loss exposure, concentration, pnl dispersion."""
    return jsonify(get_portfolio_risk())


@app.route("/api/performance/stats", methods=["GET"])
def api_performance_stats():
    """Closed-trade analytics: expectancy, profit factor, Sortino, max drawdown."""
    return jsonify(get_performance_stats())


@app.route("/api/macro/regime", methods=["GET"])
def api_macro_regime():
    """Return which strategies are currently enabled by the macro regime gate (with the live yield rule applied)."""
    m = dict(get_macro_regime())
    if live_params().get("yield_block_calls") and m.get("yield_inverted") and m.get("buy_call"):
        m["buy_call"] = False
        m["reason"]   = (m.get("reason") or "") + "; Yield curve inverted — disable Buy Call (yield_block_calls)"
        m["label"]    = (m.get("label") or "normal").replace("normal", "") + "+inv"
    return jsonify(m)


@app.route("/api/tracked", methods=["GET"])
def api_tracked_get():
    """Return all tracked positions with their latest sell signal.
    Each position is enriched with `hold_horizon` data when Hold Horizon
    Analysis results are available (Level 2 of the HH integration —
    informational only; does not change sell-signal logic)."""
    with track_lock:
        positions = load_tracked()
    # Enrich each position with hold-horizon context if HH data exists.
    now = datetime.now()
    for pos in positions:
        try:
            tracked_at = pos.get("tracked_at")
            if not tracked_at:
                continue
            days_held = max(0, (now - datetime.fromisoformat(tracked_at)).days)
            hh = _hs_targets_for_position(pos.get("strategy", ""), days_held)
            if hh is not None:
                pos["hold_horizon"] = hh
        except Exception:
            pass
    return jsonify(positions)


@app.route("/api/track/<position_id>", methods=["DELETE"])
def _auto_position_block(position_id):
    """
    Positions the autotrader opened are held at a broker: the Track tab must not
    drop or re-point them behind its back (the broker contract would be orphaned).
    Returns a JSON error response for such a position, else None.
    """
    with track_lock:
        pos = next((p for p in load_tracked() if p["id"] == position_id), None)
    if pos and pos.get("auto"):
        return jsonify({"ok": False, "message": "This position was opened by the autotrader — close it from the "
                                                "🤖 Autotrade tab so the broker position is closed as well."}), 409
    return None


def api_track_remove(position_id):
    """Remove a position from tracking."""
    blocked = _auto_position_block(position_id)
    if blocked:
        return blocked
    with track_lock:
        positions = [p for p in load_tracked() if p["id"] != position_id]
        save_tracked(positions)
    return jsonify({"ok": True})


@app.route("/api/track/<position_id>/option", methods=["PATCH"])
def api_track_update_option(position_id):
    """
    Update the option details (strike / expiration / type) for a tracked position.
    Tries to fetch a fresh price for the new option from yfinance.
    """
    blocked = _auto_position_block(position_id)
    if blocked:
        return blocked
    data       = request.get_json(silent=True) or {}
    new_strike = _as_float(data.get("strike"))
    new_exp    = str(data.get("expiration") or "").strip()
    new_type   = str(data.get("type") or "call").strip().lower()
    if new_strike <= 0 or not new_exp or new_type not in ("call", "put"):
        return jsonify({"ok": False, "message": "Strike, expiration, and type are required"})
    try:
        exp_date = date.fromisoformat(new_exp)
    except ValueError:
        return jsonify({"ok": False, "message": "Expiration must be YYYY-MM-DD"})

    with track_lock:
        pos = next((p for p in load_tracked() if p["id"] == position_id), None)
    if not pos:
        return jsonify({"ok": False, "message": "Position not found"})
    ticker = pos.get("ticker", "")

    # ── Fetch fresh option data for the new strike/exp (network, outside the lock) ──
    new_opt = dict(pos.get("option") or {})
    new_opt.update(strike=new_strike, expiration=new_exp, type=new_type,
                   dte=max((exp_date - date.today()).days, 0))
    entry_price_auto = None
    try:
        chain = _option_chain(yf.Ticker(ticker), new_exp, ticker)
        opts  = chain.calls if new_type == "call" else chain.puts
        row   = opts[(opts["strike"] - new_strike).abs() < 0.01]
        if not row.empty:
            r = row.iloc[0]
            mid, bid, ask, _, quality = _option_price_detail(r)
            if _as_float(r.get("impliedVolatility")) > 3.0:
                print(f"[PriceCheck] {ticker} edit strike={new_strike} rejected (IV > 300%, misaligned row)")
            else:
                new_opt["bid"]           = round(bid, 3)
                new_opt["ask"]           = round(ask, 3)
                new_opt["mid"]           = round(mid, 3)
                new_opt["iv"]            = round(_as_float(r.get("impliedVolatility")) * 100, 1)
                new_opt["volume"]        = _as_int(r.get("volume"))          # NaN-safe (yfinance leaves untraded rows NaN)
                new_opt["open_interest"] = _as_int(r.get("openInterest"))
                # Auto-set entry price only from a live or near-live quote — never
                # from lastPrice alone, which can be days old for thin options.
                if mid > 0 and quality in ("live", "stale_bid", "ask_only"):
                    entry_price_auto = round(mid, 3)
    except Exception as e:
        print(f"[option edit] Could not fetch live data for {ticker} {new_exp} {new_type} ${new_strike}: {e}")

    with track_lock:
        positions = load_tracked()
        pos = next((p for p in positions if p["id"] == position_id), None)
        if not pos:
            return jsonify({"ok": False, "message": "Position not found"})
        pos["option"]      = new_opt
        pos["last_signal"] = None   # will be populated by the immediate refresh below
        pos["last_checked"]= None

        # Update entry price to live mid (unless user already set it manually).
        # entry_price_manual = True means the user deliberately typed in their
        # actual fill price and it must never be auto-overwritten.
        if entry_price_auto and not pos.get("entry_price_manual"):
            pos["entry_price"] = entry_price_auto

        save_tracked(positions)
        pos_id_to_refresh = pos["id"]

    # ── Immediate sell-signal refresh for the edited position ──────
    # Don't wait 6 hours for the background pass — run get_sell_signal
    # now in a short-lived thread so the card shows live prices within seconds.
    def _refresh_after_edit(pid):
        try:
            with track_lock:
                positions = load_tracked()
                target = next((p for p in positions if p["id"] == pid), None)
            if target is None:
                return
            sig = get_sell_signal(target)
            clear_override = sig.pop("_clear_live_override", False)
            with track_lock:
                positions = load_tracked()
                for p in positions:
                    if p["id"] == pid:
                        p["last_checked"] = datetime.now().isoformat()
                        if clear_override:
                            p.pop("live_price_override",    None)
                            p.pop("live_price_override_at", None)
                        p["last_signal"] = sig
                        break
                save_tracked(positions)
            print(f"[Track] Immediate refresh after option edit for {target.get('ticker')} done.")
        except Exception as ex:
            print(f"[Track] Immediate refresh failed: {ex}")

    threading.Thread(target=_refresh_after_edit, args=(pos_id_to_refresh,), daemon=True).start()

    return jsonify({
        "ok":    True,
        "option": new_opt,
        "entry_price": pos.get("entry_price"),
    })


@app.route("/api/track/<position_id>/entry", methods=["PATCH"])
def api_track_update_entry(position_id):
    """Update the actual entry price paid for a tracked position."""
    data      = request.get_json(silent=True) or {}
    new_price = _as_float(data.get("entry_price"))
    if new_price <= 0:
        return jsonify({"ok": False, "message": "Price must be greater than 0"})

    with track_lock:
        positions = load_tracked()
        for p in positions:
            if p["id"] == position_id:
                p["entry_price"]        = new_price
                p["entry_price_manual"] = True   # flag that user set this manually
                # Recompute P&L against the new entry so the card updates at once
                sig = p.get("last_signal") or {}
                if _as_float(sig.get("current_option_mid")) > 0:
                    p["last_signal"] = _apply_price_update(p, sig, _as_float(sig.get("current_option_mid")),
                                                           _as_float(sig.get("current_bid")),
                                                           _as_float(sig.get("current_ask")),
                                                           sig.get("price_quality", "unavailable"))
                break
        else:
            return jsonify({"ok": False, "message": "Position not found"})
        save_tracked(positions)
    return jsonify({"ok": True, "entry_price": new_price})


@app.route("/api/track/<position_id>/live-price", methods=["PATCH"])
def api_track_live_price(position_id):
    """
    Store a manually-entered current option price.
    Used when yfinance data is stale/unavailable.
    The override stays active until yfinance returns a fresh two-sided
    quote (bid+ask both present), at which point it is automatically cleared.
    Pass {live_price: 0} or {clear: true} to remove the override.
    """
    data = request.get_json(silent=True) or {}

    # Allow explicit clear
    if data.get("clear") or (data.get("live_price") is not None and _as_float(data.get("live_price")) == 0):
        with track_lock:
            positions = load_tracked()
            for p in positions:
                if p["id"] == position_id:
                    p.pop("live_price_override",    None)
                    p.pop("live_price_override_at", None)
                    p["last_signal"] = None
                    p["last_checked"] = None
                    break
            save_tracked(positions)
        return jsonify({"ok": True})

    new_price = _as_float(data.get("live_price"))
    if new_price <= 0:
        return jsonify({"ok": False, "message": "Price must be greater than 0"})

    with track_lock:
        positions = load_tracked()
        for p in positions:
            if p["id"] == position_id:
                p["live_price_override"]    = round(new_price, 3)
                p["live_price_override_at"] = datetime.now().isoformat()
                # Inject into last_signal so the tracker instantly shows updated
                # P&L and re-evaluates target / stop / trailing / stagnation.
                p["last_signal"] = _apply_price_update(p, p.get("last_signal") or {}, new_price, 0.0, 0.0, "manual")
                break
        else:
            return jsonify({"ok": False, "message": "Position not found"})
        save_tracked(positions)

    return jsonify({"ok": True, "live_price": new_price})


# ── Performance tracker API ───────────────────────────────────

def _closed_trade_record(ticker, strategy, option_desc, entry, exit_, contracts,
                         entry_date, exit_date, notes, source):
    """Build a closed-trade dict with strategy-direction-aware P&L (exit 0 = expired worthless)."""
    sign      = -1.0 if strategy in SHORT_PREMIUM_STRATS else 1.0
    pnl_per   = sign * (exit_ - entry) * 100
    pnl_total = pnl_per * contracts
    pnl_pct   = sign * (exit_ - entry) / entry * 100
    return {
        "id":          str(uuid.uuid4())[:8],
        "ticker":      ticker,
        "strategy":    strategy,
        "option_desc": option_desc,
        "entry_price": round(entry, 3),
        "exit_price":  round(exit_, 3),
        "contracts":   contracts,
        "pnl_per":     round(pnl_per,   2),
        "pnl_total":   round(pnl_total, 2),
        "pnl_pct":     round(pnl_pct,   1),
        "entry_date":  entry_date,
        "exit_date":   exit_date,
        "notes":       notes,
        "source":      source,
    }


@app.route("/api/performance", methods=["GET"])
def api_performance_get():
    with _perf_lock:
        trades = load_performance()
    summary = calc_performance_summary(trades)
    return jsonify({"trades": trades, "summary": summary})


@app.route("/api/performance", methods=["POST"])
def api_performance_add():
    """Add a manually-entered closed trade."""
    data  = request.get_json(silent=True) or {}
    entry = _as_float(data.get("entry_price"))
    exit_ = _as_float(data.get("exit_price"), -1.0)
    contr = max(1, _as_int(data.get("contracts"), 1))
    strat = _norm_strategy(data.get("strategy", ""))
    if entry <= 0 or exit_ < 0:
        return jsonify({"ok": False, "error": "Entry price must be > 0 and exit price ≥ 0 (0 = expired worthless)."}), 400
    trade = _closed_trade_record(
        ticker=str(data.get("ticker") or "").upper(), strategy=strat,
        option_desc=data.get("option_desc", ""), entry=entry, exit_=exit_, contracts=contr,
        entry_date=data.get("entry_date", ""), exit_date=data.get("exit_date", ""),
        notes=data.get("notes", ""), source="manual")
    with _perf_lock:
        trades = load_performance()
        trades.append(trade)
        save_performance(trades)
    return jsonify({"ok": True, "trade": trade})


@app.route("/api/performance/from-tracker", methods=["POST"])
def api_performance_from_tracker():
    """Log a tracked position as closed and move it to performance history."""
    data       = request.get_json(silent=True) or {}
    pos_id     = data.get("position_id")
    exit_date  = data.get("exit_date", "")
    blocked = _auto_position_block(pos_id)
    if blocked:
        return blocked

    with track_lock:
        positions = load_tracked()
        pos = next((p for p in positions if p["id"] == pos_id), None)
        if not pos:
            return jsonify({"ok": False, "error": "Position not found"}), 404
        opt         = pos.get("option") or {}
        sig         = pos.get("last_signal") or {}
        entry_price = _as_float(pos.get("entry_price")) or _as_float(opt.get("mid")) or _as_float(opt.get("ask"))
        exit_price  = _as_float(data.get("exit_price"), -1.0)
        if exit_price < 0:                                   # not supplied → use the latest quote
            exit_price = _as_float(sig.get("current_option_mid"), -1.0)
        if entry_price <= 0 or exit_price < 0:
            return jsonify({"ok": False, "error": "Need a valid entry price and an exit price (0 = expired worthless). "
                                                  "Use ✏️ Edit entry first if entry is missing."}), 400
        trade = _closed_trade_record(
            ticker=pos.get("ticker", ""), strategy=_norm_strategy(pos.get("strategy", "")),
            option_desc=f"{str(opt.get('type', '')).capitalize()} ${opt.get('strike', '')} {opt.get('expiration', '')}",
            entry=entry_price, exit_=exit_price, contracts=max(1, _as_int(pos.get("contracts"), 1)),
            entry_date=str(pos.get("tracked_at") or "")[:10],
            exit_date=exit_date or datetime.now().strftime("%Y-%m-%d"),
            notes=data.get("notes", ""), source="tracker")
        positions = [p for p in positions if p["id"] != pos_id]
        save_tracked(positions)
    with _perf_lock:
        perf = load_performance()
        perf.append(trade)
        save_performance(perf)
    return jsonify({"ok": True, "trade": trade})


@app.route("/api/performance/<trade_id>", methods=["DELETE"])
def api_performance_delete(trade_id):
    with _perf_lock:
        trades = load_performance()
        trades = [t for t in trades if t["id"] != trade_id]
        save_performance(trades)
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════
#  AUTOTRADE ROUTES   (the logic lives in autotrader.py)
# ══════════════════════════════════════════════════════════════

@app.route("/api/autotrade/status")
def api_autotrade_status():
    return jsonify(autotrader.status())


@app.route("/api/autotrade/config", methods=["GET", "POST"])
def api_autotrade_config():
    if request.method == "POST":
        cfg = autotrader.set_config(request.get_json(silent=True) or {})
        return jsonify({"ok": True, "config": autotrader.public_config(cfg)})
    return jsonify(autotrader.public_config())


@app.route("/api/autotrade/toggle", methods=["POST"])
def api_autotrade_toggle():
    enabled = bool((request.get_json(silent=True) or {}).get("enabled"))
    cfg = autotrader.load_config()
    if enabled and cfg["broker"] == "alpaca" and not cfg["paper"] and os.environ.get("OPTIONS_BOT_ALLOW_LIVE") != "1":
        return jsonify({"ok": False, "message": "Live Alpaca trading is refused unless OPTIONS_BOT_ALLOW_LIVE=1 "
                                                "is set in the bot's environment. Use paper, or set it deliberately."}), 400
    cfg = autotrader.set_config({"enabled": enabled})
    autotrader._log("enabled" if enabled else "disabled", broker=cfg["broker"], paper=cfg["paper"])
    return jsonify({"ok": True, "enabled": cfg["enabled"]})


@app.route("/api/autotrade/run", methods=["POST"])
def api_autotrade_run():
    """One forced cycle now (also runs when disabled / after hours — for testing on paper)."""
    return jsonify({"ok": True, "result": autotrader.run_now()})


@app.route("/api/autotrade/log")
def api_autotrade_log():
    return jsonify(autotrader.get_log(_as_int(request.args.get("n"), 100)))


@app.route("/api/autotrade/close/<position_id>", methods=["POST"])
def api_autotrade_close(position_id):
    ok = autotrader.request_close(position_id)
    return jsonify({"ok": ok, "message": "" if ok else "No auto position with that id"}), (200 if ok else 404)


@app.route("/api/autotrade/manual", methods=["POST"])
def api_autotrade_manual():
    """Operator entry: {"ticker", "strategy", "note"} → one position now through the autotrader's own pipeline."""
    data = request.get_json(silent=True) or {}
    sym = str(data.get("ticker") or "").upper().strip()
    if not sym:
        return jsonify({"ok": False, "message": "ticker required"}), 400
    try:
        res = autotrader.manual_entry(sym, data.get("strategy") or "buy_call", str(data.get("note") or ""))
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500
    return jsonify(res), (200 if res.get("ok") else 409)


# ══════════════════════════════════════════════════════════════
#  HTML DASHBOARD
# ══════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Options Bot — Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  --bg:#0f1117;--surface:#1a1d27;--surface2:#22263a;--border:#2e3248;
  --text:#e2e8f0;--muted:#8892a4;
  --green:#22c55e;--red:#ef4444;--yellow:#f59e0b;
  --blue:#3b82f6;--purple:#a855f7;--accent:#6366f1;--orange:#f97316;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px}

header{background:var(--surface);border-bottom:1px solid var(--border);
       padding:14px 28px;display:flex;align-items:center;gap:20px;flex-wrap:wrap}
header h1{font-size:18px;font-weight:800;color:var(--accent)}
header .tagline{font-size:12px;color:var(--muted);flex:1}

.btn{border:none;border-radius:8px;padding:8px 16px;cursor:pointer;font-size:13px;font-weight:600;transition:.15s}
.btn-primary{background:var(--accent);color:#fff}
.btn-primary:hover{opacity:.85}
.btn-danger{background:var(--red);color:#fff}
.btn-danger:hover{opacity:.85}
.btn-green{background:var(--green);color:#000}
.btn-green:hover{opacity:.85}
.btn-ghost{background:transparent;border:1px solid var(--border);color:var(--muted)}
.btn-ghost:hover{border-color:var(--text);color:var(--text)}
.btn:disabled{opacity:.4;cursor:not-allowed}

/* Progress bar */
#scan-bar{height:3px;background:var(--accent);width:0%;transition:width .3s;position:fixed;top:0;left:0;z-index:999}

/* Tabs */
.tabs{display:flex;gap:0;border-bottom:1px solid var(--border);padding:0 28px;background:var(--surface)}
.tab{padding:12px 18px;cursor:pointer;font-size:13px;font-weight:600;color:var(--muted);
     border-bottom:2px solid transparent;transition:.15s;white-space:nowrap}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab .badge-count{background:var(--surface2);border-radius:10px;padding:1px 7px;
                  font-size:11px;margin-left:6px;color:var(--muted)}
.tab.active .badge-count{background:var(--accent);color:#fff}

/* Tab panels */
.panel{display:none;padding:20px 28px}
.panel.active{display:block}

/* Scan status */
.scan-status{background:var(--surface);border:1px solid var(--border);border-radius:10px;
             padding:14px 20px;margin-bottom:18px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.scan-status .label{color:var(--muted);font-size:12px}
.scan-status .value{font-weight:600;margin-top:2px}

/* Stat cards */
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 18px;flex:1;min-width:110px}
.stat .slabel{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px}
.stat .sval{font-size:22px;font-weight:800;margin-top:3px}

/* Cards grid */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:18px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.card-head{padding:12px 16px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--border)}
.ticker{font-size:17px;font-weight:800}
.cname{font-size:11px;color:var(--muted);margin-top:2px}
.cprice{font-size:20px;font-weight:700}
.csector{font-size:10px;color:var(--muted);text-align:right;margin-top:1px}
.score-ring{width:44px;height:44px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-weight:800;font-size:14px;flex-shrink:0}

.signals{display:flex;gap:6px;padding:8px 16px;flex-wrap:wrap}
.sig{border-radius:5px;padding:3px 9px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px}
.sig-red   {background:rgba(239,68,68,.15); color:var(--red);   border:1px solid rgba(239,68,68,.3)}
.sig-yellow{background:rgba(245,158,11,.15);color:var(--yellow);border:1px solid rgba(245,158,11,.3)}
.sig-blue  {background:rgba(59,130,246,.15);color:var(--blue);  border:1px solid rgba(59,130,246,.3)}
.sig-purple{background:rgba(168,85,247,.15);color:var(--purple);border:1px solid rgba(168,85,247,.3)}
.sig-green {background:rgba(34,197,94,.15); color:var(--green); border:1px solid rgba(34,197,94,.3)}
.sig-orange{background:rgba(249,115,22,.15);color:var(--orange);border:1px solid rgba(249,115,22,.3)}

.charts{display:grid;grid-template-columns:1fr 1fr;padding:0 16px 8px;gap:10px}
.chart-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px}

.opt-info{border-top:1px solid var(--border);padding:10px 16px}
.opt-title{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:7px}
.opt-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}
.opt-cell .key{font-size:9px;color:var(--muted)}
.opt-cell .val{font-size:13px;font-weight:600;margin-top:1px}
.opt-cell .val.good{color:var(--green)}

.card-footer{border-top:1px solid var(--border);padding:10px 16px;display:flex;align-items:center;gap:10px}
.qty-wrap{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}
.qty-wrap input{width:52px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;
                padding:5px 8px;color:var(--text);font-size:13px;text-align:center}
.cost-est{font-size:11px;color:var(--muted);flex:1}
.cost-est .cost-val{font-weight:600;color:var(--text)}

/* Queue panel */
.queue-empty{text-align:center;padding:50px;color:var(--muted)}
.queue-table{width:100%;border-collapse:collapse}
.queue-table th{font-size:11px;color:var(--muted);text-align:left;padding:8px 12px;
                border-bottom:1px solid var(--border);font-weight:600;text-transform:uppercase}
.queue-table td{padding:10px 12px;border-bottom:1px solid var(--border);vertical-align:middle}
.queue-table tr:last-child td{border-bottom:none}
.status-pill{border-radius:12px;padding:2px 8px;font-size:10px;font-weight:700;text-transform:uppercase}
.s-pending  {background:rgba(59,130,246,.2); color:var(--blue)}
.s-executing{background:rgba(245,158,11,.2);color:var(--yellow)}
.s-done     {background:rgba(34,197,94,.2); color:var(--green)}
.s-error    {background:rgba(239,68,68,.2); color:var(--red)}

/* Confirm modal */
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:1000;
               align-items:center;justify-content:center}
.modal-overlay.open{display:flex}
.modal{background:var(--surface);border:1px solid var(--border);border-radius:14px;
       padding:28px;max-width:480px;width:90%;box-shadow:0 20px 60px rgba(0,0,0,.5)}
.modal h2{font-size:16px;margin-bottom:8px}
.modal .sub{color:var(--muted);font-size:13px;margin-bottom:16px;line-height:1.5}
.modal .trade-summary{background:var(--surface2);border-radius:8px;padding:12px;
                      font-size:13px;margin-bottom:16px;line-height:1.8}
.modal .guardrail-warn{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);
                        border-radius:8px;padding:12px;color:var(--red);margin-bottom:14px;font-size:13px}
.modal .confirm-input{background:var(--surface2);border:1px solid var(--border);border-radius:8px;
                      padding:9px 12px;color:var(--text);font-size:14px;width:100%;margin-bottom:12px}
.modal-actions{display:flex;gap:10px;justify-content:flex-end}

.disclaimer{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.2);border-radius:8px;
            padding:12px 16px;margin-bottom:18px;font-size:12px;color:var(--muted);line-height:1.6}
.disclaimer b{color:var(--red)}

.spinner{display:inline-block;width:14px;height:14px;border:2px solid var(--accent);
         border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes livepulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.8)}}
@keyframes pulse-bar{0%,100%{opacity:.35}50%{opacity:1}}

/* Monte Carlo section */
.mc-section{border-top:1px solid var(--border);padding:12px 16px}
.mc-title{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px}
.mc-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px}
.mc-cell .mc-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
.mc-cell .mc-val{font-size:14px;font-weight:700;margin-top:3px}
.mc-cell .mc-sub{font-size:10px;color:var(--muted);margin-top:1px}
.pop-bar-track{height:6px;background:var(--surface2);border-radius:3px;margin-top:5px;overflow:hidden}
.pop-bar-fill{height:100%;border-radius:3px;transition:width .4s}
.mc-chart-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;margin-bottom:3px}

.empty-state{text-align:center;padding:60px 20px;color:var(--muted)}
.empty-state h3{color:var(--text);margin-bottom:8px}
</style>
</head>
<body>
<div id="scan-bar"></div>

<header>
  <h1>⚡ Options Bot</h1>
  <span class="tagline">⚡ Focus: ~100 liquid names · 📊 Full: S&amp;P 500 (~503) · 5 strategies · SoFi execution</span>
  <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
    <div id="scan-info" style="font-size:12px;color:var(--muted)">Not yet scanned</div>
    <div id="live-badge" style="display:none;align-items:center;gap:6px;font-size:11px;
         background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.4);
         border-radius:20px;padding:3px 10px;color:#10b981">
      <span style="width:7px;height:7px;border-radius:50%;background:#10b981;
            animation:livepulse 1.4s ease-in-out infinite;display:inline-block"></span>
      LIVE · refreshing every <span id="live-mins-display">5</span>m
      <select id="live-interval" onchange="setLiveInterval(this.value)"
        style="background:transparent;border:none;color:#10b981;font-size:11px;cursor:pointer;outline:none">
        <option value="3">3m</option>
        <option value="5" selected>5m</option>
        <option value="10">10m</option>
        <option value="15">15m</option>
      </select>
      <span onclick="stopLiveRefresh()" title="Stop live refresh"
        style="cursor:pointer;opacity:.7;margin-left:2px">✕</span>
    </div>
    <div id="last-refresh-info" style="font-size:11px;color:var(--muted);display:none"></div>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    <input id="ticker-search" type="text" placeholder="Ticker e.g. NVDA"
      style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;
             padding:7px 12px;color:var(--text);font-size:13px;width:160px;outline:none"
      onkeydown="if(event.key==='Enter')searchTicker()"/>
    <button class="btn btn-ghost" onclick="searchTicker()">🔍 Search</button>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    <div id="scan-mode-toggle" style="display:flex;border-radius:8px;overflow:hidden;border:1px solid var(--border);font-size:12px">
      <button id="mode-focus" onclick="setScanMode('focus')"
        style="padding:6px 14px;border:none;cursor:pointer;background:var(--accent);color:#fff;font-weight:600;transition:all .2s">
        ⚡ Focus
      </button>
      <button id="mode-full" onclick="setScanMode('full')"
        style="padding:6px 14px;border:none;cursor:pointer;background:var(--surface2);color:var(--muted);font-weight:600;transition:all .2s">
        📊 Full S&amp;P 500
      </button>
    </div>
    <button id="btn-green-candle" onclick="toggleGreenCandle()"
      style="padding:6px 14px;border-radius:8px;border:1px solid var(--border);cursor:pointer;background:var(--surface2);color:var(--muted);font-weight:600;font-size:12px;transition:all .2s">
      🕯️ Green Candle
    </button>
    <button id="btn-red-candle" onclick="toggleRedCandle()"
      style="padding:6px 14px;border-radius:8px;border:1px solid var(--border);cursor:pointer;background:var(--surface2);color:var(--muted);font-weight:600;font-size:12px;transition:all .2s">
      🕯️ Red Candle
    </button>
    <div style="display:flex;align-items:center;gap:6px;padding:4px 10px;border-radius:8px;
                border:1px solid var(--border);background:var(--surface2)" title="Max dollar loss you're willing to risk per trade. Sets recommended contract count on every scan card.">
      <span style="font-size:12px;color:var(--muted);white-space:nowrap">💰 Max loss $</span>
      <input id="risk-budget-input" type="number" min="0" step="5" placeholder="e.g. 75"
        style="width:70px;background:transparent;border:none;color:var(--text);font-size:13px;
               font-weight:600;outline:none;padding:0"
        onchange="setRiskBudget(this.value)"
        onkeydown="if(event.key==='Enter') setRiskBudget(this.value)"/>
    </div>
    <button class="btn btn-primary" id="btn-scan" onclick="startScan()">▶ Run Scan</button>
  </div>
</header>

<div id="search-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:500;overflow-y:auto;padding:40px 20px">
  <div style="max-width:920px;margin:0 auto;background:var(--surface);border-radius:16px;padding:28px;position:relative">
    <button onclick="closeSearch()" style="position:absolute;top:14px;right:18px;background:none;border:none;color:var(--muted);font-size:22px;cursor:pointer">✕</button>
    <h2 id="search-title" style="color:var(--accent);margin-bottom:20px;font-size:18px"></h2>
    <div id="search-results"></div>
  </div>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('sell_puts')">
    Sell Puts <span class="badge-count" id="cnt-sell_puts">0</span>
  </div>
  <div class="tab" onclick="switchTab('buy_calls')">
    Buy Calls <span class="badge-count" id="cnt-buy_calls">0</span>
  </div>
  <div class="tab" onclick="switchTab('buy_puts')">
    Buy Puts <span class="badge-count" id="cnt-buy_puts">0</span>
  </div>
  <div class="tab" onclick="switchTab('covered_calls')">
    Covered Calls <span class="badge-count" id="cnt-covered_calls">0</span>
  </div>
  <div class="tab" onclick="switchTab('iron_condors')">
    Iron Condors <span class="badge-count" id="cnt-iron_condors">0</span>
  </div>
  <div class="tab" onclick="switchTab('tracked')">
    📍 Tracked <span class="badge-count" id="cnt-tracked">0</span>
  </div>
  <div class="tab" onclick="switchTab('queue')">
    Trade Queue <span class="badge-count" id="cnt-queue">0</span>
  </div>
  <div class="tab" onclick="switchTab('history')">
    History <span class="badge-count" id="cnt-history">0</span>
  </div>
  <div class="tab" onclick="switchTab('backtest')">📈 Backtest</div>
  <div class="tab" onclick="switchTab('bsmodel')">⚖️ B-S Model</div>
  <div class="tab" onclick="switchTab('performance')">📊 Performance</div>
  <div class="tab" onclick="switchTab('calibrate')">🎯 Calibrate</div>
  <div class="tab" onclick="switchTab('autotrade')">🤖 Autotrade</div>
</div>

<!-- PANELS -->
<div class="panel active" id="panel-sell_puts"></div>
<div class="panel" id="panel-buy_calls"></div>
<div class="panel" id="panel-buy_puts"></div>
<div class="panel" id="panel-covered_calls"></div>
<div class="panel" id="panel-iron_condors">
  <div style="text-align:center;padding:40px 20px">
    <div style="font-size:36px;margin-bottom:12px">🦅</div>
    <div style="color:var(--text);font-size:16px;font-weight:600;margin-bottom:8px">Iron Condor Scanner</div>
    <div style="color:var(--muted);font-size:13px;margin-bottom:24px;max-width:420px;margin-left:auto;margin-right:auto">
      Iron condors take longer to scan (options chain required for every ticker).
      Run it separately when you're ready — results stay until the next scan.
    </div>
    <button id="btn-condor-scan" class="btn btn-primary" onclick="startCondorScan()" style="font-size:15px;padding:10px 28px">
      🔍 Scan for Iron Condors
    </button>
    <div id="condor-scan-status" style="margin-top:16px;font-size:12px;color:var(--muted)"></div>
  </div>
  <div id="condor-results"></div>
</div>

<div class="panel" id="panel-tracked">
  <div class="disclaimer">
    <b>📍 Tracked Positions</b> — Option prices and the profit-target / stop / trailing-stop checks refresh every 3 minutes; the full RSI/Bollinger sell-signal review runs every 6 hours and after every scan.
    A signal fires when: profit target hit, stop loss hit, RSI reverses through the threshold, or Bollinger Band signal fires.
  </div>
  <div id="tracked-content"><div class="queue-empty">No tracked positions yet. Click "Track" on any signal card to start monitoring it.</div></div>
</div>

<!-- ── Edit Option Modal ──────────────────────────────────── -->
<div id="edit-option-modal" style="display:none;position:fixed;inset:0;z-index:9999;
     background:rgba(0,0,0,0.6);align-items:center;justify-content:center">
  <div style="background:var(--surface);border:1px solid var(--border);border-radius:14px;
              padding:28px 32px;width:360px;max-width:95vw;box-shadow:0 8px 32px rgba(0,0,0,.5)">
    <div style="font-size:16px;font-weight:700;margin-bottom:18px">✏️ Edit Option</div>

    <label style="font-size:12px;color:var(--muted)">Option Type</label>
    <select id="eom-type" style="width:100%;padding:8px 10px;border-radius:8px;
      border:1px solid var(--border);background:var(--surface2);color:var(--text);
      font-size:14px;margin-bottom:12px;margin-top:4px">
      <option value="call">Call</option>
      <option value="put">Put</option>
    </select>

    <label style="font-size:12px;color:var(--muted)">Strike Price ($)</label>
    <input id="eom-strike" type="number" step="0.5" min="0.5"
      style="width:100%;padding:8px 10px;border-radius:8px;
             border:1px solid var(--border);background:var(--surface2);color:var(--text);
             font-size:14px;margin-bottom:12px;margin-top:4px;box-sizing:border-box"/>

    <label style="font-size:12px;color:var(--muted)">Expiration Date (YYYY-MM-DD)</label>
    <input id="eom-exp" type="date"
      style="width:100%;padding:8px 10px;border-radius:8px;
             border:1px solid var(--border);background:var(--surface2);color:var(--text);
             font-size:14px;margin-bottom:18px;margin-top:4px;box-sizing:border-box"/>

    <div id="eom-status" style="font-size:12px;color:var(--muted);margin-bottom:14px;min-height:18px"></div>

    <div style="display:flex;gap:10px">
      <button id="eom-save" class="btn btn-primary" style="flex:1"
        onclick="saveEditOption()">💾 Save &amp; Refresh Price</button>
      <button class="btn btn-ghost" style="flex:0 0 auto"
        onclick="closeEditOptionModal()">Cancel</button>
    </div>
  </div>
</div>

<div class="panel" id="panel-queue">
  <div class="disclaimer">
    <b>⚠ Safety reminder:</b> Nothing executes without your explicit approval.
    Click "Confirm &amp; Place Order" to open SoFi and fill in the trade form.
    The bot will <b>not</b> click Submit for you (unless you enable AUTO_SUBMIT in config).
    Trades over <b>$1,000</b> require you to type <b>CONFIRM</b> as an extra safeguard.
  </div>
  <div id="queue-content"><div class="queue-empty">Your trade queue is empty.</div></div>
</div>

<div class="panel" id="panel-history">
  <div id="history-content"><div class="queue-empty">No executed trades yet.</div></div>
</div>

<div class="panel" id="panel-backtest">
  <div style="padding:24px;max-width:900px;margin:0 auto">
    <h2 style="color:var(--accent);font-size:18px;margin-bottom:6px">📈 Signal Backtester</h2>
    <p style="color:var(--muted);font-size:13px;margin-bottom:20px">
      Walks 2 years of history, fires signals using the same logic as the live scanner,
      and prices options using Black-Scholes with realised-vol as IV proxy.
    </p>
    <!-- Controls -->
    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end;margin-bottom:20px">
      <div>
        <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">MODE</label>
        <select id="bt-mode" onchange="btModeChange()"
          style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;
                 padding:7px 12px;color:var(--text);font-size:13px;outline:none">
          <option value="signals">⚡ Current Signals (fastest)</option>
          <option value="focus">Focus List (~100 tickers)</option>
          <option value="ticker">Single Ticker</option>
          <option value="full">Full S&P 500 (slow)</option>
        </select>
      </div>
      <div id="bt-ticker-wrap" style="display:none">
        <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">TICKER</label>
        <input id="bt-ticker" type="text" placeholder="e.g. AAPL"
          style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;
                 padding:7px 12px;color:var(--text);font-size:13px;width:110px;outline:none"/>
      </div>
      <div>
        <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">HOLD PERIOD</label>
        <select id="bt-hold"
          style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;
                 padding:7px 12px;color:var(--text);font-size:13px;outline:none">
          <option value="7">7 days</option>
          <option value="14">14 days</option>
          <option value="21" selected>21 days (to expiry)</option>
        </select>
      </div>
      <button id="bt-run-btn" class="btn btn-primary" onclick="runBacktest()" style="padding:8px 22px">
        ▶ Run Backtest
      </button>
    </div>
    <div id="bt-status" style="font-size:12px;color:var(--muted);margin-bottom:16px"></div>

    <!-- Summary cards -->
    <div id="bt-summary" style="display:none">
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:24px" id="bt-stat-cards"></div>
      <!-- Per-strategy breakdown -->
      <div id="bt-strat-breakdown" style="margin-bottom:24px"></div>
      <!-- Equity curve (simple CSS bar chart) -->
      <div style="margin-bottom:24px">
        <div style="font-size:13px;color:var(--muted);margin-bottom:8px">Cumulative P&L (running total, capped at 500 trades)</div>
        <canvas id="bt-equity-chart" height="120"
          style="width:100%;background:var(--surface2);border-radius:8px"></canvas>
      </div>
      <!-- Trade table -->
      <div style="font-size:13px;color:var(--muted);margin-bottom:8px">
        Recent trades <span id="bt-trade-count"></span>
      </div>
      <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:12px" id="bt-trade-table">
          <thead>
            <tr style="color:var(--muted);border-bottom:1px solid var(--border)">
              <th style="text-align:left;padding:6px 8px">Date</th>
              <th style="text-align:left;padding:6px 8px">Ticker</th>
              <th style="text-align:left;padding:6px 8px">Strategy</th>
              <th style="text-align:right;padding:6px 8px">Stock In</th>
              <th style="text-align:right;padding:6px 8px">Stock Out</th>
              <th style="text-align:right;padding:6px 8px">Opt In</th>
              <th style="text-align:right;padding:6px 8px">Opt Out</th>
              <th style="text-align:right;padding:6px 8px">P&L %</th>
              <th style="text-align:right;padding:6px 8px">IV%</th>
            </tr>
          </thead>
          <tbody id="bt-trade-tbody"></tbody>
        </table>
      </div>

      <!-- Signal Quality Analysis -->
      <div id="bt-sig-quality" style="margin-top:28px;display:none">
        <div style="font-size:14px;font-weight:600;color:var(--accent);margin-bottom:10px">📊 Signal Quality Analysis</div>
        <div style="color:var(--muted);font-size:12px;margin-bottom:12px">
          Win rate per individual signal across all backtested trades.
          Higher = that signal historically preceded profitable moves.
        </div>
        <div id="bt-sig-quality-rows"></div>
      </div>
    </div>
  </div>

  <!-- ── Walk-Forward Optimizer ─────────────────────── -->
  <div style="padding:24px;max-width:900px;margin:0 auto;border-top:1px solid var(--border);margin-top:8px">
    <h2 style="color:var(--accent);font-size:18px;margin-bottom:6px">🧬 Walk-Forward Optimizer</h2>
    <p style="color:var(--muted);font-size:13px;margin-bottom:16px">
      Sweeps RSI thresholds, confluence requirements, ADX threshold, and hold period across
      <b>1,024 parameter combinations</b> (bear-signal minimum, buy-put trend gate, GARCH mode, and the exit rules: premium stop, profit target, trailing stop, stagnation, max hold, stock-based stop) using the <b>same rules the live scanner runs</b>. Trains on the first 75% of history, validates on the last 25%
      to prevent overfitting. Best settings can be applied to the live scanner.
    </p>
    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end;margin-bottom:16px">
      <div>
        <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">UNIVERSE</label>
        <select id="wfo-mode"
          style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;
                 padding:7px 12px;color:var(--text);font-size:13px;outline:none">
          <option value="signals">⚡ Current Signals (fastest)</option>
          <option value="focus">Focus List (~100 tickers)</option>
          <option value="full">Full S&P 500 (slow)</option>
        </select>
      </div>
      <button id="wfo-run-btn" class="btn btn-primary" onclick="runOptimizer()" style="padding:8px 22px">
        🧬 Run Optimizer
      </button>
    </div>
    <div id="wfo-status" style="font-size:12px;color:var(--muted);margin-bottom:16px"></div>
    <div id="wfo-results" style="display:none">
      <div id="wfo-best-card" style="background:var(--surface2);border-radius:12px;padding:16px;margin-bottom:16px"></div>
      <div style="font-size:13px;color:var(--muted);margin-bottom:8px">Top 10 parameter combos (ranked by validation Sharpe)</div>
      <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:12px">
          <thead>
            <tr style="color:var(--muted);border-bottom:1px solid var(--border)">
              <th style="text-align:left;padding:6px 8px">RSI OS</th>
              <th style="text-align:left;padding:6px 8px">RSI OB</th>
              <th style="text-align:left;padding:6px 8px">Bull Min</th>
              <th style="text-align:left;padding:6px 8px">Bear Min</th>
              <th style="text-align:left;padding:6px 8px">ADX</th>
              <th style="text-align:left;padding:6px 8px">Hold</th>
              <th style="text-align:right;padding:6px 8px">Train Sharpe</th>
              <th style="text-align:right;padding:6px 8px">Val Sharpe</th>
              <th style="text-align:right;padding:6px 8px">Val Win%</th>
              <th style="text-align:right;padding:6px 8px">Val Avg P&L</th>
            </tr>
          </thead>
          <tbody id="wfo-leaderboard-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- PERFORMANCE TRACKER PANEL -->
<div class="panel" id="panel-performance">
  <div style="padding:24px;max-width:1100px;margin:0 auto">
    <h2 style="color:var(--accent);font-size:18px;margin-bottom:6px">📊 Performance Tracker</h2>
    <p style="color:var(--muted);font-size:13px;margin-bottom:24px">
      Log closed trades from the Tracker tab or enter them manually. Tracks your realized P&amp;L over time.
    </p>

    <!-- Summary stat cards -->
    <div id="perf-summary-cards" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:14px;margin-bottom:28px">
      <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--border)">
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px">TOTAL P&L</div>
        <div id="perf-total-pnl" style="font-size:22px;font-weight:700;color:var(--text)">—</div>
        <div id="perf-avg-pnl-pct" style="font-size:12px;color:var(--muted);margin-top:4px"></div>
      </div>
      <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--border)">
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px">TOTAL P&L %</div>
        <div id="perf-total-pnl-pct" style="font-size:22px;font-weight:700;color:var(--text)">—</div>
        <div id="perf-total-invested" style="font-size:12px;color:var(--muted);margin-top:4px"></div>
      </div>
      <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--border)">
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px">WIN RATE</div>
        <div id="perf-win-rate" style="font-size:22px;font-weight:700;color:var(--text)">—</div>
        <div id="perf-wl-counts" style="font-size:11px;color:var(--muted);margin-top:4px"></div>
      </div>
      <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--border)">
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px">AVG WIN</div>
        <div id="perf-avg-win" style="font-size:22px;font-weight:700;color:var(--green)">—</div>
      </div>
      <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--border)">
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px">AVG LOSS</div>
        <div id="perf-avg-loss" style="font-size:22px;font-weight:700;color:var(--red)">—</div>
      </div>
      <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--border)">
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px">BEST TRADE</div>
        <div id="perf-best" style="font-size:18px;font-weight:700;color:var(--green)">—</div>
      </div>
      <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--border)">
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px">WORST TRADE</div>
        <div id="perf-worst" style="font-size:18px;font-weight:700;color:var(--red)">—</div>
      </div>
      <!-- Advanced metrics for options trading (replaces Sharpe) -->
      <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--accent);border-width:1px">
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px" title="Average $ made per trade. The killer metric.">EXPECTANCY / TRADE</div>
        <div id="perf-expectancy" style="font-size:22px;font-weight:700;color:var(--text)">—</div>
        <div style="font-size:11px;color:var(--muted);margin-top:4px">positive = edge</div>
      </div>
      <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--accent);border-width:1px">
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px" title="Total wins / total losses. >1.5 decent, >2.5 strong.">PROFIT FACTOR</div>
        <div id="perf-profit-factor" style="font-size:22px;font-weight:700;color:var(--text)">—</div>
        <div style="font-size:11px;color:var(--muted);margin-top:4px">target &gt; 1.5</div>
      </div>
      <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--accent);border-width:1px">
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px" title="Like Sharpe but only counts downside vol. Proper for options.">SORTINO RATIO</div>
        <div id="perf-sortino" style="font-size:22px;font-weight:700;color:var(--text)">—</div>
        <div style="font-size:11px;color:var(--muted);margin-top:4px">target &gt; 0.5</div>
      </div>
      <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--accent);border-width:1px">
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px" title="Worst peak-to-trough loss across your equity curve.">MAX DRAWDOWN</div>
        <div id="perf-max-dd" style="font-size:22px;font-weight:700;color:var(--red)">—</div>
        <div id="perf-max-dd-pct" style="font-size:11px;color:var(--muted);margin-top:4px"></div>
      </div>
    </div>

    <!-- Open portfolio risk -->
    <div id="portfolio-risk-panel" style="background:var(--surface2);border-radius:12px;padding:16px 20px;border:1px solid var(--border);margin-bottom:28px;display:none">
      <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:10px">⚖ Open Portfolio Risk</div>
      <div id="portfolio-risk-body" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px;font-size:12px">
        <!-- filled by JS -->
      </div>
    </div>

    <!-- Cumulative P&L chart -->
    <div style="background:var(--surface2);border-radius:12px;padding:20px;border:1px solid var(--border);margin-bottom:28px">
      <div style="font-size:13px;color:var(--muted);font-weight:600;margin-bottom:14px">Cumulative P&L ($)</div>
      <canvas id="perf-chart" height="90"></canvas>
    </div>

    <!-- Manual trade entry form -->
    <div style="background:var(--surface2);border-radius:12px;padding:20px;border:1px solid var(--border);margin-bottom:28px">
      <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:14px">➕ Log a Trade Manually</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-bottom:14px">
        <div>
          <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">TICKER</label>
          <input id="mt-ticker" placeholder="AAPL" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
        </div>
        <div>
          <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">STRATEGY</label>
          <select id="mt-strategy" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px">
            <option value="buy_calls">Buy Calls</option>
            <option value="buy_puts">Buy Puts</option>
            <option value="sell_puts">Sell Puts</option>
            <option value="covered_calls">Covered Calls</option>
          </select>
        </div>
        <div>
          <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">STRIKE ($)</label>
          <input id="mt-strike" type="number" step="0.50" placeholder="195.00" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
        </div>
        <div>
          <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">EXPIRATION</label>
          <input id="mt-expiration" type="text" placeholder="2025-02-21" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
        </div>
        <div>
          <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">ENTRY ($)</label>
          <input id="mt-entry" type="number" step="0.01" placeholder="1.25" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
        </div>
        <div>
          <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">EXIT ($)</label>
          <input id="mt-exit" type="number" step="0.01" placeholder="2.50" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
        </div>
        <div>
          <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">CONTRACTS</label>
          <input id="mt-contracts" type="number" step="1" min="1" placeholder="1" value="1" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
        </div>
        <div>
          <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">CLOSE DATE</label>
          <input id="mt-date" type="date" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
        </div>
        <div style="grid-column:1/-1">
          <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">NOTES (optional)</label>
          <input id="mt-notes" placeholder="e.g. Exited on earnings gap up" style="width:100%;padding:7px 10px;background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
        </div>
      </div>
      <button onclick="addManualTrade()" class="btn btn-primary" style="font-size:13px;padding:8px 20px">➕ Add Trade</button>
      <span id="mt-status" style="margin-left:12px;font-size:12px;color:var(--muted)"></span>
    </div>

    <!-- Trade history table -->
    <div style="background:var(--surface2);border-radius:12px;padding:20px;border:1px solid var(--border)">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
        <div style="font-size:13px;font-weight:600;color:var(--text)">Trade History</div>
        <button onclick="refreshPerformance()" class="btn btn-ghost" style="font-size:12px;padding:5px 12px">🔄 Refresh</button>
      </div>
      <div id="perf-table-wrap" style="overflow-x:auto">
        <div style="color:var(--muted);font-size:13px;text-align:center;padding:20px">No trades logged yet.</div>
      </div>
    </div>
  </div>
</div>

<!-- BLACK-SCHOLES MODEL PANEL -->
<div class="panel" id="panel-bsmodel">
  <div style="padding:24px;max-width:860px;margin:0 auto">
    <h2 style="color:var(--accent);font-size:18px;margin-bottom:6px">⚖️ Black-Scholes Analyser</h2>
    <p style="color:var(--muted);font-size:13px;margin-bottom:24px">
      Enter any option's details to see its fair value, Greeks, intrinsic vs extrinsic breakdown,
      and whether the market is pricing vol fairly vs our GARCH estimate.
    </p>

    <!-- Input form -->
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:14px;margin-bottom:20px">
      <div>
        <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">TICKER (for GARCH)</label>
        <input id="bs-ticker" placeholder="e.g. AAPL" style="width:100%;padding:8px 10px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
      </div>
      <div>
        <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">STOCK PRICE ($)</label>
        <input id="bs-stock" type="number" step="0.01" placeholder="195.00" style="width:100%;padding:8px 10px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
      </div>
      <div>
        <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">STRIKE PRICE ($)</label>
        <input id="bs-strike" type="number" step="0.50" placeholder="195.00" style="width:100%;padding:8px 10px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
      </div>
      <div>
        <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">DAYS TO EXPIRY</label>
        <input id="bs-dte" type="number" step="1" placeholder="30" style="width:100%;padding:8px 10px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
      </div>
      <div>
        <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">MARKET IV (%)</label>
        <input id="bs-iv" type="number" step="0.1" placeholder="35.0" style="width:100%;padding:8px 10px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
      </div>
      <div>
        <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">MARKET PRICE ($)</label>
        <input id="bs-mktprice" type="number" step="0.01" placeholder="3.50" style="width:100%;padding:8px 10px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px"/>
      </div>
      <div>
        <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">OPTION TYPE</label>
        <select id="bs-type" style="width:100%;padding:8px 10px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px">
          <option value="call">Call</option>
          <option value="put">Put</option>
        </select>
      </div>
    </div>

    <button onclick="runBSAnalysis()" class="btn btn-primary" style="margin-bottom:28px">⚖️ Analyse</button>

    <!-- Results -->
    <div id="bs-results" style="display:none">

      <!-- Mispricing banner -->
      <div id="bs-banner" style="border-radius:12px;padding:16px 20px;margin-bottom:24px;font-size:15px;font-weight:600"></div>

      <!-- Three-column summary cards -->
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:24px">
        <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--border)">
          <div style="font-size:11px;color:var(--muted);margin-bottom:6px">BS FAIR VALUE</div>
          <div id="bs-fair" style="font-size:22px;font-weight:700;color:var(--text)">—</div>
          <div id="bs-vs-market" style="font-size:12px;color:var(--muted);margin-top:4px"></div>
        </div>
        <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--border)">
          <div style="font-size:11px;color:var(--muted);margin-bottom:6px">VOL COMPARISON</div>
          <div id="bs-vol-compare" style="font-size:15px;font-weight:600;color:var(--text)">—</div>
          <div id="bs-vol-verdict" style="font-size:12px;margin-top:4px"></div>
        </div>
        <div style="background:var(--surface2);border-radius:12px;padding:16px;border:1px solid var(--border)">
          <div style="font-size:11px;color:var(--muted);margin-bottom:6px">PREMIUM BREAKDOWN</div>
          <div id="bs-intrinsic" style="font-size:13px;color:var(--text)">—</div>
          <div id="bs-extrinsic" style="font-size:13px;color:var(--muted);margin-top:4px"></div>
        </div>
      </div>

      <!-- Greeks -->
      <div style="background:var(--surface2);border-radius:12px;padding:18px;border:1px solid var(--border);margin-bottom:24px">
        <div style="font-size:12px;font-weight:600;color:var(--muted);margin-bottom:14px;letter-spacing:.05em">GREEKS</div>
        <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px" id="bs-greeks"></div>
      </div>

      <!-- Interpretation -->
      <div style="background:var(--surface2);border-radius:12px;padding:18px;border:1px solid var(--border)">
        <div style="font-size:12px;font-weight:600;color:var(--muted);margin-bottom:12px;letter-spacing:.05em">WHAT THIS MEANS FOR YOUR TRADE</div>
        <div id="bs-interpretation" style="font-size:13px;color:var(--text);line-height:1.7"></div>
      </div>

    </div>
    <div id="bs-error" style="display:none;color:var(--red);font-size:13px;margin-top:12px"></div>
  </div>
</div>

<!-- CONFIRM MODAL -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <h2>🔒 Confirm Trade Execution</h2>
    <div class="sub">Review the order below carefully. This will open SoFi in your browser and fill in the trade form. <b>No money moves until you review and submit on the SoFi page.</b></div>
    <div class="trade-summary" id="modal-summary"></div>
    <div class="guardrail-warn" id="modal-guardrail" style="display:none"></div>
    <input class="confirm-input" id="modal-confirm-input" placeholder='Type CONFIRM to override the $1,000 limit'
           style="display:none" oninput="checkConfirmInput()"/>
    <div class="modal-actions">
      <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
      <button class="btn btn-green" id="modal-btn-confirm" onclick="submitExecution()">
        ✓ Confirm &amp; Place Order
      </button>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════
     CALIBRATE PANEL
     ══════════════════════════════════════════════════════════ -->
<div class="panel" id="panel-autotrade">
  <div class="disclaimer">
    <b>🤖 Autotrade</b> — trades the scanner's signals through a broker on its own, with the same exit rules the backtest was run with
    (profit target / stop / trailing stop / stagnation / RSI-BB reversal / max hold / expiry).
    Default broker is a <b>local paper ledger</b> (no account needed). <b>Alpaca</b> runs on its paper endpoint; real money additionally
    requires <code>OPTIONS_BOT_ALLOW_LIVE=1</code> in the bot's environment. Kill switch: create a file named <code>STOP_AUTOTRADE</code>
    in the bot folder. Everything it does is in the log below and in 📊 Performance.
  </div>
  <div class="stats" id="at-stats"></div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:8px">
    <button id="at-toggle" class="btn btn-primary" onclick="atToggle()">▶ Enable autotrading</button>
    <button class="btn btn-ghost" onclick="atRunNow()">Run one cycle now</button>
    <button class="btn btn-ghost" onclick="refreshAutotrade()">↻ Refresh</button>
    <span id="at-msg" style="font-size:12px;color:var(--muted)"></span>
  </div>
  <div id="at-skipped" style="margin-bottom:16px"></div>
  <div class="card" style="padding:16px;margin-bottom:16px">
    <div style="font-weight:700;margin-bottom:10px">Open auto positions</div>
    <div id="at-positions"></div>
  </div>
  <div class="card" style="padding:16px;margin-bottom:16px">
    <div style="font-weight:700;margin-bottom:10px">Settings <span style="font-weight:400;color:var(--muted);font-size:12px">(saved to autotrade_config.json)</span></div>
    <div id="at-config" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px"></div>
    <div style="margin-top:12px;display:flex;gap:10px;align-items:center">
      <button class="btn btn-primary" onclick="atSaveConfig()">💾 Save settings</button>
      <span id="at-config-msg" style="font-size:12px;color:var(--muted)"></span>
    </div>
  </div>
  <div class="card" style="padding:16px">
    <div style="font-weight:700;margin-bottom:10px">Activity log</div>
    <div id="at-log"></div>
  </div>
</div>

<div class="panel" id="panel-calibrate">
  <div style="padding:24px;max-width:1000px;margin:0 auto">
    <h2 style="color:var(--accent);font-size:18px;margin-bottom:4px">🎯 Signal Calibration</h2>
    <p style="color:var(--muted);font-size:13px;margin-bottom:24px">
      Trains the bot on recent market data to learn which signals have actually predicted
      profitable moves. Scores update automatically once calibration completes.
    </p>

    <!-- VIX Regime Card -->
    <div id="cal-regime-card" style="background:var(--surface2);border-radius:14px;
         padding:18px 20px;border:1px solid var(--border);margin-bottom:20px">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <div style="font-size:28px" id="cal-regime-icon">⏳</div>
        <div style="flex:1">
          <div style="font-size:13px;font-weight:700;color:var(--text)" id="cal-regime-label">Loading regime…</div>
          <div style="font-size:12px;color:var(--muted);margin-top:2px" id="cal-regime-desc"></div>
        </div>
        <div style="display:flex;gap:24px;flex-wrap:wrap">
          <div style="text-align:center">
            <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px">VIX</div>
            <div id="cal-vix-val" style="font-size:20px;font-weight:800;color:var(--text)">—</div>
          </div>
          <div style="text-align:center">
            <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px">Active RSI OS</div>
            <div id="cal-rsi-os" style="font-size:20px;font-weight:800;color:var(--text)">—</div>
          </div>
          <div style="text-align:center">
            <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px">Active RSI OB</div>
            <div id="cal-rsi-ob" style="font-size:20px;font-weight:800;color:var(--text)">—</div>
          </div>
        </div>
      </div>
      <div id="cal-threshold-note" style="margin-top:10px;font-size:11px;
           color:var(--muted);padding-top:10px;border-top:1px solid var(--border)"></div>
    </div>

    <!-- Run Calibration -->
    <div style="background:var(--surface2);border-radius:14px;padding:18px 20px;
         border:1px solid var(--border);margin-bottom:20px">
      <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:14px">
        <div style="flex:1">
          <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:4px">
            📡 Run Signal Calibration
          </div>
          <div style="font-size:12px;color:var(--muted)">
            Backtests every signal combination over your chosen window using free historical data.
            Takes ~2–3 minutes. Re-run monthly to keep weights current.
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
          <select id="cal-months" style="background:var(--surface);border:1px solid var(--border);
                  color:var(--text);border-radius:8px;padding:6px 10px;font-size:13px">
            <option value="6">6 months</option>
            <option value="12" selected>12 months</option>
            <option value="18">18 months</option>
            <option value="24">24 months</option>
          </select>
          <select id="cal-universe" style="background:var(--surface);border:1px solid var(--border);
                  color:var(--text);border-radius:8px;padding:6px 10px;font-size:13px">
            <option value="focus">Focus List (~100 stocks, fast)</option>
            <option value="snp500">S&amp;P 500 (500 stocks, ~10 min)</option>
          </select>
          <button id="cal-run-btn" class="btn btn-primary" onclick="runCalibration()"
                  style="white-space:nowrap">🔬 Run Calibration</button>
          <select id="lr-source" style="background:var(--surface);border:1px solid var(--border);
                  color:var(--text);border-radius:8px;padding:6px 10px;font-size:13px"
                  onchange="document.getElementById('lr-hist-opts').style.display=this.value==='backtest'?'flex':'none'">
            <option value="signal_log">🧠 Smart: Live Signal Log</option>
            <option value="backtest">🧠 Smart: Historical Backtest</option>
          </select>
          <span id="lr-hist-opts" style="display:none;align-items:center;gap:6px;flex-wrap:wrap">
            <select id="lr-months" style="background:var(--surface);border:1px solid var(--border);
                    color:var(--text);border-radius:8px;padding:6px 10px;font-size:13px">
              <option value="6">6 months</option>
              <option value="12" selected>12 months</option>
              <option value="18">18 months</option>
            </select>
            <select id="lr-universe" style="background:var(--surface);border:1px solid var(--border);
                    color:var(--text);border-radius:8px;padding:6px 10px;font-size:13px">
              <option value="snp500" selected>S&amp;P 500</option>
              <option value="focus">Focus List</option>
            </select>
          </span>
          <button id="cal-lr-btn" class="btn" onclick="runLogisticCalibration()"
                  style="white-space:nowrap;background:var(--surface);border:1px solid var(--accent);
                         color:var(--accent)">
            🧠 Smart Calibrate
          </button>
        </div>
      </div>
      <div id="cal-progress-wrap" style="display:none">
        <div style="height:6px;background:var(--surface);border-radius:3px;overflow:hidden;margin-bottom:6px">
          <div id="cal-progress-bar" style="height:100%;background:var(--accent);
               width:0%;transition:width .4s;border-radius:3px"></div>
        </div>
        <div id="cal-status-text" style="font-size:12px;color:var(--muted)">Starting…</div>
      </div>
      <div id="cal-last-run" style="font-size:11px;color:var(--muted);margin-top:8px"></div>
    </div>

    <!-- Signal Win Rates -->
    <div id="cal-results-wrap" style="display:none">

      <!-- Simulated signal quality -->
      <div style="background:var(--surface2);border-radius:14px;padding:18px 20px;
           border:1px solid var(--border);margin-bottom:20px">
        <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:4px">
          📊 Signal Win Rates <span style="font-size:11px;font-weight:400;color:var(--muted)" id="cal-sim-meta"></span>
        </div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:14px">
          How often each signal predicted a profitable move over the last
          <span id="cal-months-label">12</span> months.
          Signals scoring ≥60% boost live scores; &lt;40% reduce them.
        </div>
        <div id="cal-signal-rows"></div>
      </div>

      <!-- Logistic Regression Weights card (shown after Smart Calibrate) -->
      <div id="cal-lr-wrap" style="display:none;background:var(--surface2);border-radius:14px;
           padding:18px 20px;border:1px solid var(--accent);margin-bottom:20px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;flex-wrap:wrap">
          <div style="font-size:13px;font-weight:700;color:var(--accent)">🧠 Smart Calibration — Logistic Regression Weights</div>
          <div style="margin-left:auto;font-size:11px;color:var(--muted)" id="cal-lr-meta"></div>
        </div>
        <div style="font-size:12px;color:var(--muted);margin-bottom:14px">
          Signals weighted by their <em>joint</em> predictive power across your actual bot recommendations.
          Unlike individual win-rates, this accounts for signal interactions — e.g. RSI oversold
          is far more valuable <em>when combined with</em> MACD bull cross.
          Bars &gt; 1.0× = signal helps when others are present; &lt; 1.0× = signal is redundant or harmful in context.
        </div>
        <div id="cal-lr-status" style="font-size:12px;color:var(--muted);margin-bottom:10px"></div>
        <div id="cal-lr-rows"></div>
      </div>

      <!-- Per-strategy sim summary -->
      <div style="background:var(--surface2);border-radius:14px;padding:18px 20px;
           border:1px solid var(--border);margin-bottom:20px">
        <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:4px">
          📈 Simulated Strategy Performance
        </div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:14px">
          Full risk/return profile across <span id="cal-sim-n">all</span> simulated trades —
          every setup the bot would have recommended. Per-trade returns shown as %;
          cumulative P&amp;L assumes $100 per signal.
        </div>
        <div id="cal-sim-overall" style="display:grid;
             grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:16px"></div>
        <div style="font-size:12px;font-weight:600;color:var(--muted);margin-bottom:8px">By strategy</div>
        <div id="cal-strat-rows" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px"></div>
      </div>

      <!-- Live signal log (unbiased) -->
      <div style="background:var(--surface2);border-radius:14px;
           padding:18px 20px;border:1px solid var(--border);margin-bottom:20px">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;flex-wrap:wrap">
          <div style="font-size:13px;font-weight:700;color:var(--text);cursor:pointer;user-select:none"
               onclick="toggleSigLog()" title="Click to collapse/expand">
            📡 All Bot Recommendations <span id="siglog-chevron" style="font-size:11px;color:var(--muted)">▼</span>
          </div>
          <div style="font-size:11px;color:var(--muted)">
            Every signal the bot has generated — whether you traded it or not
          </div>
          <div style="margin-left:auto;display:flex;align-items:center;gap:10px;flex-wrap:wrap">
            <label style="font-size:11px;color:var(--muted);display:flex;align-items:center;gap:6px">
              Show
              <select id="siglog-rows-select" onchange="rerenderSigLog()"
                      style="background:var(--surface);border:1px solid var(--border);
                             color:var(--text);border-radius:6px;padding:2px 6px;font-size:11px">
                <option value="10">10</option>
                <option value="25" selected>25</option>
                <option value="50">50</option>
                <option value="100">100</option>
                <option value="0">All</option>
              </select>
              rows
            </label>
            <div style="font-size:11px;color:var(--muted)" id="cal-siglog-meta"></div>
          </div>
        </div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:14px" id="cal-siglog-notice"></div>

        <div id="siglog-body-wrap">
          <!-- Per-strategy W/L summary cards -->
          <div id="siglog-strat-summary" style="display:none;margin-bottom:18px">
            <div style="font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;
                 letter-spacing:.5px;margin-bottom:10px">Performance by Strategy</div>
            <div id="siglog-strat-cards" style="display:flex;gap:10px;flex-wrap:wrap"></div>
          </div>

          <!-- Win rate bars (only shown once enough entries are resolved) -->
          <div id="cal-siglog-quality-wrap" style="display:none;margin-bottom:16px">
            <div style="font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;
                 letter-spacing:.5px;margin-bottom:8px">Signal Win Rates (from live recommendations)</div>
            <div id="cal-siglog-rows"></div>
          </div>

          <!-- Recommendations table -->
          <div style="overflow-x:auto">
            <table style="width:100%;border-collapse:collapse;font-size:12px">
              <thead>
                <tr style="border-bottom:1px solid var(--border)">
                  <th style="text-align:left;padding:6px 8px;color:var(--muted);font-weight:600">Date</th>
                  <th style="text-align:left;padding:6px 8px;color:var(--muted);font-weight:600">Ticker</th>
                  <th style="text-align:left;padding:6px 8px;color:var(--muted);font-weight:600">Strategy</th>
                  <th style="text-align:right;padding:6px 8px;color:var(--muted);font-weight:600">Score</th>
                  <th style="text-align:left;padding:6px 8px;color:var(--muted);font-weight:600">VIX</th>
                  <th style="text-align:left;padding:6px 8px;color:var(--muted);font-weight:600">Resolves</th>
                  <th style="text-align:right;padding:6px 8px;color:var(--muted);font-weight:600">Stock Δ</th>
                  <th style="text-align:left;padding:6px 8px;color:var(--muted);font-weight:600">Status</th>
                </tr>
              </thead>
              <tbody id="cal-siglog-tbody">
                <tr><td colspan="8" style="padding:20px;text-align:center;color:var(--muted)">
                  Run a scan to start logging recommendations
                </td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Actual user trades -->
      <div id="cal-actual-wrap" style="background:var(--surface2);border-radius:14px;
           padding:18px 20px;border:1px solid var(--border);margin-bottom:20px;display:none">
        <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:14px">
          🏆 Your Actual Trades
          <span style="font-size:11px;font-weight:400;color:var(--muted);margin-left:6px">
            Real closed trades from your Performance log — weighted more heavily than simulations
          </span>
        </div>
        <div id="cal-actual-rows" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px"></div>
      </div>

    </div><!-- /cal-results-wrap -->

    <!-- ─────────────────────────────────────────────────────────
         HOLD HORIZON ANALYSIS
         ───────────────────────────────────────────────────────── -->
    <div style="background:var(--surface2);border-radius:14px;
         padding:18px 20px;border:1px solid var(--border);margin-bottom:20px">
      <div style="display:flex;align-items:flex-start;gap:14px;flex-wrap:wrap;margin-bottom:12px">
        <div style="flex:1">
          <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:4px">
            ⏱️ Hold Horizon Analysis
          </div>
          <div style="font-size:12px;color:var(--muted)">
            Runs the backtest but records option P&amp;L at 1, 2, 3, 5, 7, 10, 14 and 21 day
            checkpoints so you can see exactly when each strategy peaks — and whether you're
            leaving money on the table by holding too long.
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
          <select id="hs-universe"
                  style="background:var(--surface);border:1px solid var(--border);
                         color:var(--text);border-radius:8px;padding:6px 10px;font-size:13px">
            <option value="focus">Focus List (~100 stocks, fast)</option>
            <option value="snp500">S&amp;P 500 (500 stocks, ~10 min)</option>
          </select>
          <select id="hs-months"
                  style="background:var(--surface);border:1px solid var(--border);
                         color:var(--text);border-radius:8px;padding:6px 10px;font-size:13px">
            <option value="6">6 months</option>
            <option value="12" selected>12 months</option>
            <option value="18">18 months</option>
          </select>
          <button id="hs-run-btn" class="btn btn-primary" onclick="runHoldSweep()"
                  style="white-space:nowrap">⏱️ Run Analysis</button>
        </div>
      </div>

      <!-- Progress bar -->
      <div id="hs-progress-wrap" style="display:none;margin-bottom:12px">
        <div style="height:6px;background:var(--surface);border-radius:3px;overflow:hidden;margin-bottom:4px">
          <div id="hs-progress-bar"
               style="height:100%;background:var(--accent);width:30%;border-radius:3px;
                      animation:pulse-bar 1.4s ease-in-out infinite"></div>
        </div>
        <div id="hs-status-text" style="font-size:12px;color:var(--muted)">Starting…</div>
      </div>

      <!-- Results -->
      <div id="hs-results-wrap" style="display:none">
        <div style="font-size:11px;color:var(--muted);margin-bottom:12px" id="hs-meta"></div>

        <!-- Chart -->
        <div style="position:relative;height:280px;margin-bottom:16px">
          <canvas id="hs-chart"></canvas>
        </div>

        <!-- Summary table -->
        <div style="overflow-x:auto">
          <table style="width:100%;border-collapse:collapse;font-size:12px" id="hs-table">
            <thead>
              <tr style="border-bottom:1px solid var(--border)">
                <th style="text-align:left;padding:6px 8px;color:var(--muted);font-weight:600">Strategy</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600" title="Number of backtested trades for this strategy">Trades</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">1d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">2d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">3d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">5d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">7d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">10d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">14d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">21d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--accent);font-weight:700">🏆 Best</th>
              </tr>
            </thead>
            <tbody id="hs-tbody"></tbody>
          </table>
        </div>

        <!-- Win-rate table -->
        <div style="margin-top:12px;overflow-x:auto">
          <div style="font-size:11px;font-weight:600;color:var(--muted);
               text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">
            Win Rate % at Each Hold Duration
          </div>
          <table style="width:100%;border-collapse:collapse;font-size:12px" id="hs-wr-table">
            <thead>
              <tr style="border-bottom:1px solid var(--border)">
                <th style="text-align:left;padding:6px 8px;color:var(--muted);font-weight:600">Strategy</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600" title="Number of backtested trades for this strategy">Trades</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">1d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">2d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">3d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">5d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">7d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">10d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">14d</th>
                <th style="text-align:center;padding:6px 8px;color:var(--muted);font-weight:600">21d</th>
              </tr>
            </thead>
            <tbody id="hs-wr-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>

  </div>
</div><!-- /panel-calibrate -->

<script>
// ── State ──────────────────────────────────────────────────────
let activeTab  = 'sell_puts';
let pendingExec = null;
let scanPoll   = null;
let resultPoll = null;
const chartRegistry = {};

// Escape text that comes from the server / third parties (company names,
// RSS headlines, publishers) before it goes into innerHTML or an attribute.
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
// Escape for use inside a single-quoted JS string in an onclick="" attribute.
function jsq(s) {
  return esc(String(s ?? '').replace(/\\/g, '\\\\').replace(/'/g, "\\'"));
}
// Only http(s) links are allowed into href (RSS feeds are untrusted input).
function safeHref(u) {
  return /^https?:\/\//i.test(String(u || '')) ? esc(u) : '#';
}

// ── Tab switching ─────────────────────────────────────────────
function switchTab(tab) {
  activeTab = tab;
  // Match the tab by the name in its onclick (a positional list drifted every
  // time a tab was added — the Calibrate tab never highlighted).
  document.querySelectorAll('.tab').forEach(el=>{
    el.classList.toggle('active', (el.getAttribute('onclick')||'').includes(`'${tab}'`));
  });
  document.querySelectorAll('.panel').forEach(el=> el.classList.remove('active'));
  document.getElementById('panel-'+tab).classList.add('active');
  if (tab==='queue')       refreshQueue();
  if (tab==='history')     refreshHistory();
  if (tab==='tracked')     refreshTracked();
  if (tab==='performance') refreshPerformance();
  if (tab==='autotrade')   refreshAutotrade();
}

// ── Autotrade tab ─────────────────────────────────────────────
const AT_FIELDS = [
  ['broker','Broker','select',['local_paper','alpaca']],
  ['paper','Alpaca: paper endpoint','bool'],
  ['alpaca_key_id','Alpaca key ID','password'],
  ['alpaca_secret','Alpaca secret','password'],
  ['allowed_strategies','Strategies (comma-separated)','text'],
  ['min_score','Min signal score','number'],
  ['max_open_positions','Max open auto positions','number'],
  ['max_new_per_cycle','Max new trades per cycle','number'],
  ['max_position_usd','Max $ premium per long trade','number'],
  ['max_collateral_usd','Max $ collateral per short put','number'],
  ['daily_loss_cap_usd','Daily loss cap $ (0 = off)','number'],
  ['max_spread_pct','Max bid/ask spread %','number'],
  ['min_dte','Min DTE','number'],
  ['max_dte','Max DTE','number'],
  ['market_hours_only','Regular hours only','bool'],
  ['interval_mins','Cycle every (minutes)','number'],
  ['scan_if_stale_hours','Rescan when results older than (h)','number'],
  ['reentry_cooldown_days','Re-entry cooldown (trading days)','number'],
  ['pending_order_timeout_mins','Cancel unfilled orders after (min)','number'],
  ['premarket_scan_time','Pre-market scan time (ET, blank = off)','text'],
  ['entry_window_et','Entry window (ET, e.g. 14:45-15:55; blank = all day)','text'],
  ['work_orders','Work orders from the mid (not cross the spread)','bool'],
  ['order_step_mins','Minutes per price rung','number'],
  ['skip_earnings_days','Skip if earnings within (days, 0 = off)','number'],
  ['skip_fomc_day','Skip FOMC decision days','bool'],
  ['max_per_sector','Max open positions per sector (0 = off)','number'],
  ['vol_scale_sizing','Scale $ per trade by volatility','bool'],
  ['conviction_boost','Conviction boost × (squeeze breakout / high-VIX reversion)','number'],
  ['weekly_loss_cap_usd','Weekly loss cap $ (0 = off)','number'],
  ['paper_starting_cash','Paper starting cash $','number'],
  ['paper_slippage_pct','Paper slippage %','number'],
];
const AT_INPUT = 'width:100%;padding:7px 10px;border-radius:8px;border:1px solid var(--border);background:var(--surface2);color:var(--text);font-size:13px;box-sizing:border-box';
function atField([key, label, type, opts], val) {
  let ctl;
  if (type === 'select') {
    ctl = `<select id="at-f-${key}" style="${AT_INPUT}">${opts.map(o=>`<option value="${o}"${o===val?' selected':''}>${o}</option>`).join('')}</select>`;
  } else if (type === 'bool') {
    ctl = `<select id="at-f-${key}" style="${AT_INPUT}"><option value="true"${val?' selected':''}>Yes</option><option value="false"${!val?' selected':''}>No</option></select>`;
  } else if (type === 'password') {
    ctl = `<input id="at-f-${key}" type="password" value="" placeholder="${esc(val || '(not set)')}" autocomplete="new-password" style="${AT_INPUT}">`;
  } else {
    const v = Array.isArray(val) ? val.join(', ') : (val ?? '');
    ctl = `<input id="at-f-${key}" type="${type==='number'?'number':'text'}" step="any" value="${esc(v)}" style="${AT_INPUT}">`;
  }
  return `<div><label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">${label}</label>${ctl}</div>`;
}
function atStat(label, val, color) {
  return `<div class="stat"><div style="font-size:11px;color:var(--muted)">${label}</div><div style="font-size:17px;font-weight:700;color:${color||'var(--text)'}">${val}</div></div>`;
}
let atStatus = null;
async function refreshAutotrade() {
  let s;
  try { s = await fetch('/api/autotrade/status').then(r=>r.json()); } catch(e) { return; }
  atStatus = s;
  const acct = s.account || {};
  const money = v => v==null ? '—' : '$' + (+v).toLocaleString('en-US',{maximumFractionDigits:0});
  const liveTag = s.broker==='alpaca' ? (s.paper ? ' (paper)' : ' ⚠ LIVE') : ' (local paper)';
  document.getElementById('at-stats').innerHTML =
    atStat('Status', s.kill_switch ? '⛔ Kill switch' : (s.enabled ? '🟢 Enabled' : '⚪ Disabled'),
           s.kill_switch ? 'var(--red)' : (s.enabled ? 'var(--green)' : 'var(--muted)')) +
    atStat('Broker', esc(s.broker) + liveTag, (s.broker==='alpaca' && !s.paper) ? 'var(--red)' : undefined) +
    atStat('Market', s.market_open ? 'Open' : 'Closed') +
    atStat('Equity', acct.equity!=null ? money(acct.equity) : (s.account_error ? '⚠ broker error' : '—')) +
    atStat('Cash', money(acct.cash)) +
    atStat('Open auto', `${s.open_auto_positions}${s.pending_entries ? ' +'+s.pending_entries+' pending' : ''}`) +
    atStat('Today realised', (s.today.realized>=0?'+':'') + '$' + (+s.today.realized).toFixed(2) + (s.daily_loss_cap_hit ? ' ⛔ cap' : ''),
           s.today.realized < 0 ? 'var(--red)' : 'var(--green)') +
    atStat('Last cycle', s.last_cycle ? new Date(s.last_cycle).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) : 'never') +
    atStat('Next cycle', (s.enabled && s.next_cycle) ? new Date(s.next_cycle).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) : '—') +
    atStat('Max hold', s.hold_days != null ? s.hold_days + ' days' : '—');
  const btn = document.getElementById('at-toggle');
  btn.textContent = s.enabled ? '⏸ Disable autotrading' : '▶ Enable autotrading';
  btn.className = 'btn ' + (s.enabled ? 'btn-danger' : 'btn-primary');
  let msg = '';
  if (s.last_result) {
    const r = s.last_result;
    msg = r.ran ? `Last cycle: ${r.exits} exit(s), ${r.entries} entry(ies)` + ((r.skipped||[]).length ? `, ${r.skipped.length} skipped` : '') + (r.scan ? ` · scan: ${r.scan}` : '')
                : `Last cycle: ${r.reason || 'idle'}`;
  }
  if (s.account_error) msg += ` · ⚠ broker: ${s.account_error}`;
  if (s.last_error)    msg += ` · ⚠ ${s.last_error}`;
  document.getElementById('at-msg').textContent = msg;
  const skipped = (s.last_result && s.last_result.skipped) || [];
  document.getElementById('at-skipped').innerHTML = skipped.length
    ? `<details><summary style="cursor:pointer;color:var(--muted);font-size:12px">Why ${skipped.length} candidate(s) were skipped last cycle</summary><div style="font-size:12px;color:var(--muted);margin-top:6px">${skipped.map(esc).join('<br>')}</div></details>` : '';
  const cfgEl = document.getElementById('at-config');
  if (!cfgEl.dataset.dirty) cfgEl.innerHTML = AT_FIELDS.map(f => atField(f, s.config[f[0]])).join('');
  cfgEl.oninput = () => { cfgEl.dataset.dirty = '1'; };
  refreshAutotradePositions();
  refreshAutotradeLog();
}
async function refreshAutotradePositions() {
  let positions = [];
  try { positions = (await fetch('/api/tracked').then(r=>r.json())).filter(p => p.auto); } catch(e) { return; }
  const el = document.getElementById('at-positions');
  if (!positions.length) { el.innerHTML = '<div style="color:var(--muted);font-size:13px">No open auto positions.</div>'; return; }
  el.innerHTML = `<table class="queue-table"><thead><tr><th>Ticker</th><th>Strategy</th><th>Contract</th><th>Qty</th><th>Entry</th><th>Now</th><th>P&amp;L</th><th>Status</th><th></th></tr></thead><tbody>
    ${positions.map(p => { const sig = p.last_signal || {}; const opt = p.option || {};
      const pnl = sig.pct_change != null ? `${sig.pct_change>=0?'+':''}${(+sig.pct_change).toFixed(1)}%` : '—';
      const status = p.pending_close ? '⏳ closing' : (sig.triggered ? '🔔 exit due' : (p.force_close ? '⏳ close requested' : 'open'));
      return `<tr><td><b>${esc(p.ticker)}</b></td><td>${esc(p.strategy_label||p.strategy)}</td>
        <td style="font-size:12px">${esc(opt.type||'')} $${(+opt.strike||0).toFixed(2)} ${esc(opt.expiration||'')}</td>
        <td>${p.contracts}</td><td>$${(+p.entry_price||0).toFixed(2)}</td>
        <td>${sig.current_option_mid!=null ? '$'+(+sig.current_option_mid).toFixed(2) : '—'}</td>
        <td style="color:${sig.pct_change>=0?'var(--green)':'var(--red)'}">${pnl}</td>
        <td style="font-size:12px">${status}<br><span style="color:var(--muted);font-size:11px">${esc((sig.reasons||[]).join('; ')).slice(0,160)}</span></td>
        <td><button class="btn btn-ghost" style="font-size:12px;padding:5px 10px" onclick="atClose('${p.id}')">Close</button></td></tr>`; }).join('')}
    </tbody></table>`;
}
async function refreshAutotradeLog() {
  let log = [];
  try { log = await fetch('/api/autotrade/log?n=60').then(r=>r.json()); } catch(e) { return; }
  const el = document.getElementById('at-log');
  if (!log.length) { el.innerHTML = '<div style="color:var(--muted);font-size:13px">Nothing logged yet.</div>'; return; }
  const color = a => /error|rejected|failed|kill/.test(a) ? 'var(--red)' : (a==='entry' ? 'var(--green)' : (a==='exit' ? 'var(--blue)' : 'var(--text)'));
  el.innerHTML = `<table class="queue-table" style="font-size:12px"><thead><tr><th>Time</th><th>Action</th><th>Details</th></tr></thead><tbody>
    ${log.map(e => { const {ts, action, detail, ...rest} = e;
      const kv = Object.entries(rest).map(([k,v]) => `${k}=${v}`).join(' ');
      return `<tr><td style="white-space:nowrap">${esc(String(ts).replace('T',' '))}</td><td style="color:${color(action)};font-weight:600">${esc(action)}</td><td>${esc(kv)}${detail ? '<br><span style="color:var(--muted)">'+esc(detail).slice(0,300)+'</span>' : ''}</td></tr>`; }).join('')}
    </tbody></table>`;
}
async function atToggle() {
  if (!atStatus) return;
  const enabling = !atStatus.enabled;
  if (enabling && atStatus.broker === 'alpaca' && !atStatus.paper) {
    if (!confirm('This will trade REAL MONEY through Alpaca. Are you sure?')) return;
  }
  const d = await fetch('/api/autotrade/toggle', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({enabled: enabling})}).then(r=>r.json());
  if (!d.ok) alert(d.message || 'Could not change the autotrader');
  refreshAutotrade();
}
async function atRunNow() {
  document.getElementById('at-msg').textContent = 'Running one cycle…';
  try {
    const d = await fetch('/api/autotrade/run', {method:'POST'}).then(r=>r.json());
    const r = d.result || {};
    document.getElementById('at-msg').textContent = r.ran ? `Cycle done: ${r.exits} exit(s), ${r.entries} entry(ies), ${(r.skipped||[]).length} skipped` : `Cycle skipped: ${r.reason}`;
  } catch(e) { document.getElementById('at-msg').textContent = 'Cycle failed: ' + e; }
  refreshAutotrade();
}
async function atSaveConfig() {
  const patch = {};
  for (const [key, , type] of AT_FIELDS) {
    const el = document.getElementById('at-f-' + key); if (!el) continue;
    const v = el.value;
    if (type === 'password') { if (v) patch[key] = v; }
    else if (type === 'bool') patch[key] = (v === 'true');
    else if (type === 'number') { const n = parseFloat(v); if (!Number.isNaN(n)) patch[key] = n; }   // blank = keep the stored value
    else if (key === 'allowed_strategies') patch[key] = v.split(',').map(s=>s.trim()).filter(Boolean);
    else patch[key] = v;
  }
  const d = await fetch('/api/autotrade/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(patch)}).then(r=>r.json());
  document.getElementById('at-config-msg').textContent = d.ok ? 'Saved ✓' : ('Error: ' + (d.message||''));
  delete document.getElementById('at-config').dataset.dirty;
  refreshAutotrade();
}
async function atClose(id) {
  if (!confirm('Close this position through the broker on the next cycle?')) return;
  await fetch('/api/autotrade/close/' + id, {method:'POST'});
  refreshAutotrade();
}
setInterval(() => { if (activeTab === 'autotrade') refreshAutotrade(); }, 30000);

// ── Scan mode toggle ─────────────────────────────────────────
let currentScanMode = 'focus';
function setScanMode(mode) {
  currentScanMode = mode;
  document.getElementById('mode-focus').style.background = mode==='focus' ? 'var(--accent)' : 'var(--surface2)';
  document.getElementById('mode-focus').style.color      = mode==='focus' ? '#fff' : 'var(--muted)';
  document.getElementById('mode-full').style.background  = mode==='full'  ? 'var(--accent)' : 'var(--surface2)';
  document.getElementById('mode-full').style.color       = mode==='full'  ? '#fff' : 'var(--muted)';
}

// ── Green candle toggle ───────────────────────────────────────
let greenCandleOn = false;
function _applyGreenCandleStyle(on) {
  const btn = document.getElementById('btn-green-candle');
  if (!btn) return;
  btn.style.background = on ? '#16a34a' : 'var(--surface2)';
  btn.style.color      = on ? '#fff'    : 'var(--muted)';
  btn.style.borderColor= on ? '#16a34a' : 'var(--border)';
  btn.title = on ? 'Green Candle filter ON — buy calls require current close > previous close'
                 : 'Green Candle filter OFF — click to require confirming green candle on buy calls';
}
async function toggleGreenCandle() {
  greenCandleOn = !greenCandleOn;
  _applyGreenCandleStyle(greenCandleOn);
  await fetch('/api/scan/green-candle', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({enabled: greenCandleOn})
  });
}
async function initGreenCandle() {
  try {
    const d = await fetch('/api/scan/green-candle').then(r=>r.json());
    greenCandleOn = d.green_candle;
    _applyGreenCandleStyle(greenCandleOn);
  } catch(e) {}
}

// ── Red candle toggle (buy puts) ──────────────────────────────
let redCandleOn = false;
function _applyRedCandleStyle(on) {
  const btn = document.getElementById('btn-red-candle');
  if (!btn) return;
  btn.style.background  = on ? '#dc2626' : 'var(--surface2)';
  btn.style.color       = on ? '#fff'    : 'var(--muted)';
  btn.style.borderColor = on ? '#dc2626' : 'var(--border)';
  btn.title = on ? 'Red Candle filter ON — buy puts require current close < previous close'
                 : 'Red Candle filter OFF — click to require confirming red candle on buy puts';
}
async function toggleRedCandle() {
  redCandleOn = !redCandleOn;
  _applyRedCandleStyle(redCandleOn);
  await fetch('/api/scan/red-candle', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({enabled: redCandleOn})
  });
}
async function initRedCandle() {
  try {
    const d = await fetch('/api/scan/red-candle').then(r=>r.json());
    redCandleOn = d.red_candle;
    _applyRedCandleStyle(redCandleOn);
  } catch(e) {}
}

// ── Scan ──────────────────────────────────────────────────────
async function startScan() {
  document.getElementById('btn-scan').disabled = true;
  const r = await fetch('/api/scan/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode: currentScanMode})
  });
  if (!r.ok) { alert('Scan already running'); document.getElementById('btn-scan').disabled=false; return; }
  pollScanStatus();
  pollResults();
}

function pollScanStatus() {
  if (scanPoll) clearInterval(scanPoll);
  let seenRunning = false;
  scanPoll = setInterval(async () => {
    let data;
    try { data = await fetch('/api/scan/status').then(r=>r.json()); }
    catch(e) { return; }
    const bar  = document.getElementById('scan-bar');
    const info = document.getElementById('scan-info');
    if (data.running) {
      seenRunning = true;
      const pct = data.total ? Math.round(data.progress/data.total*100) : 0;
      bar.style.width = pct+'%';
      info.textContent = `Scanning… ${data.progress}/${data.total} (${pct}%)`;
    } else if (seenRunning) {
      // Only stop when we've confirmed the scan actually ran
      bar.style.width = '100%';
      setTimeout(()=>{ bar.style.width='0%'; },800);
      const lastStr = data.last_scan
        ? `Last scan: ${new Date(data.last_scan).toLocaleTimeString()}`
        : 'Not yet scanned';
      const nextStr = data.next_rescan ? ` · Next: ${data.next_rescan}` : '';
      info.textContent = lastStr + nextStr;
      document.getElementById('btn-scan').disabled = false;
      clearInterval(scanPoll);
      clearInterval(resultPoll);
      ['sell_puts','buy_calls','buy_puts','covered_calls'].forEach(loadStrategy);
      updateLiveBadge(data);
    } else {
      // Scan not running — still update live badge on each poll
      updateLiveBadge(data);
    }
  }, 1500);
}

function pollResults() {
  if (resultPoll) clearInterval(resultPoll);
  resultPoll = setInterval(() => {
    ['sell_puts','buy_calls','buy_puts','covered_calls'].forEach(loadStrategy);
  }, 4000);
}

// ── Live refresh UI ────────────────────────────────────────────

function updateLiveBadge(data) {
  const badge   = document.getElementById('live-badge');
  const refInfo = document.getElementById('last-refresh-info');
  if (!badge) return;
  if (data.live_active) {
    badge.style.display = 'flex';
    document.getElementById('live-mins-display').textContent = data.live_mins;
    const sel = document.getElementById('live-interval');
    if (sel) sel.value = String(data.live_mins);
  } else {
    badge.style.display = 'none';
  }
  if (data.last_refresh) {
    const ago = Math.round((Date.now() - new Date(data.last_refresh).getTime()) / 60000);
    refInfo.style.display = 'block';
    refInfo.textContent = `Last refreshed: ${ago === 0 ? 'just now' : ago+'m ago'}`;
    // Reload cards if a refresh just happened (within last 15s)
    if (Date.now() - new Date(data.last_refresh).getTime() < 15000) {
      ['sell_puts','buy_calls','buy_puts','covered_calls'].forEach(loadStrategy);
    }
  }
}

async function setLiveInterval(mins) {
  await fetch('/api/refresh/set', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({mins: parseInt(mins)})
  });
}

async function stopLiveRefresh() {
  await fetch('/api/refresh/set', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({stop: true})
  });
  document.getElementById('live-badge').style.display = 'none';
  document.getElementById('last-refresh-info').style.display = 'none';
}

// Poll scan status every 10s even when not scanning (to update live badge)
setInterval(async () => {
  if (document.getElementById('btn-scan').disabled) return; // scan running, its own poll handles it
  try {
    const data = await fetch('/api/scan/status').then(r=>r.json());
    updateLiveBadge(data);
  } catch(e) {}
}, 10000);

// ── Iron Condor on-demand scan ────────────────────────────────
let condorPoll = null;

async function startCondorScan() {
  const btn    = document.getElementById('btn-condor-scan');
  const status = document.getElementById('condor-scan-status');
  btn.disabled = true;
  btn.textContent = '⏳ Scanning…';
  const r = await fetch('/api/condor/scan', {method:'POST'});
  if (!r.ok) { alert('Condor scan already running'); btn.disabled=false; btn.textContent='🔍 Scan for Iron Condors'; return; }
  status.textContent = 'Starting condor scan…';
  if (condorPoll) clearInterval(condorPoll);
  condorPoll = setInterval(async () => {
    try {
      const d = await fetch('/api/condor/status').then(r=>r.json());
      if (d.running) {
        const pct = d.total ? Math.round(d.progress/d.total*100) : 0;
        status.textContent = `Scanning… ${d.progress}/${d.total} (${pct}%)`;
      } else {
        clearInterval(condorPoll);
        btn.disabled = false;
        btn.textContent = '🔍 Scan for Iron Condors';
        status.textContent = `Scan complete.`;
        // Load condor results into the results div
        loadCondorResults();
      }
    } catch(e) {}
  }, 1500);
}

async function loadCondorResults() {
  const container = document.getElementById('condor-results');
  if (!container) return;
  try {
    const data = await fetch('/api/results/iron_condors').then(r=>r.json());
    if (!data.length) {
      container.innerHTML = '<div style="text-align:center;padding:20px;color:var(--muted)">No iron condor setups found.</div>';
      return;
    }
    // buildCard returns an HTML string (appendChild on it threw, so condor
    // results never rendered).  Build the cards the same way loadStrategy does.
    const rendered = [];
    let cards = '';
    data.forEach(item => {
      try { cards += buildCard(item, 'iron_condors'); rendered.push(item); }
      catch(e) { console.error('buildCard error for', item.ticker, e); }
    });
    container.innerHTML = `<div class="cards">${cards}</div>`;
    try { renderCharts(rendered); } catch(e) { console.error('renderCharts error', e); }
    for (const d of rendered) {
      renderNewsIntoCard(d.ticker, 'iron_condor', 'news-' + d.ticker + '_iron_condor');
    }
  } catch(e) { console.error('Condor results load error:', e); }
}

// ── Load strategy results ─────────────────────────────────────
async function loadStrategy(strategy) {
  let data;
  try {
    const resp = await fetch('/api/results/'+strategy);
    if (!resp.ok) { console.error('API error', strategy, resp.status); return; }
    data = await resp.json();
  } catch(e) { console.error('Fetch failed', strategy, e); return; }

  const panel = document.getElementById('panel-'+strategy);
  if (!data.length) {
    document.getElementById('cnt-'+strategy).textContent = 0;
    panel.innerHTML = `<div class="empty-state"><h3>No signals yet</h3><p>Run a scan to populate results.</p></div>`;
    return;
  }

  // Build cards — count only successfully rendered ones so badge matches display
  let cards = '';
  let rendered = 0;
  const renderedData = [];
  for (const d of data) {
    try {
      const html = buildCard(d, strategy);
      cards += html;
      rendered++;
      renderedData.push(d);
    } catch(e) {
      console.error('buildCard error for', d.ticker, '—', e.message, e);
    }
  }

  // Update badge to reflect what's actually displayed
  document.getElementById('cnt-'+strategy).textContent = rendered;

  panel.innerHTML = `<div class="disclaimer">
    <b>⚠ Not financial advice.</b> These are technical signals only.
    Options trading involves significant risk. Always do your own research.
    You must manually add trades to queue and approve each execution.
  </div>
  <div class="cards">${cards}</div>`;
  try { renderCharts(renderedData); } catch(e) { console.error('renderCharts error', e); }

  // Async: fetch news for each successfully rendered card
  for (const d of renderedData) {
    const sid = d.strategy || strategy;
    const id  = d.ticker + '_' + sid;
    renderNewsIntoCard(d.ticker, d.strategy || sid, 'news-' + id);
  }
}

// ── Card builder ──────────────────────────────────────────────
function scoreColor(s) {
  if(s>=70) return '#22c55e';
  if(s>=45) return '#f59e0b';
  return '#3b82f6';
}

function buildCard(d, strategy) {
  const sc   = scoreColor(d.score);
  const opt  = d.option || {};
  // Always use singular strategy from data for IDs so renderCharts can find canvases
  const sid  = d.strategy || strategy;
  const id   = d.ticker + '_' + sid;

  // Signals
  let sigs = '';
  if(d.rsi_oversold)  sigs += `<span class="sig sig-red">RSI ${d.rsi}</span>`;
  if(d.rsi_overbought)sigs += `<span class="sig sig-orange">RSI ${d.rsi} ↑</span>`;
  if(d.rsi_neutral)   sigs += `<span class="sig sig-blue">RSI ${d.rsi} Neutral</span>`;
  if(d.bb_oversold)   sigs += `<span class="sig sig-yellow">Below BB</span>`;
  if(d.bb_overbought) sigs += `<span class="sig sig-orange">Above BB</span>`;
  // Donchian Channel signals
  if(d.dc_at_lower)   sigs += `<span class="sig sig-red" title="Price at 20-day Donchian lower band — N-period price low">📉 DC Lower $${d.dc_lower?.toFixed(2)}</span>`;
  if(d.dc_at_upper)   sigs += `<span class="sig sig-orange" title="Price at 20-day Donchian upper band — N-period price high">📈 DC Upper $${d.dc_upper?.toFixed(2)}</span>`;
  if(d.dc_squeeze)    sigs += `<span class="sig sig-blue" title="Donchian range ${d.dc_range_pct?.toFixed(1)}% — tight consolidation, breakout likely">🗜 DC Squeeze ${d.dc_range_pct?.toFixed(1)}%</span>`;
  if(d.reversal)      sigs += `<span class="sig sig-green">RSI Turning ↑</span>`;
  if(d.iv_high)       sigs += `<span class="sig sig-purple">IV Rank ${d.iv_rank?.toFixed(0)}</span>`;
  if(d.prem_high)     sigs += `<span class="sig sig-blue">Premium ${opt.premium_pct?.toFixed(1)}%</span>`;
  if(d.macd_bull)     sigs += `<span class="sig sig-green">MACD ↑ Cross</span>`;
  if(d.macd_bear)     sigs += `<span class="sig sig-red">MACD ↓ Cross</span>`;
  if(d.pc_bearish)    sigs += `<span class="sig sig-orange">P/C ${d.pc_ratio?.toFixed(2)} Bearish</span>`;
  if(d.pc_bullish)    sigs += `<span class="sig sig-green">P/C ${d.pc_ratio?.toFixed(2)} Bullish</span>`;
  if(d.earnings_soon) sigs += `<span class="sig sig-yellow">⚠ Earnings ${d.earnings_dte}d</span>`;
  // Trend direction
  if(d.trend_bull)          sigs += `<span class="sig sig-green">Above 200 MA</span>`;
  if(d.trend_bear)          sigs += `<span class="sig sig-red">Below 200 MA</span>`;
  if(d.exhaustion_bounce)   sigs += `<span class="sig sig-green">🔥 Exhaustion Bounce</span>`;
  // ADX strength
  if(d.adx != null) {
    const adxColor = d.adx > 25 ? 'sig-orange' : 'sig-blue';
    const adxLabel = d.adx > 40 ? 'Strong Trend' : d.adx > 25 ? 'Trending' : 'Sideways';
    sigs += `<span class="sig ${adxColor}">ADX ${d.adx?.toFixed(0)} ${adxLabel}</span>`;
  }
  // VIX regime
  if(d.vix != null) {
    const vixColor = d.vix_regime==='high'?'sig-red':d.vix_regime==='low'?'sig-green':'sig-blue';
    const vixLabel = d.vix_regime==='high'?'⚡ High Vol':'🧊 Low Vol'
    if(d.vix_regime!=='normal') sigs += `<span class="sig ${vixColor}">${vixLabel} VIX ${d.vix?.toFixed(1)}</span>`;
  }
  // Broad market (SPY) regime
  if(d.market_trend) {
    const mktMap = {
      'bull_strong': ['sig-green',  '📈 Market Bull'],
      'bull_weak':   ['sig-blue',   '↗ Market Mild Bull'],
      'bear_strong': ['sig-red',    '📉 Market Bear'],
      'bear_weak':   ['sig-orange', '↘ Market Mild Bear'],
    };
    const [mktColor, mktLabel] = mktMap[d.market_trend] || [];
    if(mktColor) sigs += `<span class="sig ${mktColor}">${mktLabel}</span>`;
  }
  // Beta — how much the stock's move is driven by the broad market (SPY)
  if(d.beta != null) {
    const ab = Math.abs(d.beta);
    let bColor, bLabel;
    if(d.beta < 0)     { bColor='sig-purple'; bLabel='Inverse'; }
    else if(ab >= 1.3) { bColor='sig-orange'; bLabel='High Mkt Sensitivity'; }
    else if(ab >= 0.7) { bColor='sig-blue';   bLabel='Moves w/ Market'; }
    else               { bColor='sig-green';  bLabel='Low Mkt Sensitivity'; }
    sigs += `<span class="sig ${bColor}" title="Rolling 60-day beta vs SPY. |β|>1 amplifies market moves, ~1 tracks it, <1 dampens, <0 moves opposite. The higher |β|, the more this stock's trend is just the market scaled.">β ${d.beta.toFixed(2)} ${bLabel}</span>`;
  }
  // Fed meeting proximity + yield curve
  if(d.days_to_next_fed != null) {
    if(d.pre_fed_window) {
      sigs += `<span class="sig sig-orange">🏦 FOMC in ${d.days_to_next_fed}d — IV inflated</span>`;
    } else if(d.post_fed_window) {
      sigs += `<span class="sig sig-blue">🏦 Post-FOMC — IV crushed</span>`;
    } else if(d.days_to_next_fed <= 14) {
      sigs += `<span class="sig sig-blue">🏦 FOMC in ${d.days_to_next_fed}d</span>`;
    }
  }
  if(d.yield_inverted) {
    sigs += `<span class="sig sig-red">📉 Yield Curve Inverted ${d.yield_spread?.toFixed(2)}%</span>`;
  } else if(d.yield_steep) {
    sigs += `<span class="sig sig-green">📈 Yield Curve Steep +${d.yield_spread?.toFixed(2)}%</span>`;
  }
  // GARCH volatility estimate
  if(d.garch_vol != null) {
    const garchColor = d.garch_vol > 60 ? 'sig-red' : d.garch_vol > 35 ? 'sig-orange' : 'sig-blue';
    sigs += `<span class="sig ${garchColor}">σ GJR-GARCH ${d.garch_vol?.toFixed(1)}%</span>`;
  }
  // Volume confirmation
  if(d.vol_confirmed) sigs += `<span class="sig sig-green">📊 Vol Confirmed</span>`;
  // Short squeeze setup
  if(d.squeeze_setup) sigs += `<span class="sig sig-purple">🚀 Squeeze Setup ${d.short_pct?.toFixed(1)}% Short</span>`;
  else if(d.short_pct != null && d.short_pct >= 10)
    sigs += `<span class="sig sig-orange">Short ${d.short_pct?.toFixed(1)}%</span>`;
  // Confluence count
  const sid2 = d.strategy;
  if(sid2==='buy_call'||sid2==='sell_put') {
    sigs += `<span class="sig ${d.bull_signals>=3?'sig-green':d.bull_signals>=2?'sig-blue':'sig-yellow'}">${d.bull_signals} Bull Signal${d.bull_signals!==1?'s':''}</span>`;
  }
  if(sid2==='buy_put') {
    sigs += `<span class="sig ${d.bear_signals>=3?'sig-red':d.bear_signals>=2?'sig-orange':'sig-yellow'}">${d.bear_signals} Bear Signal${d.bear_signals!==1?'s':''}</span>`;
  }

  const optType   = opt.type === 'call' ? '📈 Call' : opt.type === 'condor' ? '🦅 Iron Condor' : '📉 Put';
  const action    = sid === 'sell_put'    ? 'Sell Put'
                  : sid === 'buy_call'    ? 'Buy Call'
                  : sid === 'buy_put'     ? 'Buy Put'
                  : sid === 'iron_condor' ? 'Iron Condor'
                  : 'Sell Covered Call';
  const isBuy     = sid === 'buy_call' || sid === 'buy_put';
  const isCondor  = sid === 'iron_condor';
  const costPer   = isBuy ? (opt.ask||0)*100 : (opt.bid||0)*100;
  const costLabel = isBuy ? 'Entry Cost / Contract' : isCondor ? 'Net Credit / Contract' : 'Premium Received / Contract';
  const costColor = isBuy ? 'var(--red)' : 'var(--green)';
  const costSign  = isBuy ? '-' : '+';
  const costBg    = isBuy ? 'rgba(239,68,68,0.08)' : 'rgba(34,197,94,0.08)';
  const costBdr   = isBuy ? 'rgba(239,68,68,0.33)' : 'rgba(34,197,94,0.33)';
  const netCredit = (d.net_credit||0)*100;

  // ── Iron Condor 4-leg table ──────────────────────────────────
  let legsHtml = '';
  if (isCondor && d.legs) {
    const L = d.legs;
    const sp = L.short_put  || {};
    const lp = L.long_put   || {};
    const sc = L.short_call || {};
    const lc = L.long_call  || {};
    const nc = d.net_credit || 0;
    const beLow  = ((sp.strike||0) - nc).toFixed(2);
    const beHigh = ((sc.strike||0) + nc).toFixed(2);
    legsHtml = `
  <div style="border-top:1px solid var(--border);padding:12px 16px">
    <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px">
      🦅 Iron Condor — 4 Trades to Execute
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <thead>
        <tr style="color:var(--muted);font-size:10px;text-transform:uppercase">
          <th style="text-align:left;padding:4px 6px">#</th>
          <th style="text-align:left;padding:4px 6px">Action</th>
          <th style="text-align:left;padding:4px 6px">Type</th>
          <th style="text-align:right;padding:4px 6px">Strike</th>
          <th style="text-align:right;padding:4px 6px">Expiry</th>
          <th style="text-align:right;padding:4px 6px">Price/Share</th>
          <th style="text-align:right;padding:4px 6px">Per Contract</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-top:1px solid var(--border)">
          <td style="padding:5px 6px;color:var(--muted)">1</td>
          <td style="padding:5px 6px"><span style="color:var(--green);font-weight:700">SELL</span></td>
          <td style="padding:5px 6px">PUT</td>
          <td style="padding:5px 6px;text-align:right;font-weight:600">$${(sp.strike||0).toFixed(2)}</td>
          <td style="padding:5px 6px;text-align:right;color:var(--muted)">${sp.expiration||''}</td>
          <td style="padding:5px 6px;text-align:right;color:var(--green)">+$${(sp.bid||0).toFixed(2)}</td>
          <td style="padding:5px 6px;text-align:right;color:var(--green)">+$${((sp.bid||0)*100).toFixed(2)}</td>
        </tr>
        <tr style="border-top:1px solid var(--border)">
          <td style="padding:5px 6px;color:var(--muted)">2</td>
          <td style="padding:5px 6px"><span style="color:var(--red);font-weight:700">BUY</span></td>
          <td style="padding:5px 6px">PUT</td>
          <td style="padding:5px 6px;text-align:right;font-weight:600">$${(lp.strike||0).toFixed(2)}</td>
          <td style="padding:5px 6px;text-align:right;color:var(--muted)">${lp.expiration||''}</td>
          <td style="padding:5px 6px;text-align:right;color:var(--red)">-$${(lp.ask||0).toFixed(2)}</td>
          <td style="padding:5px 6px;text-align:right;color:var(--red)">-$${((lp.ask||0)*100).toFixed(2)}</td>
        </tr>
        <tr style="border-top:1px solid var(--border)">
          <td style="padding:5px 6px;color:var(--muted)">3</td>
          <td style="padding:5px 6px"><span style="color:var(--green);font-weight:700">SELL</span></td>
          <td style="padding:5px 6px">CALL</td>
          <td style="padding:5px 6px;text-align:right;font-weight:600">$${(sc.strike||0).toFixed(2)}</td>
          <td style="padding:5px 6px;text-align:right;color:var(--muted)">${sc.expiration||''}</td>
          <td style="padding:5px 6px;text-align:right;color:var(--green)">+$${(sc.bid||0).toFixed(2)}</td>
          <td style="padding:5px 6px;text-align:right;color:var(--green)">+$${((sc.bid||0)*100).toFixed(2)}</td>
        </tr>
        <tr style="border-top:1px solid var(--border)">
          <td style="padding:5px 6px;color:var(--muted)">4</td>
          <td style="padding:5px 6px"><span style="color:var(--red);font-weight:700">BUY</span></td>
          <td style="padding:5px 6px">CALL</td>
          <td style="padding:5px 6px;text-align:right;font-weight:600">$${(lc.strike||0).toFixed(2)}</td>
          <td style="padding:5px 6px;text-align:right;color:var(--muted)">${lc.expiration||''}</td>
          <td style="padding:5px 6px;text-align:right;color:var(--red)">-$${(lc.ask||0).toFixed(2)}</td>
          <td style="padding:5px 6px;text-align:right;color:var(--red)">-$${((lc.ask||0)*100).toFixed(2)}</td>
        </tr>
        <tr style="border-top:2px solid var(--border);background:rgba(34,197,94,0.06)">
          <td colspan="5" style="padding:7px 6px;font-weight:700;color:var(--green)">NET CREDIT (keep if stock stays between $${beLow} – $${beHigh})</td>
          <td style="padding:7px 6px;text-align:right;font-weight:800;color:var(--green)">+$${nc.toFixed(2)}</td>
          <td style="padding:7px 6px;text-align:right;font-weight:800;color:var(--green)">+$${(nc*100).toFixed(2)}</td>
        </tr>
      </tbody>
    </table>
    <div style="display:flex;gap:16px;margin-top:8px;font-size:11px;color:var(--muted)">
      <span>📐 Put spread width: <b style="color:var(--text)">$${(d.put_width||0).toFixed(2)}</b></span>
      <span>📐 Call spread width: <b style="color:var(--text)">$${(d.call_width||0).toFixed(2)}</b></span>
      <span>🎯 Profit zone: <b style="color:var(--green)">$${beLow} – $${beHigh}</b></span>
    </div>
  </div>`;
  }

  return `
<div class="card">
  <div class="card-head">
    <div style="flex:1">
      <div class="ticker">${esc(d.ticker)}</div>
      <div class="cname">${esc(d.name)}</div>
    </div>
    <div style="text-align:right;flex:1">
      <div class="cprice" style="color:${sc}">$${d.price.toFixed(3)}</div>
      <div class="csector">${esc(d.sector)}</div>
    </div>
    <div class="score-ring" style="background:${sc}22;color:${sc};border:2px solid ${sc}">${d.score}</div>
  </div>

  <div class="signals">${sigs}</div>

  <div class="charts">
    <div>
      <div class="chart-label">Price + Bollinger Bands</div>
      <canvas id="pc-${id}" height="110"></canvas>
    </div>
    <div>
      <div class="chart-label">RSI (14)</div>
      <canvas id="rc-${id}" height="110"></canvas>
    </div>
  </div>

  ${isCondor ? legsHtml : `
  <div class="opt-info">
    <div class="opt-title">${optType} · ${opt.expiration} (${opt.dte}d) · ${action}</div>
    <div style="display:flex;align-items:center;gap:10px;margin:8px 0 4px;padding:8px 12px;border-radius:8px;background:${costBg};border:1px solid ${costBdr}">
      <span style="font-size:12px;color:var(--muted)">${costLabel}:</span>
      <span style="font-size:20px;font-weight:700;color:${costColor}">${costSign}$${costPer.toFixed(2)}</span>
      <span style="font-size:11px;color:var(--muted)">(${isBuy?'ask':'bid'} x 100 shares)</span>
    </div>
    <div class="opt-grid">
      <div class="opt-cell"><div class="key">Strike</div>
        <div class="val">$${opt.strike?.toFixed(2)}</div></div>
      <div class="opt-cell"><div class="key">Bid / Ask</div>
        <div class="val">$${opt.bid?.toFixed(2)} / $${opt.ask?.toFixed(2)}</div></div>
      <div class="opt-cell"><div class="key">IV</div>
        <div class="val ${d.iv_high?'good':''}">${opt.iv?.toFixed(1)}%${d.iv_rank?' (Rank '+d.iv_rank.toFixed(0)+')':''}</div></div>
      <div class="opt-cell"><div class="key">Premium %</div>
        <div class="val ${d.prem_high?'good':''}">${opt.premium_pct?.toFixed(2)}%</div></div>
      <div class="opt-cell"><div class="key">Volume</div>
        <div class="val">${(opt.volume||0).toLocaleString()}</div></div>
      <div class="opt-cell"><div class="key">Open Int.</div>
        <div class="val">${(opt.open_interest||0).toLocaleString()}</div></div>
      <div class="opt-cell"><div class="key">BB Low</div>
        <div class="val">$${d.bb_lower?.toFixed(3)}</div></div>
      <div class="opt-cell"><div class="key">BB High</div>
        <div class="val">$${d.bb_upper?.toFixed(3)}</div></div>
      <div class="opt-cell"><div class="key">MACD</div>
        <div class="val" style="color:${d.macd_bull?'var(--green)':d.macd_bear?'var(--red)':'var(--text)'}">
          ${d.macd_val?.toFixed(3)} ${d.macd_bull?'▲ Bull Cross':d.macd_bear?'▼ Bear Cross':''}
        </div></div>
      <div class="opt-cell"><div class="key">P/C Ratio</div>
        <div class="val" style="color:${d.pc_bearish?'var(--red)':d.pc_bullish?'var(--green)':'var(--text)'}">
          ${d.pc_ratio!=null?d.pc_ratio.toFixed(2):'—'}
          ${d.pc_bearish?' (Fear)':d.pc_bullish?' (Greed)':''}
        </div></div>
      <div class="opt-cell"><div class="key">Earnings</div>
        <div class="val" style="color:${d.earnings_soon?'var(--yellow)':'var(--text)'}">
          ${d.earnings_dte!=null?(d.earnings_dte+'d away'):'—'}
          ${d.earnings_soon?' ⚠ IV Risk':''}
        </div></div>
      <div class="opt-cell"><div class="key">200 MA / Trend</div>
        <div class="val" style="color:${d.trend_bull?'var(--green)':d.trend_bear?'var(--red)':'var(--muted)'}">
          ${d.ma200!=null?'$'+d.ma200.toFixed(2):'—'}
          ${d.trend_bull?' ↑ Bull':d.trend_bear?' ↓ Bear':''}
        </div></div>
      <div class="opt-cell"><div class="key">ADX (Strength)</div>
        <div class="val" style="color:${d.adx>40?'var(--orange)':d.adx>25?'var(--yellow)':'var(--green)'}">
          ${d.adx!=null?d.adx.toFixed(1):'—'}
          ${d.adx>40?' Very Strong':d.adx>25?' Trending':' Sideways'}
        </div></div>
      <div class="opt-cell"><div class="key">Beta (vs SPY)</div>
        <div class="val" style="color:${d.beta==null?'var(--muted)':Math.abs(d.beta)>=1.3?'var(--orange)':Math.abs(d.beta)>=0.7?'var(--text)':'var(--green)'}">
          ${d.beta!=null?d.beta.toFixed(2):'—'}
          ${d.beta==null?'':Math.abs(d.beta)>=1.3?' Amplifies Mkt':Math.abs(d.beta)>=0.7?' Tracks Mkt':d.beta<0?' Inverse':' Defensive'}
        </div></div>
    </div>
  </div>`}

  ${buildBSSection(d)}

  ${buildNewsSection(d, 'news-'+id)}

  ${buildMCSection(d, id)}

  ${buildSizingBlock(d, id, opt, sid, isCondor)}

  <div class="card-footer">
    <div class="qty-wrap">
      Contracts:
      <input type="number" min="1" max="100" value="1" id="qty-${id}"
        oninput="updateCost('${id}',${isCondor?d.net_credit||0:opt.ask||0},${opt.strike||0},'${sid}',${d.put_width||0},${d.call_width||0})"/>
    </div>
    <div class="cost-est" id="cost-${id}">Est. capital: <span class="cost-val">${calcCostDisplay(1, isCondor?d.net_credit||0:opt.ask||0, opt.strike||0, sid, d.put_width||0, d.call_width||0)}</span></div>
    <button class="btn btn-primary" onclick="addToQueue('${jsq(d.ticker)}','${jsq(d.name)}',${d.price},'${sid}','${jsq(d.strategy_label||action)}','${id}')">
      + Add to Queue
    </button>
    <button class="btn btn-ghost" style="margin-left:6px" onclick="trackPosition('${jsq(d.ticker)}','${jsq(d.name)}',${d.price},'${sid}','${jsq(d.strategy_label||action)}','${id}')">
      📍 Track
    </button>
    <button class="btn btn-ghost" style="margin-left:6px" onclick="openBSFromCard('${d.ticker}',${d.price},${opt.strike||0},${opt.dte||30},${opt.iv||0},${opt.mid||0},'${opt.type||sid}')">
      ⚖️ B-S
    </button>
  </div>
</div>`;
}

function renderNewsIntoCard(ticker, strategy, containerId) {
  // Fetch news async and inject into the card's news container
  fetch('/api/news/' + ticker)
    .then(r => r.json())
    .then(data => {
      const el = document.getElementById(containerId);
      if (!el) return;
      const news     = data.articles || [];
      const colorMap = { green: 'var(--green)', red: 'var(--red)', yellow: 'var(--yellow)' };

      // Overall sentiment
      let overallHtml = '';
      if (news.length) {
        const avg = news.reduce((s,a) => s + (a.sentiment?.compound||0), 0) / news.length;
        const oc  = avg >= 0.05 ? 'var(--green)' : avg <= -0.05 ? 'var(--red)' : 'var(--yellow)';
        const ol  = avg >= 0.05 ? '▲ Positive'   : avg <= -0.05 ? '▼ Negative' : '● Neutral';
        overallHtml = `<span style="font-size:12px;font-weight:600;color:${oc}">${ol} (${avg>=0?'+':''}${avg.toFixed(2)})</span>`;
      }

      // Conflict warning
      let conflictHtml = '';
      if (news.length) {
        const avg = news.reduce((s,a) => s + (a.sentiment?.compound||0), 0) / news.length;
        const bullStrats = ['buy_call','sell_put','covered_call'];
        const bearStrats = ['buy_put'];
        if (bullStrats.includes(strategy) && avg <= -0.15)
          conflictHtml = `<div style="margin-top:8px;padding:7px 10px;border-radius:6px;
            background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.4);
            font-size:12px;color:var(--red)">⚠ News is mostly negative (avg ${avg>=0?'+':''}${avg.toFixed(2)}) — may suppress bounce</div>`;
        else if (bearStrats.includes(strategy) && avg >= 0.15)
          conflictHtml = `<div style="margin-top:8px;padding:7px 10px;border-radius:6px;
            background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.4);
            font-size:12px;color:var(--red)">⚠ News is mostly positive (avg +${avg.toFixed(2)}) — may suppress decline</div>`;
      }

      const articleRows = news.map(a => {
        const sc  = colorMap[a.sentiment?.color] || 'var(--muted)';
        const lbl = a.sentiment?.label || '';
        const ts  = a.published ? new Date(a.published*1000).toLocaleDateString() : '';
        return `
          <div style="display:flex;gap:10px;align-items:flex-start;padding:7px 0;border-bottom:1px solid var(--border)">
            <span style="min-width:60px;font-size:10px;font-weight:700;color:${sc};
                         padding:2px 6px;border-radius:4px;background:${sc}18;text-align:center">${lbl}</span>
            <div style="flex:1;min-width:0">
              <a href="${safeHref(a.link)}" target="_blank" rel="noopener noreferrer"
                 style="font-size:12px;color:var(--text);text-decoration:none;
                        display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                 title="${esc(a.title)}">${esc(a.title)}</a>
              <span style="font-size:10px;color:var(--muted)">${esc(a.publisher)}${ts?' · '+ts:''}</span>
            </div>
          </div>`;
      }).join('');

      el.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
          <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px">📰 Recent News</div>
          ${overallHtml}
        </div>
        ${conflictHtml}
        ${news.length ? articleRows : '<div style="font-size:12px;color:var(--muted)">No recent news found.</div>'}`;
    })
    .catch(() => {
      const el = document.getElementById(containerId);
      if (el) el.innerHTML = '<div style="font-size:12px;color:var(--muted)">Could not load news.</div>';
    });
}

function buildNewsSection(d, newsContainerId) {
  // Returns a placeholder div; actual news is fetched async after render
  return `
  <div style="border-top:1px solid var(--border);padding:12px 16px"
       id="${newsContainerId}">
    <div style="font-size:12px;color:var(--muted)">📰 Loading news…</div>
  </div>`;
}

function buildBSSection(d) {
  const bs = d.bs;
  const bm = d.bs_misprice;
  if (!bs || bs.fair_value == null || bs.delta == null) return '';

  const fv      = bs.fair_value;
  const delta   = bs.delta;
  const mid     = (d.option || {}).mid || 0;
  const colorMap = { red: 'var(--red)', green: 'var(--green)', yellow: 'var(--yellow)' };
  const bmColor  = bm ? (colorMap[bm.color] || 'var(--muted)') : 'var(--muted)';
  const bmLabel  = bm ? bm.label : 'N/A';
  const bmPct    = bm ? (bm.pct > 0 ? '+' : '') + bm.pct.toFixed(1) + '%' : '';

  // Tooltip explanation
  const tip = bm
    ? (bm.color === 'green'
        ? 'Market price is BELOW the BS fair value — option may be a bargain'
        : bm.color === 'red'
        ? 'Market price is ABOVE the BS fair value — option may be expensive'
        : 'Market price is close to the BS fair value')
    : '';

  return `
  <div style="border-top:1px solid var(--border);padding:12px 16px">
    <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px">
      🧮 Black-Scholes Pricing
    </div>
    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
      <div style="display:flex;flex-direction:column;gap:2px">
        <span style="font-size:10px;color:var(--muted)">Fair Value</span>
        <span style="font-size:16px;font-weight:700;color:var(--text)">$${fv.toFixed(3)}</span>
      </div>
      <div style="display:flex;flex-direction:column;gap:2px">
        <span style="font-size:10px;color:var(--muted)">Market Mid</span>
        <span style="font-size:16px;font-weight:700;color:var(--text)">$${mid.toFixed(3)}</span>
      </div>
      <div style="display:flex;flex-direction:column;gap:2px">
        <span style="font-size:10px;color:var(--muted)">Delta</span>
        <span style="font-size:16px;font-weight:700;color:var(--text)">${delta.toFixed(3)}</span>
      </div>
      <div style="flex:1;min-width:140px">
        <div title="${tip}" style="display:inline-flex;align-items:center;gap:8px;padding:8px 14px;border-radius:8px;
            background:${bmColor}18;border:1px solid ${bmColor}55;cursor:default">
          <span style="font-size:18px;font-weight:800;color:${bmColor}">${bmPct}</span>
          <span style="font-size:12px;font-weight:600;color:${bmColor}">${bmLabel}</span>
        </div>
      </div>
    </div>
  </div>`;
}

function buildMCSection(d, id) {
  const mc = d.mc;
  if (!mc || mc.pop == null || mc.ev == null || mc.breakeven == null || mc.max_loss == null) return '';

  const pop     = mc.pop;
  const ev      = mc.ev;
  const be      = mc.breakeven;
  const maxP    = mc.max_profit;   // may be null (unlimited upside)
  const maxL    = mc.max_loss;
  const nsims   = (mc.n_sims||10000).toLocaleString();

  const popColor = pop >= 60 ? 'var(--green)' : pop >= 40 ? 'var(--yellow)' : 'var(--red)';
  const evColor  = ev >= 0   ? 'var(--green)' : 'var(--red)';
  const evStr    = (ev >= 0 ? '+' : '') + '$' + Math.abs(ev).toFixed(2);
  const maxPStr  = maxP !== null ? '+$' + maxP.toFixed(2) : 'Unlimited';
  const maxLStr  = '-$' + Math.abs(maxL).toFixed(2);

  return `
  <div class="mc-section">
    <div class="mc-title">&#x1F3B2; Monte Carlo &nbsp;&middot;&nbsp; ${nsims} simulations</div>
    <div class="mc-grid">
      <div class="mc-cell">
        <div class="mc-label">Prob. of Profit</div>
        <div class="mc-val" style="color:${popColor}">${pop.toFixed(1)}%</div>
        <div class="pop-bar-track"><div class="pop-bar-fill" style="width:${pop}%;background:${popColor}"></div></div>
      </div>
      <div class="mc-cell">
        <div class="mc-label">Exp. Value / contract</div>
        <div class="mc-val" style="color:${evColor}">${evStr}</div>
        <div class="mc-sub">avg across all paths</div>
      </div>
      <div class="mc-cell">
        <div class="mc-label">Break-even at expiry</div>
        <div class="mc-val">$${be.toFixed(3)}</div>
        <div class="mc-sub">stock price needed</div>
      </div>
      <div class="mc-cell">
        <div class="mc-label">Max Profit / Loss</div>
        <div class="mc-val" style="color:var(--green);font-size:12px">${maxPStr}</div>
        <div class="mc-sub" style="color:var(--red)">${maxLStr} max loss</div>
      </div>
    </div>
    <div class="mc-chart-label">P&amp;L Distribution per contract (${nsims} paths)</div>
    <canvas id="mc-${id}" height="70"></canvas>
  </div>`;
}

// ── Position sizing ────────────────────────────────────────────
let _riskBudget    = 0;    // loaded from backend on startup
let _stopLossPct   = 15;   // mirrors SELL_STOP_LOSS,   loaded from /api/config/risk-budget
let _profitTgtPct  = 30;   // mirrors SELL_PROFIT_TARGET, loaded from /api/config/risk-budget

function buildSizingBlock(d, id, opt, sid, isCondor) {
  if (!_riskBudget || _riskBudget <= 0) return '';

  const ask         = isCondor ? (d.net_credit || 0) : (opt.ask || 0);
  const costPerContract = isCondor
    ? Math.max((Math.max(d.put_width||0, d.call_width||0) - ask), 0) * 100
    : ask * 100;

  if (costPerContract <= 0) return '';

  const stopPct   = _stopLossPct  / 100;
  const profitPct = _profitTgtPct / 100;
  const maxLossPerContract = costPerContract * stopPct;
  const maxProfitPerContract = costPerContract * profitPct;

  // Recommended contracts = how many fit within the budget at the stop %
  const recommended = Math.max(1, Math.floor(_riskBudget / maxLossPerContract));
  const actualSpend    = recommended * costPerContract;
  const actualMaxLoss  = recommended * maxLossPerContract;
  const actualMaxProfit = recommended * maxProfitPerContract;

  // Badge: does 1 contract fit, fit tightly, or bust the budget?
  let badge, badgeColor;
  const ratio = actualMaxLoss / _riskBudget;
  if (ratio <= 1.0) {
    badge = '✅ On budget';  badgeColor = 'var(--green)';
  } else if (ratio <= 1.5) {
    badge = '⚠️ Slightly over';  badgeColor = '#f59e0b';
  } else {
    badge = '🔴 Over budget — consider skipping';  badgeColor = 'var(--red)';
  }

  // Auto-set the contracts input
  setTimeout(() => {
    const el = document.getElementById('qty-' + id);
    if (el) {
      el.value = recommended;
      updateCost(id, isCondor ? d.net_credit||0 : opt.ask||0,
                 opt.strike||0, sid, d.put_width||0, d.call_width||0);
    }
  }, 0);

  return `
  <div style="border-top:1px solid var(--border);padding:12px 16px;
              background:rgba(167,139,250,0.05)">
    <div style="font-size:10px;color:var(--muted);text-transform:uppercase;
                letter-spacing:.6px;margin-bottom:10px">💰 Position Sizing (budget $${_riskBudget})</div>
    <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap">
      <div style="display:flex;flex-direction:column;gap:2px">
        <span style="font-size:10px;color:var(--muted)">Recommended</span>
        <span style="font-size:18px;font-weight:800;color:var(--text)">${recommended} contract${recommended!==1?'s':''}</span>
      </div>
      <div style="display:flex;flex-direction:column;gap:2px">
        <span style="font-size:10px;color:var(--muted)">Total spend</span>
        <span style="font-size:14px;font-weight:700;color:var(--text)">$${actualSpend.toFixed(2)}</span>
      </div>
      <div style="display:flex;flex-direction:column;gap:2px">
        <span style="font-size:10px;color:var(--muted)">Max loss (${_stopLossPct}% stop)</span>
        <span style="font-size:14px;font-weight:700;color:var(--red)">-$${actualMaxLoss.toFixed(2)}</span>
      </div>
      <div style="display:flex;flex-direction:column;gap:2px">
        <span style="font-size:10px;color:var(--muted)">Max profit (${_profitTgtPct}% target)</span>
        <span style="font-size:14px;font-weight:700;color:var(--green)">+$${actualMaxProfit.toFixed(2)}</span>
      </div>
      <div style="margin-left:auto">
        <span style="font-size:12px;font-weight:600;color:${badgeColor}">${badge}</span>
      </div>
    </div>
  </div>`;
}

async function setRiskBudget(val) {
  const budget = parseFloat(val) || 0;
  _riskBudget = budget;
  const resp = await fetch('/api/config/risk-budget', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ risk_budget: budget })
  }).then(r => r.json()).catch(() => ({}));
  if (resp.ok) {
    // Reload all strategy cards to show updated sizing blocks
    ['sell_puts','buy_calls','buy_puts','covered_calls'].forEach(loadStrategy);
  }
}

async function initRiskBudget() {
  try {
    const d = await fetch('/api/config/risk-budget').then(r => r.json());
    _riskBudget   = d.risk_budget       || 0;
    _stopLossPct  = d.stop_loss_pct     || 15;
    _profitTgtPct = d.profit_target_pct || 30;
    if (_riskBudget > 0) {
      document.getElementById('risk-budget-input').value = _riskBudget;
    }
  } catch(e) {}
}

function calcCostDisplay(qty, ask, strike, strategy, putW=0, callW=0) {
  let cost = 0;
  if(strategy==='buy_call'||strategy==='buy_put') cost = qty * 100 * ask;
  else if(strategy==='sell_put') cost = qty * 100 * strike;
  else if(strategy==='iron_condor'){
    const spreadW = Math.max(putW, callW) || strike * 0.05;
    cost = qty * 100 * Math.max(spreadW - ask, 0);  // ask = net_credit for condor
  } else cost = 0;
  const over = cost > 1000;
  return `<span style="color:${over?'var(--yellow)':'inherit'}">${cost===0?'$0 (covered)':'$'+cost.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}${over?' ⚠':''}</span>`;
}

function updateCost(id, ask, strike, strategy, putW=0, callW=0) {
  const qty = parseInt(document.getElementById('qty-'+id)?.value)||1;
  document.getElementById('cost-'+id).innerHTML =
    `Est. capital: <span class="cost-val">${calcCostDisplay(qty,ask,strike,strategy,putW,callW)}</span>`;
}

// ── Charts ────────────────────────────────────────────────────
function renderCharts(data) {
  data.forEach(d => {
    const strategy = d.strategy;
    const id = d.ticker + '_' + strategy;
    const c  = d.chart;
    const sc = scoreColor(d.score);

    const pcEl = document.getElementById('pc-'+id);
    const rcEl = document.getElementById('rc-'+id);
    if(!pcEl || !rcEl) return;
    if(chartRegistry['pc-'+id]) chartRegistry['pc-'+id].destroy();
    if(chartRegistry['rc-'+id]) chartRegistry['rc-'+id].destroy();

    chartRegistry['pc-'+id] = new Chart(pcEl, {
      type:'line',
      data:{labels:c.dates,datasets:[
        {data:c.bb_upper,borderColor:'rgba(99,102,241,.4)',borderWidth:1,borderDash:[3,3],pointRadius:0,fill:false,label:'BB Upper'},
        {data:c.bb_mid,  borderColor:'rgba(99,102,241,.25)',borderWidth:1,borderDash:[2,4],pointRadius:0,fill:false,label:'BB Mid'},
        {data:c.bb_lower,borderColor:'rgba(99,102,241,.4)',borderWidth:1,borderDash:[3,3],pointRadius:0,fill:'+2',backgroundColor:'rgba(99,102,241,.06)',label:'BB Lower'},
        {data:c.price,   borderColor:sc,borderWidth:2,pointRadius:0,fill:false,label:'Price',tension:.3},
      ]},
      options:{responsive:true,maintainAspectRatio:true,
        plugins:{legend:{display:false}},
        scales:{x:{ticks:{color:'#8892a4',maxTicksLimit:6,font:{size:9}},grid:{color:'rgba(255,255,255,.04)'}},
                y:{ticks:{color:'#8892a4',font:{size:9}},grid:{color:'rgba(255,255,255,.04)'}}}}
    });

    chartRegistry['rc-'+id] = new Chart(rcEl, {
      type:'line',
      data:{labels:c.dates,datasets:[
        {data:c.rsi,borderColor:'#f59e0b',borderWidth:1.5,pointRadius:0,fill:false,label:'RSI',tension:.3}
      ]},
      options:{responsive:true,maintainAspectRatio:true,
        plugins:{legend:{display:false}},
        scales:{x:{ticks:{color:'#8892a4',maxTicksLimit:6,font:{size:9}},grid:{color:'rgba(255,255,255,.04)'}},
                y:{min:0,max:100,ticks:{color:'#8892a4',font:{size:9},stepSize:25},grid:{color:'rgba(255,255,255,.04)'}}}},
      plugins:[{afterDraw(chart){
        const {ctx,chartArea:{left,right},scales:{y}}=chart;
        [[30,'rgba(239,68,68,.6)'],[70,'rgba(34,197,94,.4)']].forEach(([v,col])=>{
          const yp=y.getPixelForValue(v);
          ctx.save();ctx.strokeStyle=col;ctx.lineWidth=1;ctx.setLineDash([4,3]);
          ctx.beginPath();ctx.moveTo(left,yp);ctx.lineTo(right,yp);ctx.stroke();ctx.restore();
        });
      }}]
    });

    // ── Monte Carlo histogram ─────────────────────────────────
    const mcEl = document.getElementById('mc-'+id);
    const mc   = d.mc;
    if (mcEl && mc && mc.dist) {
      if(chartRegistry['mc-'+id]) chartRegistry['mc-'+id].destroy();
      const edges  = mc.dist.edges;
      const counts = mc.dist.counts;
      const labels = counts.map((_,i) => '$'+((edges[i]+edges[i+1])/2).toFixed(0));
      const colors = counts.map((_,i) => (edges[i]+edges[i+1])/2 >= 0
        ? 'rgba(34,197,94,.7)' : 'rgba(239,68,68,.7)');

      chartRegistry['mc-'+id] = new Chart(mcEl, {
        type: 'bar',
        data: { labels, datasets: [{
          data: counts, backgroundColor: colors,
          borderWidth: 0, barPercentage: 1.0, categoryPercentage: 1.0
        }]},
        options: {
          responsive: true, maintainAspectRatio: true,
          plugins: { legend: { display: false },
                     tooltip: { callbacks: { label: ctx => ctx.raw.toLocaleString() + ' paths' }}},
          scales: {
            x: { ticks: { color:'#8892a4', maxTicksLimit:8, font:{size:8} },
                 grid: { display: false }},
            y: { ticks: { color:'#8892a4', font:{size:8} },
                 grid: { color:'rgba(255,255,255,.04)' }}
          }
        },
        plugins:[{afterDraw(chart){
          // Draw vertical line at $0 (break-even marker)
          const {ctx, chartArea:{top,bottom}, scales:{x}} = chart;
          const zeroIdx = counts.findIndex((_,i)=>(edges[i]+edges[i+1])/2 >= 0);
          if(zeroIdx > 0){
            const xPos = x.getPixelForValue(zeroIdx - 0.5);
            ctx.save();
            ctx.strokeStyle='rgba(255,255,255,.4)';
            ctx.lineWidth=1.5;
            ctx.setLineDash([4,3]);
            ctx.beginPath();ctx.moveTo(xPos,top);ctx.lineTo(xPos,bottom);ctx.stroke();
            ctx.restore();
          }
        }}]
      });
    }
  });
}

// ── Trade queue ───────────────────────────────────────────────
// Cards rendered from the search overlay aren't in /api/results — keep the
// signal objects of the last search so Add/Track can find them.
window._searchSignals = window._searchSignals || {};

function findSignal(results, ticker, strategy) {
  return results.find(d=>d.ticker===ticker) || window._searchSignals[`${ticker}_${strategy}`] || null;
}

async function addToQueue(ticker, name, price, strategy, stratLabel, id) {
  const qty = parseInt(document.getElementById('qty-'+id)?.value)||1;
  // `strategy` is the singular name (sell_put); the results URL uses the plural bucket.
  // Sending the plural to /api/queue used to make calc_trade_cost miss the
  // sell_put branch, so cash-secured puts were costed at the premium and the
  // $1,000 guardrail never fired for them.
  const results = await fetch('/api/results/'+strategy+'s').then(r=>r.json()).catch(()=>[]);
  const signal  = findSignal(results, ticker, strategy);
  if(!signal){ alert('Signal no longer available. Re-run scan.'); return; }

  const entryPrice = signal.strategy === 'iron_condor'
    ? (signal.net_credit || 0)
    : (signal.option?.mid || signal.option?.ask || 0);

  const body = {ticker, name, price, strategy, strategy_label: stratLabel,
                option: signal.option, qty,
                entry_price: entryPrice,
                // Pass condor-specific fields for cost calculation
                put_width:  signal.put_width  || 0,
                call_width: signal.call_width || 0,
                legs:       signal.legs       || null};
  let resp;
  try {
    resp = await fetch('/api/queue',{method:'POST',headers:{'Content-Type':'application/json'},
                                     body:JSON.stringify(body)}).then(r=>r.json());
  } catch(e) { alert('Could not reach the bot: '+e); return; }
  if(!resp.ok){ alert(resp.message || resp.error || 'Could not add to queue.'); return; }
  const g = resp.guardrail;

  let msg = `Added ${ticker} ${stratLabel} ×${qty} to queue.`;
  if(!g.passes){
    msg += `\n\n⚠ This trade exceeds your $1,000 guardrail (est. $${g.cost.toFixed(2)}).\nYou'll need to type CONFIRM when executing.`;
  }
  alert(msg);
  document.getElementById('cnt-queue').textContent =
    parseInt(document.getElementById('cnt-queue').textContent||0)+1;
}

// ── Tracked positions ─────────────────────────────────────────
async function trackPosition(ticker, name, price, strategy, stratLabel, id) {
  const results = await fetch('/api/results/'+strategy+'s').then(r=>r.json()).catch(()=>[]);
  const signal  = findSignal(results, ticker, strategy);
  if(!signal){ alert('Signal no longer available. Re-run scan.'); return; }

  const entryPrice = signal.strategy === 'iron_condor'
    ? (signal.net_credit || 0)
    : (signal.option?.mid || signal.option?.ask || 0);

  const body = {
    ticker, name, price,
    strategy:        signal.strategy || strategy,
    strategy_label:  stratLabel,
    option:          signal.option   || {},
    legs:            signal.legs     || null,
    net_credit:      signal.net_credit || 0,
    put_width:       signal.put_width  || 0,
    call_width:      signal.call_width || 0,
    contracts:       parseInt(document.getElementById('qty-'+id)?.value)||1,
    entry_price:     entryPrice,
  };
  const resp = await fetch('/api/track',{method:'POST',headers:{'Content-Type':'application/json'},
                                          body:JSON.stringify(body)}).then(r=>r.json());
  if(resp.ok){
    alert(`📍 Now tracking ${ticker} ${stratLabel}. Prices and stop/target checks refresh every 3 minutes; the full sell-signal review runs every 6 hours.`);
    document.getElementById('cnt-tracked').textContent =
      parseInt(document.getElementById('cnt-tracked').textContent||0)+1;
  } else {
    alert(resp.message || 'Could not add to tracked positions.');
  }
}

async function refreshTracked() {
  let positions;
  try {
    positions = await fetch('/api/tracked').then(r=>r.json());
  } catch(e) {
    console.warn('tracked refresh failed (bot restarting?)', e);   // the 60 s poll used to throw an unhandled rejection
    return;
  }
  document.getElementById('cnt-tracked').textContent = positions.length;
  const el = document.getElementById('tracked-content');
  if(!positions.length){
    el.innerHTML='<div class="queue-empty">No tracked positions yet. Click "📍 Track" on any signal card to start monitoring it.</div>';
    return;
  }

  const rows = positions.map(p => {
    const sig    = p.last_signal || {};
    const fired  = sig.triggered;
    const reasons = (sig.reasons||[]).join('; ');
    const rsi    = sig.current_rsi   != null ? sig.current_rsi.toFixed(1)   : '—';
    const cprice = sig.current_price != null ? '$'+sig.current_price.toFixed(2) : '—';
    const valid  = sig.signal_still_valid;
    const chk    = p.last_checked ? new Date(p.last_checked).toLocaleTimeString() : 'Never';
    const opt    = p.option || {};

    // ── Entry / current option price ───────────────────────
    const entryMid    = p.entry_price ? +p.entry_price : null;
    const currentMid  = sig.current_option_mid != null ? +sig.current_option_mid : null;
    const currentBid  = sig.current_bid  != null ? +sig.current_bid  : null;
    const currentAsk  = sig.current_ask  != null ? +sig.current_ask  : null;
    const priceQuality = sig.price_quality || 'unavailable';
    const isManual    = p.entry_price_manual ? true : false;
    const entryLabel  = isManual ? '✏️ ' : '';
    const entryStr    = entryMid != null ? entryLabel+'$'+entryMid.toFixed(3) : '—';

    // ── Price quality badge ─────────────────────────────────
    // "live" = Yahoo returned a two-sided bid+ask, but Yahoo option chain
    // data can lag real-time by 15–30 min. Use Live Price to override
    // with your broker's real-time quote when prices seem off.
    const qBadge = {
      live:        { icon: '🟡', label: 'Yahoo quote',  color: '#f59e0b' },
      stale_bid:   { icon: '🟡', label: 'Stale bid',    color: '#f59e0b' },
      ask_only:    { icon: '🟡', label: 'Ask only',     color: '#f59e0b' },
      last_only:   { icon: '🔴', label: 'Stale',        color: 'var(--red)' },
      manual:      { icon: '🟢', label: 'Manual price', color: 'var(--green)' },
      unavailable: { icon: '⚫', label: 'No quote',     color: 'var(--muted)' },
    };
    const qInfo = qBadge[priceQuality] || qBadge['unavailable'];
    // Build mid/bid/ask detail line
    let bidAskLine = '';
    if (currentBid != null && currentAsk != null) {
      bidAskLine = `<span style="font-size:10px;color:var(--muted)">b $${currentBid.toFixed(2)} / a $${currentAsk.toFixed(2)}</span>`;
    } else if (currentAsk != null) {
      bidAskLine = `<span style="font-size:10px;color:var(--muted)">ask $${currentAsk.toFixed(2)}</span>`;
    }
    // Show timestamp for manual prices so user knows when they entered it
    const manualAt = p.live_price_override_at
      ? new Date(p.live_price_override_at).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})
      : null;
    const manualNote = (priceQuality === 'manual' && manualAt)
      ? `<span style="font-size:10px;color:var(--muted)">entered ${manualAt} · auto-updates</span>`
      : '';
    const qualityBadge = currentMid != null ? `
      <span style="font-size:10px;color:${qInfo.color};white-space:nowrap"
            title="Price quality: ${priceQuality}">${qInfo.icon} ${qInfo.label}</span>
      ${bidAskLine}${manualNote}` : '';
    const currentStr  = currentMid != null
      ? `<span style="font-weight:600">$${currentMid.toFixed(3)}</span>`
      : '—';

    // ── Edit entry price button ─────────────────────────────
    const editBtn = `<button onclick="editEntryPrice('${p.id}', ${entryMid||0})"
      title="Set your actual purchase price"
      style="font-size:10px;padding:2px 6px;margin-top:3px;cursor:pointer;
             background:var(--surface2);border:1px solid var(--border);
             border-radius:4px;color:var(--muted)">✏️ Edit entry</button>`;

    // ── Performance calculation ─────────────────────────────
    let perfHtml = `<div>${editBtn}</div>`;
    if (entryMid != null && currentMid != null) {
      const pct    = (currentMid - entryMid) / entryMid * 100;
      const dollar = (currentMid - entryMid) * 100;
      const color  = pct >= 0 ? 'var(--green)' : 'var(--red)';
      const arrow  = pct >= 0 ? '▲' : '▼';
      const sign   = pct >= 0 ? '+' : '';
      perfHtml = `
        <div style="display:flex;flex-direction:column;gap:2px">
          <span style="font-size:13px;font-weight:700;color:${color}">${arrow} ${sign}${pct.toFixed(1)}%</span>
          <span style="font-size:11px;color:${color}">${sign}$${dollar.toFixed(2)} / contract</span>
          <span style="font-size:10px;color:var(--muted)">${entryStr} → ${currentStr}</span>
          ${qualityBadge}
          ${editBtn}
        </div>`;
    } else if (entryMid != null) {
      perfHtml = `<div style="display:flex;flex-direction:column;gap:3px">
        <span style="font-size:11px;color:var(--muted)">Entry: ${entryStr}<br>Awaiting price…</span>
        ${qualityBadge}
        ${editBtn}
      </div>`;
    } else {
      perfHtml = `<div style="display:flex;flex-direction:column;gap:3px">
        <span style="font-size:11px;color:var(--muted)">No entry price set</span>
        ${editBtn}
      </div>`;
    }

    // ── RSI Momentum badge ──────────────────────────────────
    let momentumHtml = '<span style="color:var(--muted);font-size:12px">—</span>';
    if (p.last_signal && p.last_signal.momentum_label) {
      const ml      = p.last_signal.momentum_label;
      const slope   = p.last_signal.rsi_slope   || 0;
      const eta     = p.last_signal.eta_days;
      const dist    = p.last_signal.rsi_distance || 0;
      const slopeSign = slope >= 0 ? '+' : '';

      const mColors = {
        accelerating: '#22c55e',
        on_track:     '#60a5fa',
        turning_up:   '#34d399',
        turning_down: '#f87171',
        basing:       '#94a3b8',
        stalling:     '#f59e0b',
        reversing:    '#ef4444',
      };
      const mIcons = {
        accelerating: '🚀',
        on_track:     '📈',
        turning_up:   '↗️',
        turning_down: '↘️',
        basing:       '⏳',
        stalling:     '⚠️',
        reversing:    '🔴',
      };
      const mLabels = {
        accelerating: 'Accelerating',
        on_track:     'On Track',
        turning_up:   'Turning Up',
        turning_down: 'Turning Down',
        basing:       'Still Oversold',
        stalling:     'Stalling',
        reversing:    'Reversing',
      };
      const col   = mColors[ml] || 'var(--muted)';
      const icon  = mIcons[ml]  || '';
      // "basing" label depends on strategy — oversold for buy calls, overbought for buy puts
      const strat = p.strategy || '';
      const basingLabel = strat === 'buy_put' ? 'Still Overbought' : 'Still Oversold';
      const label = ml === 'basing' ? basingLabel : (mLabels[ml] || ml);

      const etaLine  = eta != null ? `<br><span style="font-size:10px;color:var(--muted)">~${eta}d to target</span>` : '';
      const distLine = dist > 0   ? `<br><span style="font-size:10px;color:var(--muted)">${dist.toFixed(1)} RSI pts left</span>` : '';

      momentumHtml = `
        <div>
          <span style="font-size:12px;font-weight:700;color:${col}">${icon} ${label}</span>
          <br><span style="font-size:11px;color:var(--muted)">RSI ${slopeSign}${slope.toFixed(1)}/day</span>
          ${etaLine}${distLine}
        </div>`;
    }

    // ── Signal badge ────────────────────────────────────────
    let signalBadge = '<span style="color:var(--muted);font-size:12px">Waiting for check…</span>';
    if (p.last_signal) {
      if (fired) {
        signalBadge = `<span style="background:var(--red);color:#fff;padding:3px 8px;border-radius:6px;font-weight:700;font-size:12px">🔴 SELL SIGNAL</span>`;
      } else if (valid === false) {
        signalBadge = `<span style="background:var(--yellow);color:#000;padding:3px 8px;border-radius:6px;font-weight:600;font-size:12px">⚠ Signal Gone</span>`;
      } else {
        signalBadge = `<span style="background:var(--green);color:#000;padding:3px 8px;border-radius:6px;font-weight:600;font-size:12px">✅ Hold</span>`;
      }
    }

    // ── Hold Horizon hint (Level 2: informational only) ─────
    // Surfaces what the latest Hold Horizon Analysis says about this
    // strategy at the current days-held. Does NOT change sell-signal
    // logic — it's guidance you decide whether to act on.
    let hhHtml = '';
    const hh = p.hold_horizon;
    if (hh && hh.optimal_day != null && hh.optimal_avg_pnl != null) {
        const arrow = hh.days_held >= hh.optimal_day ? '🎯' : '📊';
        const wr  = hh.current_win_rate != null ? ` · ${hh.current_win_rate.toFixed(0)}% wr` : '';
        hhHtml = `
          <div style="margin-top:6px;padding:6px 8px;background:var(--surface2);border-left:2px solid var(--border);border-radius:4px;font-size:10px;line-height:1.45;color:var(--muted)">
            <div style="color:var(--text);font-weight:600">${arrow} Hold Horizon</div>
            <div>Avg at ${hh.nearest_day}d (held ${hh.days_held}d): <b style="color:${(hh.current_avg_pnl||0)>=0?'var(--green)':'var(--red)'}">${(hh.current_avg_pnl||0).toFixed(1)}%</b>${wr}</div>
            <div>Optimal: <b>${hh.optimal_day}d</b> (avg <b style="color:${(hh.optimal_avg_pnl||0)>=0?'var(--green)':'var(--red)'}">${(hh.optimal_avg_pnl||0).toFixed(1)}%</b>)</div>
            ${hh.verdict ? `<div style="margin-top:3px;font-style:italic">${hh.verdict}</div>` : ''}
          </div>`;
    }

    // ── Tracked-since ───────────────────────────────────────
    const trackedAt = p.tracked_at ? new Date(p.tracked_at).toLocaleDateString() : '—';

    return `<tr>
      <td>
        <b>${p.ticker}</b><br>
        <span style="font-size:11px;color:var(--muted)">${esc(p.name||'')}</span><br>
        <span style="font-size:10px;color:var(--muted)">Since ${trackedAt}</span>
      </td>
      <td>${p.strategy_label||p.strategy}</td>
      <td style="font-size:12px">
        ${opt.type||''} $${opt.strike?.toFixed(2)||''}<br>
        <span style="color:var(--muted)">${opt.expiration||''}</span>
      </td>
      <td>${perfHtml}${hhHtml}</td>
      <td>
        ${cprice}<br>
        <span style="font-size:11px;color:var(--muted)">RSI: ${rsi}</span>
      </td>
      <td>${momentumHtml}</td>
      <td>
        ${signalBadge}
        ${fired&&reasons?`<br><span style="font-size:11px;color:var(--muted)">${reasons}</span>`:''}
      </td>
      <td style="font-size:11px;color:var(--muted)">${chk}</td>
      <td style="white-space:nowrap">
        <button class="btn btn-ghost" style="font-size:12px;padding:5px 10px;display:block;margin-bottom:5px"
          onclick="logFromTracker('${p.id}')">✅ Close &amp; Log</button>
        <button class="btn btn-ghost" style="font-size:12px;padding:5px 10px;display:block;margin-bottom:5px"
          onclick="enterLivePrice('${p.id}', ${currentMid||0}, ${p.live_price_override||0})">📝 Live Price</button>
        <button class="btn btn-ghost" style="font-size:12px;padding:5px 10px;display:block;margin-bottom:5px"
          onclick="openEditOptionModal('${p.id}','${opt.type||'call'}',${opt.strike||0},'${opt.expiration||''}')">✏️ Edit Option</button>
        <button class="btn btn-ghost" style="font-size:12px;padding:5px 10px;display:block"
          onclick="untrackPosition('${p.id}')">✕ Remove</button>
      </td>
    </tr>`;
  });

  el.innerHTML = `<table class="queue-table">
    <thead><tr>
      <th>Ticker</th><th>Strategy</th><th>Option</th>
      <th>Performance</th><th>Stock Price</th>
      <th>Momentum</th><th>Signal</th><th>Last Check</th><th>Actions</th>
    </tr></thead>
    <tbody>${rows.join('')}</tbody>
  </table>`;
}

async function editEntryPrice(posId, currentEntry) {
  const input = prompt(
    'Enter your actual purchase price for this option (per share, e.g. 0.85):',
    currentEntry > 0 ? currentEntry.toFixed(3) : ''
  );
  if (input === null) return;  // user cancelled
  const price = parseFloat(input);
  if (isNaN(price) || price <= 0) {
    alert('Please enter a valid price greater than 0.');
    return;
  }
  const resp = await fetch(`/api/track/${posId}/entry`, {
    method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({entry_price: price})
  }).then(r => r.json());
  if (resp.ok) {
    refreshTracked();
  } else {
    alert(resp.message || 'Could not update entry price.');
  }
}

async function untrackPosition(posId) {
  if(!confirm('Stop tracking this position?')) return;
  await fetch('/api/track/'+posId,{method:'DELETE'});
  refreshTracked();
}

// ── Enter live price ──────────────────────────────────────────
async function enterLivePrice(posId, currentMid, currentOverride) {
  const existing = currentOverride > 0 ? currentOverride : (currentMid > 0 ? currentMid : '');
  const input = prompt(
    'Enter the current option price from your broker (per share, e.g. 0.52).\n' +
    'Leave blank and press OK to clear a previous manual price.',
    existing > 0 ? existing.toFixed ? existing.toFixed(3) : existing : ''
  );
  if (input === null) return;  // cancelled

  // Empty input = clear override
  if (input.trim() === '') {
    await fetch(`/api/track/${posId}/live-price`, {
      method: 'PATCH',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ clear: true })
    });
    refreshTracked();
    return;
  }

  const price = parseFloat(input);
  if (isNaN(price) || price <= 0) {
    alert('Please enter a valid price greater than 0.');
    return;
  }

  const resp = await fetch(`/api/track/${posId}/live-price`, {
    method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ live_price: price })
  }).then(r => r.json()).catch(() => ({ ok: false }));

  if (resp.ok) {
    refreshTracked();
  } else {
    alert(resp.message || 'Could not update price.');
  }
}

// ── Edit Option modal ──────────────────────────────────────────
let _eomPosId = null;

function openEditOptionModal(posId, type, strike, expiration) {
  _eomPosId = posId;
  document.getElementById('eom-type').value   = type || 'call';
  document.getElementById('eom-strike').value = strike > 0 ? strike : '';
  document.getElementById('eom-exp').value    = expiration || '';
  document.getElementById('eom-status').textContent = '';
  document.getElementById('eom-save').disabled = false;
  const modal = document.getElementById('edit-option-modal');
  modal.style.display = 'flex';
}

function closeEditOptionModal() {
  document.getElementById('edit-option-modal').style.display = 'none';
  _eomPosId = null;
}

async function saveEditOption() {
  if (!_eomPosId) return;
  const type   = document.getElementById('eom-type').value;
  const strike = parseFloat(document.getElementById('eom-strike').value);
  const exp    = document.getElementById('eom-exp').value.trim();
  const status = document.getElementById('eom-status');

  if (!strike || strike <= 0) { status.textContent = '⚠ Enter a valid strike price.'; return; }
  if (!exp)                   { status.textContent = '⚠ Enter an expiration date.'; return; }

  document.getElementById('eom-save').disabled = true;
  status.style.color = 'var(--muted)';
  status.textContent = '⏳ Fetching live price for new option…';

  const resp = await fetch(`/api/track/${_eomPosId}/option`, {
    method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ type, strike, expiration: exp })
  }).then(r => r.json()).catch(() => ({ ok: false, message: 'Network error' }));

  if (resp.ok) {
    const opt = resp.option || {};
    const midStr = opt.mid > 0 ? ` — mid $${opt.mid.toFixed(3)}` : '';
    status.style.color = 'var(--green)';
    status.textContent = `✅ Saved${midStr}. Refreshing…`;
    setTimeout(() => {
      closeEditOptionModal();
      refreshTracked();
    }, 900);
  } else {
    status.style.color = 'var(--red)';
    status.textContent = `❌ ${resp.message || 'Could not update option.'}`;
    document.getElementById('eom-save').disabled = false;
  }
}

// Close modal when clicking backdrop
document.getElementById('edit-option-modal').addEventListener('click', function(e) {
  if (e.target === this) closeEditOptionModal();
});

// ── Ticker search ─────────────────────────────────────────────
async function searchTicker() {
  const sym = document.getElementById('ticker-search').value.trim().toUpperCase();
  if (!sym) { alert('Enter a ticker symbol first.'); return; }

  const overlay   = document.getElementById('search-overlay');
  const titleEl   = document.getElementById('search-title');
  const resultsEl = document.getElementById('search-results');

  titleEl.textContent = `Searching ${sym}…`;
  resultsEl.innerHTML = '<div style="color:var(--muted);padding:16px 0">Scanning ticker, please wait…</div>';
  overlay.style.display = 'block';

  let resp;
  try {
    resp = await fetch('/api/search', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ticker: sym})
    }).then(r => r.json());
  } catch(e) {
    resp = {ok:false, message:'Could not reach the bot: '+e};
  }

  if (!resp.ok) {
    titleEl.textContent = sym;
    resultsEl.innerHTML = `<div style="color:var(--red);padding:16px 0">${esc(resp.message)}</div>`;
    return;
  }

  const signals = resp.signals;
  const count   = Object.keys(signals).length;
  titleEl.textContent = `${sym} — ${count} signal${count !== 1 ? 's' : ''} found`;

  let html = '';
  const rendered = [];
  for (const [key, d] of Object.entries(signals)) {
    try {
      html += buildCard(d, key);
      rendered.push(d);
      window._searchSignals[`${d.ticker}_${d.strategy || key}`] = d;   // so Add/Track work for searched tickers
    } catch(e) { console.error('buildCard error', key, e); }
  }
  resultsEl.innerHTML = html || '<div style="color:var(--muted)">No cards to display.</div>';
  try { renderCharts(rendered); } catch(e) { console.error('renderCharts error', e); }
  for (const d of rendered) {
    renderNewsIntoCard(d.ticker, d.strategy, 'news-' + d.ticker + '_' + d.strategy);
  }
}

function closeSearch() {
  document.getElementById('search-overlay').style.display = 'none';
  document.getElementById('ticker-search').value = '';
}

async function refreshQueue() {
  const q = await fetch('/api/queue').then(r=>r.json());
  document.getElementById('cnt-queue').textContent = q.length;
  const el = document.getElementById('queue-content');
  if(!q.length){ el.innerHTML='<div class="queue-empty">Your trade queue is empty. Add trades from the scan tabs above.</div>'; return; }

  el.innerHTML = `<table class="queue-table">
    <thead><tr>
      <th>Ticker</th><th>Strategy</th><th>Option</th><th>Qty</th>
      <th>Est. Cost</th><th>Status</th><th>Actions</th>
    </tr></thead>
    <tbody>
    ${q.map(t=>`<tr>
      <td><b>${esc(t.ticker)}</b><br><span style="font-size:11px;color:var(--muted)">${esc(t.name||'')}</span></td>
      <td>${t.strategy_label||t.strategy}</td>
      <td style="font-size:12px">${t.option?.type||''} $${t.option?.strike?.toFixed(2)||''}<br>
          ${t.option?.expiration||''} (${t.option?.dte||'?'}d)</td>
      <td>${t.qty}</td>
      <td style="color:${t.estimated_cost>1000?'var(--yellow)':'var(--text)'}">
          $${(t.estimated_cost||0).toLocaleString('en-US',{minimumFractionDigits:2})}
          ${!t.guardrail_ok?'<br><span style="font-size:10px;color:var(--yellow)">⚠ Over limit</span>':''}
      </td>
      <td><span class="status-pill s-${t.status||'pending'}">${t.status||'pending'}</span></td>
      <td>
        <button class="btn btn-green" style="font-size:12px;padding:6px 12px"
          onclick="openModal('${t.id}')">Execute</button>
        <button class="btn btn-ghost" style="font-size:12px;padding:6px 10px;margin-left:6px"
          onclick="cancelTrade('${t.id}')">✕</button>
      </td>
    </tr>`).join('')}
    </tbody></table>`;
}

async function cancelTrade(id) {
  if(!confirm('Remove this trade from the queue?')) return;
  await fetch('/api/queue/'+id,{method:'DELETE'});
  refreshQueue();
}

// ── Execution modal ───────────────────────────────────────────
async function openModal(tradeId) {
  const q = await fetch('/api/queue').then(r=>r.json());
  const t = q.find(x=>x.id===tradeId);
  if(!t) return;
  pendingExec = t;

  const opt = t.option||{};
  document.getElementById('modal-summary').innerHTML = `
    <b>${t.ticker}</b> — ${t.strategy_label} &nbsp;·&nbsp; ${t.qty} contract(s)<br>
    Option: ${opt.type} &nbsp;$${opt.strike?.toFixed(2)} &nbsp;exp ${opt.expiration} (${opt.dte}d)<br>
    Bid/Ask: $${opt.bid?.toFixed(2)} / $${opt.ask?.toFixed(2)} &nbsp;·&nbsp; IV ${opt.iv?.toFixed(1)}%<br>
    <b>Estimated capital: $${(t.estimated_cost||0).toLocaleString('en-US',{minimumFractionDigits:2})}</b>
  `;

  const over = !t.guardrail_ok;
  const warn = document.getElementById('modal-guardrail');
  const inp  = document.getElementById('modal-confirm-input');
  const btn  = document.getElementById('modal-btn-confirm');

  if(over){
    warn.textContent = `⚠ This trade exceeds your $1,000 guardrail (est. $${(t.estimated_cost||0).toFixed(2)}). Type CONFIRM in the box below to proceed.`;
    warn.style.display='block';
    inp.style.display='block';
    inp.value='';
    btn.disabled=true;
  } else {
    warn.style.display='none';
    inp.style.display='none';
    btn.disabled=false;
  }

  document.getElementById('modal').classList.add('open');
}

function closeModal() {
  document.getElementById('modal').classList.remove('open');
  pendingExec = null;
}

function checkConfirmInput() {
  const val = document.getElementById('modal-confirm-input').value.trim().toUpperCase();
  document.getElementById('modal-btn-confirm').disabled = (val !== 'CONFIRM');
}

async function submitExecution() {
  if(!pendingExec) return;
  const t    = pendingExec;
  const over = !t.guardrail_ok;
  const body = over ? { override_confirm: document.getElementById('modal-confirm-input').value.trim() } : {};

  const btn = document.getElementById('modal-btn-confirm');
  const btnLabel = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Launching SoFi…';

  let resp;
  try {
    resp = await fetch('/api/execute/'+t.id,{
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
    }).then(r=>r.json());
  } catch(e) {
    resp = {ok:false, message:'Could not reach the bot: '+e};
  } finally {
    // Restore the button — it used to stay on the spinner forever after the first click.
    btn.innerHTML = btnLabel;
    btn.disabled  = false;
    closeModal();
  }
  if(resp.ok){
    alert(`✓ SoFi has been opened and your order form is filled in.\n\nPlease review the order on screen, then click "Place Order" yourself in SoFi.\n\nTrade moved to History.`);
  } else if(resp.guardrail_fail){
    alert('Guardrail blocked: '+resp.message);
  } else {
    alert('Execution error: '+resp.message);
  }
  refreshQueue();
}

// ── Backtest ──────────────────────────────────────────────────
let btPoll = null;

function btModeChange() {
  const mode = document.getElementById('bt-mode').value;
  document.getElementById('bt-ticker-wrap').style.display = mode==='ticker' ? 'block' : 'none';
}

async function runBacktest() {
  const mode      = document.getElementById('bt-mode').value;
  const ticker    = document.getElementById('bt-ticker')?.value.trim().toUpperCase() || '';
  const hold_days = parseInt(document.getElementById('bt-hold').value);
  const btn       = document.getElementById('bt-run-btn');
  if (mode==='ticker' && !ticker) { alert('Enter a ticker first'); return; }
  btn.disabled = true; btn.textContent = '⏳ Running…';
  document.getElementById('bt-status').textContent = 'Starting backtest…';
  document.getElementById('bt-summary').style.display = 'none';
  const r = await fetch('/api/backtest/run', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({mode, ticker, hold_days})
  });
  if (!r.ok) {
    const err = await r.json().catch(()=>({}));
    alert(err.message || err.error || 'Backtest could not start');
    btn.disabled=false; btn.textContent='▶ Run Backtest'; return;
  }
  if (btPoll) clearInterval(btPoll);
  btPoll = setInterval(async () => {
    try {
      const d = await fetch('/api/backtest/status').then(r=>r.json());
      if (d.running) {
        const pct = d.total ? Math.round(d.progress/d.total*100) : 0;
        document.getElementById('bt-status').textContent = `Scanning… ${d.progress}/${d.total} (${pct}%)`;
      } else {
        clearInterval(btPoll);
        btn.disabled=false; btn.textContent='▶ Run Backtest';
        document.getElementById('bt-status').textContent = `Complete — ${d.trade_count} trades found`;
        renderBacktestResults();
      }
    } catch(e) {}
  }, 1500);
}

async function renderBacktestResults() {
  const data = await fetch('/api/backtest/results').then(r=>r.json());
  if (!data || !data.summary) return;
  const s      = data.summary;
  const trades = data.trades || [];

  // ── Empty state ──────────────────────────────────────────────
  if (!s.total_trades) {
    document.getElementById('bt-stat-cards').innerHTML =
      `<div style="grid-column:1/-1;text-align:center;padding:32px;color:var(--muted)">
         <div style="font-size:28px;margin-bottom:8px">🔍</div>
         <div style="font-size:14px;font-weight:600;color:var(--text);margin-bottom:6px">No trades found</div>
         <div style="font-size:12px;max-width:400px;margin:0 auto">
           The backtest walked 2 years of history but no tickers met the signal criteria
           (RSI, BB, MACD confluence) during that period. Try a broader universe like
           <b>Focus List</b> or <b>Full S&P 500</b> to find more signal opportunities.
         </div>
       </div>`;
    document.getElementById('bt-strat-breakdown').innerHTML = '';
    document.getElementById('bt-summary').style.display = 'block';
    return;
  }

  // Summary stat cards
  const fmt = (v, pct=true) => v == null ? '—' : (pct ? (v>0?'+':'')+v+'%' : v);
  const cards = [
    {label:'Trades',   val: s.total_trades,       color:'var(--text)'},
    {label:'Win Rate', val: fmt(s.win_rate),       color: s.win_rate>=55?'var(--green)':s.win_rate>=45?'var(--yellow)':'var(--red)'},
    {label:'Avg P&L',  val: fmt(s.avg_pnl),       color: s.avg_pnl>0?'var(--green)':'var(--red)'},
    {label:'Avg Win',  val: fmt(s.avg_win),        color:'var(--green)'},
    {label:'Avg Loss', val: fmt(s.avg_loss),       color:'var(--red)'},
    {label:'Sharpe',   val: fmt(s.sharpe, false),  color: s.sharpe>0.5?'var(--green)':s.sharpe>0?'var(--yellow)':'var(--red)'},
    {label:'Best',     val: fmt(s.best),           color:'var(--green)'},
    {label:'Worst',    val: fmt(s.worst),          color:'var(--red)'},
  ];
  document.getElementById('bt-stat-cards').innerHTML = cards.map(c=>
    `<div style="background:var(--surface2);border-radius:10px;padding:14px 10px;text-align:center">
       <div style="font-size:11px;color:var(--muted);margin-bottom:4px">${c.label}</div>
       <div style="font-size:20px;font-weight:700;color:${c.color}">${c.val}</div>
     </div>`).join('');

  // Per-strategy breakdown
  const strats = s.by_strategy || {};
  const labels = {buy_call:'Buy Call 📈', buy_put:'Buy Put 📉', sell_put:'Sell Put 💰'};
  document.getElementById('bt-strat-breakdown').innerHTML = Object.entries(strats).map(([k,v])=>
    `<div style="display:inline-block;background:var(--surface2);border-radius:10px;padding:10px 16px;margin:0 8px 8px 0;font-size:12px">
       <b style="color:var(--accent)">${labels[k]||k}</b> &nbsp;
       ${v.count} trades · <span style="color:${v.win_rate>=50?'var(--green)':'var(--red)'}">WR ${v.win_rate}%</span> · Avg ${v.avg_pnl>0?'+':''}${v.avg_pnl}%
     </div>`).join('');

  // Equity curve on canvas — the summary block must be visible first or the
  // canvas measures 0 px wide and the curve is never drawn.
  document.getElementById('bt-summary').style.display = 'block';
  const canvas = document.getElementById('bt-equity-chart');
  const ctx    = canvas.getContext('2d');
  const W = canvas.offsetWidth || canvas.parentElement?.clientWidth || 800, H = 120;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  ctx.clearRect(0,0,W,H);
  // Build cumulative P&L series (sorted by entry_date)
  const sorted = [...trades].sort((a,b)=>a.entry_date.localeCompare(b.entry_date));
  let cum = 0; const equity = [0];
  sorted.forEach(t => { cum += t.pnl_pct; equity.push(cum); });
  const minE = Math.min(...equity), maxE = Math.max(...equity);
  const range = maxE - minE || 1;
  const toY = v => H - 10 - ((v - minE) / range) * (H - 20);
  const toX = i => (i / (equity.length-1)) * (W-20) + 10;
  // Zero line
  ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(10, toY(0)); ctx.lineTo(W-10, toY(0)); ctx.stroke();
  // Equity line
  ctx.beginPath();
  equity.forEach((v,i)=>{ i===0?ctx.moveTo(toX(i),toY(v)):ctx.lineTo(toX(i),toY(v)); });
  ctx.strokeStyle = cum>=0 ? '#10b981' : '#ef4444';
  ctx.lineWidth = 2; ctx.stroke();
  // Fill
  ctx.lineTo(toX(equity.length-1), toY(0)); ctx.lineTo(toX(0), toY(0)); ctx.closePath();
  ctx.fillStyle = cum>=0 ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.12)'; ctx.fill();

  // Trade table
  document.getElementById('bt-trade-count').textContent = `(${Math.min(trades.length, 200)} of ${trades.length} shown)`;
  document.getElementById('bt-trade-tbody').innerHTML = trades.slice(0,200).map(t=>{
    const color = t.win ? 'var(--green)' : 'var(--red)';
    const sl = {buy_call:'📈 Buy Call', buy_put:'📉 Buy Put', sell_put:'💰 Sell Put'}[t.strategy]||t.strategy;
    return `<tr style="border-bottom:1px solid var(--border)">
      <td style="padding:5px 8px;color:var(--muted)">${t.entry_date}</td>
      <td style="padding:5px 8px;font-weight:600">${t.ticker}</td>
      <td style="padding:5px 8px">${sl}</td>
      <td style="padding:5px 8px;text-align:right">$${t.entry_price}</td>
      <td style="padding:5px 8px;text-align:right">$${t.exit_price}</td>
      <td style="padding:5px 8px;text-align:right">$${t.entry_opt}</td>
      <td style="padding:5px 8px;text-align:right">$${t.exit_opt}</td>
      <td style="padding:5px 8px;text-align:right;font-weight:700;color:${color}">${t.pnl_pct>0?'+':''}${t.pnl_pct}%</td>
      <td style="padding:5px 8px;text-align:right;color:var(--muted)">${t.iv_est}%</td>
    </tr>`;
  }).join('');

  // Signal quality analysis
  const sigQual = data.sig_quality || [];
  if (sigQual.length) {
    const WR_COLOR = wr => wr >= 60 ? 'var(--green)' : wr >= 50 ? 'var(--yellow)' : 'var(--red)';
    document.getElementById('bt-sig-quality-rows').innerHTML = sigQual.map(r => {
      const bar = Math.round(r.win_rate);
      return `<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:12px">
        <div style="width:140px;color:var(--text);flex-shrink:0">${r.signal}</div>
        <div style="flex:1;background:var(--surface2);border-radius:4px;height:16px;overflow:hidden">
          <div style="width:${bar}%;height:100%;background:${WR_COLOR(r.win_rate)};transition:width .4s"></div>
        </div>
        <div style="width:50px;text-align:right;color:${WR_COLOR(r.win_rate)};font-weight:600">${r.win_rate}%</div>
        <div style="width:60px;text-align:right;color:var(--muted)">${r.count} trades</div>
        <div style="width:65px;text-align:right;color:${r.avg_pnl>0?'var(--green)':'var(--red)'}">${r.avg_pnl>0?'+':''}${r.avg_pnl}% avg</div>
      </div>`;
    }).join('');
    document.getElementById('bt-sig-quality').style.display = 'block';
  }

  document.getElementById('bt-summary').style.display = 'block';
}

// ── Walk-Forward Optimizer ────────────────────────────────────
let wfoPoll = null;

async function runOptimizer() {
  const mode = document.getElementById('wfo-mode').value;
  const btn  = document.getElementById('wfo-run-btn');
  btn.disabled = true; btn.textContent = '⏳ Optimizing…';
  document.getElementById('wfo-status').textContent = 'Starting optimizer…';
  document.getElementById('wfo-results').style.display = 'none';
  const r = await fetch('/api/optimizer/run', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({mode})
  });
  if (!r.ok) {
    const err = await r.json();
    alert(err.message || 'Optimizer error');
    btn.disabled=false; btn.textContent='🧬 Run Optimizer'; return;
  }
  if (wfoPoll) clearInterval(wfoPoll);
  wfoPoll = setInterval(async () => {
    try {
      const d = await fetch('/api/optimizer/status').then(r=>r.json());
      if (d.running) {
        const pct = d.total ? Math.round(d.progress/d.total*100) : 0;
        document.getElementById('wfo-status').textContent =
          `${d.stage} — ${d.progress}/${d.total} combos (${pct}%)`;
      } else {
        clearInterval(wfoPoll);
        btn.disabled=false; btn.textContent='🧬 Run Optimizer';
        const ts = d.completed_at ? ` — last run ${new Date(d.completed_at).toLocaleString()}` : '';
        document.getElementById('wfo-status').textContent = (d.stage || 'Complete') + ts;
        renderOptimizerResults();
      }
    } catch(e) {}
  }, 2000);
}

async function renderOptimizerResults() {
  const data = await fetch('/api/optimizer/results').then(r=>r.json());
  if (!data || !data.best) return;
  const best = data.best;
  const p    = best.params;

  // Best params card
  document.getElementById('wfo-best-card').innerHTML = `
    <div style="font-size:13px;font-weight:700;color:var(--accent);margin-bottom:10px">
      🏆 Best Parameters Found
      <span style="font-size:11px;font-weight:400;color:var(--muted);margin-left:8px">
        Validation Sharpe: <b style="color:var(--green)">${best.val_sharpe}</b> ·
        Win Rate: <b>${best.val_win_rate}%</b> ·
        Avg P&L: <b>${best.val_avg_pnl>0?'+':''}${best.val_avg_pnl}%</b> ·
        ${best.val_trades} val trades
      </span>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px">
      ${Object.entries(p).map(([k,v])=>
        `<span style="background:rgba(99,102,241,.15);border:1px solid rgba(99,102,241,.3);
                border-radius:8px;padding:4px 10px;font-size:12px">
           <span style="color:var(--muted)">${k.replace(/_/g,' ')}</span>
           <b style="color:var(--accent);margin-left:4px">${v}</b>
         </span>`).join('')}
    </div>
    <button class="btn btn-primary" onclick="applyBestParams()" style="font-size:12px;padding:6px 16px">
      ⚡ Apply to Live Scanner
    </button>
    <span id="wfo-apply-msg" style="font-size:11px;color:var(--muted);margin-left:10px"></span>`;

  // Leaderboard
  document.getElementById('wfo-leaderboard-tbody').innerHTML =
    (data.leaderboard || []).map((row, i) => {
      const p2 = row.params;
      const isTop = i === 0;
      return `<tr style="border-bottom:1px solid var(--border);${isTop?'background:rgba(16,185,129,.07)':''}">
        <td style="padding:5px 8px">${p2.rsi_oversold ?? '—'}</td>
        <td style="padding:5px 8px">${p2.rsi_overbought ?? '—'}</td>
        <td style="padding:5px 8px">${p2.bull_min ?? '—'}</td>
        <td style="padding:5px 8px">${p2.bear_min ?? '—'}</td>
        <td style="padding:5px 8px">${p2.adx_threshold ?? '—'}</td>
        <td style="padding:5px 8px" title="stop ${p2.stop_loss_pct ?? '—'} / target ${p2.profit_target_pct ?? '—'} / stock stop ${p2.stock_stop_pct ?? '—'}">${p2.hold_days ?? '—'}d</td>
        <td style="padding:5px 8px;text-align:right;color:var(--muted)">${row.train_sharpe}</td>
        <td style="padding:5px 8px;text-align:right;font-weight:${isTop?700:400};
            color:${row.val_sharpe>0?'var(--green)':'var(--red)'}">${row.val_sharpe}</td>
        <td style="padding:5px 8px;text-align:right;
            color:${row.val_win_rate>=50?'var(--green)':'var(--red)'}">${row.val_win_rate}%</td>
        <td style="padding:5px 8px;text-align:right;
            color:${row.val_avg_pnl>0?'var(--green)':'var(--red)'}">
            ${row.val_avg_pnl>0?'+':''}${row.val_avg_pnl}%</td>
      </tr>`;
    }).join('');

  document.getElementById('wfo-results').style.display = 'block';
}

async function applyBestParams() {
  const r = await fetch('/api/optimizer/apply', {method:'POST'});
  const d = await r.json();
  const msg = document.getElementById('wfo-apply-msg');
  if (d.ok) {
    msg.textContent = '✓ Applied — takes effect on next scan';
    msg.style.color = 'var(--green)';
  } else {
    msg.textContent = d.message || 'Failed';
    msg.style.color = 'var(--red)';
  }
}

// ── History ───────────────────────────────────────────────────
async function refreshHistory() {
  const h = await fetch('/api/history').then(r=>r.json());
  document.getElementById('cnt-history').textContent = h.length;
  const el = document.getElementById('history-content');
  if(!h.length){ el.innerHTML='<div class="queue-empty">No trades executed yet.</div>'; return; }

  // Build table immediately with loading cells
  el.innerHTML = `<table class="queue-table">
    <thead><tr>
      <th>Ticker</th><th>Strategy</th><th>Option</th><th>Qty</th>
      <th>Entry Cost</th><th>Performance</th><th>Executed At</th>
    </tr></thead>
    <tbody>
    ${[...h].reverse().map(t=>`<tr>
      <td>
        <b>${t.ticker}</b><br>
        <span style="font-size:11px;color:var(--muted)">${esc(t.name||'')}</span>
      </td>
      <td style="font-size:12px">${t.strategy_label||t.strategy}</td>
      <td style="font-size:12px">
        ${t.option?.type||''} $${t.option?.strike?.toFixed(2)||''}<br>
        <span style="color:var(--muted)">${t.option?.expiration||''}</span>
      </td>
      <td>${t.qty}</td>
      <td>
        $${(t.estimated_cost||0).toFixed(2)}<br>
        <span style="font-size:10px;color:var(--muted)">Entry: $${(+t.entry_price||0).toFixed(3)}</span>
      </td>
      <td id="hperf-${t.id}">
        <span style="color:var(--muted);font-size:12px">Loading…</span>
      </td>
      <td style="font-size:12px;color:var(--muted)">
        ${t.executed_at?new Date(t.executed_at).toLocaleString():'-'}
      </td>
    </tr>`).join('')}
    </tbody></table>`;

  // Async fetch live performance for each trade
  try {
    const perf = await fetch('/api/history/perf').then(r=>r.json());
    h.forEach(t => {
      const cell = document.getElementById('hperf-'+t.id);
      if(!cell) return;
      const p = perf[t.id];
      if(!p) {
        cell.innerHTML = '<span style="font-size:11px;color:var(--muted)">Expired / N/A</span>';
        return;
      }
      const color = p.pct_change >= 0 ? 'var(--green)' : 'var(--red)';
      const arrow = p.pct_change >= 0 ? '▲' : '▼';
      const sign  = p.pct_change >= 0 ? '+' : '';
      cell.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:2px">
          <span style="font-size:14px;font-weight:700;color:${color}">${arrow} ${sign}${p.pct_change.toFixed(1)}%</span>
          <span style="font-size:11px;color:${color}">${sign}$${p.dollar_change.toFixed(2)} total</span>
          <span style="font-size:10px;color:var(--muted)">$${p.entry_mid.toFixed(3)} → $${p.current_mid.toFixed(3)}</span>
        </div>`;
    });
  } catch(e) {
    console.error('History perf error', e);
  }
}

// ── Performance Tracker ───────────────────────────────────────
let _perfChart = null;

async function refreshPerformance() {
  try {
    const d = await fetch('/api/performance').then(r => r.json());
    const s = d.summary;
    const trades = d.trades || [];

    // Summary cards
    const pnlColor    = s.total_pnl >= 0 ? 'var(--green)' : 'var(--red)';
    const pnlSign     = s.total_pnl >= 0 ? '+' : '';
    const avgPctSign  = (s.avg_pnl_pct||0) >= 0 ? '+' : '';
    const totalPctSign = (s.total_pnl_pct||0) >= 0 ? '+' : '';
    document.getElementById('perf-total-pnl').textContent      = s.total_trades > 0 ? `${pnlSign}$${s.total_pnl.toFixed(2)}` : '—';
    document.getElementById('perf-total-pnl').style.color      = pnlColor;
    document.getElementById('perf-avg-pnl-pct').textContent    = s.total_trades > 0 ? `Avg ${avgPctSign}${(s.avg_pnl_pct||0).toFixed(1)}% per trade` : '';
    document.getElementById('perf-avg-pnl-pct').style.color    = pnlColor;
    document.getElementById('perf-total-pnl-pct').textContent  = s.total_trades > 0 ? `${totalPctSign}${(s.total_pnl_pct||0).toFixed(1)}%` : '—';
    document.getElementById('perf-total-pnl-pct').style.color  = pnlColor;
    document.getElementById('perf-total-invested').textContent = s.total_trades > 0 ? `on $${(s.total_invested||0).toFixed(0)} deployed` : '';
    document.getElementById('perf-win-rate').textContent       = s.total_trades > 0 ? `${s.win_rate.toFixed(0)}%` : '—';
    document.getElementById('perf-wl-counts').textContent      = s.total_trades > 0 ? `${s.winners}W / ${s.losers}L (${s.total_trades} trades)` : '';
    document.getElementById('perf-avg-win').textContent        = s.winners > 0 ? `+${s.avg_win_pct.toFixed(1)}%` : '—';
    document.getElementById('perf-avg-loss').textContent       = s.losers  > 0 ? `${s.avg_loss_pct.toFixed(1)}%` : '—';
    document.getElementById('perf-best').textContent           = s.best_trade  != null ? `+${s.best_trade.toFixed(1)}%`  : '—';
    document.getElementById('perf-worst').textContent          = s.worst_trade != null ? `${s.worst_trade.toFixed(1)}%` : '—';

    // Advanced options-trader metrics (expectancy/PF/Sortino/MaxDD)
    try {
      const stats = await fetch('/api/performance/stats').then(r => r.json());
      if (stats && stats.n_trades > 0) {
        const expSign  = stats.expectancy >= 0 ? '+' : '';
        const expColor = stats.expectancy >= 0 ? 'var(--green)' : 'var(--red)';
        document.getElementById('perf-expectancy').textContent = `${expSign}$${stats.expectancy.toFixed(2)}`;
        document.getElementById('perf-expectancy').style.color = expColor;

        const pf = stats.profit_factor;
        const pfStr = pf == null ? '∞' : pf.toFixed(2);
        const pfColor = pf == null ? 'var(--green)' : pf >= 2 ? 'var(--green)' : pf >= 1.5 ? 'var(--text)' : 'var(--red)';
        document.getElementById('perf-profit-factor').textContent = pfStr;
        document.getElementById('perf-profit-factor').style.color = pfColor;

        const sor = stats.sortino;
        const sorColor = sor >= 1 ? 'var(--green)' : sor >= 0.5 ? 'var(--text)' : 'var(--red)';
        document.getElementById('perf-sortino').textContent = sor.toFixed(2);
        document.getElementById('perf-sortino').style.color = sorColor;

        document.getElementById('perf-max-dd').textContent = `-$${stats.max_drawdown.toFixed(2)}`;
        document.getElementById('perf-max-dd-pct').textContent = stats.max_drawdown_pct > 0 ? `${stats.max_drawdown_pct.toFixed(0)}% of peak` : '';
      }
    } catch (e) { /* ignore */ }

    // Open portfolio risk panel
    try {
      const risk = await fetch('/api/portfolio/risk').then(r => r.json());
      const panel = document.getElementById('portfolio-risk-panel');
      const body  = document.getElementById('portfolio-risk-body');
      if (risk.n_positions > 0) {
        panel.style.display = 'block';
        const concColor = risk.largest_concentration_pct > 40 ? 'var(--red)' : 'var(--text)';
        body.innerHTML = `
          <div>
            <div style="color:var(--muted);font-size:11px">POSITIONS</div>
            <div style="font-weight:700;font-size:16px;color:var(--text)">${risk.n_positions}</div>
          </div>
          <div>
            <div style="color:var(--muted);font-size:11px">TOTAL MAX LOSS</div>
            <div style="font-weight:700;font-size:16px;color:var(--red)">$${risk.total_max_loss.toFixed(0)}</div>
            <div style="color:var(--muted);font-size:10px">if all → $0</div>
          </div>
          <div>
            <div style="color:var(--muted);font-size:11px">STOP-LOSS EXPOSURE</div>
            <div style="font-weight:700;font-size:16px;color:var(--text)">$${risk.stop_loss_exposure.toFixed(0)}</div>
            <div style="color:var(--muted);font-size:10px">if all hit -${_stopLossPct}%</div>
          </div>
          <div>
            <div style="color:var(--muted);font-size:11px">LARGEST POSITION</div>
            <div style="font-weight:700;font-size:16px;color:${concColor}">${risk.largest_position_ticker || '—'}</div>
            <div style="color:var(--muted);font-size:10px">${risk.largest_concentration_pct.toFixed(0)}% of book ($${risk.largest_position.toFixed(0)})</div>
          </div>
          <div>
            <div style="color:var(--muted);font-size:11px">CURRENT P&L STDEV</div>
            <div style="font-weight:700;font-size:16px;color:var(--text)">${risk.pnl_stdev.toFixed(1)}%</div>
            <div style="color:var(--muted);font-size:10px">min ${risk.pnl_min.toFixed(1)}% / max ${risk.pnl_max.toFixed(1)}%</div>
          </div>
        `;
      } else {
        panel.style.display = 'none';
      }
    } catch (e) { /* ignore */ }

    // Cumulative P&L chart
    const sorted = [...trades].sort((a,b) => new Date(a.exit_date||0) - new Date(b.exit_date||0));
    let cumulative = 0;
    const labels = [], dataPoints = [];
    sorted.forEach(t => {
      cumulative += (t.pnl_total || 0);
      labels.push(t.ticker + ' ' + (t.exit_date ? t.exit_date.slice(0,10) : ''));
      dataPoints.push(+cumulative.toFixed(2));
    });

    const ctx = document.getElementById('perf-chart').getContext('2d');
    if (_perfChart) { _perfChart.destroy(); _perfChart = null; }
    _perfChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Cumulative P&L ($)',
          data: dataPoints,
          borderColor: dataPoints.length && dataPoints[dataPoints.length-1] >= 0 ? '#22c55e' : '#ef4444',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 4,
          tension: 0.3
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#94a3b8', maxRotation: 45, font: { size: 11 } }, grid: { color: '#1e293b' } },
          y: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: '#1e293b' } }
        }
      }
    });

    // Trade history table
    const wrap = document.getElementById('perf-table-wrap');
    if (!trades.length) {
      wrap.innerHTML = '<div style="color:var(--muted);font-size:13px;text-align:center;padding:20px">No trades logged yet.</div>';
      return;
    }
    const byDate = [...trades].sort((a,b) => new Date(b.exit_date||0) - new Date(a.exit_date||0));
    const tRows = byDate.map(t => {
      const pct    = t.pnl_pct   != null ? t.pnl_pct   : 0;
      const dollar = t.pnl_total != null ? t.pnl_total : 0;
      const color  = pct >= 0 ? 'var(--green)' : 'var(--red)';
      const sign   = pct >= 0 ? '+' : '';
      const contracts = t.contracts || 1;
      const stratLabel = {buy_calls:'Buy Calls', buy_puts:'Buy Puts', sell_puts:'Sell Puts', covered_calls:'Covered Calls'}[t.strategy] || t.strategy || '—';
      return `<tr>
        <td>${t.exit_date ? t.exit_date.slice(0,10) : '—'}</td>
        <td><b>${t.ticker||'—'}</b></td>
        <td>${stratLabel}</td>
        <td style="font-size:12px">${t.option_desc||'—'}</td>
        <td>$${(t.entry_price||0).toFixed(3)}</td>
        <td>$${(t.exit_price||0).toFixed(3)}</td>
        <td style="font-weight:700;color:${color}">${sign}${pct.toFixed(1)}%</td>
        <td style="color:${color}">${sign}$${dollar.toFixed(2)}<br><span style="font-size:10px;color:var(--muted)">${contracts} contract${contracts>1?'s':''}</span></td>
        <td style="font-size:12px;color:var(--muted)">${t.notes||''}</td>
        <td>
          <button class="btn btn-ghost" style="font-size:11px;padding:3px 8px"
            onclick="deletePerformanceTrade('${t.id}')">✕</button>
        </td>
      </tr>`;
    }).join('');
    wrap.innerHTML = `<table class="queue-table">
      <thead><tr>
        <th>Date</th><th>Ticker</th><th>Strategy</th><th>Option</th>
        <th>Entry</th><th>Exit</th><th>P&L %</th><th>P&L $</th><th>Notes</th><th></th>
      </tr></thead>
      <tbody>${tRows}</tbody>
    </table>`;
  } catch(e) {
    console.error('refreshPerformance error', e);
  }
}

async function logFromTracker(posId) {
  const exitStr = prompt('Enter your exit price for this option (per share, e.g. 2.50):');
  if (exitStr === null) return;
  const exitPrice = parseFloat(exitStr);
  if (isNaN(exitPrice) || exitPrice < 0) {
    alert('Please enter a valid exit price (0 or more).');
    return;
  }
  const notesStr = prompt('Add a note (optional, press Enter to skip):') || '';
  const resp = await fetch('/api/performance/from-tracker', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ position_id: posId, exit_price: exitPrice, notes: notesStr })
  }).then(r => r.json());
  if (resp.ok) {
    refreshTracked();
    refreshPerformance();
    // Update tracked count badge
    const tracked = await fetch('/api/tracked').then(r=>r.json());
    document.getElementById('cnt-tracked').textContent = tracked.length;
  } else {
    alert(resp.message || resp.error || 'Could not log trade.');
  }
}

async function addManualTrade() {
  const ticker     = document.getElementById('mt-ticker').value.trim().toUpperCase();
  const strategy   = document.getElementById('mt-strategy').value;
  const strike     = parseFloat(document.getElementById('mt-strike').value) || 0;
  const expiration = document.getElementById('mt-expiration').value.trim();
  const entry      = parseFloat(document.getElementById('mt-entry').value);
  const exit       = parseFloat(document.getElementById('mt-exit').value);
  const contracts  = parseInt(document.getElementById('mt-contracts').value) || 1;
  const closeDate  = document.getElementById('mt-date').value;
  const notes      = document.getElementById('mt-notes').value.trim();
  const status     = document.getElementById('mt-status');

  if (!ticker || isNaN(entry) || isNaN(exit)) {
    status.textContent = '⚠ Ticker, entry and exit price are required.';
    status.style.color = 'var(--red)';
    return;
  }
  const resp = await fetch('/api/performance', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      ticker, strategy,
      option_desc: `${['buy_calls','covered_calls'].includes(strategy)?'Call':'Put'} $${strike} ${expiration}`.trim(),
      entry_price: entry, exit_price: exit,
      contracts, exit_date: closeDate, notes
    })
  }).then(r => r.json());

  if (resp.ok) {
    status.textContent = '✅ Trade logged!';
    status.style.color = 'var(--green)';
    // Clear form
    ['mt-ticker','mt-strike','mt-expiration','mt-entry','mt-exit','mt-notes'].forEach(id => {
      document.getElementById(id).value = '';
    });
    document.getElementById('mt-contracts').value = '1';
    document.getElementById('mt-date').value = '';
    refreshPerformance();
    setTimeout(() => { status.textContent = ''; }, 3000);
  } else {
    status.textContent = resp.message || resp.error || 'Error logging trade.';
    status.style.color = 'var(--red)';
  }
}

async function deletePerformanceTrade(tradeId) {
  if (!confirm('Delete this trade from history? This cannot be undone.')) return;
  const resp = await fetch(`/api/performance/${tradeId}`, { method: 'DELETE' }).then(r => r.json());
  if (resp.ok) {
    refreshPerformance();
  } else {
    alert(resp.message || resp.error || 'Could not delete trade.');
  }
}

// ── Init ──────────────────────────────────────────────────────
// ── Black-Scholes Analyser ────────────────────────────────────
function openBSFromCard(ticker, stockPrice, strike, dte, iv, marketPrice, optType) {
  // Switch to the BS tab
  switchTab('bsmodel');
  // Populate fields from scan data
  document.getElementById('bs-ticker').value    = ticker;
  document.getElementById('bs-stock').value     = stockPrice;
  document.getElementById('bs-strike').value    = strike;
  document.getElementById('bs-dte').value       = dte;
  document.getElementById('bs-iv').value        = iv;
  document.getElementById('bs-mktprice').value  = marketPrice;
  // Map strategy name to call/put
  const t = (optType || '').toLowerCase();
  document.getElementById('bs-type').value = t.includes('put') ? 'put' : 'call';
  // Auto-run the analysis
  runBSAnalysis();
}

async function runBSAnalysis() {
  const ticker    = document.getElementById('bs-ticker').value.trim();
  const stock     = parseFloat(document.getElementById('bs-stock').value);
  const strike    = parseFloat(document.getElementById('bs-strike').value);
  const dte       = parseFloat(document.getElementById('bs-dte').value);
  const iv        = parseFloat(document.getElementById('bs-iv').value);
  const mktPrice  = parseFloat(document.getElementById('bs-mktprice').value) || 0;
  const optType   = document.getElementById('bs-type').value;

  document.getElementById('bs-results').style.display = 'none';
  document.getElementById('bs-error').style.display   = 'none';

  const res = await fetch('/api/bs-analyze', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ticker, stock_price: stock, strike, dte, iv_pct: iv,
                          market_price: mktPrice, option_type: optType})
  });
  const d = await res.json();

  if (!d.ok) {
    document.getElementById('bs-error').textContent = d.error;
    document.getElementById('bs-error').style.display = 'block';
    return;
  }

  // Banner
  const bannerColors = {overpriced:'#7f1d1d', underpriced:'#14532d', 'fairly priced':'#1e3a5f', unavailable:'#1e293b'};
  const bannerLabels = {
    overpriced:     `⚠️ Option is OVERPRICED — market price is ${d.mispricing_pct}% above BS fair value`,
    underpriced:    `✅ Option is UNDERPRICED — market price is ${Math.abs(d.mispricing_pct)}% below BS fair value`,
    'fairly priced':`✔️ Option is FAIRLY PRICED — market price is within 5% of BS fair value`,
    unavailable:    `ℹ️ Enter a market price to see overall mispricing`
  };
  const bannerKey = d.mispricing_label || 'unavailable';
  const banner = document.getElementById('bs-banner');
  banner.style.background = bannerColors[bannerKey] || '#1e293b';
  banner.style.color = '#fff';
  banner.textContent = bannerLabels[bannerKey] || '';

  // Summary cards
  document.getElementById('bs-fair').textContent = `$${d.fair_value.toFixed(3)}`;
  document.getElementById('bs-vs-market').textContent =
    mktPrice > 0 ? `Market: $${mktPrice.toFixed(2)} (${d.mispricing_pct > 0 ? '+' : ''}${d.mispricing_pct}%)` : 'Enter market price for comparison';

  const volColors = {overpriced:'#ef4444', underpriced:'#22c55e', fair:'#60a5fa', unavailable:'#94a3b8'};
  const volLabels = {
    overpriced:  `IV ${iv}% vs GJR-GARCH ${d.garch_vol_pct}% → +${d.vol_diff_pct}% overpriced`,
    underpriced: `IV ${iv}% vs GJR-GARCH ${d.garch_vol_pct}% → ${d.vol_diff_pct}% underpriced`,
    fair:        `IV ${iv}% ≈ GJR-GARCH ${d.garch_vol_pct}% → fairly priced vol`,
    unavailable: `Enter ticker to compare IV vs GARCH vol`
  };
  document.getElementById('bs-vol-compare').style.color = volColors[d.vol_verdict] || '#94a3b8';
  document.getElementById('bs-vol-compare').textContent = volLabels[d.vol_verdict] || '';
  document.getElementById('bs-vol-verdict').textContent =
    d.vol_verdict === 'overpriced' ? 'Better for sellers. Buyers overpaying for vol.' :
    d.vol_verdict === 'underpriced'? 'Better for buyers. Cheap vol.' :
    d.vol_verdict === 'fair'       ? 'No vol edge. Trade on direction.' : '';
  document.getElementById('bs-vol-verdict').style.color =
    d.vol_verdict === 'overpriced' ? '#ef4444' : d.vol_verdict === 'underpriced' ? '#22c55e' : '#94a3b8';

  document.getElementById('bs-intrinsic').innerHTML =
    `<span style="color:var(--green)">Intrinsic: $${d.intrinsic.toFixed(3)}</span>`;
  document.getElementById('bs-extrinsic').innerHTML =
    `<span>Extrinsic (time+vol): $${d.extrinsic.toFixed(3)}</span>`;

  // Greeks
  const gk = d.greeks;
  const greekDefs = [
    {name:'Delta', val: gk.delta.toFixed(4), desc:'$ move per $1 stock move'},
    {name:'Gamma', val: gk.gamma.toFixed(5), desc:'Delta change per $1 stock move'},
    {name:'Theta', val: gk.theta.toFixed(4), desc:'$ lost per day (time decay)'},
    {name:'Vega',  val: gk.vega.toFixed(4),  desc:'$ gained per 1% IV rise'},
    {name:'Rho',   val: gk.rho.toFixed(4),   desc:'$ change per 1% rate move'},
  ];
  document.getElementById('bs-greeks').innerHTML = greekDefs.map(g => `
    <div style="text-align:center">
      <div style="font-size:11px;color:var(--muted);margin-bottom:4px">${g.name}</div>
      <div style="font-size:18px;font-weight:700;color:var(--accent)">${g.val}</div>
      <div style="font-size:10px;color:var(--muted);margin-top:4px">${g.desc}</div>
    </div>`).join('');

  // Interpretation
  document.getElementById('bs-interpretation').innerHTML = d.interpretation;

  document.getElementById('bs-results').style.display = 'block';
}

window.onload = async () => {
  // Guarded: if this first call failed nothing below it ever initialised.
  try {
    const s = await fetch('/api/scan/status').then(r=>r.json());
    if(s.last_scan) {
      const _nextStr = s.next_rescan ? ` · Next: ${s.next_rescan}` : '';
      document.getElementById('scan-info').textContent = `Last scan: ${new Date(s.last_scan).toLocaleTimeString()}${_nextStr}`;
      ['sell_puts','buy_calls','buy_puts','covered_calls'].forEach(loadStrategy);
      loadCondorResults();
    }
  } catch(e) { console.error('status load failed', e); }
  // Restore WFO results if saved from a previous session
  try {
    const wfo = await fetch('/api/optimizer/status').then(r=>r.json());
    if (wfo.best && wfo.completed_at) {
      const ts = new Date(wfo.completed_at).toLocaleString();
      document.getElementById('wfo-status').textContent = `Restored from last run — ${ts} (params auto-applied)`;
      renderOptimizerResults();
    }
  } catch(e) {}
  refreshQueue();
  initGreenCandle();
  initRedCandle();
  initRiskBudget();
  // Load tracked count on startup
  try {
    const tracked = await fetch('/api/tracked').then(r=>r.json());
    document.getElementById('cnt-tracked').textContent = tracked.length;
  } catch(e) { console.error('tracked load failed', e); }
  // Auto-refresh tracked tab every 60s if it's active
  setInterval(()=>{ if(activeTab==='tracked') refreshTracked(); }, 60000);
  // Load VIX regime on startup
  loadCalRegime();
  // Load any saved calibration results and signal log stats
  loadCalResults();
  loadSignalLogStats();
};

// ── Calibration ───────────────────────────────────────────────
let calPoll = null;

async function loadCalRegime() {
  try {
    const d = await fetch('/api/vix/regime').then(r => r.json());
    document.getElementById('cal-regime-icon').textContent  = d.icon  || '⚪';
    document.getElementById('cal-regime-label').textContent =
      `${d.icon} VIX Regime: ${(d.regime || 'unknown').toUpperCase()}`;
    document.getElementById('cal-regime-label').style.color = d.color || 'var(--text)';
    document.getElementById('cal-regime-desc').textContent  = d.description || '';
    document.getElementById('cal-vix-val').textContent      = d.vix != null ? d.vix.toFixed(1) : '—';
    document.getElementById('cal-rsi-os').textContent       = d.rsi_oversold  != null ? d.rsi_oversold  : '—';
    document.getElementById('cal-rsi-ob').textContent       = d.rsi_overbought != null ? d.rsi_overbought : '—';

    const baseOs = d.base_rsi_os;
    const baseOb = d.base_rsi_ob;
    const adjOs  = d.rsi_oversold;
    const adjOb  = d.rsi_overbought;
    let note = '';
    if (adjOs !== baseOs || adjOb !== baseOb) {
      note = `Thresholds adjusted from defaults (OS: ${baseOs} → ${adjOs}, OB: ${baseOb} → ${adjOb}) due to ${d.regime} VIX environment.`;
    } else {
      note = `Thresholds at defaults (OS: ${baseOs}, OB: ${baseOb}) — no VIX adjustment needed.`;
    }
    if (d.cal_active) note += '  •  🎯 Calibration weights active — signal scores adjusted.';
    if (d.cal_date)   note += `  Last calibrated: ${d.cal_date}.`;
    document.getElementById('cal-threshold-note').textContent = note;
  } catch(e) {
    console.error('loadCalRegime:', e);
  }
}

async function runCalibration() {
  const months   = parseInt(document.getElementById('cal-months').value) || 12;
  const universe = document.getElementById('cal-universe').value;
  const btn      = document.getElementById('cal-run-btn');
  btn.disabled = true; btn.textContent = '⏳ Running…';
  document.getElementById('cal-progress-wrap').style.display = 'block';
  document.getElementById('cal-status-text').textContent = 'Starting calibration…';
  document.getElementById('cal-results-wrap').style.display = 'none';

  const r = await fetch('/api/calibrate/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({months, universe})
  });
  if (!r.ok) {
    const err = await r.json();
    alert(err.message || 'Calibration error');
    btn.disabled = false; btn.textContent = '🔬 Run Calibration';
    return;
  }

  if (calPoll) clearInterval(calPoll);
  calPoll = setInterval(async () => {
    try {
      const d = await fetch('/api/calibrate/status').then(r => r.json());
      if (d.running) {
        const pct = d.total ? Math.round(d.progress / d.total * 100) : 0;
        document.getElementById('cal-progress-bar').style.width = pct + '%';
        document.getElementById('cal-status-text').textContent =
          `${d.stage} — ${d.progress}/${d.total} tickers (${pct}%)`;
      } else {
        clearInterval(calPoll);
        btn.disabled = false; btn.textContent = '🔬 Run Calibration';
        document.getElementById('cal-progress-bar').style.width = '100%';
        document.getElementById('cal-status-text').textContent = d.stage || 'Complete';
        if (d.completed_at) {
          document.getElementById('cal-last-run').textContent =
            `Last run: ${new Date(d.completed_at).toLocaleString()} · ${d.tickers_used} tickers · ${d.total_trades} trades`;
        }
        await loadCalResults();
        loadCalRegime();
        setTimeout(() => {
          document.getElementById('cal-progress-wrap').style.display = 'none';
        }, 2000);
      }
    } catch(e) {}
  }, 2000);
}

async function runLogisticCalibration() {
  const btn      = document.getElementById('cal-lr-btn');
  const source   = document.getElementById('lr-source')?.value   || 'signal_log';
  const months   = parseInt(document.getElementById('lr-months')?.value  || '12');
  const universe = document.getElementById('lr-universe')?.value || 'snp500';
  btn.disabled = true; btn.textContent = '⏳ Starting…';

  try {
    const r = await fetch('/api/calibrate/logistic', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source, months, universe}),
    });
    if (!r.ok) {
      const err = await r.json();
      alert(err.message || 'Smart calibration error');
      btn.disabled = false; btn.textContent = '🧠 Smart Calibrate';
      return;
    }
  } catch(e) {
    alert('Could not reach server'); btn.disabled = false; btn.textContent = '🧠 Smart Calibrate';
    return;
  }

  // Poll until done
  const poll = setInterval(async () => {
    try {
      const d = await fetch('/api/calibrate/logistic/results').then(r => r.json());
      if (d.running) {
        btn.textContent = '⏳ ' + (d.stage || 'Fitting…').slice(0, 30) + '…';
      } else {
        clearInterval(poll);
        btn.disabled = false; btn.textContent = '🧠 Smart Calibrate';
        if (d.results) renderLRResults(d.results);
      }
    } catch(e) {}
  }, 1500);
}

function renderLRResults(data) {
  const wrap = document.getElementById('cal-lr-wrap');
  if (!wrap) return;
  wrap.style.display = 'block';
  // The LR block lives inside the classic-calibration results wrapper, which
  // stayed hidden until a classic calibration had run — make it visible.
  const outer = document.getElementById('cal-results-wrap');
  if (outer) outer.style.display = 'block';

  const meta = document.getElementById('cal-lr-meta');
  if (meta) {
    const src = data.source === 'backtest'
      ? `${data.months}m ${data.universe} backtest`
      : 'live signal log';
    meta.textContent =
      `${data.total_entries || 0} samples · ${src} · run ${(data.completed_at||'').slice(0,10)}`;
  }

  const STRAT_LABELS = {
    buy_call: '📈 Buy Call', sell_put: '💰 Sell Put',
    buy_put:  '📉 Buy Put',  covered_call: '🛡 Covered Call',
  };

  // Weight bar color: green >1.2, amber ~1.0, red <0.8
  const WC = w => w >= 1.3 ? 'var(--green)' : w >= 1.0 ? '#22c55e' :
                   w >= 0.8 ? '#f59e0b'      : w >= 0.6 ? '#f97316' : 'var(--red)';
  // Bar width: map [0.3 … 2.0] → [15% … 100%]
  const WW = w => Math.round(Math.max(15, Math.min(100, (w - 0.3) / 1.7 * 100)));

  const rows = document.getElementById('cal-lr-rows');
  if (!rows) return;

  const strats = data.strategies || {};
  let html = '';

  for (const [strat, res] of Object.entries(strats)) {
    const label = STRAT_LABELS[strat] || strat;
    html += `<div style="margin-bottom:18px">
      <div style="font-size:12px;font-weight:700;color:var(--text);margin-bottom:6px;
                  display:flex;align-items:center;gap:10px">
        ${label}`;

    if (res.status === 'ok') {
      const splitNote = res.split_method && res.split_method.includes('median')
        ? ` · <span style="color:var(--accent)" title="All trades had the same outcome so the model learned what separates above-median from below-median P&L trades">median split</span>`
        : '';
      html += `<span style="font-size:11px;font-weight:400;color:var(--muted)">
        ${res.count} entries · ${res.win_rate}% win rate · model accuracy ${res.accuracy}%${splitNote}
      </span>`;
    } else {
      const msg = res.message || res.status || 'unavailable';
      html += `<span style="font-size:11px;color:var(--muted);font-style:italic">${msg}</span>`;
    }
    html += `</div>`;

    if (res.status === 'ok' && res.coef_details) {
      // Sort by weight descending for readability
      const details = [...res.coef_details].sort((a,b) => b.weight - a.weight);
      for (const d of details) {
        const col = WC(d.weight);
        const ww  = WW(d.weight);
        const dir = d.coef >= 0 ? '▲' : '▼';
        html += `<div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;font-size:12px">
          <div style="width:150px;color:var(--text);flex-shrink:0">${d.label}</div>
          <div style="flex:1;background:var(--surface);border-radius:4px;height:16px;overflow:hidden;min-width:60px;position:relative">
            <div style="width:${ww}%;height:100%;background:${col};transition:width .5s;border-radius:4px"></div>
            <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--muted);opacity:.3"></div>
          </div>
          <div style="width:44px;text-align:right;font-weight:700;color:${col}">${d.weight.toFixed(2)}×</div>
          <div style="width:56px;text-align:right;color:var(--muted);font-size:11px">${dir} ${Math.abs(d.odds_ratio).toFixed(2)} OR</div>
        </div>`;
      }
    }
    html += `</div>`;
  }

  if (!html) html = '<div style="color:var(--muted);font-size:12px">No strategy results available.</div>';
  rows.innerHTML = html;

  // Show status note
  const statusEl = document.getElementById('cal-lr-status');
  if (statusEl) {
    const fitted = Object.values(strats).filter(r => r.status === 'ok').length;
    const waiting = Object.values(strats).filter(r => r.status === 'insufficient_data').length;
    if (waiting > 0) {
      statusEl.innerHTML = `<span style="color:var(--accent)">⚠ ${waiting} strategy/strategies need more resolved entries (min 20).
        Keep running scans — entries resolve automatically after 7–21 days.</span>`;
    } else {
      statusEl.innerHTML = '';
    }
  }
}

async function loadCalResults() {
  try {
    const d = await fetch('/api/calibrate/results').then(r => r.json());
    if (!d || !d.sim_quality) return;
    renderCalResults(d);
  } catch(e) {}
  // Also load live signal log stats (updates independently of calibration)
  loadSignalLogStats();
  // Load LR calibration results if available
  try {
    const lr = await fetch('/api/calibrate/logistic/results').then(r => r.json());
    if (lr.results) renderLRResults(lr.results);
  } catch(e) {}
}

async function loadSignalLogStats() {
  try {
    const [d, entries] = await Promise.all([
      fetch('/api/signal-log/summary').then(r => r.json()),
      fetch('/api/signal-log/entries').then(r => r.json()),
    ]);

    // ── Meta line ──────────────────────────────────────────────────────
    const meta = document.getElementById('cal-siglog-meta');
    if (meta) {
      const res = d.resolved || 0;
      const pen = d.pending  || 0;
      meta.textContent = `${res} resolved · ${pen} pending · bot logs every scan automatically`;
    }
    if ((d.resolved || 0) + (d.pending || 0) > 0) {
      const outer = document.getElementById('cal-results-wrap');   // signal-log block lives inside it
      if (outer) outer.style.display = 'block';
    }

    // ── Win-rate bars ──────────────────────────────────────────────────
    const rows = document.getElementById('cal-siglog-rows');
    if (rows) {
      const WR_COLOR = wr => wr >= 70 ? 'var(--green)' : wr >= 60 ? '#22c55e' : wr >= 50 ? '#f59e0b' : wr >= 40 ? '#f97316' : 'var(--red)';
      const sigQ = d.sig_quality || [];
      if (sigQ.length === 0 && (d.resolved || 0) < 3) {
        rows.innerHTML = `<div style="color:var(--muted);font-size:12px">
          Run a few scans and wait for the first outcomes to resolve.
          The log grows automatically after each scan.
        </div>`;
      } else {
        rows.innerHTML = sigQ.map(r => {
          const col = WR_COLOR(r.win_rate);
          return `<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:12px">
            <div style="width:150px;color:var(--text);flex-shrink:0">${r.signal}</div>
            <div style="flex:1;background:var(--surface);border-radius:4px;height:16px;overflow:hidden;min-width:60px">
              <div style="width:${Math.round(r.win_rate)}%;height:100%;background:${col};border-radius:4px"></div>
            </div>
            <div style="width:44px;text-align:right;font-weight:700;color:${col}">${r.win_rate}%</div>
            <div style="width:65px;text-align:right;color:var(--muted)">${r.count} live recs</div>
          </div>`;
        }).join('') || '<div style="color:var(--muted);font-size:12px">Not enough resolved entries yet.</div>';
      }
    }

    // ── Signal log table ───────────────────────────────────────────────
    window._sigLogEntries = entries.entries || [];  // cache for re-render
    rerenderSigLog();
  } catch(e) { console.error('loadSignalLogStats error', e); }
}

// Cached entries so row-count changes don't need a re-fetch
window._sigLogEntries = [];
window._sigLogCollapsed = false;

function toggleSigLog() {
  window._sigLogCollapsed = !window._sigLogCollapsed;
  const wrap = document.getElementById('siglog-body-wrap');
  const chev = document.getElementById('siglog-chevron');
  if (wrap) wrap.style.display = window._sigLogCollapsed ? 'none' : 'block';
  if (chev) chev.textContent = window._sigLogCollapsed ? '▶' : '▼';
}

function rerenderSigLog() {
  const tbody = document.getElementById('cal-siglog-tbody');
  if (!tbody) return;
  const list = window._sigLogEntries || [];
  if (list.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="padding:20px;text-align:center;color:var(--muted)">
      Run a scan to start logging recommendations
    </td></tr>`;
    const tally = document.getElementById('siglog-tally');
    if (tally) tally.style.display = 'none';
    return;
  }

  // ── Per-strategy W/L summary cards (all resolved entries) ──
  const STRAT_META = {
    buy_call:     { label: '📈 Buy Call',      short: 'Buy Call' },
    buy_put:      { label: '📉 Buy Put',       short: 'Buy Put' },
    sell_put:     { label: '💰 Sell Put',      short: 'Sell Put' },
    covered_call: { label: '🛡 Covered Call',  short: 'Cvrd Call' },
    iron_condor:  { label: '🦅 Iron Condor',   short: 'Condor' },
  };
  const resolved = list.filter(e => e.outcome && e.outcome !== 'pending' && e.outcome !== 'no_price');
  const summaryWrap = document.getElementById('siglog-strat-summary');
  const cardsEl     = document.getElementById('siglog-strat-cards');
  if (resolved.length > 0 && cardsEl) {
    // Group by strategy
    const byStrat = {};
    for (const e of resolved) {
      const s = e.strategy || 'unknown';
      if (!byStrat[s]) byStrat[s] = { wins: 0, losses: 0, returns: [] };
      if (e.outcome === 'win') byStrat[s].wins++;
      else byStrat[s].losses++;
      const ret = e.stock_return_pct !== undefined && e.stock_return_pct !== null
                  ? parseFloat(e.stock_return_pct) : null;
      if (ret !== null) byStrat[s].returns.push(ret);
    }
    const stratOrder = ['buy_call','buy_put','sell_put','covered_call','iron_condor'];
    cardsEl.innerHTML = stratOrder
      .filter(s => byStrat[s])
      .map(s => {
        const d      = byStrat[s];
        const total  = d.wins + d.losses;
        const pct    = Math.round(d.wins / total * 100);
        const avgRet = d.returns.length
                       ? (d.returns.reduce((a,b)=>a+b,0) / d.returns.length).toFixed(1)
                       : null;
        const pctCol = pct >= 60 ? 'var(--green)' : pct >= 45 ? '#f59e0b' : 'var(--red)';
        const retStr = avgRet !== null
                       ? `<div style="font-size:11px;color:${parseFloat(avgRet)>=0?'var(--green)':'var(--red)'};
                              margin-top:3px;font-weight:600">
                            avg ${parseFloat(avgRet)>=0?'+':''}${avgRet}% stock
                          </div>`
                       : '';
        const meta   = STRAT_META[s] || { label: s, short: s };
        return `<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;
                     padding:10px 14px;min-width:120px;flex:1">
          <div style="font-size:11px;color:var(--muted);margin-bottom:6px">${meta.label}</div>
          <div style="display:flex;align-items:baseline;gap:6px">
            <span style="font-size:16px;font-weight:700;color:${pctCol}">${pct}%</span>
            <span style="font-size:11px;color:var(--muted)">${total} trades</span>
          </div>
          <div style="font-size:11px;color:var(--muted);margin-top:3px">
            <span style="color:var(--green)">${d.wins}W</span>
            <span style="color:var(--border)"> / </span>
            <span style="color:var(--red)">${d.losses}L</span>
          </div>
          ${retStr}
        </div>`;
      }).join('');
    summaryWrap.style.display = 'block';
  } else if (summaryWrap) {
    summaryWrap.style.display = 'none';
  }

  const limitSel = document.getElementById('siglog-rows-select');
  const limit    = limitSel ? parseInt(limitSel.value) : 25;
  const shown    = limit === 0 ? list : list.slice(0, limit);

  const STRAT_LABEL = {
    buy_call: '📈 Buy Call', buy_put: '📉 Buy Put',
    sell_put: '💰 Sell Put', covered_call: '🛡 Covered Call',
    iron_condor: '🦅 Iron Condor',
  };
  const REGIME_COLOR = { high: '#ef4444', normal: '#f59e0b', low: '#22c55e' };

  tbody.innerHTML = shown.map(e => {
    const strat  = STRAT_LABEL[e.strategy] || e.strategy;
    const regime = e.vix_regime || 'normal';
    const regCol = REGIME_COLOR[regime] || 'var(--muted)';
    const regLbl = regime.charAt(0).toUpperCase() + regime.slice(1);

    let retCell     = `<td style="padding:5px 8px;text-align:right;color:var(--muted)">—</td>`;
    let outcomeCell = `<td style="padding:5px 8px"><span style="color:var(--muted)">⏳ Pending</span></td>`;

    if (e.outcome && e.outcome !== 'pending') {
      // ── Final resolved outcome ──
      const ret   = e.stock_return_pct !== null && e.stock_return_pct !== undefined
                     ? parseFloat(e.stock_return_pct).toFixed(1)
                     : (e.stock_return !== null && e.stock_return !== undefined
                        ? parseFloat(e.stock_return).toFixed(1) : null);
      const isWin = e.outcome === 'win';
      retCell = `<td style="padding:5px 8px;text-align:right;font-weight:700;
                    color:${isWin ? 'var(--green)' : 'var(--red)'}">
                  ${ret !== null ? (ret >= 0 ? '+' : '') + ret + '%' : '—'}
                 </td>`;
      outcomeCell = `<td style="padding:5px 8px">
                      <span style="color:${isWin ? 'var(--green)' : 'var(--red)'}">
                        ${isWin ? '🟢 Win' : '🔴 Loss'}
                      </span>
                     </td>`;
    } else if (e.current_return_pct !== null && e.current_return_pct !== undefined) {
      // ── Live unrealized P&L (hold period not yet complete) ──
      const liveRet = parseFloat(e.current_return_pct).toFixed(1);
      const isUp    = e.current_return_pct >= 0;
      // For buy_put the direction is inverted — stock down is good
      const strat_key = e.strategy || '';
      const isGood  = strat_key === 'buy_put' ? !isUp
                    : strat_key === 'sell_put' || strat_key === 'covered_call' ? e.current_return_pct > -5
                    : isUp;
      const liveCol = isGood ? 'var(--green)' : 'var(--red)';
      retCell = `<td style="padding:5px 8px;text-align:right;font-weight:700;color:${liveCol}">
                  ${isUp ? '+' : ''}${liveRet}%
                 </td>`;
      outcomeCell = `<td style="padding:5px 8px">
                      <span style="color:var(--muted);font-size:11px">📊 Live</span>
                     </td>`;
    }

    const signalDate = (e.signal_date || '').slice(0, 10);
    const checkAfter = (e.check_after || '').slice(0, 10);
    const score      = e.score !== undefined ? Math.round(e.score) : '—';

    return `<tr style="border-bottom:1px solid var(--border)">
      <td style="padding:5px 8px;color:var(--muted);white-space:nowrap">${signalDate}</td>
      <td style="padding:5px 8px;font-weight:700;color:var(--text)">${e.ticker || '—'}</td>
      <td style="padding:5px 8px;white-space:nowrap">${strat}</td>
      <td style="padding:5px 8px;text-align:right;font-weight:700;color:var(--accent)">${score}</td>
      <td style="padding:5px 8px;text-align:center">
        <span style="color:${regCol};font-size:11px;font-weight:600;
               background:${regCol}22;border-radius:4px;padding:2px 6px">${regLbl}</span>
      </td>
      <td style="padding:5px 8px;color:var(--muted);white-space:nowrap">${checkAfter}</td>
      ${retCell}
      ${outcomeCell}
    </tr>`;
  }).join('');

  // Append a "showing X of Y" footer row if limited
  if (limit > 0 && list.length > limit) {
    tbody.innerHTML += `<tr><td colspan="8" style="padding:8px;text-align:center;
      color:var(--muted);font-size:11px;font-style:italic">
      Showing ${limit} of ${list.length} entries — change the row selector above to see more
    </td></tr>`;
  }
}

function renderCalResults(d) {
  document.getElementById('cal-results-wrap').style.display = 'block';
  document.getElementById('cal-months-label').textContent = d.months || 12;

  const meta = `· ${d.tickers_used} tickers · ${d.total_trades} trades · run ${(d.completed_at||'').slice(0,10)}`;
  document.getElementById('cal-sim-meta').textContent = meta;

  // Signal win-rate bars
  const WR_COLOR = wr => wr >= 70 ? 'var(--green)' : wr >= 60 ? '#22c55e' : wr >= 50 ? '#f59e0b' : wr >= 40 ? '#f97316' : 'var(--red)';
  const WEIGHT_LABEL = wr => wr >= 70 ? '▲ +30%' : wr >= 60 ? '▲ +10%' : wr >= 50 ? '→ neutral' : wr >= 40 ? '▼ −25%' : '▼ −50%';

  const sigRows = (d.sim_quality || []).map(r => {
    const bar  = Math.round(r.win_rate);
    const col  = WR_COLOR(r.win_rate);
    const wlbl = WEIGHT_LABEL(r.win_rate);
    return `<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;font-size:12px">
      <div style="width:150px;color:var(--text);flex-shrink:0;font-weight:500">${r.signal}</div>
      <div style="flex:1;background:var(--surface);border-radius:4px;height:18px;overflow:hidden;min-width:80px">
        <div style="width:${bar}%;height:100%;background:${col};transition:width .5s;border-radius:4px"></div>
      </div>
      <div style="width:44px;text-align:right;font-weight:700;color:${col}">${r.win_rate}%</div>
      <div style="width:70px;text-align:right;color:var(--muted)">${r.count} trades</div>
      <div style="width:60px;text-align:right;color:${r.avg_pnl>0?'var(--green)':'var(--red)'};font-weight:600">
        ${r.avg_pnl>0?'+':''}${r.avg_pnl}%
      </div>
      <div style="width:72px;text-align:right;font-size:11px;
           color:${col};background:${col}22;border-radius:4px;padding:2px 5px">${wlbl}</div>
    </div>`;
  }).join('');
  document.getElementById('cal-signal-rows').innerHTML = sigRows ||
    '<div style="color:var(--muted);font-size:13px">Not enough signal data yet.</div>';

  // ── Simulated Strategy Performance — overall stat cards ─────────────
  const ss  = d.sim_summary || {};
  // Safe formatters — old calibration files predate the newer metrics,
  // so any missing field renders as "—" instead of the string "undefined".
  const has = v => v !== undefined && v !== null && !(typeof v === 'number' && Number.isNaN(v));
  const pct = v => has(v) ? (v > 0 ? '+' : '') + v + '%' : '—';   // signed %
  const npc = v => has(v) ? v + '%' : '—';                         // unsigned %
  const raw = v => has(v) ? v : '—';                               // bare number
  // $-formatter: each percentage point of pnl == $1 on a $100 bet,
  // so total_pnl (sum of pct returns) directly converts to dollars.
  const dollar = v => has(v) ? (v >= 0 ? '+' : '−') + '$' + Math.abs(Math.round(v)).toLocaleString() : '—';
  const pos = v => v > 0 ? 'var(--green)' : v < 0 ? 'var(--red)' : 'var(--muted)';
  const nLabel = document.getElementById('cal-sim-n');
  if (nLabel) nLabel.textContent = (ss.total_trades || 0) + ' ';
  const statCard = (label, value, sub, col) =>
    `<div style="background:var(--surface);border-radius:10px;padding:12px 14px;border:1px solid var(--border)">
       <div style="font-size:10px;font-weight:600;color:var(--muted);letter-spacing:.4px;margin-bottom:6px">${label}</div>
       <div style="font-size:20px;font-weight:800;color:${col||'var(--text)'}">${value}</div>
       <div style="font-size:10px;color:var(--muted);margin-top:2px">${sub||''}</div>
     </div>`;
  const overallEl = document.getElementById('cal-sim-overall');
  if (overallEl) {
    if (ss.total_trades) {
      const sExp = ss.expectancy ?? ss.avg_pnl;   // expectancy == mean P&L%; fall back for old files
      overallEl.innerHTML = [
        statCard('P&L @ $100/TRADE', dollar(ss.total_pnl), 'if you bet $100 on every signal', pos(ss.total_pnl)),
        statCard('WIN RATE',      npc(ss.win_rate),    (ss.total_trades||0)+' trades',
                 ss.win_rate>=55?'var(--green)':ss.win_rate>=45?'#f59e0b':'var(--red)'),
        statCard('EXPECTANCY',    pct(sExp),           'avg per trade · +ve = edge', pos(sExp)),
        statCard('AVG WIN',       pct(ss.avg_win),     '', 'var(--green)'),
        statCard('AVG LOSS',      npc(ss.avg_loss),    '', 'var(--red)'),
        statCard('PROFIT FACTOR', raw(ss.profit_factor),   'target > 1.5',
                 ss.profit_factor>=1.5?'var(--green)':ss.profit_factor>=1?'#f59e0b':'var(--red)'),
        statCard('SORTINO RATIO', raw(ss.sortino),     'target > 0.5',
                 ss.sortino>=0.5?'var(--green)':ss.sortino>=0?'#f59e0b':'var(--red)'),
        statCard('SHARPE RATIO',  raw(ss.sharpe),      'mean / stdev',
                 ss.sharpe>=0.5?'var(--green)':ss.sharpe>=0?'#f59e0b':'var(--red)'),
        statCard('MAX DRAWDOWN',
                 has(ss.max_drawdown_pct) ? ss.max_drawdown_pct + '% off peak' : npc(ss.max_drawdown),
                 has(ss.max_drawdown_pct) ? 'worst dip from running peak' : 'deepest equity dip', 'var(--red)'),
        statCard('RECOVERY FACTOR',
                 has(ss.recovery_factor) ? ss.recovery_factor + '×' : '—',
                 'total gain / worst dip · >3 healthy',
                 ss.recovery_factor>=3?'var(--green)':ss.recovery_factor>=1.5?'#f59e0b':'var(--red)'),
        statCard('BEST TRADE',    pct(ss.best),        '', 'var(--green)'),
        statCard('WORST TRADE',   npc(ss.worst),       '', 'var(--red)'),
      ].join('');
    } else {
      overallEl.innerHTML =
        '<div style="color:var(--muted);font-size:13px">No simulated trades yet — run a calibration.</div>';
    }
  }

  // Per-strategy simulated summary
  const byStrat = ss.by_strategy || {};
  const stratLabels = {buy_call:'📈 Buy Calls', buy_put:'📉 Buy Puts', sell_put:'💰 Sell Puts'};
  const miniRow = (lbl, val, c) =>
    `<div style="display:flex;justify-content:space-between;font-size:11px;margin-top:3px">
       <span style="color:var(--muted)">${lbl}</span>
       <span style="font-weight:700;color:${c||'var(--text)'}">${val}</span></div>`;
  document.getElementById('cal-strat-rows').innerHTML =
    Object.entries(byStrat).map(([s, v]) => {
      const col = v.win_rate >= 55 ? 'var(--green)' : v.win_rate >= 45 ? '#f59e0b' : 'var(--red)';
      return `<div style="background:var(--surface);border-radius:10px;padding:14px;border:1px solid var(--border)">
        <div style="font-size:12px;font-weight:700;color:var(--text);margin-bottom:8px">${stratLabels[s]||s}</div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:4px">${v.count} trades</div>
        <div style="font-size:18px;font-weight:800;color:${col}">${npc(v.win_rate)}</div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px">win rate</div>
        ${miniRow('Expectancy',   pct(v.expectancy ?? v.avg_pnl), pos(v.expectancy ?? v.avg_pnl))}
        ${miniRow('P&L @ $100/tr', dollar(v.total_pnl), pos(v.total_pnl))}
        ${miniRow('Profit factor',raw(v.profit_factor),
                  v.profit_factor>=1.5?'var(--green)':v.profit_factor>=1?'#f59e0b':'var(--red)')}
        ${miniRow('Sortino',      raw(v.sortino),
                  v.sortino>=0.5?'var(--green)':v.sortino>=0?'#f59e0b':'var(--red)')}
        ${miniRow('Max drawdown', has(v.max_drawdown_pct) ? v.max_drawdown_pct + '% off peak' : npc(v.max_drawdown), 'var(--red)')}
        ${miniRow('Recovery', has(v.recovery_factor) ? v.recovery_factor + '×' : '—',
                  v.recovery_factor>=3?'var(--green)':v.recovery_factor>=1.5?'#f59e0b':'var(--red)')}
        ${miniRow('Avg win / loss', pct(v.avg_win)+' / '+npc(v.avg_loss))}
      </div>`;
    }).join('') || '<div style="color:var(--muted);font-size:13px">No strategy data.</div>';

  // Actual user trades
  const actual = d.actual_trades || {};
  if (Object.keys(actual).length > 0) {
    document.getElementById('cal-actual-wrap').style.display = 'block';
    document.getElementById('cal-actual-rows').innerHTML =
      Object.entries(actual).map(([s, v]) => {
        const col = v.win_rate >= 55 ? 'var(--green)' : v.win_rate >= 45 ? '#f59e0b' : 'var(--red)';
        return `<div style="background:var(--surface);border-radius:10px;padding:14px;border:1px solid var(--border)">
          <div style="font-size:12px;font-weight:700;color:var(--text);margin-bottom:8px">${stratLabels[s]||s}</div>
          <div style="font-size:11px;color:var(--muted);margin-bottom:4px">${v.count} real trades</div>
          <div style="font-size:18px;font-weight:800;color:${col}">${v.win_rate}%</div>
          <div style="font-size:11px;color:var(--muted)">actual win rate</div>
          <div style="font-size:13px;font-weight:700;margin-top:4px;
               color:${v.avg_pnl>0?'var(--green)':'var(--red)'}">${v.avg_pnl>0?'+':''}${v.avg_pnl}% avg P&L</div>
        </div>`;
      }).join('');
  }
}

// ── Hold Horizon Analysis ──────────────────────────────────────────────────

let _hsChartInstance = null;
let _hsPollTimer     = null;

async function runHoldSweep() {
  const universe = document.getElementById('hs-universe').value;
  const months   = parseInt(document.getElementById('hs-months').value);
  document.getElementById('hs-run-btn').disabled = true;
  document.getElementById('hs-progress-wrap').style.display = 'block';
  document.getElementById('hs-results-wrap').style.display  = 'none';
  document.getElementById('hs-status-text').textContent = 'Starting…';

  try {
    const r = await fetch('/api/calibrate/hold-sweep', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({universe, months}),
    });
    if (!r.ok) {
      const e = await r.json();
      alert('Hold sweep error: ' + (e.error || r.status));
      document.getElementById('hs-run-btn').disabled = false;
      return;
    }
    pollHoldSweep();
  } catch(e) {
    alert('Hold sweep failed: ' + e);
    document.getElementById('hs-run-btn').disabled = false;
  }
}

function pollHoldSweep() {
  if (_hsPollTimer) clearInterval(_hsPollTimer);
  _hsPollTimer = setInterval(async () => {
    try {
      const d = await fetch('/api/calibrate/hold-sweep/results').then(r => r.json());
      document.getElementById('hs-status-text').textContent = d.stage || '…';
      if (!d.running) {
        clearInterval(_hsPollTimer);
        _hsPollTimer = null;
        document.getElementById('hs-run-btn').disabled = false;
        document.getElementById('hs-progress-wrap').style.display = 'none';
        if (d.results) renderHoldSweepResults(d.results);
      }
    } catch(e) { /* ignore transient errors */ }
  }, 1500);
}

function renderHoldSweepResults(r) {
  document.getElementById('hs-results-wrap').style.display = 'block';

  const ts  = r.completed_at ? new Date(r.completed_at).toLocaleString() : '';
  const uni = r.universe === 'snp500' ? 'S&P 500' : 'Focus List';
  document.getElementById('hs-meta').textContent =
    `${r.n_trades?.toLocaleString() || '?'} trades · ${r.months}m · ${uni} · ${ts}`;

  const INTERVALS = r.intervals || [1,2,3,5,7,10,14,21];
  const LABELS    = INTERVALS.map(d => d + 'd');

  const COLORS = {
    buy_call: { border:'#4ade80', bg:'rgba(74,222,128,0.15)' },
    buy_put:  { border:'#f87171', bg:'rgba(248,113,113,0.15)' },
    sell_put: { border:'#60a5fa', bg:'rgba(96,165,250,0.15)' },
  };
  const NAMES = { buy_call: 'Buy Call', buy_put: 'Buy Put', sell_put: 'Sell Put' };

  const datasets = [];
  const strats   = ['buy_call', 'buy_put', 'sell_put'];

  for (const strat of strats) {
    const sd = r.by_strategy?.[strat];
    if (!sd) continue;
    const data = INTERVALS.map(d => sd.by_day?.[String(d)]?.avg_pnl ?? null);
    datasets.push({
      label: NAMES[strat],
      data,
      borderColor:      COLORS[strat].border,
      backgroundColor:  COLORS[strat].bg,
      borderWidth: 2.5,
      pointRadius: INTERVALS.map((d, i) => d === sd.optimal_day ? 7 : 4),
      pointBackgroundColor: INTERVALS.map((d, i) =>
        d === sd.optimal_day ? COLORS[strat].border : 'transparent'),
      tension: 0.35,
      fill: false,
      spanGaps: true,
    });
  }

  // Chart
  const ctx = document.getElementById('hs-chart').getContext('2d');
  if (_hsChartInstance) { _hsChartInstance.destroy(); _hsChartInstance = null; }
  _hsChartInstance = new Chart(ctx, {
    type: 'line',
    data: { labels: LABELS, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: getComputedStyle(document.body).getPropertyValue('--text') || '#e2e8f0' }
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y != null ? ctx.parsed.y.toFixed(1) + '%' : 'n/a'}`
          }
        },
      },
      scales: {
        x: {
          ticks: { color: '#94a3b8' },
          grid:  { color: 'rgba(148,163,184,0.12)' },
          title: { display: true, text: 'Hold Duration (trading days)', color: '#94a3b8' },
        },
        y: {
          ticks: { color: '#94a3b8', callback: v => v + '%' },
          grid:  { color: 'rgba(148,163,184,0.12)' },
          title: { display: true, text: 'Avg P&L %', color: '#94a3b8' },
        },
      },
    }
  });

  // P&L table
  let pnlRows = '';
  for (const strat of strats) {
    const sd = r.by_strategy?.[strat];
    if (!sd) continue;
    const cells = INTERVALS.map(d => {
      const dd = sd.by_day?.[String(d)];
      if (!dd) return '<td style="text-align:center;padding:5px 8px;color:var(--muted)">—</td>';
      const isOpt = d === sd.optimal_day;
      const color = dd.avg_pnl > 0 ? 'var(--green)' : dd.avg_pnl < 0 ? 'var(--red)' : 'var(--muted)';
      const bg    = isOpt ? 'background:rgba(99,102,241,0.18);border-radius:4px;' : '';
      return `<td style="text-align:center;padding:5px 8px;${bg}">
        <span style="color:${color};font-weight:${isOpt?'800':'400'}">${dd.avg_pnl > 0 ? '+' : ''}${dd.avg_pnl}%</span>
      </td>`;
    }).join('');
    const optLabel = `${sd.optimal_day}d`;
    // Per-strategy trade count — small N is a sample-size warning sign
    const nTrades = sd.n_trades || 0;
    const nColor  = nTrades < 30 ? 'var(--red)' : nTrades < 100 ? 'var(--yellow)' : 'var(--muted)';
    const nLabel  = `<span style="color:${nColor};font-weight:600" title="${nTrades < 30 ? 'Small sample — averages may not generalise' : ''}">${nTrades.toLocaleString()}</span>`;
    pnlRows += `<tr style="border-bottom:1px solid var(--border)">
      <td style="padding:5px 8px;font-weight:600;color:var(--text)">${NAMES[strat]}</td>
      <td style="text-align:center;padding:5px 8px;font-size:11px">${nLabel}</td>
      ${cells}
      <td style="text-align:center;padding:5px 8px;font-weight:800;color:var(--accent)">${optLabel}</td>
    </tr>`;
  }
  document.getElementById('hs-tbody').innerHTML = pnlRows;

  // Win-rate table
  let wrRows = '';
  for (const strat of strats) {
    const sd = r.by_strategy?.[strat];
    if (!sd) continue;
    const cells = INTERVALS.map(d => {
      const dd = sd.by_day?.[String(d)];
      if (!dd) return '<td style="text-align:center;padding:5px 8px;color:var(--muted)">—</td>';
      const wr  = dd.win_rate;
      const color = wr >= 55 ? 'var(--green)' : wr >= 45 ? 'var(--text)' : 'var(--red)';
      return `<td style="text-align:center;padding:5px 8px">
        <span style="color:${color}">${wr}%</span>
      </td>`;
    }).join('');
    const nTrades = sd.n_trades || 0;
    const nColor  = nTrades < 30 ? 'var(--red)' : nTrades < 100 ? 'var(--yellow)' : 'var(--muted)';
    const nLabel  = `<span style="color:${nColor};font-weight:600">${nTrades.toLocaleString()}</span>`;
    wrRows += `<tr style="border-bottom:1px solid var(--border)">
      <td style="padding:5px 8px;font-weight:600;color:var(--text)">${NAMES[strat]}</td>
      <td style="text-align:center;padding:5px 8px;font-size:11px">${nLabel}</td>
      ${cells}
    </tr>`;
  }
  document.getElementById('hs-wr-tbody').innerHTML = wrRows;
}
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

HEADLESS = ("--headless" in sys.argv) or os.environ.get("OPTIONS_BOT_HEADLESS") == "1"


class _TeeStream:
    """stdout/stderr → the console AND logs\\bot_<date>.log (a new file each day; never raises)."""
    def __init__(self, stream):
        self.stream, self._day, self._fh, self._lock = stream, None, None, threading.RLock()   # re-entrant: a warning raised mid-write must not deadlock

    def _file(self):
        day = date.today().isoformat()
        if day != self._day:
            try:
                os.makedirs("logs", exist_ok=True)
                if self._fh:
                    self._fh.close()
                self._fh = open(os.path.join("logs", f"bot_{day}.log"), "a", encoding="utf-8", errors="replace", buffering=1)
                self._day = day
            except Exception:
                self._fh = None
        return self._fh

    def write(self, s):
        try:
            self.stream.write(s)
        except Exception:
            pass
        with self._lock:
            fh = self._file()
            if fh:
                try:
                    fh.write(s)
                except Exception:
                    pass
        return len(s)

    def flush(self):
        for f in (self.stream, self._fh):
            try:
                f.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self.stream, name)


def main():
    # Refuse to start twice: the second copy would fail to bind the port and the
    # headless launcher would restart it forever.  Exit code 3 tells it to stop.
    with socket.socket() as _s:
        if _s.connect_ex(("127.0.0.1", SERVER_PORT)) == 0:
            print(f"  Another Options Bot is already listening on port {SERVER_PORT} — exiting.")
            sys.exit(3)
    sys.stdout, sys.stderr = _TeeStream(sys.stdout), _TeeStream(sys.stderr)
    print("\n" + "═" * 56)
    print("  OPTIONS BOT v2.0" + ("  (headless)" if HEADLESS else ""))
    print("═" * 56)
    print(f"  Guardrail   : ${GUARDRAIL_LIMIT:,.0f} per trade")
    print(f"  Auto-submit : {'ENABLED ⚠' if AUTO_SUBMIT else 'DISABLED (safe mode)'}")
    print(f"  Dashboard   : http://localhost:{SERVER_PORT}")
    print("═" * 56 + "\n")
    print("  Starting web server…")

    # Cached GJR-GARCH fits (every backtest / optimizer job refits from scratch without them)
    _load_garch_cache()

    # Restore last WFO run from disk (auto-applies best params if found)
    _load_wfo_results()

    # Restore last backtest results from disk
    _load_bt_results()

    # Restore last calibration weights from disk (adjusts live scores immediately)
    _load_calibration_on_startup()
    _load_lr_calibration_on_startup()   # overlay LR weights on top (more precise)

    # Restore Hold Horizon Analysis results from disk (used by tracked-position UI)
    _load_hs_results()

    # Restore prefetch cache (info/calendar 24h TTL — saves ~50% of repeat scan calls)
    _load_prefetch_cache()

    # Autotrader loop (does nothing until enabled in the 🤖 Autotrade tab / autotrade_config.json)
    autotrader.init(sys.modules[__name__])
    autotrader.start()
    _at_cfg = autotrader.load_config()
    print(f"  Autotrade   : {'ENABLED — ' + _at_cfg['broker'] + ('' if _at_cfg['broker'] != 'alpaca' else (' paper' if _at_cfg['paper'] else ' LIVE ⚠')) if _at_cfg['enabled'] else 'disabled'}"
          + ("  (kill switch file present)" if autotrader.kill_switch_active() else ""))

    # Fetch live 6-month T-bill rate from FRED before first scan
    threading.Thread(target=fetch_risk_free_rate, daemon=True).start()

    # Start sell-signal tracker in the background
    threading.Thread(target=_schedule_track_checker, daemon=True).start()

    # Lightweight option price refresh every TRACK_PRICE_REFRESH_MINS minutes
    threading.Thread(target=_refresh_tracked_prices, daemon=True).start()

    # Resolve pending signal log entries (checks outcomes every 6 hours)
    threading.Thread(target=_schedule_signal_log_resolver, daemon=True).start()

    # Scheduled full re-scan every AUTO_RESCAN_HOURS (this thread was defined
    # but never started before, so the "next rescan" in the UI was always empty)
    threading.Thread(target=_auto_rescan_loop, daemon=True).start()

    # Open browser after short delay so Flask has time to start (not when unattended)
    if HEADLESS:
        print("  Headless: not opening a browser — the dashboard is still at the URL above")
    else:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{SERVER_PORT}")).start()

    # Bind to localhost only: the dashboard has no authentication and can
    # change config / queue trades, so it must not be reachable from the LAN.
    app.run(host="127.0.0.1", port=SERVER_PORT, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
