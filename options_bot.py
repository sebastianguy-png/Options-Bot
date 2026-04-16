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
  • Bot NEVER executes anything on its own
  • Every trade requires you to manually click "Add to Queue"
  • Every execution requires a final "Confirm & Place Order" in the dashboard
  • Trades over $1,000 require typing the word CONFIRM as a second factor
  • The SoFi browser window fills in your order and lands on the Review page;
    YOU click Submit — or enable AUTO_SUBMIT only if you're comfortable

Run:
    python options_bot.py
    → Opens http://localhost:5000 in your browser automatically
"""

# ──────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────
import os, sys, json, uuid, threading, webbrowser, time, warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading as _threading

import yfinance as yf
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Yahoo Finance rate-limit guard ────────────────────────────
# All yfinance calls that hit Yahoo's API (options chain, calendar,
# expiry lists) go through this semaphore so we never fire more
# than 2 concurrent requests regardless of thread count.
_YF_SEM = _threading.Semaphore(2)

def _yf_call(fn, *args, retries=4, base_delay=6, **kwargs):
    """
    Call any yfinance method with rate-limit retry + semaphore.
    Usage:  chain = _yf_call(tkr.option_chain, exp)
    """
    import time
    for attempt in range(retries + 1):
        with _YF_SEM:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e).lower()
                is_rate = any(x in msg for x in
                              ("too many requests", "rate limit", "429", "rate limited"))
                if is_rate and attempt < retries:
                    wait = base_delay * (2 ** attempt)
                    print(f"  [Rate limit] retrying in {wait}s (attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                else:
                    raise

# Flask — install with: pip install flask
try:
    from flask import Flask, jsonify, request, render_template_string
except ImportError:
    print("Flask not found. Run:  pip install flask")
    sys.exit(1)

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
MIN_DTE              = 14      # Min days to expiry
MAX_DTE              = 45      # Max days to expiry
MIN_OPEN_INTEREST    = 10      # Min OI on selected option
MAX_WORKERS          = 2       # Parallel scan threads (keep low to avoid rate limits)
SCAN_DELAY_S         = 0.15    # Seconds to sleep between ticker fetches (throttle)
SERVER_PORT          = 5000    # Web dashboard port
LIVE_REFRESH_MINS    = 5       # How often to re-check current signal tickers (minutes)

# ── Scan modes ────────────────────────────────────────────────
# "focus"  → ~80 high-liquidity names with active options markets
# "full"   → entire S&P 500 universe (~503 tickers, slower)
SCAN_MODE            = "focus"   # default; overridden by UI toggle

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
    "WMT","COST","TGT","HD","MCD","SBUX","NKE","AMZN","BKNG","CMG",
    # Communication / Media
    "NFLX","DIS","CMCSA","T","VZ","TMUS","SNAP","SPOT","RBLX","TTD",
    # Industrial / Defence
    "GE","HON","CAT","DE","BA","RTX","LMT","UPS","FDX",
    # EV / Clean energy
    "RIVN","NIO","PLUG","FCEL","BLNK","CHPT",
    # Fintech / Crypto
    "PYPL","SQ","COIN","HOOD","SOFI","AFRM","UPST",
    # ETFs (very liquid options)
    "SPY","QQQ","IWM","XLF","XLE","XLK","ARKK","GLD","SLV","TLT",
]

# ── Monte Carlo ───────────────────────────────────────────────
N_MC_SIMS            = 10_000  # Simulations per signal (higher = slower but more accurate)
RISK_FREE_RATE       = 0.05    # Annual risk-free rate used in GBM price model

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

# ── Tracked position sell-signal thresholds ───────────────────
# A sell signal fires when ANY of these conditions are met:
SELL_PROFIT_TARGET   = 40    # % gain on the option premium  → take profit
SELL_STOP_LOSS       = 35    # % loss on the option premium  → cut loss
# RSI reversal thresholds (original signal must be gone):
SELL_RSI_BULL_TARGET = 58    # Buy Call: RSI back above this → signal consumed
SELL_RSI_BEAR_TARGET = 42    # Buy Put:  RSI back below this → signal consumed
TRACK_CHECK_MINS     = 10    # How often (minutes) to re-scan tracked tickers

# ──────────────────────────────────────────────────────────────
#  FLASK APP
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)

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
    return 100 - 100 / (1 + rs)


def calc_bb(closes, period=20, nstd=2):
    sma = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    return sma + nstd * std, sma, sma - nstd * std


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
        from math import log, sqrt, exp
        from scipy.stats import norm

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


# ── VIX regime cache (refreshed every 30 min) ────────────────
_vix_cache = {"value": None, "ts": None}

def get_vix():
    """Return current VIX level, cached for 30 min."""
    import time as _time
    now = _time.time()
    if _vix_cache["value"] is not None and (now - _vix_cache["ts"]) < 1800:
        return _vix_cache["value"]
    try:
        v = yf.Ticker("^VIX").history(period="5d")
        if not v.empty:
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


def get_short_interest(tkr_obj):
    """
    Return (short_pct_float, short_ratio) from yfinance info.
    short_pct_float: e.g. 0.15 = 15% of float is short.
    short_ratio: days-to-cover.
    """
    try:
        info = tkr_obj.info
        pct  = info.get("shortPercentOfFloat")   # 0.0–1.0
        ratio = info.get("shortRatio")
        return (round(float(pct) * 100, 1) if pct else None,
                round(float(ratio), 1) if ratio else None)
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
        t = text.lower()
        bull = sum(1 for w in ["beat","upgrade","growth","record","raised","surge",
                                "gain","profit","strong","buy","bullish","positive"] if w in t)
        bear = sum(1 for w in ["miss","downgrade","cut","loss","recall","investigation",
                                "drop","fall","weak","sell","bearish","negative","layoff"] if w in t)
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
        url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            _news_cache[sym] = (now, [])
            return []

        # <link> in RSS 2.0 is a text node between tags, not an attribute —
        # ElementTree struggles with it; use string splitting as a fallback
        root  = ET.fromstring(resp.text)
        items = root.findall(".//item")
        articles = []
        for item in items[:max_articles]:
            title    = (item.findtext("title")   or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            source   = (item.findtext("source")  or "Yahoo Finance").strip()
            if not title:
                continue

            # Extract link — RSS <link> is awkward in ElementTree,
            # grab it directly from the raw item XML
            raw_item = ET.tostring(item, encoding="unicode")
            link = "#"
            try:
                # Pattern: <link>URL</link>
                import re
                m = re.search(r"<link>(.*?)</link>", raw_item)
                if m:
                    link = m.group(1).strip()
            except Exception:
                pass

            # Parse pub date to unix timestamp
            pub_ts = 0
            try:
                from email.utils import parsedate_to_datetime
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

        pdm = (high - high.shift(1)).clip(lower=0)
        ndm = (low.shift(1) - low).clip(lower=0)
        # Zero out where the other DM dominates
        pdm = pdm.where(pdm > ndm, 0.0)
        ndm = ndm.where(ndm > pdm, 0.0)

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
    for i in range(-lookback, 0):
        prev = float(m.iloc[i - 1]) - float(s.iloc[i - 1])
        curr = float(m.iloc[i])     - float(s.iloc[i])
        if prev < 0 and curr >= 0:
            return "bull"
        if prev > 0 and curr <= 0:
            return "bear"
    return None


def calc_put_call_ratio(tkr_obj):
    """
    Put/Call volume ratio for the nearest eligible expiry.
    > 1.2  → bearish sentiment (fear / hedging)
    < 0.7  → bullish sentiment (greed / speculation)
    """
    try:
        exp, _ = _best_exp(tkr_obj)
        if not exp:
            return None
        chain    = _yf_call(tkr_obj.option_chain, exp)
        put_vol  = float(chain.puts["volume"].fillna(0).sum())
        call_vol = float(chain.calls["volume"].fillna(0).sum())
        if call_vol <= 0:
            return None
        return round(put_vol / call_vol, 2)
    except Exception:
        return None


def get_next_earnings_dte(tkr_obj):
    """
    Return days until the next earnings date, or None if unavailable.
    Works with both dict and DataFrame variants of yfinance calendar.
    """
    try:
        cal = tkr_obj.calendar
        if cal is None:
            return None
        from datetime import date as _date, datetime as _dt

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


def safe_list(series, n=60):
    return [round(float(v), 4) if not (v != v) else None for v in series.iloc[-n:]]


# ══════════════════════════════════════════════════════════════
#  OPTIONS-CHAIN HELPERS
# ══════════════════════════════════════════════════════════════

def _best_exp(tkr_obj):
    """Return the expiry closest to 30 DTE within [MIN_DTE, MAX_DTE]."""
    exps  = _yf_call(lambda: tkr_obj.options)
    if not exps:
        return None, None
    today = datetime.now()
    best_exp, best_dte = None, None
    for exp in exps:
        dte = (datetime.strptime(exp, "%Y-%m-%d") - today).days
        if MIN_DTE <= dte <= MAX_DTE:
            if best_dte is None or abs(dte - 30) < abs(best_dte - 30):
                best_exp, best_dte = exp, dte
    return best_exp, best_dte


def pick_option(tkr_obj, stock_price, option_type="put",
                otm_factor=1.0, require_oi=True):
    """
    Find ATM (or slightly OTM) option for the best expiry.
    otm_factor > 1.0 → OTM call; < 1.0 → OTM put.
    """
    exp, dte = _best_exp(tkr_obj)
    if not exp:
        return None

    try:
        chain = _yf_call(tkr_obj.option_chain, exp)
        opts  = chain.puts if option_type == "put" else chain.calls
        if opts.empty:
            return None

        if require_oi:
            opts = opts[opts["openInterest"].fillna(0) >= MIN_OPEN_INTEREST].copy()
        if opts.empty:
            return None

        target_strike = stock_price * otm_factor
        opts["dist"] = abs(opts["strike"] - target_strike)
        row = opts.loc[opts["dist"].idxmin()]

        bid = float(row.get("bid") or 0)
        ask = float(row.get("ask") or 0)
        mid = (bid + ask) / 2 if bid > 0 else float(row.get("lastPrice") or 0)
        iv  = float(row.get("impliedVolatility") or 0) * 100

        if mid < 0.01 or iv < 1:
            return None

        return {
            "type":          option_type,
            "expiration":    exp,
            "dte":           dte,
            "strike":        round(float(row["strike"]), 2),
            "bid":           round(bid, 3),
            "ask":           round(ask, 3),
            "mid":           round(mid, 3),
            "iv":            round(iv, 1),
            "volume":        int(row.get("volume",        0) or 0),
            "open_interest": int(row.get("openInterest",  0) or 0),
            "premium_pct":   round(mid / stock_price * 100, 2),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
#  UNIVERSE LOADER
# ══════════════════════════════════════════════════════════════

def get_universe():
    """
    Fetch ticker universe from multiple sources, falling back gracefully.
    Priority: GitHub CSV → Wikipedia → built-in expanded list
    Respects the global SCAN_MODE — 'focus' returns FOCUS_LIST directly.
    """
    import requests
    if SCAN_MODE == "focus":
        tickers = list(dict.fromkeys(FOCUS_LIST))
        print(f"  → Focus mode: {len(tickers)} tickers")
        return tickers
    tickers = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/124.0.0.0 Safari/537.36"}

    # ── Source 1: GitHub CSV (maintained S&P 500 list, very reliable) ──
    if not tickers:
        try:
            url = ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
                   "/main/data/constituents.csv")
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")[1:]   # skip header
                t = [l.split(",")[0].strip().replace(".", "-")
                     for l in lines if l.strip()]
                t = [x for x in t if 1 < len(x) <= 6]
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
                t = pd.read_html(html)[0][col].tolist()
                t = [str(x).replace(".", "-") for x in t if isinstance(x, str) and 1 < len(str(x)) <= 6]
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
            "TFC","BK","STT","ALLY","SYF","DFS","FITB","RF","HBAN","KEY","CFG",
            # Healthcare
            "UNH","JNJ","LLY","ABBV","MRK","PFE","TMO","ABT","DHR","BMY","AMGN",
            "GILD","VRTX","REGN","ISRG","HUM","CI","CVS","MCK","CAH",
            # Energy
            "XOM","CVX","COP","SLB","EOG","PXD","MPC","VLO","PSX","OXY","HAL",
            "DVN","BKR","FANG","APA","MRO","HES","EQT","AR","RRC",
            # Consumer
            "WMT","COST","TGT","HD","LOW","MCD","SBUX","NKE","YUM","CMG","DRI",
            "AMZN","BKNG","MAR","HLT","MGM","LVS","WYNN","CZR","DKNG","PENN",
            # Communication
            "META","GOOGL","NFLX","DIS","CMCSA","T","VZ","TMUS","WBD","PARA",
            "SNAP","PINS","SPOT","RBLX","TTD","ZM","MTCH",
            # Industrial
            "GE","HON","CAT","DE","BA","RTX","LMT","NOC","GD","MMM",
            "UPS","FDX","DAL","UAL","AAL","LUV","ALK","JBLU","NCLH","RCL","CCL",
            # Materials
            "FCX","NEM","AA","X","CLF","MT","NUE","STLD","CMC","HCC",
            # EV / Clean energy
            "TSLA","RIVN","LCID","NIO","XPEV","LI","PLUG","FCEL","BLNK","CHPT",
            # Fintech / Crypto
            "PYPL","SQ","COIN","HOOD","SOFI","AFRM","UPST","LC","NU","MQ",
            # Retail / E-comm
            "SHOP","ETSY","EBAY","W","CVNA","KSS","M","JWN","GPS","ANF",
            # Biotech
            "MRNA","BNTX","BIIB","ALNY","SGEN","RARE","IONS","ACAD","SAGE","INCY",
            # REITs / Utilities
            "AMT","PLD","EQIX","CCI","SPG","O","VTR","WELL","EXR","AVB",
            "NEE","DUK","SO","D","AEP","XEL","PCG","EIX","ES","AWK",
            # Small/mid popular
            "PLTR","HOOD","SOFI","MARA","RIOT","SPCE","VALE","BB","NOK","SIRI",
            "F","GM","FORD","LCID","RIVN","CCL","AAL","NCLH","AMC","GME",
        ]
        tickers = list(dict.fromkeys(tickers))   # dedupe, preserve order

    tickers = list({str(t).replace(".", "-") for t in tickers
                    if isinstance(t, str) and 1 < len(t) <= 6})
    print(f"  → Universe: {len(tickers)} tickers total")
    return tickers


# ══════════════════════════════════════════════════════════════
#  PER-TICKER SCAN  (runs four strategies simultaneously)
# ══════════════════════════════════════════════════════════════

def _yf_history_with_retry(sym, period="1y", retries=3, base_delay=5):
    """Fetch yfinance history with exponential back-off on rate-limit errors."""
    import time
    tkr = yf.Ticker(sym)
    for attempt in range(retries):
        try:
            hist = tkr.history(period=period)
            return tkr, hist
        except Exception as e:
            msg = str(e).lower()
            if "too many requests" in msg or "rate limit" in msg or "429" in msg:
                wait = base_delay * (2 ** attempt)
                print(f"  [Rate limit] {sym} — waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise
    # Final attempt after back-off
    tkr = yf.Ticker(sym)
    return tkr, tkr.history(period=period)


def scan_ticker(sym, pre_hist=None, scan_condors=False):
    """
    Returns dict with keys: sell_put, buy_call, buy_put, covered_call
    Each value is either a signal dict or None.
    pre_hist:      optional pre-fetched DataFrame (from bulk yf.download)
    scan_condors:  if True, also evaluate iron condor setups (slow — on demand only)
    """
    import time
    try:
        if pre_hist is not None and not pre_hist.empty and len(pre_hist) >= 60:
            hist = pre_hist
            tkr  = yf.Ticker(sym)   # still needed for options chain
        else:
            time.sleep(SCAN_DELAY_S)   # throttle only when fetching individually
            tkr, hist = _yf_history_with_retry(sym)

        if hist.empty or len(hist) < 60:
            return None

        price = float(hist["Close"].iloc[-1])
        # Minimum price filter applies to all strategies (skip sub-penny stocks)
        if price < MIN_PRICE:
            return None
        # NOTE: MAX_PRICE_SELL is enforced per-strategy below —
        #   Sell Puts & Covered Calls: price ≤ MAX_PRICE_SELL
        #   Buy Calls & Buy Puts: no upper price cap

        closes        = hist["Close"]
        rsi           = calc_rsi(closes, RSI_PERIOD)
        bb_u, bb_m, bb_l = calc_bb(closes, BB_PERIOD, BB_STD_DEV)

        cur_rsi  = float(rsi.iloc[-1])
        cur_bb_u = float(bb_u.iloc[-1])
        cur_bb_m = float(bb_m.iloc[-1])
        cur_bb_l = float(bb_l.iloc[-1])

        oversold    = cur_rsi < RSI_OVERSOLD or price < cur_bb_l
        overbought  = cur_rsi > RSI_OVERBOUGHT or price > cur_bb_u
        reversal_up = rsi_turning_up(rsi)

        # ── 200-day MA trend direction ────────────────────────
        ma200_ser = closes.rolling(200).mean()
        ma200     = float(ma200_ser.iloc[-1]) if not ma200_ser.iloc[-1] != ma200_ser.iloc[-1] else None
        trend_bull = ma200 is not None and price > ma200   # stock above 200 MA
        trend_bear = ma200 is not None and price < ma200   # stock below 200 MA

        # ── ADX — trend strength ──────────────────────────────
        adx_ser, pdi_ser, ndi_ser = calc_adx(hist)
        cur_adx = round(float(adx_ser.iloc[-1]), 1) if adx_ser is not None else None
        cur_pdi = round(float(pdi_ser.iloc[-1]), 1) if pdi_ser is not None else None
        cur_ndi = round(float(ndi_ser.iloc[-1]), 1) if ndi_ser is not None else None
        strong_trend = cur_adx is not None and cur_adx > 30   # raised from 25 → 30

        # Knife / roar guards:
        #   falling_knife = confirmed strong downtrend → protect Buy Call and Sell Put
        _raw_knife = trend_bear and strong_trend and (cur_ndi or 0) > (cur_pdi or 0)
        # Exception: extreme oversold (RSI < 25) with RSI turning up = exhaustion bounce
        #   — allow even in a downtrend, the selling is likely exhausted
        exhaustion_bounce = cur_rsi < 25 and reversal_up
        falling_knife = _raw_knife and not exhaustion_bounce

        #   melt_up = confirmed strong uptrend → protect Buy Put
        melt_up = trend_bull and strong_trend and (cur_pdi or 0) > (cur_ndi or 0)

        # ── MACD (uses existing history data — no extra network call) ──
        macd_line, macd_sig_line, macd_hist_ser = calc_macd(closes)
        macd_cross   = macd_crossover(macd_line, macd_sig_line)
        macd_bull    = macd_cross == "bull"
        macd_bear    = macd_cross == "bear"
        cur_macd     = round(float(macd_line.iloc[-1]),   4)
        cur_macd_sig = round(float(macd_sig_line.iloc[-1]), 4)
        cur_macd_h   = round(float(macd_hist_ser.iloc[-1]), 4)

        # ── VIX regime ────────────────────────────────────────
        cur_vix    = get_vix()
        regime     = vix_regime(cur_vix)
        # In high-VIX environments prefer selling premium; in low-VIX prefer buying
        regime_favors_selling = regime == "high"
        regime_favors_buying  = regime == "low"

        # ── Volume confirmation ───────────────────────────────
        vol_confirmed, cur_vol, avg_vol = volume_confirmation(hist)

        # ── Confluence counters ────────────────────────────────
        # Buy Call needs 2+ bullish signals to avoid catching a falling knife
        bull_signals = sum([
            cur_rsi < 30,
            price < cur_bb_l,
            reversal_up,
            macd_bull,
            vol_confirmed,          # volume backing the move adds a signal
            regime_favors_buying,   # low VIX = cheap options, good time to buy
        ])
        # Buy Put needs 2+ bearish signals to avoid shorting a roaring bull
        bear_signals = sum([
            cur_rsi > RSI_OVERBOUGHT,
            price > cur_bb_u,
            macd_bear,
            vol_confirmed,
        ])

        # ── WFO param overrides (applied via /api/optimizer/apply) ──
        _wfo_p    = scan_state.get("wfo_params", {})
        _bull_min = _wfo_p.get("bull_min", 1)
        _bear_min = _wfo_p.get("bear_min", 2)

        # Early exit — skip meta + options fetch if no strategy can trigger
        has_sell_put     = (oversold or macd_bear) and price <= MAX_PRICE_SELL and not falling_knife
        has_buy_call     = oversold and bull_signals >= _bull_min and not falling_knife
        has_buy_put      = (overbought or macd_bear) and bear_signals >= _bear_min and not melt_up
        has_covered_call = price <= MAX_PRICE_SELL
        # Iron condors are skipped during main scan (slow); run separately on demand
        has_iron_condor  = (not oversold) and (not overbought) and (35 <= cur_rsi <= 65) and scan_condors
        if not any([has_sell_put, has_buy_call, has_buy_put, has_covered_call, has_iron_condor]):
            return None

        # ── Company meta (only fetched when a signal is possible) ─
        try:
            info   = tkr.info
            name   = info.get("shortName", sym)
            sector = info.get("sector", "N/A")
        except Exception:
            name, sector = sym, "N/A"

        # ── Short interest ────────────────────────────────────
        short_pct, short_ratio = get_short_interest(tkr)
        # Squeeze setup: high short interest + oversold = potential short squeeze
        squeeze_setup = (short_pct is not None and short_pct >= 10 and oversold)

        # ── Put/Call ratio for nearest expiry ─────────────────
        pc_ratio = calc_put_call_ratio(tkr)
        pc_bearish = pc_ratio is not None and pc_ratio > 1.2
        pc_bullish = pc_ratio is not None and pc_ratio < 0.7

        # ── Next earnings date ────────────────────────────────
        earnings_dte  = get_next_earnings_dte(tkr)
        earnings_soon = (earnings_dte is not None and
                         MIN_DTE <= earnings_dte <= MAX_DTE)

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
            "rsi":   round(cur_rsi, 1),
            "bb_upper": round(cur_bb_u, 3),
            "bb_mid":   round(cur_bb_m, 3),
            "bb_lower": round(cur_bb_l, 3),
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
            # VIX regime
            "vix":           cur_vix,
            "vix_regime":    regime,
            # Volume confirmation
            "vol_confirmed": vol_confirmed,
            "cur_vol":       int(cur_vol) if cur_vol else None,
            "avg_vol":       int(avg_vol) if avg_vol else None,
            # Short interest
            "short_pct":     short_pct,
            "short_ratio":   short_ratio,
            "squeeze_setup": squeeze_setup,
            "chart": chart,
        }

        results = {}

        # ── 1. SELL PUT ───────────────────────────────────────
        if (oversold or macd_bear) and price <= MAX_PRICE_SELL and not falling_knife:
            opt = pick_option(tkr, price, "put", otm_factor=1.0)
            if opt and opt["bid"] >= 0.01:
                iv_rank = calc_iv_rank(closes, opt["iv"])
                iv_high  = iv_rank is not None and iv_rank >= IV_RANK_THRESHOLD
                prem_high = opt["premium_pct"] >= PREMIUM_PCT_MIN
                if iv_high or prem_high:
                    score = _score_sell_put(cur_rsi, price, cur_bb_l, iv_rank, opt["premium_pct"])
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
        if oversold and bull_signals >= 1 and not falling_knife:
            opt = pick_option(tkr, price, "call", otm_factor=1.0)
            if opt and opt["ask"] >= 0.01:
                score = _score_buy_call(cur_rsi, price, cur_bb_l, reversal_up)
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
        if (overbought or macd_bear) and bear_signals >= 2 and not melt_up:
            opt = pick_option(tkr, price, "put", otm_factor=1.0)
            if opt and opt["ask"] >= 0.01:
                score = _score_buy_put(cur_rsi, price, cur_bb_u)
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
        if price <= MAX_PRICE_SELL:
            opt_cc = pick_option(tkr, price, "call", otm_factor=1.05)
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
        # Signal: stock is range-bound, IV is elevated
        neutral = (not oversold) and (not overbought) and (35 <= cur_rsi <= 65)
        iv_rank_ic = None
        if neutral:
            opt_sp = pick_option(tkr, price, "put",  otm_factor=0.95)  # short put
            opt_lp = pick_option(tkr, price, "put",  otm_factor=0.90)  # long put
            opt_sc = pick_option(tkr, price, "call", otm_factor=1.05)  # short call
            opt_lc = pick_option(tkr, price, "call", otm_factor=1.10)  # long call

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

def _score_sell_put(rsi, price, bb_l, iv_rank, prem_pct):
    s = 0
    if rsi < RSI_OVERSOLD: s += max(0, 35 + (RSI_OVERSOLD - rsi))
    if price < bb_l:        s += 25
    if iv_rank and iv_rank >= IV_RANK_THRESHOLD: s += min(int(iv_rank - IV_RANK_THRESHOLD), 25)
    if prem_pct >= PREMIUM_PCT_MIN: s += min(int((prem_pct - PREMIUM_PCT_MIN) * 3), 15)
    return min(max(round(s), 0), 100)

def _score_buy_call(rsi, price, bb_l, reversal):
    s = 0
    if rsi < 30:    s += 40
    elif rsi < 35:  s += 25
    if price < bb_l: s += 25
    if reversal:     s += 20
    return min(max(round(s), 0), 100)

def _score_buy_put(rsi, price, bb_u):
    s = 0
    if rsi > 75:     s += 40
    elif rsi > 65:   s += 25
    if price > bb_u: s += 30
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

def load_tracked():
    if os.path.exists(TRACK_FILE):
        try:
            with open(TRACK_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_tracked(positions):
    with open(TRACK_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


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
    entry    = position.get("option", {})
    entry_price = float(entry.get("mid") or entry.get("ask") or 0)

    result = {
        "triggered":          False,
        "reasons":            [],
        "current_rsi":        None,
        "current_price":      None,
        "signal_still_valid": True,
    }

    try:
        tkr  = yf.Ticker(sym)
        hist = tkr.history(period="3mo")
        if hist.empty or len(hist) < 30:
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

        # ── Fetch current option price ─────────────────────────
        current_mid = 0.0
        try:
            opt    = position.get("option", {})
            exp    = opt.get("expiration")
            strike = opt.get("strike")
            otype  = opt.get("type", "call")
            if exp and strike:
                chain = _yf_call(tkr.option_chain, exp)
                opts  = chain.calls if otype == "call" else chain.puts
                row   = opts[abs(opts["strike"] - strike) < 0.01]
                if not row.empty:
                    bid = float(row.iloc[0].get("bid") or 0)
                    ask = float(row.iloc[0].get("ask") or 0)
                    current_mid = (bid + ask) / 2
        except Exception:
            pass

        # ── Profit / loss check (if we have entry & current price) ──
        result["current_option_mid"] = round(current_mid, 3) if current_mid > 0 else None
        if entry_price > 0 and current_mid > 0:
            pct_change = (current_mid - entry_price) / entry_price * 100
            dollar_change = (current_mid - entry_price) * 100   # per contract
            result["pct_change"]    = round(pct_change, 1)
            result["dollar_change"] = round(dollar_change, 2)
            if pct_change >= SELL_PROFIT_TARGET:
                result["triggered"] = True
                result["reasons"].append(
                    f"🎯 Profit target hit: +{pct_change:.1f}% (target +{SELL_PROFIT_TARGET}%)"
                )
            elif pct_change <= -SELL_STOP_LOSS:
                result["triggered"] = True
                result["reasons"].append(
                    f"🛑 Stop loss hit: {pct_change:.1f}% (limit -{SELL_STOP_LOSS}%)"
                )

        # ── RSI / BB reversal checks ───────────────────────────
        if strategy == "buy_call":
            # Original signal: oversold RSI — sell when RSI recovers
            if cur_rsi >= SELL_RSI_BULL_TARGET:
                result["triggered"] = True
                result["reasons"].append(
                    f"📈 RSI recovered to {cur_rsi:.1f} (target ≥{SELL_RSI_BULL_TARGET}) — take profit"
                )
            if price >= cur_bb_u:
                result["triggered"] = True
                result["reasons"].append(
                    f"📈 Price ${price:.2f} hit upper Bollinger Band ${cur_bb_u:.2f}"
                )
            result["signal_still_valid"] = cur_rsi < RSI_OVERSOLD or price < cur_bb_l

        elif strategy == "buy_put":
            # Original signal: overbought RSI — sell when RSI pulls back
            if cur_rsi <= SELL_RSI_BEAR_TARGET:
                result["triggered"] = True
                result["reasons"].append(
                    f"📉 RSI pulled back to {cur_rsi:.1f} (target ≤{SELL_RSI_BEAR_TARGET}) — take profit"
                )
            if price <= cur_bb_l:
                result["triggered"] = True
                result["reasons"].append(
                    f"📉 Price ${price:.2f} hit lower Bollinger Band ${cur_bb_l:.2f}"
                )
            result["signal_still_valid"] = cur_rsi > RSI_OVERBOUGHT or price > cur_bb_u

        elif strategy == "sell_put":
            # Original signal: oversold — close early if stock drops further
            if price < cur_bb_l * 0.95:
                result["triggered"] = True
                result["reasons"].append(
                    f"⚠️ Stock ${price:.2f} fell significantly below BB lower ${cur_bb_l:.2f} — consider closing"
                )
            if cur_rsi > 55:
                result["reasons"].append(
                    f"✅ RSI {cur_rsi:.1f} — put likely expired worthless or near max profit"
                )
            result["signal_still_valid"] = cur_rsi < RSI_OVERSOLD

        elif strategy == "covered_call":
            if price > float(entry.get("strike", 0)) * 1.05:
                result["triggered"] = True
                result["reasons"].append(
                    f"⚠️ Stock ${price:.2f} running past strike ${entry.get('strike')} — call may be assigned"
                )
            result["signal_still_valid"] = True

        elif strategy == "iron_condor":
            sp = float(position.get("legs", {}).get("short_put", {}).get("strike", 0))
            sc = float(position.get("legs", {}).get("short_call", {}).get("strike", 0))
            if sp and sc:
                if price <= sp:
                    result["triggered"] = True
                    result["reasons"].append(
                        f"⚠️ Stock ${price:.2f} broke below short put ${sp:.2f}"
                    )
                elif price >= sc:
                    result["triggered"] = True
                    result["reasons"].append(
                        f"⚠️ Stock ${price:.2f} broke above short call ${sc:.2f}"
                    )
            result["signal_still_valid"] = sp < price < sc if sp and sc else True

    except Exception:
        pass

    return result


def _run_tracked_check():
    """Background thread — re-check sell signals on all tracked positions."""
    positions = load_tracked()
    if not positions:
        return
    print(f"\n[Track] Checking {len(positions)} tracked position(s)…")
    for pos in positions:
        try:
            sig = get_sell_signal(pos)
            with track_lock:
                pos["last_checked"]  = datetime.now().isoformat()
                pos["last_signal"]   = sig
        except Exception:
            pass
    with track_lock:
        save_tracked(positions)
    print(f"[Track] Done checking tracked positions.")


def _schedule_track_checker():
    """Re-run the sell signal checker every TRACK_CHECK_MINS minutes."""
    _run_tracked_check()
    t = threading.Timer(TRACK_CHECK_MINS * 60, _schedule_track_checker)
    t.daemon = True
    t.start()


# ══════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════

_bt_state = {"running": False, "progress": 0, "total": 0, "results": None, "ticker": None}
_bt_lock  = threading.Lock()


def _bt_signals(closes, highs, lows, volumes, idx, params=None):
    """
    Compute signal booleans at position `idx` in the OHLCV series.
    Mirrors the logic in scan_ticker so backtest uses same rules.
    `params` dict overrides live config values — used by the WFO engine.
    Returns dict of booleans + individual signal flags, or None if not enough data.
    """
    if idx < 60:
        return None

    # ── Parameter overrides (defaults = live config) ──────────
    p = params or {}
    rsi_os  = p.get("rsi_oversold",  RSI_OVERSOLD)
    rsi_ob  = p.get("rsi_overbought",RSI_OVERBOUGHT)
    bull_min= p.get("bull_min",      1)
    bear_min= p.get("bear_min",      2)
    adx_thr = p.get("adx_threshold", 30)

    c = closes.iloc[:idx+1]
    h = highs.iloc[:idx+1]
    l = lows.iloc[:idx+1]
    v = volumes.iloc[:idx+1]

    rsi = calc_rsi(c, RSI_PERIOD)
    bb_u, bb_m, bb_l = calc_bb(c, BB_PERIOD, BB_STD_DEV)
    cur_rsi  = float(rsi.iloc[-1])
    price    = float(c.iloc[-1])
    cur_bb_l = float(bb_l.iloc[-1])
    cur_bb_u = float(bb_u.iloc[-1])

    sig_rsi_os  = cur_rsi < rsi_os       # RSI oversold
    sig_rsi_ob  = cur_rsi > rsi_ob       # RSI overbought
    sig_bb_low  = price < cur_bb_l       # below lower BB
    sig_bb_high = price > cur_bb_u       # above upper BB

    oversold   = sig_rsi_os or sig_bb_low
    overbought = sig_rsi_ob or sig_bb_high
    reversal_up = rsi_turning_up(rsi)

    ma200_ser = c.rolling(200).mean()
    ma200     = float(ma200_ser.iloc[-1]) if not pd.isna(ma200_ser.iloc[-1]) else None
    trend_bull = ma200 is not None and price > ma200
    trend_bear = ma200 is not None and price < ma200

    hist_mini = pd.DataFrame({"Close": c, "High": h, "Low": l})
    adx_s, pdi_s, ndi_s = calc_adx(hist_mini)
    cur_adx = float(adx_s.iloc[-1]) if adx_s is not None else None
    cur_pdi = float(pdi_s.iloc[-1]) if pdi_s is not None else None
    cur_ndi = float(ndi_s.iloc[-1]) if ndi_s is not None else None
    strong_trend = cur_adx is not None and cur_adx > adx_thr

    macd_line, macd_sig, _ = calc_macd(c)
    macd_cross  = macd_crossover(macd_line, macd_sig)
    sig_macd_bull = macd_cross == "bull"
    sig_macd_bear = macd_cross == "bear"

    sig_vol_ok = False
    if len(v) > 20:
        avg_vol = float(v.iloc[-21:-1].mean())
        sig_vol_ok = float(v.iloc[-1]) >= avg_vol * 1.2

    _raw_knife    = trend_bear and strong_trend and (cur_ndi or 0) > (cur_pdi or 0)
    exhaustion    = cur_rsi < 25 and reversal_up
    falling_knife = _raw_knife and not exhaustion
    melt_up       = trend_bull and strong_trend and (cur_pdi or 0) > (cur_ndi or 0)

    bull_n = sum([sig_rsi_os, sig_bb_low, reversal_up, sig_macd_bull, sig_vol_ok])
    bear_n = sum([sig_rsi_ob, sig_bb_high, sig_macd_bear, sig_vol_ok])

    return {
        "price":         price,
        "rsi":           cur_rsi,
        "oversold":      oversold,
        "overbought":    overbought,
        "falling_knife": falling_knife,
        "melt_up":       melt_up,
        "bull_signals":  bull_n,
        "bear_signals":  bear_n,
        # Individual signal flags (used by signal quality analysis)
        "sig_rsi_os":    sig_rsi_os,
        "sig_bb_low":    sig_bb_low,
        "sig_reversal":  reversal_up,
        "sig_macd_bull": sig_macd_bull,
        "sig_vol_ok":    sig_vol_ok,
        "sig_rsi_ob":    sig_rsi_ob,
        "sig_bb_high":   sig_bb_high,
        "sig_macd_bear": sig_macd_bear,
        "has_buy_call":  oversold and bull_n >= bull_min and not falling_knife,
        "has_buy_put":   (overbought or sig_macd_bear) and bear_n >= bear_min and not melt_up,
        "has_sell_put":  (oversold or sig_macd_bear) and price <= MAX_PRICE_SELL and not falling_knife,
        "vol_ok":        sig_vol_ok,
    }


def _bt_option_price(S, K, T, sigma, option_type):
    """Black-Scholes option price. T in years. Returns 0 on error."""
    try:
        return max(0.0, black_scholes(S, K, T, RISK_FREE_RATE, sigma, option_type))
    except Exception:
        return 0.0


def run_backtest_ticker(ticker, hold_days=21, params=None, hist=None,
                        idx_start=60, idx_end=None):
    """
    Walk history for `ticker`.
    hold_days : days to hold the option
    params    : optional parameter overrides for _bt_signals (WFO use)
    hist      : pre-downloaded DataFrame (skip yfinance fetch if provided)
    idx_start / idx_end : window slice for walk-forward splits
    """
    if hist is None:
        try:
            hist = yf.Ticker(ticker).history(period="2y")
            if hist.empty or len(hist) < 120:
                return []
        except Exception:
            return []

    closes  = hist["Close"]
    highs   = hist["High"]
    lows    = hist["Low"]
    volumes = hist["Volume"]
    trades  = []
    end     = (idx_end or len(hist)) - hold_days

    for idx in range(max(60, idx_start), end):
        sigs = _bt_signals(closes, highs, lows, volumes, idx, params)
        if not sigs:
            continue

        price      = sigs["price"]
        ret_win    = np.log(closes.iloc[max(0,idx-30):idx+1] /
                            closes.iloc[max(0,idx-30):idx+1].shift(1)).dropna()
        iv_est     = float(ret_win.std() * np.sqrt(252)) if len(ret_win) > 5 else 0.20
        iv_est     = max(0.05, min(iv_est, 2.0))
        T_entry    = hold_days / 365.0
        T_exit     = 0.001
        exit_price = float(closes.iloc[idx + hold_days])
        entry_date = hist.index[idx].strftime("%Y-%m-%d")
        exit_date  = hist.index[idx + hold_days].strftime("%Y-%m-%d")

        for strategy, flag in [("buy_call", sigs["has_buy_call"]),
                                ("buy_put",  sigs["has_buy_put"]),
                                ("sell_put", sigs["has_sell_put"])]:
            if not flag:
                continue
            K = round(price)

            if strategy == "sell_put":
                entry_opt = _bt_option_price(price,      K, T_entry, iv_est, "put")
                exit_opt  = _bt_option_price(exit_price, K, T_exit,  iv_est, "put")
                if entry_opt <= 0.01:
                    continue
                pnl_pct = round((entry_opt - exit_opt) / entry_opt * 100, 1)
            elif strategy == "buy_call":
                entry_opt = _bt_option_price(price,      K, T_entry, iv_est, "call")
                exit_opt  = _bt_option_price(exit_price, K, T_exit,  iv_est, "call")
                if entry_opt <= 0.01:
                    continue
                pnl_pct = round((exit_opt - entry_opt) / entry_opt * 100, 1)
            else:  # buy_put
                entry_opt = _bt_option_price(price,      K, T_entry, iv_est, "put")
                exit_opt  = _bt_option_price(exit_price, K, T_exit,  iv_est, "put")
                if entry_opt <= 0.01:
                    continue
                pnl_pct = round((exit_opt - entry_opt) / entry_opt * 100, 1)

            trades.append({
                "ticker":      ticker,
                "strategy":    strategy,
                "entry_date":  entry_date,
                "exit_date":   exit_date,
                "entry_price": round(price, 2),
                "exit_price":  round(exit_price, 2),
                "entry_opt":   round(entry_opt, 3),
                "exit_opt":    round(exit_opt, 3),
                "pnl_pct":     pnl_pct,
                "win":         pnl_pct > 0,
                "iv_est":      round(iv_est * 100, 1),
                # Individual signal flags for quality analysis
                "sig_rsi_os":    sigs["sig_rsi_os"],
                "sig_bb_low":    sigs["sig_bb_low"],
                "sig_reversal":  sigs["sig_reversal"],
                "sig_macd_bull": sigs["sig_macd_bull"],
                "sig_vol_ok":    sigs["sig_vol_ok"],
                "sig_rsi_ob":    sigs["sig_rsi_ob"],
                "sig_bb_high":   sigs["sig_bb_high"],
                "sig_macd_bear": sigs["sig_macd_bear"],
            })
    return trades


def _bt_summary(trades):
    """Aggregate stats over a list of trade dicts."""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "avg_pnl": 0,
            "avg_win": 0, "avg_loss": 0, "sharpe": 0,
            "best": 0, "worst": 0, "by_strategy": {},
        }
    wins    = [t for t in trades if t["win"]]
    losses  = [t for t in trades if not t["win"]]
    pnls    = [t["pnl_pct"] for t in trades]
    win_rate = round(len(wins) / len(trades) * 100, 1)
    avg_pnl  = round(sum(pnls) / len(pnls), 1)
    avg_win  = round(sum(t["pnl_pct"] for t in wins)  / len(wins),   1) if wins   else 0
    avg_loss = round(sum(t["pnl_pct"] for t in losses) / len(losses), 1) if losses else 0
    # Sharpe-like: mean / std of P&L
    if len(pnls) > 1:
        std = float(np.std(pnls))
        sharpe = round(avg_pnl / std, 2) if std > 0 else 0
    else:
        sharpe = 0
    # Per-strategy breakdown
    by_strat = {}
    for strat in ("buy_call", "buy_put", "sell_put"):
        st = [t for t in trades if t["strategy"] == strat]
        if st:
            w = [t for t in st if t["win"]]
            by_strat[strat] = {
                "count": len(st),
                "win_rate": round(len(w)/len(st)*100, 1),
                "avg_pnl": round(sum(t["pnl_pct"] for t in st)/len(st), 1),
            }
    return {
        "total_trades": len(trades),
        "win_rate":     win_rate,
        "avg_pnl":      avg_pnl,
        "avg_win":      avg_win,
        "avg_loss":     avg_loss,
        "sharpe":       sharpe,
        "best":         max(pnls),
        "worst":        min(pnls),
        "by_strategy":  by_strat,
    }


def _run_backtest_job(tickers, hold_days):
    global _bt_state
    all_trades = []
    with _bt_lock:
        _bt_state.update(running=True, progress=0, total=len(tickers),
                         results=None, stage="Downloading history…")

    # ── Bulk download 2y OHLCV for all tickers in one shot ───────
    # (avoids per-ticker rate limits that caused silent failures)
    print(f"  [Backtest] Bulk downloading 2y history for {len(tickers)} tickers…")
    hist_map = {}
    try:
        raw = yf.download(
            tickers, period="2y", group_by="ticker",
            auto_adjust=False, threads=True, progress=False,
        )
        for sym in tickers:
            try:
                df = (raw[sym] if len(tickers) > 1 else raw).dropna(how="all")
                if len(df) >= 120:
                    hist_map[sym] = df
            except Exception:
                pass
        print(f"  [Backtest] Got data for {len(hist_map)}/{len(tickers)} tickers")
    except Exception as e:
        print(f"  [Backtest] Bulk download failed ({e}), falling back to per-ticker")

    with _bt_lock:
        _bt_state["stage"] = "Scanning signals…"

    for i, sym in enumerate(tickers):
        try:
            pre_hist = hist_map.get(sym)           # use bulk data if available
            trades   = run_backtest_ticker(sym, hold_days, hist=pre_hist)
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
    with _bt_lock:
        _bt_state.update(running=False, stage="Complete", results={
            "trades":      all_trades[:500],
            "summary":     summary,
            "sig_quality": sig_quality,
        })
    print(f"  [Backtest] Done — {len(all_trades)} trades, "
          f"win rate {summary.get('win_rate', 'N/A')}%")


# ══════════════════════════════════════════════════════════════
#  SIGNAL QUALITY ANALYSIS
# ══════════════════════════════════════════════════════════════

def signal_quality_analysis(trades):
    """
    For each individual signal flag, compute win rate across all trades
    where that signal was active.  Returned as sorted list of dicts.
    """
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

# Parameter grid — each combination is tested independently
WFO_PARAM_GRID = {
    "rsi_oversold":  [25, 30, 35],
    "rsi_overbought":[65, 70, 75],
    "bull_min":      [1, 2, 3],
    "bear_min":      [1, 2],
    "adx_threshold": [25, 30, 35],
    "hold_days":     [14, 21, 30],
}

_wfo_state = {"running": False, "progress": 0, "total": 0, "results": None, "stage": ""}
_wfo_lock  = threading.Lock()


def _wfo_sharpe(trades):
    """Sharpe-like ratio: mean P&L / std P&L.  Returns -99 on empty."""
    if len(trades) < 3:
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


def _run_wfo_job(tickers):
    """
    Walk-forward optimization across `tickers`.

    Algorithm:
      1. Download 2y history for each ticker once.
      2. Split each history: train = first 75%, validation = last 25%.
      3. For each param combo, run backtest on ALL tickers × training window.
         Aggregate Sharpe across all resulting trades.
      4. Take top-10 combos by training Sharpe.
      5. Validate each on ALL tickers × validation window.
      6. Rank by validation Sharpe (guards against overfitting).
      7. Return best combo + full leaderboard.
    """
    global _wfo_state

    combos = _expand_grid(WFO_PARAM_GRID)
    with _wfo_lock:
        _wfo_state.update(running=True, progress=0, total=len(combos),
                          results=None, stage="Downloading history…")

    # ── Step 1: bulk-download all histories once ────────────────
    print(f"  [WFO] Downloading 2y history for {len(tickers)} tickers…")
    hist_map = {}
    try:
        raw = yf.download(tickers, period="2y", group_by="ticker",
                          auto_adjust=False, threads=True, progress=False)
        for sym in tickers:
            try:
                df = (raw[sym] if len(tickers) > 1 else raw).dropna(how="all")
                if len(df) >= 120:
                    hist_map[sym] = df
            except Exception:
                pass
    except Exception as e:
        print(f"  [WFO] Bulk download failed: {e}")

    if not hist_map:
        with _wfo_lock:
            _wfo_state.update(running=False, stage="No data available")
        return

    # ── Step 2: compute split index per ticker ──────────────────
    split_map = {sym: int(len(h) * 0.75) for sym, h in hist_map.items()}

    # ── Step 3: sweep param grid on training window ─────────────
    with _wfo_lock:
        _wfo_state["stage"] = f"Training {len(combos)} param combos…"

    def _eval_combo_train(combo):
        hold = combo.get("hold_days", 21)
        params = {k: v for k, v in combo.items() if k != "hold_days"}
        all_t = []
        for sym, hist in hist_map.items():
            split = split_map[sym]
            trades = run_backtest_ticker(sym, hold_days=hold, params=params,
                                         hist=hist, idx_end=split)
            all_t.extend(trades)
        return _wfo_sharpe(all_t), len(all_t), all_t

    train_scores = []
    for i, combo in enumerate(combos):
        sharpe, n_trades, _ = _eval_combo_train(combo)
        train_scores.append((sharpe, n_trades, combo))
        with _wfo_lock:
            _wfo_state["progress"] = i + 1

    # Sort by training Sharpe, keep top 10
    train_scores.sort(key=lambda x: x[0], reverse=True)
    top10 = train_scores[:10]

    # ── Step 4: validate top-10 on held-out window ──────────────
    with _wfo_lock:
        _wfo_state["stage"] = "Validating top 10 combos…"

    leaderboard = []
    for train_sharpe, train_n, combo in top10:
        hold   = combo.get("hold_days", 21)
        params = {k: v for k, v in combo.items() if k != "hold_days"}
        val_trades = []
        for sym, hist in hist_map.items():
            split = split_map[sym]
            trades = run_backtest_ticker(sym, hold_days=hold, params=params,
                                          hist=hist, idx_start=split)
            val_trades.extend(trades)
        val_sharpe = _wfo_sharpe(val_trades)
        val_wr     = round(len([t for t in val_trades if t["win"]]) /
                           max(len(val_trades), 1) * 100, 1)
        val_avg    = round(sum(t["pnl_pct"] for t in val_trades) /
                           max(len(val_trades), 1), 1)
        leaderboard.append({
            "params":       combo,
            "train_sharpe": train_sharpe,
            "train_trades": train_n,
            "val_sharpe":   val_sharpe,
            "val_trades":   len(val_trades),
            "val_win_rate": val_wr,
            "val_avg_pnl":  val_avg,
        })

    leaderboard.sort(key=lambda x: x["val_sharpe"], reverse=True)
    best = leaderboard[0] if leaderboard else None

    # ── Live config suggestion (don't auto-apply, show to user) ─
    suggestion = None
    if best:
        suggestion = {k: v for k, v in best["params"].items()}

    with _wfo_lock:
        _wfo_state.update(
            running=False,
            stage="Complete",
            results={
                "leaderboard": leaderboard,
                "best":        best,
                "suggestion":  suggestion,
                "tickers_used": list(hist_map.keys()),
            }
        )
    print(f"  [WFO] Done — best val Sharpe {best['val_sharpe'] if best else 'N/A'}")


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
    global scan_state
    import time

    print(f"  [Live] Refresh loop started — interval: {_live_refresh_mins} min")
    while not _live_refresh_stop.wait(timeout=_live_refresh_mins * 60):
        with scan_lock:
            if scan_state["running"]:
                continue   # full scan in progress — skip refresh
            current = scan_state["results"]

        signal_tickers = list({r["ticker"] for strat in current.values() for r in strat})
        if not signal_tickers:
            print("  [Live] No signal tickers to refresh — waiting")
            continue

        print(f"  [Live] Refreshing {len(signal_tickers)} signal ticker(s): {', '.join(signal_tickers)}")

        # Bulk download for the small signal set
        bulk_hist = {}
        try:
            raw = yf.download(
                signal_tickers, period="1y", group_by="ticker",
                auto_adjust=False, threads=True, progress=False,
            )
            if len(signal_tickers) == 1:
                bulk_hist[signal_tickers[0]] = raw
            else:
                for sym in signal_tickers:
                    try:
                        df = raw[sym].dropna(how="all")
                        if not df.empty:
                            bulk_hist[sym] = df
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [Live] Bulk download failed: {e} — using per-ticker fallback")

        new_totals = {"sell_puts": [], "buy_calls": [], "buy_puts": [],
                      "covered_calls": [], "iron_condors": []}
        for sym in signal_tickers:
            if _live_refresh_stop.is_set():
                return
            try:
                res_raw = scan_ticker(sym, bulk_hist.get(sym), False)
                if res_raw:
                    res = res_raw.get("_signals", {})
                    if "sell_put"     in res: new_totals["sell_puts"].append(res["sell_put"])
                    if "buy_call"     in res: new_totals["buy_calls"].append(res["buy_call"])
                    if "buy_put"      in res: new_totals["buy_puts"].append(res["buy_put"])
                    if "covered_call" in res: new_totals["covered_calls"].append(res["covered_call"])
                    if "iron_condor"  in res: new_totals["iron_condors"].append(res["iron_condor"])
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
            scan_state["results"]      = new_totals
            scan_state["last_refresh"] = refreshed_at

        print(f"  [Live] Refresh complete at {refreshed_at}")

    print("  [Live] Refresh loop stopped.")


def _start_live_refresh():
    """Start (or restart) the live refresh daemon thread."""
    global _live_refresh_stop
    _live_refresh_stop.set()          # stop any existing loop
    _live_refresh_stop = threading.Event()
    t = threading.Thread(target=_live_refresh_loop, daemon=True)
    t.start()


# ══════════════════════════════════════════════════════════════
#  BACKGROUND SCAN THREAD
# ══════════════════════════════════════════════════════════════

def _run_scan(mode=None):
    global scan_state, SCAN_MODE
    # Stop any running live refresh before starting a full scan
    _live_refresh_stop.set()

    if mode in ("focus", "full"):
        SCAN_MODE = mode
    tickers = get_universe()
    totals  = {"sell_puts": [], "buy_calls": [], "buy_puts": [], "covered_calls": [], "iron_condors": []}

    with scan_lock:
        scan_state.update(running=True, progress=0, total=len(tickers),
                          results={"sell_puts":[], "buy_calls":[], "buy_puts":[], "covered_calls":[], "iron_condors":[]})

    # ── Bulk OHLCV download (single API call for all tickers) ────────────
    # This avoids 500 individual history() calls and massively reduces
    # rate-limit exposure. Tickers with no data will have all-NaN rows.
    print(f"  [Bulk download] Fetching 1-year OHLCV for {len(tickers)} tickers…")
    bulk_hist = {}
    try:
        raw = yf.download(
            tickers,
            period="1y",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        # With multiple tickers, raw is a MultiIndex DataFrame: raw[sym] → OHLCV
        # With a single ticker it's a plain DataFrame — handle both cases
        if len(tickers) == 1:
            bulk_hist[tickers[0]] = raw
        else:
            for sym in tickers:
                try:
                    df = raw[sym].dropna(how="all")
                    if not df.empty:
                        bulk_hist[sym] = df
                except Exception:
                    pass
        print(f"  [Bulk download] Done — {len(bulk_hist)}/{len(tickers)} tickers have data")
    except Exception as e:
        print(f"  [Bulk download] Failed ({e}) — will fall back to per-ticker fetches")

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(scan_ticker, sym, bulk_hist.get(sym), False): sym for sym in tickers}
        for fut in as_completed(futures):
            done += 1
            raw = fut.result()
            log = raw.get("_log") if raw else None
            res = raw.get("_signals", {}) if raw else {}
            if log:
                print(log)
            if res:
                if "sell_put"    in res: totals["sell_puts"].append(res["sell_put"])
                if "buy_call"    in res: totals["buy_calls"].append(res["buy_call"])
                if "buy_put"     in res: totals["buy_puts"].append(res["buy_put"])
                if "covered_call"in res: totals["covered_calls"].append(res["covered_call"])
                if "iron_condor" in res: totals["iron_condors"].append(res["iron_condor"])

            with scan_lock:
                scan_state["progress"] = done
                # Stream partial results live
                scan_state["results"] = {
                    k: sorted(v, key=lambda x: x["score"], reverse=True)
                    for k, v in totals.items()
                }

    for key in totals:
        totals[key] = sorted(totals[key], key=lambda x: x["score"], reverse=True)

    with scan_lock:
        scan_state.update(running=False, last_scan=datetime.now().isoformat(),
                          results=totals)
    print(f"\n  Scan complete — sell_puts:{len(totals['sell_puts'])}  "
          f"buy_calls:{len(totals['buy_calls'])}  "
          f"buy_puts:{len(totals['buy_puts'])}  "
          f"covered_calls:{len(totals['covered_calls'])}  "
          f"iron_condors:{len(totals['iron_condors'])}")

    # ── News enrichment (only for signal tickers, in parallel) ──
    signal_tickers = {r["ticker"] for strat in totals.values() for r in strat}
    if signal_tickers:
        print(f"  Fetching news for {len(signal_tickers)} signal ticker(s)…")
        # Clear cache for signal tickers so we always get fresh news
        for _sym in signal_tickers:
            _news_cache.pop(_sym, None)
        news_map = {}
        def _fetch_news(sym):
            news_map[sym] = fetch_ticker_news(sym)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            pool.map(_fetch_news, signal_tickers)
        # Attach news + conflict warning to each result
        for strat_list in totals.values():
            for r in strat_list:
                sym   = r["ticker"]
                news  = news_map.get(sym, [])
                r["news"]          = news
                r["news_conflict"] = news_signal_conflict(news, r.get("strategy",""))
        with scan_lock:
            scan_state["results"] = totals
        print(f"  News enrichment done.")

    # Also refresh sell signals on all tracked positions after every scan
    _run_tracked_check()

    # ── Kick off live refresh loop ───────────────────────────────
    _start_live_refresh()


# ══════════════════════════════════════════════════════════════
#  TRADE QUEUE & GUARDRAIL
# ══════════════════════════════════════════════════════════════

def load_queue():
    if os.path.exists(QUEUE_FILE):
        try:
            with open(QUEUE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_queue(q):
    with open(QUEUE_FILE, "w") as f:
        json.dump(q, f, indent=2, default=str)


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_history(h):
    with open(HISTORY_FILE, "w") as f:
        json.dump(h, f, indent=2, default=str)


def calc_trade_cost(trade):
    """
    Returns the dollar amount that counts against the guardrail.
    • BUY  option  → debit = qty × 100 × ask
    • SELL put     → capital reserved = qty × 100 × strike
    • SELL call (covered) → $0 additional capital
    """
    qty    = int(trade.get("qty", 1))
    strat  = trade.get("strategy", "")
    opt    = trade.get("option", {})
    strike = float(opt.get("strike", 0))
    ask    = float(opt.get("ask", 0))

    if strat == "buy_call" or strat == "buy_put":
        return qty * 100 * ask
    elif strat == "sell_put":
        return qty * 100 * strike
    elif strat == "covered_call":
        return 0.0   # no additional capital required
    elif strat == "iron_condor":
        # Max loss = spread_width - net_credit (per contract × qty)
        put_width  = float(trade.get("put_width",  0))
        call_width = float(trade.get("call_width", 0))
        net_credit = float(opt.get("net_credit", ask))
        spread_w   = max(put_width, call_width) if (put_width or call_width) else strike * 0.05
        return qty * 100 * max(spread_w - net_credit, 0)
    return qty * 100 * ask


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
            "running":      scan_state["running"],
            "progress":     scan_state["progress"],
            "total":        scan_state["total"],
            "last_scan":    scan_state["last_scan"],
            "last_refresh": scan_state.get("last_refresh"),
            "live_active":  not _live_refresh_stop.is_set(),
            "live_mins":    _live_refresh_mins,
            "counts": {k: len(v) for k, v in scan_state["results"].items()},
        })


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
    global _condor_state
    tickers = get_universe()
    condors = []

    with _condor_lock:
        _condor_state.update(running=True, progress=0, total=len(tickers))

    print(f"  [Condor] Scanning {len(tickers)} tickers for iron condors…")

    # Bulk OHLCV
    bulk_hist = {}
    try:
        raw = yf.download(tickers, period="1y", group_by="ticker",
                          auto_adjust=False, threads=True, progress=False)
        if len(tickers) == 1:
            bulk_hist[tickers[0]] = raw
        else:
            for sym in tickers:
                try:
                    df = raw[sym].dropna(how="all")
                    if not df.empty:
                        bulk_hist[sym] = df
                except Exception:
                    pass
    except Exception as e:
        print(f"  [Condor] Bulk download failed: {e}")

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(scan_ticker, sym, bulk_hist.get(sym), True): sym for sym in tickers}
        for fut in as_completed(futures):
            done += 1
            raw = fut.result()
            if raw:
                res = raw.get("_signals", {})
                if "iron_condor" in res:
                    condors.append(res["iron_condor"])
            with _condor_lock:
                _condor_state["progress"] = done

    condors = sorted(condors, key=lambda x: x["score"], reverse=True)
    with scan_lock:
        scan_state["results"]["iron_condors"] = condors
    with _condor_lock:
        _condor_state.update(running=False, progress=done)
    print(f"  [Condor] Done — {len(condors)} iron condor(s) found")


@app.route("/api/condor/scan", methods=["POST"])
def api_condor_scan():
    with _condor_lock:
        if _condor_state["running"]:
            return jsonify({"ok": False, "message": "Condor scan already running"}), 409
    t = threading.Thread(target=_run_condor_scan, daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/api/condor/status")
def api_condor_status():
    with _condor_lock:
        return jsonify(_condor_state)


@app.route("/api/backtest/run", methods=["POST"])
def api_backtest_run():
    data      = request.get_json(silent=True) or {}
    hold_days = max(7, min(60, int(data.get("hold_days", 21))))
    ticker    = data.get("ticker", "").strip().upper()
    mode      = data.get("mode", "focus")   # "ticker" | "focus" | "full"
    with _bt_lock:
        if _bt_state["running"]:
            return jsonify({"ok": False, "message": "Backtest already running"}), 409
    if mode == "ticker" and ticker:
        tickers = [ticker]
    elif mode == "full":
        tickers = get_universe()
    elif mode == "signals":
        # Pull unique tickers from whatever is currently showing signals
        with scan_lock:
            tickers = list({r["ticker"]
                            for strat in scan_state["results"].values()
                            for r in strat})
        if not tickers:
            return jsonify({"ok": False, "message": "No signals found — run a scan first"}), 400
    else:
        tickers = list(dict.fromkeys(FOCUS_LIST))
    t = threading.Thread(target=_run_backtest_job, args=(tickers, hold_days), daemon=True)
    t.start()
    return jsonify({"ok": True, "tickers": len(tickers), "hold_days": hold_days})


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


# ── Walk-forward optimizer routes ─────────────────────────────
@app.route("/api/optimizer/run", methods=["POST"])
def api_optimizer_run():
    data   = request.get_json(silent=True) or {}
    mode   = data.get("mode", "focus")
    ticker = data.get("ticker", "").strip().upper()
    with _wfo_lock:
        if _wfo_state["running"]:
            return jsonify({"ok": False, "message": "Optimizer already running"}), 409
    if mode == "ticker" and ticker:
        tickers = [ticker]
    elif mode == "signals":
        with scan_lock:
            tickers = list({r["ticker"]
                            for strat in scan_state["results"].values()
                            for r in strat})
        if not tickers:
            return jsonify({"ok": False, "message": "No signals — run a scan first"}), 400
    elif mode == "full":
        tickers = get_universe()
    else:
        tickers = list(dict.fromkeys(FOCUS_LIST))
    t = threading.Thread(target=_run_wfo_job, args=(tickers,), daemon=True)
    t.start()
    return jsonify({"ok": True, "tickers": len(tickers)})


@app.route("/api/optimizer/status")
def api_optimizer_status():
    with _wfo_lock:
        s = dict(_wfo_state)
    if s.get("results"):
        s["best"]         = s["results"].get("best")
        s["leaderboard"]  = s["results"].get("leaderboard", [])[:5]
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
    global RSI_OVERSOLD, RSI_OVERBOUGHT, LIVE_REFRESH_MINS
    with _wfo_lock:
        r = _wfo_state.get("results")
    if not r or not r.get("suggestion"):
        return jsonify({"ok": False, "message": "No optimization results yet"}), 400
    s = r["suggestion"]
    applied = {}
    if "rsi_oversold"   in s:
        RSI_OVERSOLD   = s["rsi_oversold"];   applied["RSI_OVERSOLD"]   = RSI_OVERSOLD
    if "rsi_overbought" in s:
        RSI_OVERBOUGHT = s["rsi_overbought"]; applied["RSI_OVERBOUGHT"] = RSI_OVERBOUGHT
    # bull_min, bear_min, adx_threshold are used at signal-time via _bt_signals params;
    # for the live scanner they go into scan_state so scan_ticker can read them
    scan_state["wfo_params"] = {k: v for k, v in s.items()
                                if k not in ("rsi_oversold","rsi_overbought","hold_days")}
    applied.update(scan_state["wfo_params"])
    print(f"  [WFO] Applied best params: {applied}")
    return jsonify({"ok": True, "applied": applied})


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
    return jsonify(load_queue())


@app.route("/api/queue", methods=["POST"])
def api_queue_add():
    """
    Add a trade to the queue.
    Body: { ticker, strategy, option, qty, name, price }
    Returns guardrail info so the UI can warn the user immediately.
    """
    data  = request.json or {}
    trade = {
        "id":         str(uuid.uuid4())[:8],
        "queued_at":  datetime.now().isoformat(),
        "status":     "pending",    # pending | executing | done | cancelled
        "ticker":     data.get("ticker", ""),
        "name":       data.get("name", ""),
        "price":      data.get("price", 0),
        "strategy":   data.get("strategy", ""),
        "strategy_label": data.get("strategy_label", ""),
        "option":     data.get("option", {}),
        "qty":        int(data.get("qty", 1)),
        "put_width":  data.get("put_width",  0),
        "call_width": data.get("call_width", 0),
        "legs":       data.get("legs",       None),
        "entry_price": data.get("entry_price", 0),
    }

    passes, cost, msg = guardrail_check(trade)
    trade["estimated_cost"] = round(cost, 2)
    trade["guardrail_ok"]   = passes

    q = load_queue()
    q.append(trade)
    save_queue(q)
    return jsonify({"ok": True, "trade": trade, "guardrail": {"passes": passes, "cost": cost, "message": msg}})


@app.route("/api/queue/<trade_id>", methods=["DELETE"])
def api_queue_remove(trade_id):
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
    q     = load_queue()
    trade = next((t for t in q if t["id"] == trade_id), None)
    if not trade:
        return jsonify({"ok": False, "message": "Trade not found in queue"}), 404

    if trade["status"] == "executing":
        return jsonify({"ok": False, "message": "Already executing"}), 409

    # ── Guardrail check ───────────────────────────────────────
    passes, cost, msg = guardrail_check(trade)
    if not passes:
        body     = request.json or {}
        override = str(body.get("override_confirm", "")).strip().upper()
        if override != "CONFIRM":
            return jsonify({"ok": False, "guardrail_fail": True, "cost": cost, "message": msg}), 403

    # ── Mark as executing ─────────────────────────────────────
    trade["status"] = "executing"
    save_queue(q)

    # ── Hand off to SoFi automation ───────────────────────────
    try:
        from sofi_trader import place_order_on_sofi
        result = place_order_on_sofi(trade, auto_submit=AUTO_SUBMIT)
    except ImportError:
        result = {"ok": False, "message":
                  "sofi_trader.py not found. Make sure it is in the same folder."}
    except Exception as e:
        result = {"ok": False, "message": str(e)}

    # ── Update queue + history ────────────────────────────────
    trade["status"]     = "done" if result.get("ok") else "error"
    trade["executed_at"]= datetime.now().isoformat()
    trade["result"]     = result.get("message", "")
    save_queue(q)

    if result.get("ok"):
        h = load_history()
        h.append(trade)
        save_history(h)
        q = [t for t in q if t["id"] != trade_id]
        save_queue(q)

    return jsonify({"ok": result.get("ok"), "message": result.get("message", "")})


@app.route("/api/history")
def api_history():
    return jsonify(load_history())


@app.route("/api/history/perf")
def api_history_perf():
    """
    Fetch current option mid prices for all history entries in parallel.
    Returns { trade_id: { entry_mid, current_mid, pct_change, dollar_change } }
    """
    history = load_history()
    perf    = {}

    def _fetch(t):
        tid = t.get("id")
        if not tid:
            return
        try:
            opt          = t.get("option", {})
            exp          = opt.get("expiration")
            strike       = opt.get("strike")
            otype        = opt.get("type", "call")
            entry_price  = float(t.get("entry_price") or opt.get("mid") or opt.get("ask") or 0)
            qty          = int(t.get("qty", 1))
            if not exp or not strike or entry_price <= 0:
                return
            tkr   = yf.Ticker(t["ticker"])
            chain = _yf_call(tkr.option_chain, exp)
            opts  = chain.calls if otype == "call" else chain.puts
            row   = opts[abs(opts["strike"] - float(strike)) < 0.01]
            if row.empty:
                return
            bid = float(row.iloc[0].get("bid") or 0)
            ask = float(row.iloc[0].get("ask") or 0)
            current_mid = (bid + ask) / 2
            if current_mid <= 0:
                return
            pct    = (current_mid - entry_price) / entry_price * 100
            dollar = (current_mid - entry_price) * 100 * qty
            perf[tid] = {
                "entry_mid":    round(entry_price,  3),
                "current_mid":  round(current_mid,  3),
                "pct_change":   round(pct,          1),
                "dollar_change":round(dollar,       2),
            }
        except Exception:
            pass

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
        import requests, xml.etree.ElementTree as ET
        url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"
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
    sym = (request.json or {}).get("ticker", "").strip().upper()
    if not sym:
        return jsonify({"ok": False, "message": "No ticker provided"}), 400
    raw     = scan_ticker(sym)
    signals = raw.get("_signals", {}) if raw else {}
    if not signals:
        return jsonify({"ok": False, "message": f"No signals found for {sym} — it may not meet any strategy criteria right now."})
    return jsonify({"ok": True, "ticker": sym, "signals": signals})


@app.route("/api/track", methods=["POST"])
def api_track_add():
    """Add a signal to the tracked positions list."""
    data = request.json or {}
    position = {
        "id":           str(uuid.uuid4())[:8],
        "tracked_at":   datetime.now().isoformat(),
        "ticker":       data.get("ticker", ""),
        "name":         data.get("name", ""),
        "price":        data.get("price", 0),
        "strategy":     data.get("strategy", ""),
        "strategy_label": data.get("strategy_label", ""),
        "option":       data.get("option", {}),
        "legs":         data.get("legs", None),
        "net_credit":   data.get("net_credit", 0),
        "entry_price":  data.get("entry_price", 0),   # option mid at time of tracking
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


@app.route("/api/tracked", methods=["GET"])
def api_tracked_get():
    """Return all tracked positions with their latest sell signal."""
    with track_lock:
        positions = load_tracked()
    return jsonify(positions)


@app.route("/api/track/<position_id>", methods=["DELETE"])
def api_track_remove(position_id):
    """Remove a position from tracking."""
    with track_lock:
        positions = [p for p in load_tracked() if p["id"] != position_id]
        save_tracked(positions)
    return jsonify({"ok": True})


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
  <span class="tagline">⚡ Focus: ~80 liquid names · 📊 Full: S&amp;P 500 (~503) · 4 strategies · SoFi execution</span>
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
    <b>📍 Tracked Positions</b> — The bot checks these every 10 minutes and flags sell signals.
    A signal fires when: profit target hit, stop loss hit, RSI reverses through the threshold, or Bollinger Band signal fires.
  </div>
  <div id="tracked-content"><div class="queue-empty">No tracked positions yet. Click "Track" on any signal card to start monitoring it.</div></div>
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
          <option value="focus">Focus List (~80 tickers)</option>
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
          <option value="14">14 days</option>
          <option value="21" selected>21 days</option>
          <option value="30">30 days</option>
          <option value="45">45 days</option>
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
      <b>162 parameter combinations</b>. Trains on the first 75% of history, validates on the last 25%
      to prevent overfitting. Best settings can be applied to the live scanner.
    </p>
    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end;margin-bottom:16px">
      <div>
        <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">UNIVERSE</label>
        <select id="wfo-mode"
          style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;
                 padding:7px 12px;color:var(--text);font-size:13px;outline:none">
          <option value="signals">⚡ Current Signals (fastest)</option>
          <option value="focus">Focus List (~80 tickers)</option>
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

<script>
// ── State ──────────────────────────────────────────────────────
let activeTab  = 'sell_puts';
let pendingExec = null;
let scanPoll   = null;
let resultPoll = null;
const chartRegistry = {};

// ── Tab switching ─────────────────────────────────────────────
function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach((el,i)=>{
    el.classList.toggle('active',
      ['sell_puts','buy_calls','buy_puts','covered_calls','iron_condors','tracked','queue','history','backtest'][i]===tab);
  });
  document.querySelectorAll('.panel').forEach(el=> el.classList.remove('active'));
  document.getElementById('panel-'+tab).classList.add('active');
  if (tab==='queue')   refreshQueue();
  if (tab==='history') refreshHistory();
  if (tab==='tracked') refreshTracked();
}

// ── Scan mode toggle ─────────────────────────────────────────
let currentScanMode = 'focus';
function setScanMode(mode) {
  currentScanMode = mode;
  document.getElementById('mode-focus').style.background = mode==='focus' ? 'var(--accent)' : 'var(--surface2)';
  document.getElementById('mode-focus').style.color      = mode==='focus' ? '#fff' : 'var(--muted)';
  document.getElementById('mode-full').style.background  = mode==='full'  ? 'var(--accent)' : 'var(--surface2)';
  document.getElementById('mode-full').style.color       = mode==='full'  ? '#fff' : 'var(--muted)';
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
      info.textContent = data.last_scan
        ? `Last scan: ${new Date(data.last_scan).toLocaleTimeString()}`
        : 'Not yet scanned';
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
let livePoll = null;

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
    // Reuse the standard card builder by temporarily loading into the panel
    const tempDiv = document.createElement('div');
    data.forEach(item => {
      const card = buildCard(item, 'iron_condors');
      tempDiv.appendChild(card);
    });
    container.innerHTML = '';
    container.appendChild(tempDiv);
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

  document.getElementById('cnt-'+strategy).textContent = data.length;
  const panel = document.getElementById('panel-'+strategy);
  if (!data.length) {
    panel.innerHTML = `<div class="empty-state"><h3>No signals yet</h3><p>Run a scan to populate results.</p></div>`;
    return;
  }

  let cards = '';
  for (const d of data) {
    try { cards += buildCard(d, strategy); }
    catch(e) { console.error('buildCard error', d.ticker, e); }
  }

  panel.innerHTML = `<div class="disclaimer">
    <b>⚠ Not financial advice.</b> These are technical signals only.
    Options trading involves significant risk. Always do your own research.
    You must manually add trades to queue and approve each execution.
  </div>
  <div class="cards">${cards}</div>`;
  try { renderCharts(data); } catch(e) { console.error('renderCharts error', e); }

  // Async: fetch news for each card and inject into its placeholder
  for (const d of data) {
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
      <div class="ticker">${d.ticker}</div>
      <div class="cname">${d.name}</div>
    </div>
    <div style="text-align:right;flex:1">
      <div class="cprice" style="color:${sc}">$${d.price.toFixed(3)}</div>
      <div class="csector">${d.sector}</div>
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
    </div>
  </div>`}

  ${buildBSSection(d)}

  ${buildNewsSection(d, 'news-'+id)}

  ${buildMCSection(d, id)}

  <div class="card-footer">
    <div class="qty-wrap">
      Contracts:
      <input type="number" min="1" max="100" value="1" id="qty-${id}"
        oninput="updateCost('${id}',${isCondor?d.net_credit||0:opt.ask||0},${opt.strike||0},'${sid}',${d.put_width||0},${d.call_width||0})"/>
    </div>
    <div class="cost-est" id="cost-${id}">Est. capital: <span class="cost-val">${calcCostDisplay(1, isCondor?d.net_credit||0:opt.ask||0, opt.strike||0, sid, d.put_width||0, d.call_width||0)}</span></div>
    <button class="btn btn-primary" onclick="addToQueue('${d.ticker}','${d.name}',${d.price},'${sid}s','${d.strategy_label||action}','${id}')">
      + Add to Queue
    </button>
    <button class="btn btn-ghost" style="margin-left:6px" onclick="trackPosition('${d.ticker}','${d.name}',${d.price},'${sid}','${d.strategy_label||action}','${id}')">
      📍 Track
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
              <a href="${a.link}" target="_blank"
                 style="font-size:12px;color:var(--text);text-decoration:none;
                        display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                 title="${a.title}">${a.title}</a>
              <span style="font-size:10px;color:var(--muted)">${a.publisher}${ts?' · '+ts:''}</span>
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
  if (!bs) return '';

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
  if (!mc) return '';

  const pop     = mc.pop;
  const ev      = mc.ev;
  const be      = mc.breakeven;
  const maxP    = mc.max_profit;
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
async function addToQueue(ticker, name, price, strategy, stratLabel, id) {
  const qty = parseInt(document.getElementById('qty-'+id)?.value)||1;
  // Build option object from the card's data — we'll read it from /api/results
  const results = await fetch('/api/results/'+strategy).then(r=>r.json());
  const signal  = results.find(d=>d.ticker===ticker);
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
  const resp = await fetch('/api/queue',{method:'POST',headers:{'Content-Type':'application/json'},
                                          body:JSON.stringify(body)}).then(r=>r.json());

  const trade = resp.trade;
  const g     = resp.guardrail;

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
  const results = await fetch('/api/results/'+strategy+'s').then(r=>r.json());
  const signal  = results.find(d=>d.ticker===ticker);
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
    entry_price:     entryPrice,
  };
  const resp = await fetch('/api/track',{method:'POST',headers:{'Content-Type':'application/json'},
                                          body:JSON.stringify(body)}).then(r=>r.json());
  if(resp.ok){
    alert(`📍 Now tracking ${ticker} ${stratLabel}. The bot will check for sell signals every 10 minutes.`);
    document.getElementById('cnt-tracked').textContent =
      parseInt(document.getElementById('cnt-tracked').textContent||0)+1;
  } else {
    alert(resp.message || 'Could not add to tracked positions.');
  }
}

async function refreshTracked() {
  const positions = await fetch('/api/tracked').then(r=>r.json());
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
    const entryMid   = p.entry_price ? +p.entry_price : null;
    const currentMid = sig.current_option_mid != null ? +sig.current_option_mid : null;
    const entryStr   = entryMid   != null ? '$'+entryMid.toFixed(3)   : '—';
    const currentStr = currentMid != null ? '$'+currentMid.toFixed(3) : '—';

    // ── Performance calculation ─────────────────────────────
    let perfHtml = '<span style="color:var(--muted);font-size:12px">Waiting…</span>';
    if (entryMid != null && currentMid != null) {
      const pct    = sig.pct_change    != null ? sig.pct_change    : (currentMid - entryMid) / entryMid * 100;
      const dollar = sig.dollar_change != null ? sig.dollar_change : (currentMid - entryMid) * 100;
      const color  = pct >= 0 ? 'var(--green)' : 'var(--red)';
      const arrow  = pct >= 0 ? '▲' : '▼';
      const sign   = pct >= 0 ? '+' : '';
      perfHtml = `
        <div style="display:flex;flex-direction:column;gap:2px">
          <span style="font-size:13px;font-weight:700;color:${color}">${arrow} ${sign}${pct.toFixed(1)}%</span>
          <span style="font-size:11px;color:${color}">${sign}$${dollar.toFixed(2)} / contract</span>
          <span style="font-size:10px;color:var(--muted)">${entryStr} → ${currentStr}</span>
        </div>`;
    } else if (entryMid != null) {
      perfHtml = `<span style="font-size:11px;color:var(--muted)">Entry: ${entryStr}<br>Awaiting price…</span>`;
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

    // ── Tracked-since ───────────────────────────────────────
    const trackedAt = p.tracked_at ? new Date(p.tracked_at).toLocaleDateString() : '—';

    return `<tr>
      <td>
        <b>${p.ticker}</b><br>
        <span style="font-size:11px;color:var(--muted)">${p.name||''}</span><br>
        <span style="font-size:10px;color:var(--muted)">Since ${trackedAt}</span>
      </td>
      <td>${p.strategy_label||p.strategy}</td>
      <td style="font-size:12px">
        ${opt.type||''} $${opt.strike?.toFixed(2)||''}<br>
        <span style="color:var(--muted)">${opt.expiration||''}</span>
      </td>
      <td>${perfHtml}</td>
      <td>
        ${cprice}<br>
        <span style="font-size:11px;color:var(--muted)">RSI: ${rsi}</span>
      </td>
      <td>
        ${signalBadge}
        ${fired&&reasons?`<br><span style="font-size:11px;color:var(--muted)">${reasons}</span>`:''}
      </td>
      <td style="font-size:11px;color:var(--muted)">${chk}</td>
      <td>
        <button class="btn btn-ghost" style="font-size:12px;padding:5px 10px"
          onclick="untrackPosition('${p.id}')">✕ Remove</button>
      </td>
    </tr>`;
  });

  el.innerHTML = `<table class="queue-table">
    <thead><tr>
      <th>Ticker</th><th>Strategy</th><th>Option</th>
      <th>Performance</th><th>Stock Price</th>
      <th>Signal</th><th>Last Check</th><th>Actions</th>
    </tr></thead>
    <tbody>${rows.join('')}</tbody>
  </table>`;
}

async function untrackPosition(posId) {
  if(!confirm('Stop tracking this position?')) return;
  await fetch('/api/track/'+posId,{method:'DELETE'});
  refreshTracked();
}

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

  const resp = await fetch('/api/search', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ticker: sym})
  }).then(r => r.json());

  if (!resp.ok) {
    titleEl.textContent = sym;
    resultsEl.innerHTML = `<div style="color:var(--red);padding:16px 0">${resp.message}</div>`;
    return;
  }

  const signals = resp.signals;
  const count   = Object.keys(signals).length;
  titleEl.textContent = `${sym} — ${count} signal${count !== 1 ? 's' : ''} found`;

  const keyMap = {
    sell_put: 'sell_puts', buy_call: 'buy_calls',
    buy_put: 'buy_puts', covered_call: 'covered_calls', iron_condor: 'iron_condors'
  };
  let html = '';
  for (const [key, d] of Object.entries(signals)) {
    try { html += buildCard(d, keyMap[key] || key); }
    catch(e) { console.error('buildCard error', key, e); }
  }
  resultsEl.innerHTML = html || '<div style="color:var(--muted)">No cards to display.</div>';
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
      <td><b>${t.ticker}</b><br><span style="font-size:11px;color:var(--muted)">${t.name||''}</span></td>
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

  document.getElementById('modal-btn-confirm').disabled=true;
  document.getElementById('modal-btn-confirm').innerHTML='<span class="spinner"></span>Launching SoFi…';

  const resp = await fetch('/api/execute/'+t.id,{
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
  }).then(r=>r.json());

  closeModal();
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
  if (!r.ok) { alert('Backtest already running'); btn.disabled=false; btn.textContent='▶ Run Backtest'; return; }
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

  // Equity curve on canvas
  const canvas = document.getElementById('bt-equity-chart');
  const ctx    = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * window.devicePixelRatio;
  canvas.height = 120 * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  const W = canvas.offsetWidth, H = 120;
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
  document.getElementById('bt-trade-count').textContent = `(${trades.length} shown)`;
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
        document.getElementById('wfo-status').textContent = d.stage || 'Complete';
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
        <td style="padding:5px 8px">${p2.rsi_oversold}</td>
        <td style="padding:5px 8px">${p2.rsi_overbought}</td>
        <td style="padding:5px 8px">${p2.bull_min}</td>
        <td style="padding:5px 8px">${p2.bear_min}</td>
        <td style="padding:5px 8px">${p2.adx_threshold}</td>
        <td style="padding:5px 8px">${p2.hold_days}d</td>
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
        <span style="font-size:11px;color:var(--muted)">${t.name||''}</span>
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

// ── Init ──────────────────────────────────────────────────────
window.onload = async () => {
  const s = await fetch('/api/scan/status').then(r=>r.json());
  if(s.last_scan) {
    document.getElementById('scan-info').textContent = `Last scan: ${new Date(s.last_scan).toLocaleTimeString()}`;
    ['sell_puts','buy_calls','buy_puts','covered_calls'].forEach(loadStrategy);
  }
  refreshQueue();
  // Load tracked count on startup
  const tracked = await fetch('/api/tracked').then(r=>r.json());
  document.getElementById('cnt-tracked').textContent = tracked.length;
  // Auto-refresh tracked tab every 60s if it's active
  setInterval(()=>{ if(activeTab==='tracked') refreshTracked(); }, 60000);
};
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 56)
    print("  OPTIONS BOT v2.0")
    print("═" * 56)
    print(f"  Guardrail   : ${GUARDRAIL_LIMIT:,.0f} per trade")
    print(f"  Auto-submit : {'ENABLED ⚠' if AUTO_SUBMIT else 'DISABLED (safe mode)'}")
    print(f"  Dashboard   : http://localhost:{SERVER_PORT}")
    print("═" * 56 + "\n")
    print("  Starting web server…")

    # Start sell-signal tracker in the background
    threading.Thread(target=_schedule_track_checker, daemon=True).start()

    # Open browser after short delay so Flask has time to start
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{SERVER_PORT}")).start()

    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
