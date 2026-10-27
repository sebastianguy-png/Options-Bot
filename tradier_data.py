"""
Tradier market data for the scanner (and, later, the broker).

Why: the Tradier brokerage account (Oct 27 2026) comes with real-time OPRA
quotes and full option chains carrying ORATS implied vols and Greeks - better
than Alpaca's free indicative feed (quotes only, IV solved locally) and Yahoo
(blank bid/ask off-hours).  options_bot wraps this behind _market_data():
Tradier when the token is in autotrade_config.json and a probe answers, else
Alpaca, else Yahoo.  Chains come back in the same shape yfinance returns (an
object with .calls / .puts DataFrames and the columns pick_option reads), plus
the Greeks columns the daily chain recorder keeps.

Only market-data endpoints live here.  Nothing in this module can place an
order; the trading endpoints get their own adapter in broker.py once the
live gate passes.

Rate limit: 120 requests / minute on market data (X-Ratelimit-* headers);
the client sleeps when the window is nearly used up.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, date, timedelta, timezone
from typing import Callable, Optional

import pandas as pd
import requests

from alpaca_data import Chain, CHAIN_COLUMNS, DataError

BASE = {False: "https://api.tradier.com/v1", True: "https://sandbox.tradier.com/v1"}
GREEK_COLUMNS = ["delta", "gamma", "theta", "vega", "bid_iv", "ask_iv", "mid_iv", "smv_vol", "greeks_updated"]


def _ms_to_dt(ms):
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc) if ms else None
    except (TypeError, ValueError, OSError):
        return None


def _as_list(x):
    """Tradier returns a dict for one element, a list for several, None for none."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


class TradierData:
    name = "tradier"

    def __init__(self, token: str, sandbox: bool = False, iv_solver: Optional[Callable] = None,
                 risk_free: Callable[[], float] = lambda: 0.04, timeout: int = 20,
                 ttl_open: int = 120, ttl_closed: int = 1800):
        if not token:
            raise DataError("Tradier token missing")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})
        self.base = BASE[bool(sandbox)]
        self.sandbox = bool(sandbox)
        self.timeout = timeout
        self.iv_solver, self.risk_free = iv_solver, risk_free
        self.ttl_open, self.ttl_closed = ttl_open, ttl_closed
        self._cache: dict = {}                      # key -> {"ts", "val"}
        self._lock = threading.Lock()
        self._rl = {"available": None, "expiry": 0.0}

    # -- HTTP --
    def _throttle(self):
        with self._lock:
            avail, exp = self._rl["available"], self._rl["expiry"]
        if avail is not None and avail <= 3 and exp > time.time():
            time.sleep(min(exp - time.time() + 0.2, 65))

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = self.base + path
        for attempt in range(3):
            self._throttle()
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == 2:
                    raise DataError(f"GET {path}: {e}")
                time.sleep(1 + attempt)
                continue
            try:
                avail, exp = r.headers.get("X-Ratelimit-Available"), r.headers.get("X-Ratelimit-Expiry")
                if avail is not None:
                    with self._lock:
                        self._rl["available"] = int(avail)
                        self._rl["expiry"] = int(exp) / 1000 if exp else 0.0
            except (TypeError, ValueError):
                pass
            if r.status_code == 429 and attempt < 2:
                time.sleep(3 + 3 * attempt)
                continue
            if r.status_code >= 400:
                raise DataError(f"GET {path} -> {r.status_code}: {r.text[:200]}", status=r.status_code)
            try:
                return r.json() if r.text else {}
            except ValueError:
                raise DataError(f"GET {path}: non-JSON reply {r.text[:120]!r}")
        raise DataError(f"GET {path}: retries exhausted")

    def _cached(self, key, loader):
        ttl = self.ttl_open if self._session_open() else self.ttl_closed
        with self._lock:
            c = self._cache.get(key)
            if c and time.time() - c["ts"] < ttl:
                return c["val"]
        val = loader()
        with self._lock:
            self._cache[key] = {"ts": time.time(), "val": val}
        return val

    def probe(self) -> dict:
        """One cheap authenticated call; raises DataError when the token doesn't work."""
        return self._get("/markets/quotes", {"symbols": "SPY"})

    # -- Market clock --
    @staticmethod
    def _eastern_now() -> datetime:
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            return datetime.now(timezone.utc) - timedelta(hours=5)

    def _session_open(self) -> bool:
        now = self._eastern_now()
        return now.weekday() < 5 and (9 * 60 + 30) <= now.hour * 60 + now.minute < 16 * 60

    def clock(self) -> dict:
        return (self._get("/markets/clock") or {}).get("clock") or {}

    # -- Quotes --
    def quotes(self, symbols, greeks: bool = False) -> dict:
        """{symbol: quote dict} for stocks and/or OCC option symbols (one request per <=100 symbols)."""
        syms = [s for s in ([symbols] if isinstance(symbols, str) else list(symbols)) if s]
        out = {}
        for i in range(0, len(syms), 100):
            r = self._get("/markets/quotes", {"symbols": ",".join(syms[i:i + 100]), "greeks": "true" if greeks else "false"})
            for q in _as_list(((r or {}).get("quotes") or {}).get("quote")):
                if q.get("symbol"):
                    out[q["symbol"]] = q
        return out

    def stock_last(self, sym: str, cached: bool = True):
        """(price, timestamp) of the latest trade."""
        sym = sym.upper()
        key = ("last", sym)
        if cached:
            with self._lock:
                c = self._cache.get(key)
            if c and time.time() - c["ts"] < 60:
                return c["val"]
        q = self.quotes([sym]).get(sym) or {}
        price = float(q.get("last") or 0) or None
        val = (price, _ms_to_dt(q.get("trade_date")))
        with self._lock:
            self._cache[key] = {"ts": time.time(), "val": val}
        return val

    def quote(self, occ: str) -> Optional[dict]:
        """Latest NBBO for one contract."""
        q = self.quotes([occ]).get(occ)
        if not q:
            return None
        bid, ask = float(q.get("bid") or 0), float(q.get("ask") or 0)
        return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else (ask or bid or None),
                "time": _ms_to_dt(q.get("ask_date") or q.get("bid_date")), "last": float(q.get("last") or 0) or None}

    # -- Options --
    def expirations(self, sym: str) -> list:
        sym = sym.upper()

        def load():
            r = self._get("/markets/options/expirations", {"symbol": sym, "includeAllRoots": "true", "strikes": "false"})
            return sorted(str(d)[:10] for d in _as_list(((r or {}).get("expirations") or {}).get("date")))
        return list(self._cached(("exps", sym), load))

    def chain_raw(self, sym: str, exp: str) -> list:
        """The chain as Tradier sends it (list of dicts with a 'greeks' dict), cached."""
        sym, exp = sym.upper(), str(exp)[:10]

        def load():
            r = self._get("/markets/options/chains", {"symbol": sym, "expiration": exp, "greeks": "true"})
            return _as_list(((r or {}).get("options") or {}).get("option"))
        return self._cached(("chain", sym, exp), load)

    def chain(self, sym: str, exp: str) -> Chain:
        """One expiry as calls / puts DataFrames with yfinance's column names plus the Greeks."""
        sym, exp = sym.upper(), str(exp)[:10]
        raw = self.chain_raw(sym, exp)
        S, _ = self.stock_last(sym)
        S = float(S or 0.0)
        T = max((date.fromisoformat(exp) - date.today()).days, 0.5) / 365.0
        rf = float(self.risk_free() or 0.04)
        rows = {"call": [], "put": []}
        for o in raw:
            typ = o.get("option_type")
            if typ not in ("call", "put"):
                continue
            g = o.get("greeks") or {}
            bid, ask = float(o.get("bid") or 0), float(o.get("ask") or 0)
            last = float(o.get("last") or 0) or float(o.get("close") or 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else (ask or bid or last)
            iv = g.get("mid_iv") or g.get("smv_vol")
            if (not iv) and self.iv_solver and S > 0 and mid > 0:
                iv = self.iv_solver(mid, S, float(o["strike"]), T, rf, typ)
            K = float(o["strike"])
            rows[typ].append({
                "contractSymbol": o.get("symbol"), "strike": K, "bid": bid, "ask": ask, "lastPrice": last,
                "volume": int(o.get("volume") or 0), "openInterest": int(o.get("open_interest") or 0),
                "impliedVolatility": float(iv) if iv else float("nan"),
                "inTheMoney": (S > K) if typ == "call" else (S < K) if S else False,
                "lastTradeDate": _ms_to_dt(o.get("trade_date")), "quoteTime": _ms_to_dt(o.get("ask_date") or o.get("bid_date")),
                "expiration": exp, "type": typ,
                "delta": g.get("delta"), "gamma": g.get("gamma"), "theta": g.get("theta"), "vega": g.get("vega"),
                "bid_iv": g.get("bid_iv"), "ask_iv": g.get("ask_iv"), "mid_iv": g.get("mid_iv"), "smv_vol": g.get("smv_vol"),
                "greeks_updated": g.get("updated_at"),
            })
        cols = CHAIN_COLUMNS + GREEK_COLUMNS
        calls = pd.DataFrame(rows["call"], columns=cols).sort_values("strike").reset_index(drop=True)
        puts  = pd.DataFrame(rows["put"],  columns=cols).sort_values("strike").reset_index(drop=True)
        return Chain(calls, puts, exp, S, source="tradier")

    # -- History (stocks and listed option contracts) --
    def history(self, symbol: str, start: str, end: str | None = None, interval: str = "daily") -> pd.DataFrame:
        """Daily OHLCV between start and end (YYYY-MM-DD).  Works for OCC symbols while the contract is listed."""
        params = {"symbol": symbol.upper(), "interval": interval, "start": str(start)[:10],
                  "end": str(end or date.today())[:10], "session_filter": "open"}
        r = self._get("/markets/history", params)
        days = _as_list(((r or {}).get("history") or {}).get("day"))
        if not days:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame(days)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
