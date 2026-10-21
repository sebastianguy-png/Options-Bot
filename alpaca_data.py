"""
Alpaca market data for the scanner.

Why: Yahoo blanks every option bid/ask outside 09:30–16:00 ET, so pre-market
research shows no quotes.  Alpaca keeps the LAST quote of the session (plus the
daily bar, last trade and open interest) around the clock, and its IEX stock
feed carries pre-market trades.  Both come free with any Alpaca account's API
keys — the paper keys in autotrade_config.json are enough.

options_bot wraps this behind _option_expirations() / _option_chain(): when
the feed is configured and answers, chains come from here in the same shape
yfinance returns (an object with .calls / .puts DataFrames and the columns
pick_option reads); otherwise everything falls back to Yahoo.

Implied volatility: the free "indicative" options feed does not always carry
IV, so the caller passes a Black-Scholes solver and the chain fills it from
the mid price and the underlying's last trade.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, date, timedelta, timezone
from typing import Callable, Optional

import pandas as pd
import requests

from broker import parse_occ

DATA_URL  = "https://data.alpaca.markets"
TRADE_URL = {True: "https://paper-api.alpaca.markets", False: "https://api.alpaca.markets"}
CHAIN_COLUMNS = ["contractSymbol", "strike", "bid", "ask", "lastPrice", "volume", "openInterest",
                 "impliedVolatility", "inTheMoney", "lastTradeDate", "quoteTime", "expiration", "type"]


class DataError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class Chain:
    """What yfinance's option_chain() returns, as far as the bot reads it."""
    __slots__ = ("calls", "puts", "expiration", "underlying_price", "source")

    def __init__(self, calls, puts, expiration, underlying_price, source="alpaca"):
        self.calls, self.puts = calls, puts
        self.expiration, self.underlying_price, self.source = expiration, underlying_price, source


class AlpacaData:
    def __init__(self, key_id: str, secret: str, paper: bool = True, feed: str = "indicative",
                 iv_solver: Optional[Callable] = None, risk_free: Callable[[], float] = lambda: 0.04,
                 timeout: int = 20, ttl_open: int = 300, ttl_closed: int = 1800, max_dte: int = 60):
        if not key_id or not secret:
            raise DataError("Alpaca key_id / secret missing")
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret, "Accept": "application/json"})
        self.trade_base = TRADE_URL[bool(paper)]
        self.feed, self.timeout = feed, timeout
        self.iv_solver, self.risk_free = iv_solver, risk_free
        self.ttl_open, self.ttl_closed, self.max_dte = ttl_open, ttl_closed, max_dte
        self._cache: dict = {}            # sym → {"ts", "contracts", "snapshots", "exps", "price", "price_ts"}
        self._lock = threading.Lock()

    # ── HTTP ──
    def _get(self, url: str, params: dict | None = None) -> dict:
        for attempt in range(3):
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == 2:
                    raise DataError(f"GET {url}: {e}")
                time.sleep(1 + attempt)
                continue
            if r.status_code == 429 and attempt < 2:
                time.sleep(2 + 2 * attempt)
                continue
            if r.status_code >= 400:
                raise DataError(f"GET {url} → {r.status_code}: {r.text[:200]}", status=r.status_code)
            return r.json() if r.text else {}
        raise DataError(f"GET {url}: retries exhausted")

    def probe(self) -> dict:
        """One cheap authenticated call; raises DataError when the keys or the feed don't work."""
        return self._get(f"{DATA_URL}/v2/stocks/SPY/trades/latest", {"feed": "iex"})

    # ── Market clock (regular session) ──
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

    # ── Per-symbol load (contracts + snapshots, paginated, cached) ──
    def _load(self, sym: str, force: bool = False) -> dict:
        sym = sym.upper()
        ttl = self.ttl_open if self._session_open() else self.ttl_closed
        with self._lock:
            c = self._cache.get(sym)
            if c and not force and time.time() - c["ts"] < ttl:
                return c
        today = date.today()
        gte, lte = today.isoformat(), (today + timedelta(days=self.max_dte)).isoformat()
        # contracts: strike / type / expiration / open interest
        contracts, token = {}, None
        while True:
            params = {"underlying_symbols": sym, "status": "active", "expiration_date_gte": gte,
                      "expiration_date_lte": lte, "limit": 10000}
            if token:
                params["page_token"] = token
            r = self._get(f"{self.trade_base}/v2/options/contracts", params)
            for k in r.get("option_contracts") or []:
                contracts[k["symbol"]] = {"strike": float(k["strike_price"]), "type": k["type"],
                                          "expiration": k["expiration_date"],
                                          "open_interest": int(float(k["open_interest"])) if k.get("open_interest") not in (None, "") else 0,
                                          "close_price": float(k["close_price"]) if k.get("close_price") not in (None, "") else None}
            token = r.get("next_page_token")
            if not token:
                break
        # snapshots: latest quote / trade / daily bar (and IV when the feed carries it)
        snaps, token = {}, None
        while True:
            params = {"feed": self.feed, "limit": 1000, "expiration_date_gte": gte, "expiration_date_lte": lte}
            if token:
                params["page_token"] = token
            r = self._get(f"{DATA_URL}/v1beta1/options/snapshots/{sym}", params)
            snaps.update(r.get("snapshots") or {})
            token = r.get("next_page_token")
            if not token:
                break
        price, price_ts = self.stock_last(sym, cached=False)
        entry = {"ts": time.time(), "contracts": contracts, "snapshots": snaps,
                 "exps": sorted({v["expiration"] for v in contracts.values()}), "price": price, "price_ts": price_ts}
        with self._lock:
            self._cache[sym] = entry
        return entry

    # ── Public ──
    def expirations(self, sym: str) -> list:
        return list(self._load(sym)["exps"])

    def stock_last(self, sym: str, cached: bool = True):
        """(price, timestamp) of the latest IEX trade — carries pre-market and after-hours prints."""
        if cached:
            with self._lock:
                c = self._cache.get(sym.upper())
            if c and c.get("price") and time.time() - c["ts"] < 120:
                return c["price"], c["price_ts"]
        r = self._get(f"{DATA_URL}/v2/stocks/{sym.upper()}/trades/latest", {"feed": "iex"})
        t = r.get("trade") or {}
        price = float(t.get("p") or 0) or None
        ts = None
        if t.get("t"):
            try:
                ts = datetime.fromisoformat(str(t["t"]).replace("Z", "+00:00")[:26] + "+00:00") if "+" not in str(t["t"]) else datetime.fromisoformat(str(t["t"]))
            except ValueError:
                ts = None
        return price, ts

    def chain(self, sym: str, exp: str) -> Chain:
        """The contracts expiring `exp`, as calls / puts DataFrames with yfinance's column names."""
        c = self._load(sym)
        S = c.get("price") or 0.0
        exp = str(exp)[:10]
        T = max((date.fromisoformat(exp) - date.today()).days, 0.5) / 365.0
        r = float(self.risk_free() or 0.04)
        rows = {"call": [], "put": []}
        for occ, meta in c["contracts"].items():
            if meta["expiration"] != exp:
                continue
            snap = c["snapshots"].get(occ) or {}
            q, t, d = snap.get("latestQuote") or {}, snap.get("latestTrade") or {}, snap.get("dailyBar") or {}
            bid, ask = float(q.get("bp") or 0), float(q.get("ap") or 0)
            last = float(t.get("p") or 0) or (meta.get("close_price") or 0.0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else (ask or bid or last)
            iv = snap.get("impliedVolatility")
            if (iv is None or not iv) and self.iv_solver and S > 0 and mid > 0:
                iv = self.iv_solver(mid, S, meta["strike"], T, r, meta["type"])
            rows[meta["type"]].append({
                "contractSymbol": occ, "strike": meta["strike"], "bid": bid, "ask": ask, "lastPrice": last,
                "volume": int(d.get("v") or 0), "openInterest": int(meta.get("open_interest") or 0),
                "impliedVolatility": float(iv) if iv else float("nan"),
                "inTheMoney": (S > meta["strike"]) if meta["type"] == "call" else (S < meta["strike"]) if S else False,
                "lastTradeDate": t.get("t"), "quoteTime": q.get("t"), "expiration": exp, "type": meta["type"],
            })
        calls = pd.DataFrame(rows["call"], columns=CHAIN_COLUMNS).sort_values("strike").reset_index(drop=True)
        puts  = pd.DataFrame(rows["put"],  columns=CHAIN_COLUMNS).sort_values("strike").reset_index(drop=True)
        return Chain(calls, puts, exp, S)

    def quote(self, occ: str) -> Optional[dict]:
        """Latest quote for one contract (from the symbol's cached snapshot, loading it if needed)."""
        try:
            sym = parse_occ(occ)["ticker"]
        except Exception:
            return None
        snap = (self._load(sym)["snapshots"] or {}).get(occ)
        if not snap:
            return None
        q = snap.get("latestQuote") or {}
        bid, ask = float(q.get("bp") or 0), float(q.get("ap") or 0)
        return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else (ask or bid or None),
                "time": q.get("t")}
