"""
Broker adapters for the Options Bot autotrader.

Two implementations behind one small interface:

  LocalPaperBroker  — no network at all.  Fills at the price the bot passes in
                      (the option mid from the scan / tracker) with a little
                      slippage, keeps a ledger in paper_ledger.json and reserves
                      strike × 100 of cash for every short put (cash-secured).
                      This is the default and the only mode that needs nothing
                      from you.
  AlpacaBroker      — Alpaca Markets REST API (https://docs.alpaca.markets).
                      Paper trading by default (paper-api.alpaca.markets); the
                      live endpoint is only used when the config says
                      "paper": false AND the environment variable
                      OPTIONS_BOT_ALLOW_LIVE=1 is set.  Needs an Alpaca account
                      with options trading enabled and its API key / secret.

Option contracts are addressed by their OCC symbol, e.g. AAPL251121P00340000
(ticker, YYMMDD expiry, C/P, strike × 1000 zero-padded to 8 digits).

Contract the autotrader relies on:
  submit_option_order  → an Order whose status is "filled" (paper), or an open
                         status the caller must poll with get_order()
  get_order            → None ONLY when the broker has no such order (404);
                         raises BrokerError when the broker cannot be reached,
                         so a blip is never mistaken for "order gone"
  cancel_order         → True when the cancel was accepted; a cancel can still
                         race a fill, so callers re-fetch the order afterwards

Nothing in here decides WHAT to trade — the autotrader in autotrader.py does
that; this module only knows how to place, query and close orders.
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional

try:
    import requests
except ImportError:                       # requests is already a yfinance dependency
    requests = None


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def occ_symbol(ticker: str, expiration: str, opt_type: str, strike: float) -> str:
    """OCC option symbol: ticker + YYMMDD + C/P + strike*1000 (8 digits)."""
    d = datetime.strptime(str(expiration)[:10], "%Y-%m-%d")
    cp = "C" if str(opt_type).lower().startswith("c") else "P"
    return f"{ticker.upper()}{d.strftime('%y%m%d')}{cp}{int(round(float(strike) * 1000)):08d}"


def parse_occ(symbol: str) -> dict:
    """Inverse of occ_symbol: {'ticker','expiration','type','strike'}."""
    s = symbol.strip().upper()
    strike = int(s[-8:]) / 1000.0
    cp = s[-9]
    yymmdd = s[-15:-9]
    ticker = s[:-15]
    exp = datetime.strptime(yymmdd, "%y%m%d").strftime("%Y-%m-%d")
    return {"ticker": ticker, "expiration": exp, "type": "call" if cp == "C" else "put", "strike": strike}


def option_tick(price: float, side: str) -> float:
    """
    Snap a limit price to a valid option tick: $0.01 below $3, $0.05 at or
    above (every penny-increment class accepts $0.05 steps too).  Buys round
    up and sells round down, so the order is never less aggressive than meant.
    """
    p = float(price)
    step = 0.01 if p < 3 else 0.05
    n = p / step
    n = math.ceil(n - 1e-9) if side == "buy" else math.floor(n + 1e-9)
    return round(max(n * step, step), 2)


def _atomic_write(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


@dataclass
class Order:
    id: str
    symbol: str              # OCC option symbol
    side: str                # "buy" | "sell"
    qty: int
    order_type: str          # "limit" | "market"
    limit_price: Optional[float]
    status: str              # "filled" | "accepted" | "rejected" | "canceled" | ...
    filled_qty: int = 0
    filled_avg_price: Optional[float] = None
    submitted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    message: str = ""
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


class BrokerError(Exception):
    """status = HTTP status for a rejected request, None for a transport failure."""
    def __init__(self, message, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


# ──────────────────────────────────────────────────────────────
#  Interface
# ──────────────────────────────────────────────────────────────

class Broker:
    name  = "base"
    paper = True

    def account(self) -> dict:
        """{'cash': float, 'equity': float, 'buying_power': float}  (buying_power = what options can be bought/secured with)"""
        raise NotImplementedError

    def positions(self) -> list:
        """[{'symbol': OCC, 'qty': int (negative = short), 'avg_price': float, 'market_value': float}]"""
        raise NotImplementedError

    def submit_option_order(self, symbol: str, side: str, qty: int, limit_price: Optional[float] = None,
                            order_type: str = "limit", tif: str = "day", ref_price: Optional[float] = None) -> Order:
        """side: 'buy' (open long / close short) or 'sell' (open short / close long)."""
        raise NotImplementedError

    def get_order(self, order_id: str) -> Optional[Order]:
        """None only when the broker has no such order; raises BrokerError when unreachable."""
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def is_market_open(self) -> bool:
        """Regular session only (no extended hours — options don't trade there)."""
        return _nyse_regular_session_open()

    def option_quote(self, symbol: str) -> Optional[dict]:
        """{'bid': float, 'ask': float, 'mid': float} or None when unavailable."""
        return None


# ──────────────────────────────────────────────────────────────
#  Market-hours fallback (no broker clock): NYSE regular session in US/Eastern
# ──────────────────────────────────────────────────────────────

_NYSE_HOLIDAYS = {
    # observed closures — extend yearly (source: nyse.com/markets/hours-calendars)
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


def _eastern_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now(timezone.utc) - timedelta(hours=5)     # crude fallback


def _nyse_regular_session_open(now: Optional[datetime] = None) -> bool:
    now = now or _eastern_now()
    if now.weekday() >= 5 or now.strftime("%Y-%m-%d") in _NYSE_HOLIDAYS:
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= t < 16 * 60


# ──────────────────────────────────────────────────────────────
#  Local paper broker — simulated fills, JSON ledger, no network
# ──────────────────────────────────────────────────────────────

class LocalPaperBroker(Broker):
    name  = "local_paper"
    paper = True

    def __init__(self, ledger_path: str = "paper_ledger.json", starting_cash: float = 10_000.0,
                 slippage_pct: float = 1.0):
        self.path = ledger_path
        self.slippage = slippage_pct / 100.0      # fills 1% worse than the reference price
        self.ledger = _load(self.path, None) or {
            "cash": starting_cash, "starting_cash": starting_cash, "positions": {}, "orders": [],
            "created_at": datetime.now().isoformat(),
        }

    # -- persistence
    def _save(self):
        _atomic_write(self.path, self.ledger)

    @staticmethod
    def _collateral(symbol: str, qty: int) -> float:
        """Cash reserved for a short: strike × 100 per contract (a short call is treated the same)."""
        try:
            return abs(int(qty)) * parse_occ(symbol)["strike"] * 100
        except Exception:
            return 0.0

    def _reserved(self) -> float:
        return sum(self._collateral(s, p["qty"]) for s, p in self.ledger["positions"].items() if p["qty"] < 0)

    # -- interface
    def account(self) -> dict:
        mv = sum(p["qty"] * p.get("last_price", p["avg_price"]) * 100 for p in self.ledger["positions"].values())
        cash = self.ledger["cash"]
        return {"cash": round(cash, 2), "equity": round(cash + mv, 2),
                "buying_power": round(cash - self._reserved(), 2),       # cash not already securing a short
                "collateral_reserved": round(self._reserved(), 2),
                "starting_cash": self.ledger.get("starting_cash")}

    def positions(self) -> list:
        return [{"symbol": s, "qty": p["qty"], "avg_price": p["avg_price"],
                 "market_value": round(p["qty"] * p.get("last_price", p["avg_price"]) * 100, 2)}
                for s, p in self.ledger["positions"].items() if p["qty"] != 0]

    def mark(self, symbol: str, price: float) -> None:
        """Update the mark price of an open position (the tracker's refreshed mid)."""
        p = self.ledger["positions"].get(symbol)
        if p and price and price > 0:
            p["last_price"] = float(price)
            self._save()

    def _reject(self, symbol, side, qty, order_type, limit_price, message) -> Order:
        o = Order(str(uuid.uuid4())[:8], symbol, side, qty, order_type, limit_price, "rejected", message=message)
        self.ledger["orders"].append(o.to_dict()); self._save()
        return o

    def submit_option_order(self, symbol, side, qty, limit_price=None, order_type="limit", tif="day", ref_price=None):
        qty = int(qty)
        price = ref_price if ref_price else limit_price
        if not price or price <= 0 or qty <= 0:
            return self._reject(symbol, side, qty, order_type, limit_price, "no reference price / zero qty")
        fill = round(price * (1 + self.slippage) if side == "buy" else price * (1 - self.slippage), 3)
        if limit_price and ((side == "buy" and fill > limit_price) or (side == "sell" and fill < limit_price)):
            fill = float(limit_price)                       # a limit order fills no worse than its limit
        cost = fill * 100 * qty
        pos  = self.ledger["positions"].setdefault(symbol, {"qty": 0, "avg_price": 0.0, "last_price": fill})
        if side == "buy":
            if pos["qty"] < 0:                               # closing a short
                self.ledger["cash"] -= cost
            else:
                if cost > self.ledger["cash"] - self._reserved():
                    if pos["qty"] == 0:
                        self.ledger["positions"].pop(symbol, None)
                    return self._reject(symbol, side, qty, order_type, limit_price,
                                        f"insufficient buying power ({self.ledger['cash'] - self._reserved():.2f} < {cost:.2f})")
                new_qty = pos["qty"] + qty
                pos["avg_price"] = (pos["avg_price"] * pos["qty"] + fill * qty) / new_qty if new_qty else fill
                self.ledger["cash"] -= cost
            pos["qty"] += qty
        else:
            if pos["qty"] > 0:                               # closing a long
                self.ledger["cash"] += cost
            else:                                            # opening a short: premium received, strike × 100 reserved
                need = self._collateral(symbol, qty)
                if need > self.ledger["cash"] - self._reserved():
                    if pos["qty"] == 0:
                        self.ledger["positions"].pop(symbol, None)
                    return self._reject(symbol, side, qty, order_type, limit_price,
                                        f"insufficient cash to secure the short ({self.ledger['cash'] - self._reserved():.2f} < {need:.2f})")
                new_qty = pos["qty"] - qty
                pos["avg_price"] = (pos["avg_price"] * abs(pos["qty"]) + fill * qty) / abs(new_qty) if new_qty else fill
                self.ledger["cash"] += cost
            pos["qty"] -= qty
        pos["last_price"] = fill
        if pos["qty"] == 0:
            self.ledger["positions"].pop(symbol, None)
        o = Order(str(uuid.uuid4())[:8], symbol, side, qty, order_type, limit_price, "filled",
                  filled_qty=qty, filled_avg_price=fill)
        self.ledger["orders"].append(o.to_dict())
        self.ledger["orders"] = self.ledger["orders"][-500:]
        self._save()
        return o

    def settle(self, symbol, price):
        """
        Expiry settlement: remove the position at `price` (intrinsic value, 0 for
        worthless).  A long receives price × 100 × qty, a short pays it.
        Returns the settlement Order, or None when nothing is open.
        """
        pos = self.ledger["positions"].pop(symbol, None)
        if not pos or not pos.get("qty"):
            return None
        qty   = int(pos["qty"])
        price = max(0.0, float(price or 0.0))
        self.ledger["cash"] += qty * price * 100
        o = Order(str(uuid.uuid4())[:8], symbol, "sell" if qty > 0 else "buy", abs(qty), "settle", None,
                  "filled", filled_qty=abs(qty), filled_avg_price=price, message="expiry settlement")
        self.ledger["orders"].append(o.to_dict())
        self._save()
        return o

    def get_order(self, order_id):
        for o in self.ledger["orders"]:
            if o["id"] == order_id:
                return Order(**{k: v for k, v in o.items() if k in Order.__dataclass_fields__})
        return None

    def cancel_order(self, order_id):
        return False                                         # paper fills are immediate


# ──────────────────────────────────────────────────────────────
#  Alpaca — REST, paper by default
# ──────────────────────────────────────────────────────────────

class AlpacaBroker(Broker):
    name = "alpaca"
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL  = "https://api.alpaca.markets"
    DATA_URL  = "https://data.alpaca.markets"
    _OPEN_STATUSES = ("new", "accepted", "pending_new", "partially_filled", "pending_cancel", "pending_replace",
                      "accepted_for_bidding", "calculated", "held")

    def __init__(self, key_id: str, secret: str, paper: bool = True, timeout: int = 15):
        if requests is None:
            raise BrokerError("the `requests` package is required for AlpacaBroker")
        if not key_id or not secret:
            raise BrokerError("Alpaca key_id / secret missing (autotrade_config.json or ALPACA_KEY_ID / ALPACA_SECRET)")
        if not paper and os.environ.get("OPTIONS_BOT_ALLOW_LIVE") != "1":
            raise BrokerError("Live trading refused: set paper=true, or OPTIONS_BOT_ALLOW_LIVE=1 to allow real money")
        self.paper   = bool(paper)
        self.base    = self.PAPER_URL if paper else self.LIVE_URL
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret,
                               "Accept": "application/json", "Content-Type": "application/json"})

    def _req(self, method: str, path: str, base: Optional[str] = None, retry: bool = True, **kw):
        """retry=False for anything that must not be resent blindly (order submission)."""
        url = (base or self.base) + path
        attempts = 3 if retry else 1
        for attempt in range(attempts):
            try:
                r = self.s.request(method, url, timeout=self.timeout, **kw)
            except requests.RequestException as e:
                if attempt == attempts - 1:
                    raise BrokerError(f"{method} {path}: {e}")
                time.sleep(1 + attempt)
                continue
            if r.status_code == 429 and attempt < attempts - 1:
                time.sleep(2 + attempt * 2)
                continue
            if r.status_code >= 400:
                raise BrokerError(f"{method} {path} → {r.status_code}: {r.text[:300]}", status=r.status_code)
            return r.json() if r.text else {}
        raise BrokerError(f"{method} {path}: retries exhausted")

    def account(self) -> dict:
        a = self._req("GET", "/v2/account")
        regt = float(a.get("buying_power", 0) or 0)
        opt  = a.get("options_buying_power")                 # options are not marginable: this is the real limit
        return {"cash": float(a.get("cash", 0) or 0), "equity": float(a.get("equity", 0) or 0),
                "buying_power": float(opt) if opt not in (None, "") else regt,
                "regt_buying_power": regt,
                "options_approved_level": a.get("options_approved_level"), "status": a.get("status")}

    def positions(self) -> list:
        out = []
        for p in self._req("GET", "/v2/positions"):
            if p.get("asset_class") != "us_option":
                continue
            qty = int(float(p.get("qty", 0)))
            if p.get("side") == "short":
                qty = -abs(qty)
            out.append({"symbol": p["symbol"], "qty": qty, "avg_price": float(p.get("avg_entry_price", 0) or 0),
                        "market_value": float(p.get("market_value", 0) or 0),
                        "unrealized_pl": float(p.get("unrealized_pl", 0) or 0)})
        return out

    def submit_option_order(self, symbol, side, qty, limit_price=None, order_type="limit", tif="day", ref_price=None):
        coid = f"ob-{uuid.uuid4().hex[:20]}"                 # lets us find the order again if the POST times out
        body = {"symbol": symbol, "qty": str(int(qty)), "side": side, "type": order_type, "time_in_force": tif,
                "client_order_id": coid}
        if order_type == "limit":
            if not limit_price:
                raise BrokerError("limit order without limit_price")
            body["limit_price"] = f"{option_tick(limit_price, side):.2f}"
        try:
            r = self._req("POST", "/v2/orders", retry=False, json=body)
        except BrokerError as e:
            if e.status is not None and e.status < 500:
                raise                                        # a real rejection: nothing was placed
            # transport failure / 5xx: Alpaca may have accepted the order — look it up before giving up
            time.sleep(1.0)
            try:
                r = self._req("GET", f"/v2/orders:by_client_order_id?client_order_id={coid}")
            except BrokerError:
                raise e
        return self._order_from(r)

    def _order_from(self, r: dict) -> Order:
        return Order(r.get("id", ""), r.get("symbol", ""), r.get("side", ""), int(float(r.get("qty") or 0)),
                     r.get("type", ""), float(r["limit_price"]) if r.get("limit_price") else None,
                     r.get("status", ""), int(float(r.get("filled_qty") or 0)),
                     float(r["filled_avg_price"]) if r.get("filled_avg_price") else None,
                     r.get("submitted_at", ""), "", r)

    def get_order(self, order_id):
        try:
            return self._order_from(self._req("GET", f"/v2/orders/{order_id}"))
        except BrokerError as e:
            if e.status == 404:
                return None
            raise

    def cancel_order(self, order_id):
        try:
            self._req("DELETE", f"/v2/orders/{order_id}")
            return True
        except BrokerError:
            return False                                     # 422 = not cancelable any more (filled): caller re-fetches

    def is_market_open(self) -> bool:
        try:
            return bool(self._req("GET", "/v2/clock").get("is_open"))
        except BrokerError:
            return _nyse_regular_session_open()

    def option_quote(self, symbol: str) -> Optional[dict]:
        try:
            r = self._req("GET", "/v1beta1/options/quotes/latest", base=self.DATA_URL,
                          params={"symbols": symbol, "feed": "indicative"})
            q = (r.get("quotes") or {}).get(symbol)
            if not q:
                return None
            bid, ask = float(q.get("bp", 0)), float(q.get("ap", 0))
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else (ask or bid or None)
            return {"bid": bid, "ask": ask, "mid": mid}
        except BrokerError:
            return None


# ──────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────

def make_broker(cfg: dict) -> Broker:
    """
    cfg keys: broker ("local_paper" | "alpaca"), paper (bool),
              alpaca_key_id / alpaca_secret (or env ALPACA_KEY_ID / ALPACA_SECRET),
              paper_starting_cash, paper_slippage_pct, paper_ledger_path
    """
    kind = (cfg.get("broker") or "local_paper").lower()
    if kind == "alpaca":
        return AlpacaBroker(cfg.get("alpaca_key_id") or os.environ.get("ALPACA_KEY_ID", ""),
                            cfg.get("alpaca_secret") or os.environ.get("ALPACA_SECRET", ""),
                            paper=bool(cfg.get("paper", True)))
    return LocalPaperBroker(ledger_path=cfg.get("paper_ledger_path") or "paper_ledger.json",
                            starting_cash=float(cfg.get("paper_starting_cash", 10_000)),
                            slippage_pct=float(cfg.get("paper_slippage_pct", 1.0)))
