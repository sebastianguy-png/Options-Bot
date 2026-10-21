"""
Autotrader — runs the bot's own signals through a broker with nobody at the
keyboard.

What it does, every `interval_mins` minutes while enabled:
  1. settles orders that were still open at the broker (fills / cancels)
  2. EXITS: every auto position whose tracker signal says `triggered`
     (profit target, stop, trailing stop, stagnation, max hold, RSI/BB
     reversal, expiry) is closed through the broker and logged to
     performance.json — exactly the rules the backtest was run with
  3. kicks off a fresh scan when the last one is older than `scan_if_stale_hours`
  4. ENTRIES: the highest-scoring fresh signals that pass the quote checks are
     bought/sold through the broker and added to the tracked positions with
     auto=True, so the ordinary tracker takes over from there

What it will NOT do without you changing it:
  - trade real money: the broker is a local paper ledger by default; Alpaca
    runs against its paper endpoint; the live endpoint additionally needs
    OPTIONS_BOT_ALLOW_LIVE=1 in the environment (see broker.py)
  - exceed the limits in autotrade_config.json (positions, $ per trade,
    daily loss cap) or trade outside regular hours
  - keep running once the file STOP_AUTOTRADE exists next to the bot (kill
    switch — delete the file to resume)

Everything it decides is appended to autotrade_log.json.  The bot module is
injected with init() so this file never imports options_bot (no import cycle).
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
import traceback
import uuid
from datetime import datetime, date, timedelta

from broker import make_broker, occ_symbol, BrokerError, LocalPaperBroker

CONFIG_FILE = "autotrade_config.json"
STATE_FILE  = "autotrade_state.json"
LOG_FILE    = "autotrade_log.json"
LEDGER_FILE = "paper_ledger.json"
KILL_FILE   = "STOP_AUTOTRADE"
LOG_KEEP    = 2000

DEFAULT_CONFIG = {
    "enabled":                False,
    "broker":                 "local_paper",    # local_paper | alpaca
    "paper":                  True,             # alpaca: paper endpoint (live also needs OPTIONS_BOT_ALLOW_LIVE=1)
    "alpaca_key_id":          "",
    "alpaca_secret":          "",
    "paper_starting_cash":    10000.0,
    "paper_slippage_pct":     1.0,              # local paper fills this much worse than the quote mid
    "allowed_strategies":     ["buy_call", "buy_put", "sell_put"],
    "min_score":              60,               # signal score needed to be traded
    "max_open_positions":     5,                # auto positions open at once
    "max_new_per_cycle":      2,
    "max_position_usd":       500.0,            # most premium paid per long trade (sets the contract count)
    "max_collateral_usd":     5000.0,           # most cash secured per short put (strike × 100 × qty)
    "daily_loss_cap_usd":     300.0,            # realised auto P&L today ≤ −this → no new entries until tomorrow
    "max_spread_pct":         15.0,             # (ask − bid) / mid; wider quotes are skipped
    "min_dte":                14,
    "max_dte":                60,
    "market_hours_only":      True,
    "interval_mins":          5,
    "scan_if_stale_hours":    3.0,              # start a scan when the newest results are older than this
    "reentry_cooldown_days":  3,                # trading days before the same ticker+strategy can be re-entered
    "pending_order_timeout_mins": 20,           # cancel an unfilled broker order after this long
}
_SECRET_KEYS   = ("alpaca_key_id", "alpaca_secret")
_STRAT_BUCKET  = {"sell_put": "sell_puts", "buy_call": "buy_calls", "buy_put": "buy_puts"}   # covered calls need shares, condors are 4 legs
_ORDER_DONE    = {"filled"}
_ORDER_FAILED  = {"canceled", "cancelled", "expired", "rejected", "stopped", "suspended", "done_for_day"}

_ob          = None                       # the options_bot module (set by init)
_cycle_lock  = threading.RLock()          # one cycle at a time (loop vs "run now" vs manual close)
_cfg_lock    = threading.Lock()
_state_lock  = threading.Lock()
_log_lock    = threading.Lock()
_wake        = threading.Event()
_thread      = None
_broker      = None
_broker_key  = None
_account     = {"ts": 0.0, "data": None, "error": None}
_state = {
    "in_cycle":     False,
    "last_cycle":   None,
    "next_cycle":   None,
    "last_result":  None,
    "last_error":   None,
    "market_open":  None,
    "kill_switch":  False,
    "cycles":       0,
    "pending":      [],        # entry orders still open at the broker: {order_id, candidate, qty, submitted_at}
    "recent_exits": {},        # "TICKER|strategy" → ISO date of the last auto exit (re-entry cooldown)
    "daily":        {},        # "YYYY-MM-DD" → {"realized": $, "entries": n, "exits": n}
}


# ──────────────────────────────────────────────────────────────
#  Wiring / persistence
# ──────────────────────────────────────────────────────────────

def init(bot_module):
    """Give the autotrader the running options_bot module (scan results, tracker, performance log)."""
    global _ob
    _ob = bot_module
    _load_state()


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _today():
    return date.today().isoformat()


def load_config() -> dict:
    raw = _ob._load_json(CONFIG_FILE, {}, "Autotrade") if _ob else {}
    cfg = dict(DEFAULT_CONFIG)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in DEFAULT_CONFIG:
                cfg[k] = v
    return _coerce(cfg)


def _coerce(cfg: dict) -> dict:
    """Types and ranges — a typo in the JSON must not turn into an unbounded limit."""
    f = lambda k, lo, hi: max(lo, min(hi, float(cfg.get(k, DEFAULT_CONFIG[k]) or 0)))
    i = lambda k, lo, hi: int(max(lo, min(hi, int(float(cfg.get(k, DEFAULT_CONFIG[k]) or 0)))))
    cfg["enabled"]            = bool(cfg.get("enabled", False))
    cfg["broker"]             = str(cfg.get("broker") or "local_paper").lower()
    if cfg["broker"] not in ("local_paper", "alpaca"):
        cfg["broker"] = "local_paper"
    cfg["paper"]              = bool(cfg.get("paper", True))
    cfg["alpaca_key_id"]      = str(cfg.get("alpaca_key_id") or "")
    cfg["alpaca_secret"]      = str(cfg.get("alpaca_secret") or "")
    cfg["paper_starting_cash"]= f("paper_starting_cash", 100.0, 1e9)
    cfg["paper_slippage_pct"] = f("paper_slippage_pct", 0.0, 10.0)
    strats = cfg.get("allowed_strategies") or []
    cfg["allowed_strategies"] = [s for s in strats if s in _STRAT_BUCKET] if isinstance(strats, list) else list(_STRAT_BUCKET)
    cfg["min_score"]          = f("min_score", 0.0, 100.0)
    cfg["max_open_positions"] = i("max_open_positions", 0, 100)
    cfg["max_new_per_cycle"]  = i("max_new_per_cycle", 0, 20)
    cfg["max_position_usd"]   = f("max_position_usd", 0.0, 1e7)
    cfg["max_collateral_usd"] = f("max_collateral_usd", 0.0, 1e8)
    cfg["daily_loss_cap_usd"] = f("daily_loss_cap_usd", 0.0, 1e8)
    cfg["max_spread_pct"]     = f("max_spread_pct", 0.0, 100.0)
    cfg["min_dte"]            = i("min_dte", 0, 365)
    cfg["max_dte"]            = max(cfg["min_dte"], i("max_dte", 0, 730))
    cfg["market_hours_only"]  = bool(cfg.get("market_hours_only", True))
    cfg["interval_mins"]      = i("interval_mins", 1, 240)
    cfg["scan_if_stale_hours"]= f("scan_if_stale_hours", 0.25, 48.0)
    cfg["reentry_cooldown_days"] = i("reentry_cooldown_days", 0, 30)
    cfg["pending_order_timeout_mins"] = i("pending_order_timeout_mins", 1, 390)
    return cfg


def save_config(cfg: dict) -> None:
    with _cfg_lock:
        _ob._atomic_json_write(CONFIG_FILE, _coerce(dict(cfg)))


def set_config(patch: dict) -> dict:
    """Merge a (partial) config from the dashboard; blank secrets keep the stored ones."""
    global _broker, _broker_key
    cfg = load_config()
    for k, v in (patch or {}).items():
        if k not in DEFAULT_CONFIG:
            continue
        if k in _SECRET_KEYS and not v:
            continue
        cfg[k] = v
    cfg = _coerce(cfg)
    save_config(cfg)
    with _cycle_lock:                       # a broker change takes effect on the next cycle
        _broker, _broker_key = None, None
    _wake.set()
    return cfg


def public_config(cfg: dict | None = None) -> dict:
    cfg = dict(cfg or load_config())
    for k in _SECRET_KEYS:
        cfg[k] = ("•" * 8 + cfg[k][-4:]) if cfg.get(k) else ""
    return cfg


def _load_state():
    saved = _ob._load_json(STATE_FILE, {}, "Autotrade") if _ob else {}
    if isinstance(saved, dict):
        with _state_lock:
            for k in ("pending", "recent_exits", "daily"):
                if isinstance(saved.get(k), type(_state[k])):
                    _state[k] = saved[k]


def _save_state():
    with _state_lock:
        snap = {k: copy.deepcopy(_state[k]) for k in ("pending", "recent_exits", "daily")}
        # keep the daily table short
        keys = sorted(snap["daily"])[-60:]
        snap["daily"] = {k: snap["daily"][k] for k in keys}
        _state["daily"] = snap["daily"]
    try:
        _ob._atomic_json_write(STATE_FILE, snap)
    except Exception as e:
        print(f"  [Autotrade] could not save state: {e}")


def _log(action: str, **fields):
    entry = {"ts": _now(), "action": action, **fields}
    line  = " ".join(f"{k}={v}" for k, v in fields.items() if k != "detail")
    print(f"  [Autotrade] {action}: {line}" + (f" — {fields['detail']}" if fields.get("detail") else ""))
    try:
        with _log_lock:
            log = _ob._load_json(LOG_FILE, [], "Autotrade")
            log = log if isinstance(log, list) else []
            log.append(entry)
            _ob._atomic_json_write(LOG_FILE, log[-LOG_KEEP:])
    except Exception as e:
        print(f"  [Autotrade] could not write log: {e}")
    return entry


def get_log(n: int = 100) -> list:
    with _log_lock:
        log = _ob._load_json(LOG_FILE, [], "Autotrade")
    log = log if isinstance(log, list) else []
    return list(reversed(log[-max(1, int(n)):]))


def kill_switch_active() -> bool:
    return os.path.exists(KILL_FILE)


def _daily(day: str | None = None) -> dict:
    """Today's counters (caller holds _state_lock or accepts a snapshot)."""
    day = day or _today()
    return _state["daily"].setdefault(day, {"realized": 0.0, "entries": 0, "exits": 0})


# ──────────────────────────────────────────────────────────────
#  Broker
# ──────────────────────────────────────────────────────────────

def _get_broker(cfg: dict):
    global _broker, _broker_key
    key = (cfg["broker"], cfg["paper"], cfg["alpaca_key_id"], cfg["alpaca_secret"],
           cfg["paper_starting_cash"], cfg["paper_slippage_pct"])
    if _broker is None or _broker_key != key:
        _broker = make_broker({**cfg, "paper_ledger_path": LEDGER_FILE})
        _broker_key = key
        _account.update(ts=0.0, data=None, error=None)
    return _broker


def _account_info(br, max_age=60.0) -> dict | None:
    if time.time() - _account["ts"] < max_age and _account["data"] is not None:
        return _account["data"]
    try:
        _account.update(ts=time.time(), data=br.account(), error=None)
    except Exception as e:
        _account.update(ts=time.time(), data=None, error=str(e))
    return _account["data"]


def _occ(pos_or_cand: dict) -> str | None:
    opt = pos_or_cand.get("option") or {}
    try:
        return occ_symbol(pos_or_cand["ticker"], opt["expiration"], opt.get("type") or "call", opt["strike"])
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
#  The cycle
# ──────────────────────────────────────────────────────────────

def run_cycle(force: bool = False) -> dict:
    """
    One pass: settle pending orders → exits → scan if stale → entries.
    force=True runs even when disabled / outside market hours (the dashboard's
    "Run now" and the dry-run test).  Returns a short summary dict.
    """
    if _ob is None:
        raise RuntimeError("autotrader.init(options_bot) was never called")
    with _cycle_lock:
        cfg = load_config()
        summary = {"ts": _now(), "ran": False, "reason": None, "exits": 0, "entries": 0, "skipped": []}
        _state["in_cycle"] = True
        try:
            if kill_switch_active():
                if not _state["kill_switch"]:
                    _log("kill_switch", detail=f"{KILL_FILE} exists — autotrader paused until it is deleted")
                _state["kill_switch"] = True
                summary["reason"] = "kill switch"
                return summary
            if _state["kill_switch"]:
                _log("kill_switch_cleared")
            _state["kill_switch"] = False
            if not cfg["enabled"] and not force:
                summary["reason"] = "disabled"
                return summary

            br = _get_broker(cfg)
            market_open = bool(br.is_market_open())
            _state["market_open"] = market_open
            if cfg["market_hours_only"] and not market_open and not force:
                summary["reason"] = "market closed"
                return summary

            summary["ran"] = True
            summary["settled"] = _settle_pending(br, cfg)
            summary["exits"]   = _manage_exits(br, cfg)
            summary["scan"]    = _maybe_scan(cfg, market_open or force)
            summary["entries"], summary["skipped"] = _open_entries(br, cfg)
            _mark_paper(br)
            _state["last_error"] = None
        except Exception as e:
            _state["last_error"] = f"{_now()} {e}"
            _log("error", detail=f"{e}\n{traceback.format_exc()[-800:]}")
            summary["reason"] = f"error: {e}"
        finally:
            _state["in_cycle"]    = False
            _state["last_cycle"]  = _now()
            _state["last_result"] = summary
            _state["cycles"]     += 1
            _save_state()
        return summary


def _settle_pending(br, cfg) -> int:
    """Poll orders left open at the broker: entries in _state['pending'], closes on the positions."""
    n = 0
    timeout = timedelta(minutes=cfg["pending_order_timeout_mins"])
    # entries
    with _state_lock:
        pending = list(_state["pending"])
    keep = []
    for p in pending:
        o = br.get_order(p["order_id"])
        age = datetime.now() - datetime.fromisoformat(p["submitted_at"])
        if o is None:
            if age > timeout:
                _log("entry_dropped", ticker=p["candidate"]["ticker"], strategy=p["candidate"]["strategy"],
                     detail="order not found at broker after timeout")
                continue
            keep.append(p); continue
        st = (o.status or "").lower()
        if st in _ORDER_DONE and o.filled_qty:
            _add_tracked(p["candidate"], int(o.filled_qty), float(o.filled_avg_price or p["limit"]), o.id, br)
            n += 1
        elif st in _ORDER_FAILED:
            _log("entry_failed", ticker=p["candidate"]["ticker"], strategy=p["candidate"]["strategy"], status=st)
        elif age > timeout:
            br.cancel_order(p["order_id"])
            _log("entry_timeout", ticker=p["candidate"]["ticker"], strategy=p["candidate"]["strategy"],
                 detail=f"unfilled after {cfg['pending_order_timeout_mins']} min — cancelled")
        else:
            keep.append(p)
    with _state_lock:
        _state["pending"] = keep
    # closes
    with _ob.track_lock:
        positions = _ob.load_tracked()
    for pos in positions:
        pc = pos.get("pending_close")
        if not (pos.get("auto") and pc):
            continue
        o = br.get_order(pc["order_id"])
        age = datetime.now() - datetime.fromisoformat(pc["submitted_at"])
        st = (o.status or "").lower() if o else ""
        if o is not None and st in _ORDER_DONE and o.filled_qty:
            _finalize_exit(pos["id"], float(o.filled_avg_price or pc["limit"]), pc.get("reason", ""), o.id)
            n += 1
        elif (o is None and age > timeout) or st in _ORDER_FAILED or age > timeout:
            if o is not None and st not in _ORDER_FAILED:
                br.cancel_order(pc["order_id"])
            _set_position_field(pos["id"], "pending_close", None)
            _log("close_retry", ticker=pos["ticker"], strategy=pos["strategy"], status=st or "missing",
                 detail="close order not filled — will resubmit at the current quote")
    return n


def _manage_exits(br, cfg) -> int:
    """Close every auto position whose tracker signal fired (or that the user asked to close)."""
    with _ob.track_lock:
        positions = _ob.load_tracked()
    closed = 0
    today = date.today()
    for pos in positions:
        if not pos.get("auto") or pos.get("pending_close"):
            continue
        sig  = pos.get("last_signal") or {}
        opt  = pos.get("option") or {}
        exp  = str(opt.get("expiration") or "")[:10]
        try:
            expired = bool(sig.get("expired")) or (bool(exp) and date.fromisoformat(exp) < today)
        except ValueError:
            expired = False
        forced = bool(pos.get("force_close"))
        if not (sig.get("triggered") or expired or forced):
            continue
        reason = "manual close" if forced else ("expired" if expired and not sig.get("reasons") else "; ".join(sig.get("reasons") or []))
        sym = pos.get("occ") or _occ(pos)                # the contract actually held at the broker
        if not sym:
            _log("close_skipped", ticker=pos.get("ticker"), detail="position has no usable option fields")
            continue
        is_short = pos.get("strategy") in _ob.SHORT_PREMIUM_STRATS
        qty = max(1, _ob._as_int(pos.get("contracts"), 1))
        if expired:
            stock = _ob._as_float(sig.get("current_price")) or _ob._as_float(pos.get("price"))
            k     = _ob._as_float(opt.get("strike"))
            intrinsic = max(0.0, (stock - k) if (opt.get("type") == "call") else (k - stock)) if stock > 0 else 0.0
            if isinstance(br, LocalPaperBroker):
                br.settle(sym, intrinsic)
            _finalize_exit(pos["id"], round(intrinsic, 3), reason or "expired", "settle")
            closed += 1
            continue
        mid = _ob._as_float(sig.get("current_option_mid"))
        bid = _ob._as_float(sig.get("current_bid"))
        ask = _ob._as_float(sig.get("current_ask"))
        if is_short:                                     # buy to close
            side, limit = "buy", (ask if ask > 0 else (mid * 1.03 if mid > 0 else 0))
        else:                                            # sell to close
            side, limit = "sell", (bid if bid > 0 else (mid * 0.97 if mid > 0 else 0))
        if limit <= 0:
            _log("close_skipped", ticker=pos["ticker"], strategy=pos["strategy"], detail="no quote for the contract yet")
            continue
        try:
            o = br.submit_option_order(sym, side, qty, limit_price=round(limit, 2), ref_price=mid if mid > 0 else limit)
        except BrokerError as e:
            _log("close_error", ticker=pos["ticker"], strategy=pos["strategy"], detail=str(e))
            continue
        st = (o.status or "").lower()
        if st in _ORDER_DONE and o.filled_qty:
            _finalize_exit(pos["id"], float(o.filled_avg_price or limit), reason, o.id)
            closed += 1
        elif st in _ORDER_FAILED:
            _log("close_rejected", ticker=pos["ticker"], strategy=pos["strategy"], status=st, detail=o.message)
        else:
            _set_position_field(pos["id"], "pending_close",
                                {"order_id": o.id, "submitted_at": _now(), "side": side, "qty": qty,
                                 "limit": round(limit, 2), "reason": reason})
            _log("close_submitted", ticker=pos["ticker"], strategy=pos["strategy"], qty=qty, limit=round(limit, 2),
                 order=o.id, status=st)
    return closed


def _finalize_exit(pos_id: str, exit_price: float, reason: str, order_id: str):
    """Move a closed auto position into performance.json and update the daily counters."""
    with _ob.track_lock:
        positions = _ob.load_tracked()
        pos = next((p for p in positions if p["id"] == pos_id), None)
        if pos is None:
            return
        opt   = pos.get("option") or {}
        entry = _ob._as_float(pos.get("entry_price")) or _ob._as_float(opt.get("mid")) or _ob._as_float(opt.get("ask"))
        if entry <= 0:
            entry = max(exit_price, 0.01)
        trade = _ob._closed_trade_record(
            ticker=pos.get("ticker", ""), strategy=_ob._norm_strategy(pos.get("strategy", "")),
            option_desc=f"{str(opt.get('type', '')).capitalize()} ${opt.get('strike', '')} {opt.get('expiration', '')}",
            entry=entry, exit_=float(exit_price), contracts=max(1, _ob._as_int(pos.get("contracts"), 1)),
            entry_date=str(pos.get("tracked_at") or "")[:10], exit_date=_today(),
            notes=f"autotrade · {reason}", source="autotrade")
        trade["broker"] = pos.get("broker"); trade["order_id"] = order_id
        _ob.save_tracked([p for p in positions if p["id"] != pos_id])
    with _ob._perf_lock:
        perf = _ob.load_performance()
        perf.append(trade)
        _ob.save_performance(perf)
    with _state_lock:
        d = _daily()
        d["realized"] = round(d["realized"] + float(trade["pnl_total"]), 2)
        d["exits"]   += 1
        _state["recent_exits"][f"{trade['ticker']}|{trade['strategy']}"] = _today()
    _log("exit", ticker=trade["ticker"], strategy=trade["strategy"], qty=trade["contracts"],
         entry=trade["entry_price"], exit=trade["exit_price"], pnl=trade["pnl_total"], pnl_pct=trade["pnl_pct"],
         detail=reason)


def _set_position_field(pos_id: str, key: str, value):
    with _ob.track_lock:
        positions = _ob.load_tracked()
        for p in positions:
            if p["id"] == pos_id:
                if value is None:
                    p.pop(key, None)
                else:
                    p[key] = value
                break
        _ob.save_tracked(positions)


def _maybe_scan(cfg, allowed: bool) -> str:
    """Start a full scan in the background when the newest results are stale."""
    with _ob.scan_lock:
        running = bool(_ob.scan_state.get("running"))
        last    = _ob.scan_state.get("last_refresh") or _ob.scan_state.get("last_scan")
    if running:
        return "scan running"
    if last:
        try:
            age_h = (datetime.now() - datetime.fromisoformat(last)).total_seconds() / 3600
        except ValueError:
            age_h = 1e9
    else:
        age_h = 1e9
    if age_h < cfg["scan_if_stale_hours"] or not allowed:
        return f"results {age_h:.1f}h old" if last else "no scan yet"
    with _ob.scan_lock:
        if _ob.scan_state.get("running"):
            return "scan running"
        _ob.scan_state["running"] = True                     # claim the slot like /api/scan/start does
    threading.Thread(target=_ob._run_scan, daemon=True, name="autotrade-scan").start()
    _log("scan_started", detail=f"results were {age_h:.1f}h old" if last else "no results yet")
    return "scan started"


def _candidates(cfg) -> tuple[list, list]:
    """Fresh signals that pass the quote / size / dedupe checks, best score first."""
    with _ob.scan_lock:
        results = copy.deepcopy(_ob.scan_state.get("results") or {})
        last    = _ob.scan_state.get("last_refresh") or _ob.scan_state.get("last_scan")
    skipped = []
    if not last:
        return [], ["no scan results yet"]
    try:
        age_h = (datetime.now() - datetime.fromisoformat(last)).total_seconds() / 3600
    except ValueError:
        age_h = 1e9
    if age_h > cfg["scan_if_stale_hours"] * 2:
        return [], [f"scan results are {age_h:.1f}h old — waiting for a fresh scan"]
    with _ob.track_lock:
        positions = _ob.load_tracked()
    open_keys = {f"{p.get('ticker')}|{_ob._norm_strategy(p.get('strategy', ''))}" for p in positions}
    with _state_lock:
        open_keys |= {f"{p['candidate']['ticker']}|{p['candidate']['strategy']}" for p in _state["pending"]}
        recent = dict(_state["recent_exits"])
    cooldown_cut = datetime.now().date() - timedelta(days=int(cfg["reentry_cooldown_days"] * 1.5))
    out = []
    for strat in cfg["allowed_strategies"]:
        for r in results.get(_STRAT_BUCKET[strat], []) or []:
            key = f"{r.get('ticker')}|{strat}"
            opt = r.get("option") or {}
            why = None
            if _ob._as_float(r.get("score")) < cfg["min_score"]:
                why = f"score {r.get('score')} < {cfg['min_score']}"
            elif key in open_keys:
                why = "already open / pending"
            elif key in recent and date.fromisoformat(recent[key]) >= cooldown_cut:
                why = f"re-entry cooldown (exited {recent[key]})"
            elif not (opt.get("strike") and opt.get("expiration")):
                why = "no contract"
            elif not (_ob._as_float(opt.get("bid")) > 0 and _ob._as_float(opt.get("ask")) > 0):
                why = "no two-sided quote"
            elif not (cfg["min_dte"] <= _ob._as_int(opt.get("dte"), 0) <= cfg["max_dte"]):
                why = f"dte {opt.get('dte')} outside {cfg['min_dte']}-{cfg['max_dte']}"
            else:
                bid, ask = _ob._as_float(opt.get("bid")), _ob._as_float(opt.get("ask"))
                mid = (bid + ask) / 2
                spread = (ask - bid) / mid * 100 if mid > 0 else 1e9
                if spread > cfg["max_spread_pct"]:
                    why = f"spread {spread:.0f}% > {cfg['max_spread_pct']:.0f}%"
            if why:
                skipped.append(f"{r.get('ticker')} {strat}: {why}")
                continue
            out.append({"ticker": r["ticker"], "name": r.get("name", ""), "strategy": strat,
                        "strategy_label": r.get("strategy_label", strat), "score": r.get("score"),
                        "price": r.get("price"), "option": opt, "bar_date": r.get("bar_date")})
    out.sort(key=lambda c: _ob._as_float(c.get("score")), reverse=True)
    return out, skipped


def _size(cand: dict, cfg: dict, account: dict | None) -> tuple[int, float, float, str | None]:
    """→ (qty, limit, ref_mid, why_not).  Long: premium budget; short put: collateral budget; both: cash."""
    opt = cand["option"]
    bid, ask = _ob._as_float(opt.get("bid")), _ob._as_float(opt.get("ask"))
    mid = round((bid + ask) / 2, 3)
    cash = _ob._as_float((account or {}).get("buying_power")) or _ob._as_float((account or {}).get("cash"))
    if cand["strategy"] in _ob.SHORT_PREMIUM_STRATS:
        collateral = _ob._as_float(opt.get("strike")) * 100
        if collateral <= 0 or collateral > cfg["max_collateral_usd"]:
            return 0, 0.0, mid, f"collateral ${collateral:,.0f} > max ${cfg['max_collateral_usd']:,.0f}"
        qty = int(cfg["max_collateral_usd"] // collateral)
        if account is not None and cash > 0:
            qty = min(qty, int(cash // collateral))              # cash-secured
        return (qty, bid, mid, None) if qty >= 1 else (0, 0.0, mid, "not enough cash for one cash-secured contract")
    cost = ask * 100
    if cost <= 0 or cost > cfg["max_position_usd"]:
        return 0, 0.0, mid, f"premium ${cost:,.0f} > max ${cfg['max_position_usd']:,.0f} per trade"
    qty = int(cfg["max_position_usd"] // cost)
    if account is not None and cash > 0:
        qty = min(qty, int(cash // cost))
    return (qty, ask, mid, None) if qty >= 1 else (0, 0.0, mid, "not enough cash for one contract")


def _open_entries(br, cfg) -> tuple[int, list]:
    with _state_lock:
        d = dict(_daily())
        n_pending = len(_state["pending"])
    if cfg["daily_loss_cap_usd"] > 0 and d["realized"] <= -cfg["daily_loss_cap_usd"]:
        return 0, [f"daily loss cap hit (realised {d['realized']:+,.2f})"]
    with _ob.track_lock:
        n_open = sum(1 for p in _ob.load_tracked() if p.get("auto"))
    slots = min(cfg["max_open_positions"] - n_open - n_pending, cfg["max_new_per_cycle"])
    if slots <= 0:
        return 0, [f"{n_open} open + {n_pending} pending ≥ max {cfg['max_open_positions']}"]
    cands, skipped = _candidates(cfg)
    account = _account_info(br)
    if account is None and _account["error"]:
        return 0, skipped + [f"broker account unavailable: {_account['error']}"]
    entered = 0
    for c in cands:
        if entered >= slots:
            break
        qty, limit, mid, why = _size(c, cfg, account)
        if why:
            skipped.append(f"{c['ticker']} {c['strategy']}: {why}")
            continue
        sym  = _occ(c)
        side = "sell" if c["strategy"] in _ob.SHORT_PREMIUM_STRATS else "buy"
        try:
            o = br.submit_option_order(sym, side, qty, limit_price=round(limit, 2), ref_price=mid)
        except BrokerError as e:
            _log("entry_error", ticker=c["ticker"], strategy=c["strategy"], detail=str(e))
            continue
        st = (o.status or "").lower()
        if st in _ORDER_DONE and o.filled_qty:
            _add_tracked(c, int(o.filled_qty), float(o.filled_avg_price or limit), o.id, br)
            entered += 1
            if account is not None:                       # keep the cash figure honest within the cycle
                spent = (o.filled_avg_price or limit) * 100 * o.filled_qty * (1 if side == "buy" else 0)
                account = {**account, "cash": _ob._as_float(account.get("cash")) - spent,
                           "buying_power": _ob._as_float(account.get("buying_power")) - spent}
        elif st in _ORDER_FAILED:
            _log("entry_rejected", ticker=c["ticker"], strategy=c["strategy"], status=st, detail=o.message)
        else:
            with _state_lock:
                _state["pending"].append({"order_id": o.id, "candidate": c, "qty": qty, "limit": round(limit, 2),
                                          "submitted_at": _now()})
            entered += 1
            _log("entry_submitted", ticker=c["ticker"], strategy=c["strategy"], qty=qty, limit=round(limit, 2),
                 order=o.id, status=st)
    return entered, skipped


def _add_tracked(cand: dict, qty: int, fill: float, order_id: str, br):
    """Register a filled entry as a tracked position (same shape as the dashboard's Track button)."""
    position = {
        "id":            str(uuid.uuid4())[:8],
        "tracked_at":    datetime.now().isoformat(),
        "ticker":        cand["ticker"],
        "name":          cand.get("name", ""),
        "price":         _ob._as_float(cand.get("price")),
        "strategy":      cand["strategy"],
        "strategy_label": cand.get("strategy_label", cand["strategy"]),
        "option":        cand["option"],
        "legs":          None,
        "net_credit":    0.0, "put_width": 0.0, "call_width": 0.0,
        "contracts":     int(qty),
        "entry_price":   round(float(fill), 3),
        "last_signal":   None,
        "last_checked":  None,
        "auto":          True,
        "broker":        br.name,
        "paper":         bool(br.paper),
        "occ":           _occ(cand),
        "broker_order_id": order_id,
        "score_at_entry": cand.get("score"),
        "bar_date":      cand.get("bar_date"),
    }
    with _ob.track_lock:
        positions = _ob.load_tracked()
        if any(p["ticker"] == position["ticker"] and p["strategy"] == position["strategy"] for p in positions):
            _log("entry_duplicate", ticker=cand["ticker"], strategy=cand["strategy"],
                 detail="filled but an identical position is already tracked — broker position left as is")
            return
        positions.append(position)
        _ob.save_tracked(positions)
    with _state_lock:
        _daily()["entries"] += 1
    _log("entry", ticker=cand["ticker"], strategy=cand["strategy"], qty=qty, fill=round(fill, 3),
         strike=cand["option"].get("strike"), exp=cand["option"].get("expiration"), score=cand.get("score"),
         order=order_id, broker=br.name)


def _mark_paper(br):
    """Keep the local paper ledger's marks in step with the tracker's refreshed mids."""
    if not isinstance(br, LocalPaperBroker):
        return
    with _ob.track_lock:
        positions = _ob.load_tracked()
    for p in positions:
        if p.get("auto"):
            mid = _ob._as_float((p.get("last_signal") or {}).get("current_option_mid"))
            sym = p.get("occ") or _occ(p)
            if sym and mid > 0:
                br.mark(sym, mid)


# ──────────────────────────────────────────────────────────────
#  Loop / API surface
# ──────────────────────────────────────────────────────────────

def _loop():
    while True:
        cfg = load_config()
        try:
            run_cycle()
        except Exception as e:                           # run_cycle already logs; never let the thread die
            print(f"  [Autotrade] loop error: {e}")
        interval = cfg["interval_mins"] * 60
        _state["next_cycle"] = (datetime.now() + timedelta(seconds=interval)).isoformat(timespec="seconds")
        _wake.wait(timeout=interval)
        _wake.clear()


def start():
    """Start the background loop once (idempotent)."""
    global _thread
    if _thread and _thread.is_alive():
        return _thread
    _thread = threading.Thread(target=_loop, daemon=True, name="autotrader")
    _thread.start()
    return _thread


def wake():
    _wake.set()


def run_now() -> dict:
    """One forced cycle (dashboard button / tests) — runs even if disabled or after hours."""
    return run_cycle(force=True)


def request_close(pos_id: str) -> bool:
    """Ask the next cycle to close one auto position at the market quote."""
    found = False
    with _ob.track_lock:
        positions = _ob.load_tracked()
        for p in positions:
            if p["id"] == pos_id and p.get("auto"):
                p["force_close"] = True
                found = True
        if found:
            _ob.save_tracked(positions)
    if found:
        _log("close_requested", position=pos_id)
        _wake.set()
    return found


def status() -> dict:
    cfg = load_config()
    br = None
    account, acct_err = None, None
    try:
        br = _get_broker(cfg)
        account = _account_info(br)
        acct_err = _account["error"]
    except Exception as e:
        acct_err = str(e)
    with _ob.track_lock:
        positions = _ob.load_tracked()
    auto = [p for p in positions if p.get("auto")]
    with _ob.scan_lock:
        scan = {"last_scan": _ob.scan_state.get("last_scan"), "last_refresh": _ob.scan_state.get("last_refresh"),
                "running": bool(_ob.scan_state.get("running")),
                "signals": sum(len(v) for k, v in (_ob.scan_state.get("results") or {}).items())}
    with _state_lock:
        d = dict(_daily())
        st = {k: copy.deepcopy(v) for k, v in _state.items() if k != "daily"}
    return {
        "enabled":            cfg["enabled"],
        "broker":             cfg["broker"],
        "paper":              cfg["paper"] if cfg["broker"] == "alpaca" else True,
        "live_allowed_env":   os.environ.get("OPTIONS_BOT_ALLOW_LIVE") == "1",
        "kill_switch":        kill_switch_active(),
        "loop_running":       bool(_thread and _thread.is_alive()),
        "in_cycle":           st["in_cycle"],
        "last_cycle":         st["last_cycle"],
        "next_cycle":         st["next_cycle"],
        "last_result":        st["last_result"],
        "last_error":         st["last_error"],
        "market_open":        (br.is_market_open() if br else None),
        "cycles":             st["cycles"],
        "account":            account,
        "account_error":      acct_err,
        "open_auto_positions": len(auto),
        "pending_entries":    len(st["pending"]),
        "pending_closes":     sum(1 for p in auto if p.get("pending_close")),
        "today":              d,
        "daily_loss_cap_hit": cfg["daily_loss_cap_usd"] > 0 and d["realized"] <= -cfg["daily_loss_cap_usd"],
        "scan":               scan,
        "config":             public_config(cfg),
        "hold_days":          getattr(_ob, "HOLD_DAYS", None),
    }
