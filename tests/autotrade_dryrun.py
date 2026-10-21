"""
End-to-end dry run of the autotrader against the local paper broker — no
network, no real files: every file the run touches is redirected to a temp
folder, and the scan results are synthetic candidates.

    py tests/autotrade_dryrun.py

Exercises: entries with sizing + quote filters, short-put collateral, exits on a
triggered signal and on expiry, performance logging, re-entry cooldown, the
daily loss cap, the kill switch, pending (non-instant) broker orders for both
entries and closes including a broker outage and partial fills, a manual close
request, option tick rounding, config normalisation and the status() payload.
"""
import json, os, shutil, sys, tempfile, uuid, warnings
from datetime import date, datetime, timedelta
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import options_bot as ob                              # noqa: E402
import autotrader as at                               # noqa: E402
from broker import LocalPaperBroker, Order, BrokerError, option_tick   # noqa: E402

TMP = tempfile.mkdtemp(prefix="autotrade_dry_")
ob.TRACK_FILE       = os.path.join(TMP, "tracked.json")
ob.PERFORMANCE_FILE = os.path.join(TMP, "performance.json")
at.CONFIG_FILE      = os.path.join(TMP, "config.json")
at.STATE_FILE       = os.path.join(TMP, "state.json")
at.LOG_FILE         = os.path.join(TMP, "log.json")
at.LEDGER_FILE      = os.path.join(TMP, "ledger.json")
at.KILL_FILE        = os.path.join(TMP, "STOP_AUTOTRADE")

FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def cand(ticker, strat, price, bid, ask, score, dte=30, exp=None):
    exp = exp or (date.today() + timedelta(days=dte)).isoformat()
    return {"ticker": ticker, "name": ticker + " Inc", "strategy": strat, "strategy_label": strat,
            "score": score, "price": price, "bar_date": date.today().isoformat(),
            "option": {"type": "call" if strat == "buy_call" else "put", "expiration": exp, "dte": dte,
                       "strike": float(round(price)), "bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 3),
                       "iv": 35.0, "volume": 500, "open_interest": 2000, "premium_pct": 2.0}}


def set_results(**buckets):
    with ob.scan_lock:
        ob.scan_state["results"] = {"sell_puts": [], "buy_calls": [], "buy_puts": [], "covered_calls": [], "iron_condors": [],
                                    **buckets}
        ob.scan_state["last_scan"] = datetime.now().isoformat()
        ob.scan_state["running"] = False


def tracked():
    with ob.track_lock:
        return ob.load_tracked()


def perf():
    with ob._perf_lock:
        return ob.load_performance()


def log_actions():
    return [e["action"] for e in reversed(at.get_log(500))]


class FakeSlowBroker(LocalPaperBroker):
    """
    Accepts an order first and fills it on the next get_order() — the Alpaca-like
    pending path.  partial_next=True makes the next poll report "canceled" with
    all but one contract filled (partial fill, then cancelled).  fail_next=True
    makes the next poll raise (broker unreachable).
    """
    name = "fake_slow"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._queue = {}
        self.partial_next = False
        self.fail_next = False

    def submit_option_order(self, symbol, side, qty, limit_price=None, order_type="limit", tif="day", ref_price=None):
        oid = str(uuid.uuid4())[:8]
        self._queue[oid] = (symbol, side, qty, limit_price, ref_price)
        return Order(oid, symbol, side, qty, order_type, limit_price, "accepted")

    def get_order(self, order_id):
        if self.fail_next:
            self.fail_next = False
            raise BrokerError("simulated outage")
        if order_id in self._queue:
            sym, side, qty, lim, ref = self._queue.pop(order_id)
            if self.partial_next and qty > 1:
                self.partial_next = False
                o = super().submit_option_order(sym, side, qty - 1, limit_price=lim, ref_price=ref)
                o.id, o.status, o.qty = order_id, "canceled", qty
                return o
            o = super().submit_option_order(sym, side, qty, limit_price=lim, ref_price=ref)
            o.id = order_id
            return o
        return super().get_order(order_id)


def main():
    at.init(ob)
    cfg = at.set_config({"enabled": True, "broker": "local_paper", "min_score": 50, "max_open_positions": 3,
                         "max_new_per_cycle": 3, "paper_starting_cash": 20000, "max_position_usd": 500,
                         "max_collateral_usd": 5000, "daily_loss_cap_usd": 300, "max_spread_pct": 15,
                         "reentry_cooldown_days": 3, "market_hours_only": True, "premarket_scan_time": ""})
    check(cfg["enabled"] and cfg["paper_starting_cash"] == 20000, "config saved and coerced")

    print("\n1. entries")
    set_results(buy_calls=[cand("AAPL", "buy_call", 190, 2.0, 2.2, 80), cand("MSFT", "buy_call", 400, 3.0, 3.3, 40)],
                buy_puts=[cand("NVDA", "buy_put", 120, 1.5, 1.7, 75), cand("AMD", "buy_put", 150, 1.0, 1.6, 90)],
                sell_puts=[cand("F", "sell_put", 12, 0.30, 0.34, 70)])
    r = at.run_now()
    print("   result:", json.dumps(r, default=str)[:300])
    pos = tracked()
    check(r["ran"] and r["entries"] == 3, f"3 entries placed (got {r['entries']})")
    check(len(pos) == 3 and all(p.get("auto") for p in pos), "3 auto positions tracked")
    check(any("AMD" in s and "spread" in s for s in r["skipped"]), "wide spread skipped")
    check(any("MSFT" in s and "score" in s for s in r["skipped"]), "low score skipped")
    by = {p["ticker"]: p for p in pos}
    check(by["AAPL"]["contracts"] == 2 and by["NVDA"]["contracts"] == 2, "long sizing: $500 budget → 2 contracts")
    check(by["F"]["contracts"] == 4, "short put sizing: $5000 collateral / $1200 → 4 contracts")
    check(2.1 < by["AAPL"]["entry_price"] <= 2.2, f"long fill between mid and ask ({by['AAPL']['entry_price']})")
    check(0.30 <= by["F"]["entry_price"] < 0.32, f"short fill between bid and mid ({by['F']['entry_price']})")
    acct = at._get_broker(at.load_config()).account()
    check(acct["cash"] < 20000, f"paper cash moved ({acct['cash']})")
    check(abs(acct["collateral_reserved"] - 4800) < 0.01 and acct["buying_power"] < acct["cash"] - 4700,
          f"short put collateral reserved (bp {acct['buying_power']} vs cash {acct['cash']})")
    rej = at._get_broker(at.load_config()).submit_option_order("XYZ261120P00990000", "sell", 100, limit_price=1.0, ref_price=1.0)
    check(rej.status == "rejected" and "secure" in rej.message, "paper broker rejects a short it cannot secure")

    print("\n2. exits: profit target + expiry")
    with ob.track_lock:
        ps = ob.load_tracked()
        for p in ps:
            if p["ticker"] == "AAPL":
                p["last_signal"] = {"triggered": True, "reasons": ["🎯 Profit target hit: +41.0%"],
                                    "current_option_mid": 3.0, "current_bid": 2.9, "current_ask": 3.1, "current_price": 199.0}
            if p["ticker"] == "F":
                p["option"]["expiration"] = (date.today() - timedelta(days=1)).isoformat()
                p["last_signal"] = {"triggered": False, "reasons": [], "current_price": 11.0}
        ob.save_tracked(ps)
    r = at.run_now()
    pf = perf()
    check(r["exits"] == 2, f"2 exits (got {r['exits']})")
    check(len(tracked()) == 1 and tracked()[0]["ticker"] == "NVDA", "NVDA still open")
    check(len(pf) == 2 and all(t["source"] == "autotrade" for t in pf), "2 autotrade trades in performance")
    t = {x["ticker"]: x for x in pf}
    check(t["AAPL"]["pnl_total"] > 0 and 2.9 <= t["AAPL"]["exit_price"] <= 3.0, f"AAPL win logged (+{t['AAPL']['pnl_total']})")
    check(t["F"]["pnl_total"] < 0 and t["F"]["exit_price"] == 1.0, f"F expired ITM: short put loss logged ({t['F']['pnl_total']})")
    check(abs(at.status()["today"]["realized"] - (t["AAPL"]["pnl_total"] + t["F"]["pnl_total"])) < 0.01, "daily realised P&L tallied")
    check("F" not in {p["symbol"][:1] for p in at._get_broker(at.load_config()).positions()}, "ledger position settled")

    print("\n3. re-entry cooldown + already-open")
    set_results(buy_calls=[cand("AAPL", "buy_call", 190, 2.0, 2.2, 80)], buy_puts=[cand("NVDA", "buy_put", 120, 1.5, 1.7, 75)],
                sell_puts=[cand("F", "sell_put", 12, 0.30, 0.34, 70)])
    r = at.run_now()
    print("   result:", json.dumps(r, default=str)[:400])
    check(r["entries"] == 0, "no re-entries")
    check(any("AAPL" in s and "cooldown" in s for s in r["skipped"]), "AAPL blocked by cooldown")
    check(any("NVDA" in s and "already" in s for s in r["skipped"]), "NVDA blocked as already open")

    print("\n4. daily loss cap")
    with at._state_lock:
        at._daily()["realized"] = -1000.0
    set_results(buy_calls=[cand("TSLA", "buy_call", 250, 2.0, 2.2, 90)])
    r = at.run_now()
    check(r["entries"] == 0 and any("loss cap" in s for s in r["skipped"]), "entries blocked by the daily loss cap")
    with at._state_lock:
        at._daily()["realized"] = 0.0

    print("\n5. kill switch")
    set_results()                                         # nothing to enter — isolates the switch itself
    open(at.KILL_FILE, "w").close()
    r = at.run_now()
    check(not r["ran"] and r["reason"] == "kill switch", "kill switch stops the cycle")
    os.remove(at.KILL_FILE)
    r = at.run_now()
    check(r["ran"], "resumes once the file is gone")

    print("\n6. pending orders (broker that fills on the next poll)")
    cfg = at.load_config()
    key = (cfg["broker"], cfg["paper"], cfg["alpaca_key_id"], cfg["alpaca_secret"], cfg["paper_starting_cash"], cfg["paper_slippage_pct"])
    at._broker, at._broker_key = FakeSlowBroker(ledger_path=os.path.join(TMP, "ledger2.json"), starting_cash=20000), key
    set_results(buy_calls=[cand("META", "buy_call", 250, 2.0, 2.2, 90)])
    r = at.run_now()
    check(r["entries"] == 1 and at.status()["pending_entries"] == 1, "entry accepted → pending")
    check(not any(p["ticker"] == "META" for p in tracked()), "not tracked until filled")
    r = at.run_now()
    check(at.status()["pending_entries"] == 0 and any(p["ticker"] == "META" for p in tracked()), "pending entry filled → tracked")
    with ob.track_lock:                                   # give it a quote so the close can price
        ps = ob.load_tracked()
        for p in ps:
            if p["ticker"] == "META":
                p["last_signal"] = {"triggered": False, "reasons": [], "current_option_mid": 2.0, "current_bid": 1.9, "current_ask": 2.1}
        ob.save_tracked(ps)
    meta = next(p for p in tracked() if p["ticker"] == "META")
    check(meta["contracts"] == 2, "META holds 2 contracts")
    check(at.request_close(meta["id"]), "manual close requested")
    r = at.run_now()
    meta = next(p for p in tracked() if p["ticker"] == "META")
    check(meta.get("pending_close") is not None, "close accepted → pending_close on the position")
    at._broker.fail_next = True                           # broker unreachable on the poll
    r = at.run_now()
    meta = next(p for p in tracked() if p["ticker"] == "META")
    check(meta.get("pending_close") is not None and "close_check_error" in log_actions(), "broker outage keeps pending_close (no second order)")
    at._broker.partial_next = True                        # 1 of 2 fills, then the order is cancelled
    r = at.run_now()
    meta = next((p for p in tracked() if p["ticker"] == "META"), None)
    check(meta is not None and meta["contracts"] == 1 and meta.get("pending_close") is not None,
          "partial close: 1 contract booked, 1 still open and already re-submitted")
    check(any(x["ticker"] == "META" and x["contracts"] == 1 and "partial" in x["notes"] for x in perf()), "partial exit logged to performance")
    r = at.run_now()                                      # the remainder fills on this poll
    check(not any(p["ticker"] == "META" for p in tracked()), "remaining contract closed → position gone")
    check(sum(x["contracts"] for x in perf() if x["ticker"] == "META") == 2, "both META closes in performance (1 + 1)")
    check(any(x["ticker"] == "META" and "manual close" in x["notes"] for x in perf()), "manual close logged to performance")

    print("\n6b. pending entry that partially fills, then is cancelled")
    set_results(buy_calls=[cand("GOOG", "buy_call", 170, 2.0, 2.2, 90)])
    r = at.run_now()                                      # accepted → pending
    at._broker.partial_next = True
    r = at.run_now()
    goog = next((p for p in tracked() if p["ticker"] == "GOOG"), None)
    check(goog is not None and goog["contracts"] == 1 and at.status()["pending_entries"] == 0, "partial entry tracked with the filled qty only")
    check("entry_partial" in log_actions(), "partial entry logged")

    print("\n6c. option tick rounding")
    check(option_tick(2.123, "buy") == 2.13 and option_tick(2.123, "sell") == 2.12, "penny ticks below $3 (buy up, sell down)")
    check(option_tick(3.27, "buy") == 3.30 and option_tick(3.27, "sell") == 3.25, "nickel ticks at/above $3")

    print("\n7. status + log")
    s = at.status()
    json.dumps(s, default=str)
    check(s["broker"] == "local_paper" and s["open_auto_positions"] == 2 and s["cycles"] >= 8, "status() complete")
    acts = log_actions()
    for a in ("entry", "exit", "kill_switch", "entry_submitted", "close_submitted", "close_requested", "exit_partial"):
        check(a in acts, f"log has '{a}'")
    cfg2 = at.set_config({"allowed_strategies": "Sell Put, buy_calls, nonsense", "min_score": None})
    check(cfg2["allowed_strategies"] == ["sell_put", "buy_call"] and cfg2["min_score"] == 50, "config normalises strategies; null keeps the stored number")
    saved = json.load(open(at.STATE_FILE))
    check("recent_exits" in saved and "daily" in saved and "pending" in saved, "state persisted to disk")

    print()
    if FAILS:
        print(f"DRY RUN FAILED — {len(FAILS)} check(s):"); [print("   -", f) for f in FAILS]
    else:
        print("DRY RUN OK — every check passed")
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
