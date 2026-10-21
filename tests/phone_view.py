"""
Build a phone-friendly snapshot page of the running bot (status, account, open
auto positions, today's signals, recent decisions) from its local API, so it
can be published as a private page and read from a phone.

    py tests/phone_view.py <output.html>
"""
import html, json, os, sys
from datetime import datetime
import requests

BASE = "http://127.0.0.1:5000"


def get(path, default=None):
    try:
        r = requests.get(BASE + path, timeout=25)
        return r.json() if r.ok else default
    except Exception:
        return default


def esc(x):
    return html.escape("" if x is None else str(x))


def money(v, dp=0):
    try:
        return f"${float(v):,.{dp}f}"
    except (TypeError, ValueError):
        return "—"


def pct(v):
    try:
        v = float(v); return f"{v:+.1f}%"
    except (TypeError, ValueError):
        return "—"


def when(iso):
    if not iso:
        return "never"
    try:
        return datetime.fromisoformat(str(iso)[:19]).strftime("%H:%M")
    except ValueError:
        return str(iso)[:16]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "phone_view.html"
    s     = get("/api/autotrade/status", {}) or {}
    pos   = [p for p in (get("/api/tracked", []) or []) if p.get("auto")]
    log   = get("/api/autotrade/log?n=25", []) or []
    perf  = (get("/api/performance", {}) or {})
    scan  = get("/api/scan/status", {}) or {}
    sigs  = {b: (get(f"/api/results/{b}", []) or []) for b in ("buy_calls", "sell_puts", "buy_puts")}
    acct  = s.get("account") or {}
    today = s.get("today") or {}
    cfg   = s.get("config") or {}
    now   = datetime.now().strftime("%a %d %b, %H:%M ET")
    alive = bool(s)

    if not alive:
        state_pill = '<span class="pill bad">bot not reachable</span>'
    elif s.get("kill_switch"):
        state_pill = '<span class="pill bad">kill switch on</span>'
    elif s.get("enabled"):
        state_pill = '<span class="pill good">autotrading on</span>'
    else:
        state_pill = '<span class="pill">autotrading off</span>'
    market = '<span class="pill good">market open</span>' if s.get("market_open") else '<span class="pill">market closed</span>'
    broker = f"{esc(s.get('broker', '—'))}{' · paper' if s.get('paper') else ' · LIVE'}"
    lr = s.get("last_result") or {}
    last_line = (f"{lr.get('exits', 0)} exit(s), {lr.get('entries', 0)} entry(ies)" + (f", scan: {esc(lr.get('scan'))}" if lr.get("scan") else "")
                 if lr.get("ran") else f"idle ({esc(lr.get('reason') or 'no cycle yet')})")
    skipped = lr.get("skipped") or []

    # positions
    if pos:
        rows = []
        for p in pos:
            sig = p.get("last_signal") or {}; opt = p.get("option") or {}
            pnl = sig.get("pct_change")
            cls = "good" if (pnl or 0) > 0 else ("bad" if (pnl or 0) < 0 else "")
            status = "closing" if p.get("pending_close") else ("exit due" if sig.get("triggered") else "open")
            rows.append(f"<tr><td><b>{esc(p.get('ticker'))}</b><br><span class='muted'>{esc(p.get('strategy_label') or p.get('strategy'))}</span></td>"
                        f"<td>{esc(opt.get('type'))} {money(opt.get('strike'), 2)}<br><span class='muted'>{esc(opt.get('expiration'))}</span></td>"
                        f"<td class='num'>{p.get('contracts')}</td><td class='num'>{money(p.get('entry_price'), 2)}</td>"
                        f"<td class='num'>{money(sig.get('current_option_mid'), 2) if sig.get('current_option_mid') is not None else '—'}</td>"
                        f"<td class='num {cls}'>{pct(pnl) if pnl is not None else '—'}</td><td>{esc(status)}</td></tr>")
        positions_html = ("<table><thead><tr><th>Ticker</th><th>Contract</th><th class='num'>Qty</th><th class='num'>Entry</th>"
                          "<th class='num'>Now</th><th class='num'>P&amp;L</th><th>State</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
    else:
        positions_html = "<p class='empty'>No open auto positions.</p>"

    # closed trades today (autotrade)
    closed = [t for t in (perf.get("trades") or []) if t.get("source") == "autotrade"]
    closed_today = [t for t in closed if str(t.get("exit_date", "")) == datetime.now().strftime("%Y-%m-%d")]
    if closed:
        rows = []
        for t in list(reversed(closed))[:8]:
            cls = "good" if (t.get("pnl_total") or 0) > 0 else "bad"
            rows.append(f"<tr><td><b>{esc(t.get('ticker'))}</b> <span class='muted'>{esc(t.get('strategy'))}</span></td>"
                        f"<td>{esc(t.get('option_desc'))}</td><td class='num'>{money(t.get('entry_price'), 2)} → {money(t.get('exit_price'), 2)}</td>"
                        f"<td class='num {cls}'>{money(t.get('pnl_total'), 2)}<br><span class='muted'>{pct(t.get('pnl_pct'))}</span></td>"
                        f"<td>{esc(t.get('exit_date'))}</td></tr>")
        closed_html = "<table><thead><tr><th>Trade</th><th>Contract</th><th class='num'>Entry → exit</th><th class='num'>P&amp;L</th><th>Closed</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    else:
        closed_html = "<p class='empty'>No closed auto trades yet.</p>"

    # signals
    sig_rows = []
    for bucket, label in (("buy_calls", "Buy call"), ("sell_puts", "Sell put"), ("buy_puts", "Buy put (not traded)")):
        for r in sorted(sigs.get(bucket) or [], key=lambda x: -(x.get("score") or 0))[:6]:
            opt = r.get("option") or {}
            spread = ""
            try:
                b, a = float(opt.get("bid") or 0), float(opt.get("ask") or 0)
                if b > 0 and a > 0:
                    spread = f"{(a - b) / ((a + b) / 2) * 100:.0f}%"
            except (TypeError, ValueError):
                pass
            sig_rows.append(f"<tr><td><b>{esc(r.get('ticker'))}</b><br><span class='muted'>{esc(label)}</span></td>"
                            f"<td class='num'>{r.get('score')}</td><td class='num'>{money(r.get('price'), 2)}"
                            f"{'<br><span class=muted>pre-mkt</span>' if r.get('premarket_price') else ''}</td>"
                            f"<td>{esc(opt.get('type'))} {money(opt.get('strike'), 2)} {esc(opt.get('expiration'))}</td>"
                            f"<td class='num'>{money(opt.get('bid'), 2)} / {money(opt.get('ask'), 2)}<br><span class='muted'>spread {spread or '—'}</span></td></tr>")
    signals_html = ("<table><thead><tr><th>Ticker</th><th class='num'>Score</th><th class='num'>Stock</th><th>Contract</th><th class='num'>Bid / ask</th></tr></thead><tbody>"
                    + "".join(sig_rows) + "</tbody></table>") if sig_rows else "<p class='empty'>No signals in the last scan.</p>"
    scan_line = (f"last scan {when(scan.get('last_scan'))}" + (", scanning now…" if scan.get("running") else "")
                 + (f" · {s.get('scan', {}).get('signals', 0)} signals" if s.get("scan") else ""))

    # log
    log_rows = []
    for e in log[:20]:
        a = e.get("action", "")
        cls = "bad" if any(k in a for k in ("error", "rejected", "failed", "kill")) else ("good" if a in ("entry", "exit") else "")
        kv = " ".join(f"{k}={v}" for k, v in e.items() if k not in ("ts", "action", "detail"))
        log_rows.append(f"<tr><td class='mono'>{esc(str(e.get('ts', ''))[11:16])}</td><td class='{cls}'><b>{esc(a)}</b></td>"
                        f"<td>{esc(kv)}{('<br><span class=muted>' + esc(e.get('detail'))[:220] + '</span>') if e.get('detail') else ''}</td></tr>")
    log_html = "<table><tbody>" + "".join(log_rows) + "</tbody></table>" if log_rows else "<p class='empty'>Nothing logged yet.</p>"
    skipped_html = ("<details><summary>Why candidates were skipped last cycle (" + str(len(skipped)) + ")</summary><ul>"
                    + "".join(f"<li>{esc(x)}</li>" for x in skipped[:30]) + "</ul></details>") if skipped else ""

    page = f"""<title>Options Bot Pocket View</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a single column of status strips and compact tables, built for a phone screen. */
:root {{
  --bg:#f6f5f1; --surface:#ffffff; --ink:#1d2430; --muted:#6b7280; --line:#dcdad2;
  --accent:#2b5fd9; --good:#1f8a4c; --bad:#c43d3d; --warn:#b7791f;
  --font:"IBM Plex Sans",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme:dark; --bg:#13161c; --surface:#1b1f27; --ink:#e7e9ee; --muted:#9aa3b2; --line:#2b313d; --accent:#7ea2ff; --good:#4cc47f; --bad:#ef6b6b; --warn:#e0a84a; }} }}
:root[data-theme="dark"] {{ color-scheme:dark; --bg:#13161c; --surface:#1b1f27; --ink:#e7e9ee; --muted:#9aa3b2; --line:#2b313d; --accent:#7ea2ff; --good:#4cc47f; --bad:#ef6b6b; --warn:#e0a84a; }}
body {{ background:var(--bg); color:var(--ink); font-family:var(--font); font-size:15px; line-height:1.45;
  padding-block:16px 40px; padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px)); }}
h1 {{ font-size:22px; font-weight:600; margin:0 0 4px; text-wrap:balance; }}
h2 {{ font-size:13px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin:26px 0 8px; }}
.sub {{ color:var(--muted); font-size:13px; }}
.pills {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }}
.pill {{ border:1px solid var(--line); border-radius:999px; padding:3px 10px; font-size:12px; font-weight:500; background:var(--surface); }}
.pill.good {{ border-color:var(--good); color:var(--good); }} .pill.bad {{ border-color:var(--bad); color:var(--bad); }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; }}
.tile {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }}
.tile .k {{ font-size:11px; color:var(--muted); letter-spacing:.04em; text-transform:uppercase; }}
.tile .v {{ font-size:20px; font-weight:600; font-variant-numeric:tabular-nums; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; background:var(--surface); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
th,td {{ padding:7px 8px; text-align:left; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; font-weight:500; }}
tr:last-child td {{ border-bottom:none; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.muted {{ color:var(--muted); font-size:12px; }} .mono {{ font-family:var(--mono); font-size:12px; }}
.good {{ color:var(--good); }} .bad {{ color:var(--bad); }}
.empty {{ color:var(--muted); background:var(--surface); border:1px dashed var(--line); border-radius:10px; padding:12px; margin:0; }}
.wrap {{ overflow-x:auto; }}
details {{ margin-top:10px; font-size:13px; }} summary {{ cursor:pointer; color:var(--accent); }} ul {{ margin:6px 0 0 18px; padding:0; }} li {{ color:var(--muted); }}
footer {{ margin-top:28px; color:var(--muted); font-size:12px; }}
</style>
<h1>Options Bot</h1>
<div class="sub">Snapshot {esc(now)} · {broker}</div>
<div class="pills">{state_pill}{market}<span class="pill">last cycle {when(s.get('last_cycle'))}</span><span class="pill">{esc(last_line)}</span></div>
<div class="tiles">
  <div class="tile"><div class="k">Equity</div><div class="v">{money(acct.get('equity'))}</div></div>
  <div class="tile"><div class="k">Buying power</div><div class="v">{money(acct.get('buying_power'))}</div></div>
  <div class="tile"><div class="k">Today realised</div><div class="v {'good' if (today.get('realized') or 0) >= 0 else 'bad'}">{money(today.get('realized'), 2)}</div></div>
  <div class="tile"><div class="k">Open · entries · exits</div><div class="v">{len(pos)} · {today.get('entries', 0)} · {today.get('exits', 0)}</div></div>
</div>
{skipped_html}
<h2>Open auto positions</h2><div class="wrap">{positions_html}</div>
<h2>Closed auto trades{f' · today {len(closed_today)}' if closed_today else ''}</h2><div class="wrap">{closed_html}</div>
<h2>Signals · {esc(scan_line)}</h2><div class="wrap">{signals_html}</div>
<h2>Recent decisions</h2><div class="wrap">{log_html}</div>
<footer>Rules: GARCH gate · no premium stop · 21-day max hold · spread ≤ {esc(cfg.get('max_spread_pct', '—'))}% · trades {esc(', '.join(cfg.get('allowed_strategies') or []))} · min score {esc(cfg.get('min_score', '—'))} · max {esc(cfg.get('max_open_positions', '—'))} open.
Refreshed on a schedule during market hours; ask Claude for a fresh one any time.</footer>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {out_path}: alive={alive} positions={len(pos)} signals={sum(len(v) for v in sigs.values())} log={len(log)}")


if __name__ == "__main__":
    main()
