"""Tabbed, drill-down HTML dashboard for bottom_scan_v3 results."""
import json, html
from datetime import datetime
import numpy as np, pandas as pd
import plotly.io as pio
from bottom_scan_v3 import (fmt, tier, pct_color, table_html, fig_heatmap, fig_flags, fig_ticker, fig_pctile_bars,
                            PCTL_YEARS, HIST_YEARS, DOWNLOAD_YEARS, BURN_IN_YEARS, RECENT_D, RECENT_W)

def fig_json(fig):
    return json.loads(pio.to_json(fig))

def tlink(t):
    return f"<a href='#' class='tl' data-t='{t}'><b>{t}</b></a>"

def ranking_html(results, names=None):
    names = names or {}
    rows = []
    for i, r in enumerate(results, 1):
        t0, t5 = r["tests"][0], r["tests"][5]
        rows.append([(i, ""), (tlink(r["ticker"]), ""), (names.get(r["ticker"], ""), "font-size:12px"),
                     (f"<b>{r['n_flags']}</b>", f"background:{'#f6c9c9' if r['n_flags']>=7 else '#fbe7c6' if r['n_flags']>=4 else '#eee'}"),
                     (r["n_turn"], ""), (f"{r['avg_pctile']:.0f}", f"background:{pct_color(r['avg_pctile'])}"), (tier(r), ""),
                     (" ".join(x["id"] for x in r["tests"] if x["flag"]) or "—", "font-family:monospace"), (f"{r['price']:.2f}", ""),
                     (f"{r['dd1y']:.0f}%", ""), (f"{r['ddath']:.0f}%", ""),
                     (f"{t0['cur']:.0f} <small>({t0['pctile']:.0f}th)</small>", f"background:{pct_color(t0['pctile'])}"),
                     (f"{t5['cur']:.0f} <small>({t5['pctile']:.0f}th)</small>", f"background:{pct_color(t5['pctile'])}")])
    return ("<p class='hint'>Click any ticker to drill into its detail. Avg pctile = mean of the 10 indicator percentiles (lower = more extreme vs own history).</p>" +
            table_html(["#", "Ticker", "Name", "Flags /10", "Turn /4", "Avg pctile", "Tier", "Flags firing", "Price", "DD 1-yr", "DD ATH", "RSI-D (pctile)", "RSI-W (pctile)"], rows) +
            "<div id='fig-flags' class='fig'></div>")

def methodology_html(results):
    tests0 = results[0]["tests"]
    defs = table_html(["ID", "Oversold test", "Threshold", "Frame", "Recent window", "Rationale"],
        [[(x["id"], "font-family:monospace"), (x["label"], ""),
          ("stock-relative: median at own prior swing lows" if x["id"] in ("D4", "D5") else f"≤ {fmt(x['thr'],x['fmt'])}", ""),
          ("Daily" if x["frame"] == "D" else "Weekly", ""), (f"{RECENT_D} sessions" if x["frame"] == "D" else f"{RECENT_W} weeks", ""),
          ({"D1": "Classic short-term washout", "D2": "Close pinned at bottom of 3-week range", "D3": "2σ statistical extreme",
            "D4": "Has it fallen as far as at previous bottoms?", "D5": "Trend stretch comparable to previous bottoms",
            "W1": "Intermediate-term washout, rare for quality names", "W2": "Pinned at bottom of 14-week range",
            "W3": "Hugging/below weekly lower band", "W4": "Rare reading for this specific stock", "W5": "Momentum crash in one quarter"}[x["id"]], "")] for x in tests0])
    return f"""
<h3>Checklist (identical for every stock)</h3>{defs}
<h3>Turn signals (bonus, counted separately)</h3>
<ul><li><b>T1</b> Daily MACD histogram rising over 3 bars</li><li><b>T2</b> Weekly MACD histogram rising over 3 bars</li>
<li><b>T3</b> Bullish RSI divergence: lower price low but higher RSI low over ~6 months</li><li><b>T4</b> Daily RSI dipped ≤30 in last 10 sessions and has hooked back above 30</li></ul>
<h3>Tiers</h3><ul><li>≥7 flags → <b>Tier 1</b> (≥2 turn signals) or <b>Tier 2</b> (no confirmed turn)</li><li>4–6 → Tier 3 partially oversold</li><li>2–3 → Tier 4 mild pullback</li><li>≤1 → Tier 5 not oversold</li></ul>
<h3>Percentiles</h3><p>Percentile = share of the last {PCTL_YEARS} years in which the indicator reading was <i>more</i> oversold than today. 0 = most extreme reading in the window, 50 = median. "Pctile recent" applies the same to the lowest reading in the recent window.</p>
<h3>Data hygiene</h3><ul>
<li>Source: Yahoo Finance, split/dividend-adjusted daily OHLCV; weekly bars derived from daily (Friday close).</li>
<li>{DOWNLOAD_YEARS} years downloaded; the first {BURN_IN_YEARS} years serve only as warm-up so 200-day SMA, 252-day rolling high, Wilder RSI recursion and 40-week SMA are fully formed before the {PCTL_YEARS}-year percentile window begins.</li>
<li>Every indicator is computed from raw price/volume only. The single exception, by design, is W4 (percentile rank of weekly RSI).</li>
<li>Swing lows = bar whose low is the lowest within ±30 trading days (daily) / ±8 weeks (weekly) and ≥15% below the trailing 1-yr high; last {HIST_YEARS} years; the current 45 days are excluded.</li>
<li>Episodes = clusters of bars (gap ≤15 d / ≤35 d) when the indicator was at/below the threshold; forward returns measured from the last bar of the episode.</li>
<li>Today's daily and weekly bars are partial (intraday) until the close.</li></ul>
<h3>Data audit per ticker</h3>""" + table_html(["Ticker", "Raw from", "Raw bars", "Pctile window from", "Daily bars", "Weekly bars", "Missing indicator values in window"],
        [[(tlink(r["ticker"]), ""), (str(r["audit"]["first_raw_bar"]), ""), (r["audit"]["raw_bars"], ""), (str(r["audit"]["window_start"]), ""),
          (r["audit"]["window_bars_d"], ""), (r["audit"]["window_bars_w"], ""), (r["audit"]["nan_in_window"], "color:crimson;font-weight:bold" if r["audit"]["nan_in_window"] else "")] for r in results], "small")

def stock_blocks(r):
    a = r["audit"]
    head = (f"<p>Price <b>{r['price']:.2f}</b> · {r['dd1y']:.1f}% from 1-yr high · {r['ddath']:.1f}% from ATH · as of {r['asof']} · "
            f"<b>{r['n_flags']}/10 oversold flags · {r['n_turn']}/4 turn signals · {tier(r)}</b><br>"
            f"<small>Data audit: raw from {a['first_raw_bar']} ({a['raw_bars']} bars); percentile window from {a['window_start']} ({a['window_bars_d']} d / {a['window_bars_w']} w bars); "
            f"missing indicator values in window: <span style='{'color:crimson;font-weight:bold' if a['nan_in_window'] else ''}'>{a['nan_in_window']}</span></small></p>")
    rows = []
    for x in r["tests"]:
        f = x["fmt"]
        rows.append([(x["id"], "font-family:monospace"), (x["label"], ""), (f"≤ {fmt(x['thr'],f)}", ""), (f"<b>{fmt(x['cur'],f)}</b>", ""),
                     (f"<b>{fmt(x['pctile'],'{:.0f}')}</b>", f"background:{pct_color(x['pctile'])}"), (fmt(x["recent_min"], f), ""),
                     (fmt(x["pctile_recent"], "{:.0f}"), f"background:{pct_color(x['pctile_recent'])}"),
                     ("✅ now" if x["flag_now"] else ("🟡 recent" if x["flag"] else "—"), ""), (fmt(x["bot_med"], f), ""), (fmt(x["bot_min"], f), "")])
    why = [f"<b>{x['id']}</b> {x['label']} = {fmt(x['cur'],x['fmt'])}" + ("" if x["flag_now"] else f" (touched {fmt(x['recent_min'],x['fmt'])})") for x in r["tests"] if x["flag"]]
    why_not = [f"<b>{x['id']}</b> {fmt(x['cur'],x['fmt'])} vs ≤ {fmt(x['thr'],x['fmt'])}" for x in r["tests"] if not x["flag"]]
    readings = (head + "<h4>Checklist readings</h4>" +
                table_html(["ID", "Test", "Threshold", "Now", "Pctile now", "Recent low", "Pctile recent", "Flag", "Median @ prior lows", "Min @ prior lows"], rows) +
                "<div class='row'><div class='col'><h4>Why oversold</h4><ul>" + "".join(f"<li>{s}</li>" for s in why or ["nothing firing"]) + "</ul></div>"
                "<div class='col'><h4>Not (yet) oversold on</h4><ul>" + "".join(f"<li>{s}</li>" for s in why_not or ["all tests firing"]) + "</ul></div></div>"
                "<h4>Turn signals</h4>" + table_html(["Turn", "Signal", "Status", "Detail"], [[(t_, "font-family:monospace"), (lab, ""), ("✅" if ok else "—", ""), (det, "")] for t_, lab, ok, det in r["turns"]], "small") +
                "<div id='fig-pct' class='fig'></div>")
    hist = [f"<p class='hint'>Each box: prior {HIST_YEARS}-yr episodes when the indicator reached the same threshold it is tested against today, with the price change over the following 1 and 3 months (episodes with n/a are too recent).</p><div class='row'>"]
    for x in r["tests"]:
        if not x["episodes"]:
            hist.append(f"<div class='ep'><b>{x['id']}</b> {x['label']} (≤ {fmt(x['thr'],x['fmt'])}) — today {fmt(x['cur'],x['fmt'])}<br>No prior episode" + (" — <b>today is unprecedented in window</b>" if x["flag"] else "") + "</div>"); continue
        ep_rows = [[(f"{e['start']:%Y-%m-%d}" + ("" if e["start"] == e["end"] else f" → {e['end']:%Y-%m-%d}"), ""), (fmt(e["min"], x["fmt"]), ""), (f"{e['px']:.2f}", ""),
                    (fmt(e["fwd1m"], "{:+.1f}%"), f"color:{'green' if (e['fwd1m'] or 0) > 0 else 'crimson'}"), (fmt(e["fwd3m"], "{:+.1f}%"), f"color:{'green' if (e['fwd3m'] or 0) > 0 else 'crimson'}")] for e in x["episodes"]]
        f1 = [e["fwd1m"] for e in x["episodes"] if not pd.isna(e["fwd1m"])]; f3 = [e["fwd3m"] for e in x["episodes"] if not pd.isna(e["fwd3m"])]
        summ = f"<small>median +1 mo {fmt(np.median(f1) if f1 else np.nan,'{:+.1f}%')} · +3 mo {fmt(np.median(f3) if f3 else np.nan,'{:+.1f}%')} · {sum(v>0 for v in f3)}/{len(f3)} positive at 3 mo</small>"
        hist.append(f"<div class='ep'><b>{x['id']}</b> {x['label']} (≤ {fmt(x['thr'],x['fmt'])}) — today <b>{fmt(x['cur'],x['fmt'])}</b> {'✅' if x['flag_now'] else ('🟡' if x['flag'] else '')}<br>{len(x['episodes'])} episode(s) · {summ}" +
                    table_html(["Episode", "Extreme", "Price", "+1 mo", "+3 mo"], ep_rows, "small") + "</div>")
    hist.append("</div>")
    lows = []
    for name, bot, cols in (("daily", r["d_bot"], ["RSI14", "StochK", "BB_pctB", "Drawdown_1y_pct", "Dist_long_pct"]), ("weekly", r["w_bot"], ["RSI14", "StochK", "BB_pctB", "RSI14_pctile", "ROC_3m_pct"])):
        hdr = {"RSI14": "RSI-14", "StochK": "Stoch %K", "BB_pctB": "BB %B", "Drawdown_1y_pct": "DD 1-yr", "Dist_long_pct": "vs 200d" if name == "daily" else "vs 40w", "RSI14_pctile": "RSI pctile", "ROC_3m_pct": "13w ROC"}
        src = r["d"] if name == "daily" else r["w"]
        now = [(f"<b>NOW {src.index[-1]:%Y-%m-%d}</b>", "background:#fff3cd"), (f"{src['Close'].iloc[-1]:.2f}", "background:#fff3cd")] + \
              [(f"<b>{fmt(src[c].iloc[-1] if c in src else np.nan, '{:.2f}' if c == 'BB_pctB' else ('{:.1f}%' if ('pct' in c and c != 'RSI14_pctile') else '{:.1f}'))}</b>", "background:#fff3cd") for c in cols]
        lows.append(f"<h4>Historical {name} swing lows (≥15% off 1-yr high, past {HIST_YEARS} yrs) vs today</h4>")
        if bot.empty: lows.append(table_html(["Date", "Low"] + [hdr[c] for c in cols], [now], "small") + "<p><i>No qualifying historical swing lows.</i></p>"); continue
        rows = [[(f"{ts:%Y-%m-%d}", ""), (f"{b['Low']:.2f}", "")] + [(fmt(b.get(c, np.nan), "{:.2f}" if c == "BB_pctB" else ("{:.1f}%" if ("pct" in c and c != "RSI14_pctile") else "{:.1f}")), "") for c in cols] for ts, b in bot.iterrows()]
        lows.append(table_html(["Date", "Low"] + [hdr[c] for c in cols], rows + [now], "small"))
    return {"readings": readings, "history": "".join(hist), "lows": "".join(lows) + "<p class='hint'>Highlighted row = today's readings on the same indicators, for direct comparison.</p>"}

def write_dashboard(results, path):
    pio.templates.default = "plotly_white"
    data = {"order": [r["ticker"] for r in results], "html": {}, "figs": {}}
    for r in results:
        data["html"][r["ticker"]] = stock_blocks(r)
        data["figs"][r["ticker"]] = {"main": fig_json(fig_ticker(r)), "pct": fig_json(fig_pctile_bars(r))}
    data["figs"]["_heat"] = fig_json(fig_heatmap(results)); data["figs"]["_flags"] = fig_json(fig_flags(results))
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    options = "".join(f"<option value='{t}'>{t} — {r['n_flags']}/10</option>" for t, r in zip(data["order"], results))
    css = """
    body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f9;color:#222}
    header{background:#1f2f44;color:#fff;padding:10px 24px;display:flex;align-items:center;gap:18px;flex-wrap:wrap;position:sticky;top:0;z-index:10}
    header h1{font-size:18px;margin:0} header .meta{font-size:12px;color:#c9d3df}
    nav.tabs{display:flex;gap:4px;background:#2b3e55;padding:6px 24px 0;position:sticky;top:46px;z-index:9}
    nav.tabs button{background:#3b5273;color:#dde;border:none;padding:9px 18px;border-radius:8px 8px 0 0;cursor:pointer;font-size:14px}
    nav.tabs button.active{background:#f4f6f9;color:#1f2f44;font-weight:600}
    .wrap{max-width:1500px;margin:0 auto;padding:14px 24px 40px} .pane{display:none} .pane.active{display:block}
    .toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:6px 0 12px;background:#fff;padding:10px 14px;border-radius:8px;border:1px solid #e1e5ea}
    .toolbar select{font-size:15px;padding:5px 8px} .toolbar button{padding:5px 12px;cursor:pointer;border:1px solid #bbb;background:#fff;border-radius:6px}
    #back{background:#2b7bba;color:#fff;border-color:#2b7bba;display:none} #back.show{display:inline-block}
    nav.sub{display:flex;gap:4px;margin:4px 0 10px} nav.sub button{background:#e3e8ef;border:none;padding:7px 14px;border-radius:6px;cursor:pointer}
    nav.sub button.active{background:#2b7bba;color:#fff;font-weight:600}
    .tbl{border-collapse:collapse;width:100%;margin:8px 0 16px;font-size:13.5px;background:#fff} .tbl th{background:#2b3e55;color:#fff;padding:6px 8px;text-align:left}
    .tbl td{padding:5px 8px;border-bottom:1px solid #e5e5e5} .tbl.small{font-size:12.5px} .tbl tr:hover td{filter:brightness(.97)}
    .row{display:flex;flex-wrap:wrap;gap:16px} .col{flex:1 1 440px} .ep{background:#fff;border:1px solid #e3e3e3;border-radius:6px;padding:8px 12px;flex:1 1 440px;max-width:720px}
    a{color:#2b7bba} a.tl{text-decoration:none} a.tl:hover{text-decoration:underline} .hint{color:#555;font-size:13px} h3{margin-top:28px} h4{margin:18px 0 6px}
    .fig{background:#fff;border:1px solid #e1e5ea;border-radius:8px;margin:10px 0}
    """
    js = r"""
const DATA = __DATA__;
const TABS = {ranking:'Ranking', heat:'Heat-map', stock:'Stock detail', method:'Methodology'};
const SUBS = {readings:'Readings', charts:'Charts', history:'Episode history', lows:'Swing lows'};
let S = {tab:'ranking', ticker:DATA.order[0], sub:'readings'}; let stack = []; const drawn = {};
function plot(id, fig){ const el=document.getElementById(id); Plotly.react(el, fig.data, fig.layout, {displaylogo:false, responsive:true}); return el; }
function render(){
  document.querySelectorAll('nav.tabs button').forEach(b=>b.classList.toggle('active', b.dataset.tab===S.tab));
  document.querySelectorAll('.pane').forEach(p=>p.classList.toggle('active', p.id==='pane-'+S.tab));
  const back=document.getElementById('back'); back.classList.toggle('show', stack.length>0);
  if(stack.length){ const p=stack[stack.length-1]; back.textContent='← Back to '+TABS[p.tab]+(p.tab==='stock'?' ('+p.ticker+')':''); }
  if(S.tab==='ranking' && !drawn.flags){ const el=plot('fig-flags', DATA.figs._flags); drawn.flags=1; el.on('plotly_click', e=>go('stock',{ticker:e.points[0].x})); }
  if(S.tab==='heat' && !drawn.heat){ const el=plot('fig-heat', DATA.figs._heat); drawn.heat=1; el.on('plotly_click', e=>go('stock',{ticker:e.points[0].y})); }
  if(S.tab==='stock'){
    document.getElementById('tsel').value=S.ticker;
    document.querySelectorAll('nav.sub button').forEach(b=>b.classList.toggle('active', b.dataset.sub===S.sub));
    document.getElementById('stock-title').textContent=S.ticker;
    const h=DATA.html[S.ticker]; const body=document.getElementById('stock-body');
    if(S.sub==='charts'){ body.innerHTML="<div id='fig-main' class='fig'></div>"; plot('fig-main', DATA.figs[S.ticker].main); }
    else { body.innerHTML=h[S.sub]; if(S.sub==='readings') plot('fig-pct', DATA.figs[S.ticker].pct); }
  }
  location.hash = S.tab==='stock' ? `stock/${S.ticker}/${S.sub}` : S.tab;
  window.scrollTo(0,0);
}
function go(tab, opts={}, push=true){ if(push) stack.push({...S}); if(stack.length>30) stack.shift(); S={...S, tab, ...opts}; render(); }
function back(){ const p=stack.pop(); if(p){ S=p; render(); } }
function step(d){ const i=DATA.order.indexOf(S.ticker); S.ticker=DATA.order[(i+d+DATA.order.length)%DATA.order.length]; render(); }
document.addEventListener('click', e=>{ const a=e.target.closest('a.tl'); if(a){ e.preventDefault(); go('stock',{ticker:a.dataset.t, sub:'readings'}); } });
document.querySelectorAll('nav.tabs button').forEach(b=>b.onclick=()=>go(b.dataset.tab));
document.querySelectorAll('nav.sub button').forEach(b=>b.onclick=()=>{ S.sub=b.dataset.sub; render(); });
document.getElementById('tsel').onchange=e=>{ S.ticker=e.target.value; render(); };
document.getElementById('back').onclick=back;
(function init(){ const h=location.hash.replace('#',''); if(h){ const p=h.split('/'); if(p[0]==='stock' && DATA.order.includes(p[1])){ S={tab:'stock', ticker:p[1], sub:SUBS[p[2]]?p[2]:'readings'}; } else if(TABS[p[0]]) S.tab=p[0]; } render(); })();
"""
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>Oversold Checklist Dashboard</title><style>{css}</style>
<script>{pio.get_plotlyjs() if hasattr(pio,'get_plotlyjs') else __import__('plotly.offline',fromlist=['get_plotlyjs']).get_plotlyjs()}</script></head><body>
<header><h1>Oversold Checklist Dashboard</h1><span class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} ET · Yahoo Finance adjusted OHLCV · {DOWNLOAD_YEARS} yrs downloaded, {BURN_IN_YEARS} yrs warm-up, percentiles over last {PCTL_YEARS} yrs · today's bars are intraday/partial</span></header>
<nav class='tabs'><button data-tab='ranking'>1 · Ranking</button><button data-tab='heat'>2 · Heat-map</button><button data-tab='stock'>3 · Stock detail</button><button data-tab='method'>4 · Methodology</button></nav>
<div class='wrap'>
<div class='toolbar'><button id='back'>← Back</button><span id='stock-tools' style='display:contents'></span></div>
<div id='pane-ranking' class='pane'><h2>Ranking – most oversold to least</h2>{ranking_html(results)}</div>
<div id='pane-heat' class='pane'><h2>Percentile heat-map</h2><p class='hint'>Each cell = today's indicator reading as a percentile of its own {PCTL_YEARS}-yr history (0 = most oversold ever). ✅ flag firing now · 🟡 fired in recent window. Click a cell to drill into the stock.</p><div id='fig-heat' class='fig'></div></div>
<div id='pane-stock' class='pane'>
  <div class='toolbar'><label>Ticker <select id='tsel'>{options}</select></label><button onclick='step(-1)'>◀ prev</button><button onclick='step(1)'>next ▶</button><h2 style='margin:0 0 0 12px' id='stock-title'></h2></div>
  <nav class='sub'><button data-sub='readings'>Readings & flags</button><button data-sub='charts'>Charts</button><button data-sub='history'>Episode history</button><button data-sub='lows'>Swing lows vs today</button></nav>
  <div id='stock-body'></div></div>
<div id='pane-method' class='pane'><h2>Methodology & data audit</h2>{methodology_html(results)}</div>
</div><script>{js.replace('__DATA__', payload)}</script></body></html>"""
    with open(path, "w", encoding="utf-8") as f: f.write(doc)
