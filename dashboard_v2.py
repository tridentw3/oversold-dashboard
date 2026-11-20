"""
Multi-portfolio dashboard: shell index.html + data/portfolios.json + data/p/<pid>.json + data/s/<T>.json
Build:  python dashboard_v2.py  (after portfolios.py)   ->  site/
"""
import json, os, pickle, shutil, html as H
from datetime import datetime
import numpy as np, pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import bottom_scan_v3 as bs
from bottom_scan_v3 import fmt, tier, pct_color, table_html, fig_heatmap, fig_flags, fig_ticker, fig_pctile_bars, PCTL_YEARS
import dashboard as d1

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache"); SITE = os.path.join(HERE, "site")
pio.templates.default = "plotly_white"


def fj(fig): return json.loads(pio.to_json(fig))
def tl(t): return d1.tlink(t)
def pc(x, nd=0): return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:+.{nd}f}%"
def mn(x, nd=2): return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"
def mu(x): return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}x"
def up_col(u): return "" if u is None else f"background:{'#c9e9c9' if u > 0.5 else '#e8f5e8' if u > 0 else '#f6c9c9'}"
def esc(s, n=None):
    s = s or ""; return H.escape(s if n is None or len(s) <= n else s[:n - 1] + "…")


def status(r):
    if r["tier"] == 2: return "Tier 2 – not rankable on earnings"
    if r["anomaly"]: return "ANOMALY – target inflated by 30x-cap / 10x-floor"
    if r["demoted"]: return f"Demoted – {r['demote_reason']}"
    return "Tier 1"


def eps_str(r):
    tg = r["tag"]
    def one(v, t): return "—" if v is None else f"{v:,.2f}" + ("" if t in (None, "S") else f"<sup>{t}</sup>")
    return f"{one(r['eps26'], tg['eps26'])} / {one(r['eps27'], tg['eps27'])} / <b>{one(r['eps28'], tg['eps28'])}</b>"


# ----------------------------------------------------------------------------- combined ranking
# names for tickers that have no IBES/Refinitiv row anywhere (indices, un-covered single names)
NAMES = {"SPX": "S&P 500 index", "NDX": "Nasdaq-100 index", "COMP": "Nasdaq Composite index", "DJI": "Dow Jones Industrial Average",
         "PL": "Planet Labs PBC", "TEAM": "Atlassian Corp", "JPM": "JPMorgan Chase & Co"}


def name_of(t, r=None):
    return (r or {}).get("company") or NAMES.get(t, "")


def combined_rows(pf, rk_rows, scan):
    """Merge price-target rows and oversold results; compute both ranks and the combined order."""
    tickers = pf["tickers"]
    rk = {r["ticker"]: r for r in (rk_rows or [])}
    rows = []
    for t in tickers:
        s = scan.get(t); r = rk.get(t)
        rows.append({"ticker": t, "rk": r, "sc": s})
    # upside rank: Tier-1 (incl. demoted), not anomaly
    rankable = sorted([x for x in rows if x["rk"] and x["rk"]["tier"] == 1 and not x["rk"]["anomaly"]], key=lambda x: -x["rk"]["up27"])
    for i, x in enumerate(rankable, 1): x["up_rank"] = i
    scanned = sorted([x for x in rows if x["sc"]], key=lambda x: (-x["sc"]["n_flags"], x["sc"]["avg_pctile"], -x["sc"]["n_turn"]))
    for i, x in enumerate(scanned, 1): x["os_rank"] = i
    n_up, n_os = len(rankable), len(scanned)
    for x in rows:
        ur, orr = x.get("up_rank"), x.get("os_rank")
        x["both"] = ur is not None and orr is not None
        x["combo"] = (ur + orr) if x["both"] else 10 ** 6 + (orr if orr else 10 ** 5) + (0 if ur else 10 ** 4)
    rows.sort(key=lambda x: (x["combo"], -(x["rk"]["up27"] if x["rk"] and x["rk"].get("up27") is not None else -99)))
    for i, x in enumerate(rows, 1): x["c_rank"] = i
    return rows, n_up, n_os


def combined_html(pf, rows, n_up, n_os, asof):
    body = []
    for x in rows:
        r, s, t = x["rk"], x["sc"], x["ticker"]
        if r:
            px = f"{r['price']:,.2f}" + ("" if r["price_note"] == "override" else " <small>(report)</small>")
            tgt = f"<b>{mn(r.get('target27'))}</b>" if r.get("target27") else "—"
            up = r.get("up27"); bear = r.get("bear_up")
            exitpe = (mu(r.get("exit_pe")) + (" ✂" if r["haircut"] else "")) if r.get("exit_pe") else "—"
            drv = esc(r.get("driver") or r.get("haircut_reason") or "", 110)
            drv_full = H.escape((r.get("driver") or "") + ((" | Haircut: " + r["haircut_reason"]) if r.get("haircut_reason") else ""))
        else:
            px = f"{s['price']:,.2f}" if s else "—"; tgt = up = bear = None; exitpe = "—"; drv = "no IBES/Refinitiv report in this portfolio"; drv_full = drv
        st = status(r) if r else "no report"
        st_style = ("background:#eee" if not r or r["tier"] == 2 else "background:#ddd;color:#555" if r["anomaly"] else "background:#fbe7c6" if r["demoted"] else "background:#dff0d8")
        if r and r.get("source_pf"): drv_full += f" [report from {r['source_pf']} portfolio]"; st += f" <small>({r['source_pf']})</small>"
        body.append([
            (x["c_rank"] if x["both"] else "—", "font-weight:bold"), (tl(t), ""), (esc(name_of(t, r), 28), "font-size:12px"),
            (x.get("up_rank") or "n/r", f"background:{'#d9ead3' if x.get('up_rank') and x['up_rank'] <= max(3, n_up // 4) else ''}"),
            (x.get("os_rank") or "n/a", f"background:{'#f6c9c9' if x.get('os_rank') and x['os_rank'] <= max(3, n_os // 4) else ''}"),
            (f"{px} → {tgt}", "white-space:nowrap"), (pc(up), up_col(up) + ";font-weight:600"), (pc(bear), up_col(bear) if bear is not None else ""),
            (f"<b>{s['n_flags']}</b> · ↑{s['n_turn']}" if s else "—", f"white-space:nowrap;background:{'#f6c9c9' if s and s['n_flags'] >= 7 else '#fbe7c6' if s and s['n_flags'] >= 4 else '#eee'}"),
            (f"{s['avg_pctile']:.0f}" if s else "—", f"background:{pct_color(s['avg_pctile'])}" if s else ""),
            (st, st_style + ";font-size:12px;white-space:nowrap"), (f"<span title='{drv_full}'>{drv}</span>", "font-size:12px;min-width:360px")])
    hdr = ["Combined #", "Ticker", "Company", "Upside rank", "Oversold rank", "Price → target end-2027", "Upside", "Bear case", "Flags /10 · turn /4", "Avg pctile", "Status", "Driver / risk"]
    intro = (f"<p class='hint'><b>How to read:</b> <i>Upside rank</i> = position by upside to the engine's end-2027 target (2028E EPS × PEG-0.8 exit P/E, 30x cap) among the {n_up} names that are "
             f"rankable on earnings (Tier 1, not ANOMALY). <i>Oversold rank</i> = position by the 10-test checklist (flags, then average percentile, then turn signals) among the {n_os} names with price history. "
             f"<i>Combined #</i> = order by the sum of the two ranks (lower = better on both). Names that cannot be ranked on one dimension are listed after, with n/r (no rankable target) or n/a (no price data). "
             f"<i>Bear case</i> = (2028E − 1 SD) × trough P/E; <i>Flags</i> = oversold tests firing, <i>↑turn</i> = reversal signals; <i>Avg pctile</i> = today's readings vs own 4-yr history (0 = most oversold ever). "
             f"Full EPS / multiples detail is on the Price targets tab, full indicator detail on the Oversold ranking tab. Click a column header to sort, a ticker to drill in. Prices as of {asof}.</p>")
    return intro + "<div id='fig-scatter' class='fig'></div>" + table_html(hdr, body, "sortable")


def fig_scatter(rows, title):
    pts = [x for x in rows if x["both"]]
    if not pts: return go.Figure(layout=dict(title="No names rankable on both dimensions", height=300))
    ys = [min(x["rk"]["up27"] * 100, 400) for x in pts]
    col = ["#e8a33c" if x["rk"]["demoted"] else "#2b7bba" for x in pts]
    hover = [f"<b>{x['ticker']}</b> {x['rk'].get('company') or ''}<br>Target end-2027 {mn(x['rk']['target27'])} ({pc(x['rk']['up27'])}{' – clipped at +400%' if x['rk']['up27'] > 4 else ''})<br>"
             f"Bear {pc(x['rk'].get('bear_up'))} · {status(x['rk'])}<br>Oversold flags {x['sc']['n_flags']}/10 · avg pctile {x['sc']['avg_pctile']:.0f} · turn {x['sc']['n_turn']}/4<br>"
             f"Upside rank {x['up_rank']} · Oversold rank {x['os_rank']} · Combined #{x['c_rank']}" for x in pts]
    fig = go.Figure(go.Scatter(x=[x["sc"]["avg_pctile"] for x in pts], y=ys, mode="markers+text", text=[x["ticker"] for x in pts], textposition="top center",
                               textfont=dict(size=10), hovertext=hover, hoverinfo="text",
                               marker=dict(size=[8 + 2 * x["sc"]["n_flags"] for x in pts], color=col, line=dict(width=1, color="#333"), opacity=0.85)))
    fig.add_vline(x=20, line_dash="dot", line_color="#c82828"); fig.add_hline(y=50, line_dash="dot", line_color="#2a7a2a")
    fig.add_annotation(x=99, y=395, text="more oversold (own-history percentile) ▶ · ▲ more upside — the top-right corner is the sweet spot", showarrow=False, xanchor="left", font=dict(size=11, color="#555"))
    fig.update_layout(title=title, height=560, xaxis=dict(title="Average indicator percentile vs own 4-yr history (0 = most oversold ever)", autorange="reversed", range=[100, 0]),
                      yaxis=dict(title="Upside to end-2027 target (%, clipped at 400)"), margin=dict(t=60, b=50))
    fig.update_layout(annotations=fig.layout.annotations + (dict(x=0.99, y=0.02, xref="paper", yref="paper", showarrow=False, align="right", font=dict(size=11),
                                                                  text="marker size = oversold flags · blue = Tier 1 · orange = demoted (bear < −30%)<br>click a marker to drill into the stock"),))
    return fig


# ----------------------------------------------------------------------------- price-target tab
def targets_html(pf, rows, asof, missing_scan):
    rk = [x["rk"] for x in sorted(rows, key=lambda x: (x["rk"] or {}).get("rank", 9999)) if x["rk"]]
    if not rk: return "<p>This portfolio has no IBES / Refinitiv reports, so no earnings-based price targets are available. See the Oversold ranking tab.</p>"
    t1 = [r for r in rk if r["tier"] == 1 and not r["demoted"] and not r["anomaly"]]
    dem = [r for r in rk if r["tier"] == 1 and r["demoted"] and not r["anomaly"]]
    an = [r for r in rk if r["tier"] == 1 and r["anomaly"]]
    t2 = [r for r in rk if r["tier"] == 2]
    rep_dates = sorted({str(r["report_date"]) for r in rk if r.get("report_date")})
    hair = [r["ticker"] for r in rk if r["haircut"]]
    notes = [f"<b>{r['ticker']}</b>: {r['split_note']}" for r in rk if r.get("split_note")] + [f"<b>{r['ticker']}</b>: {r['currency_note']}" for r in rk if r.get("currency_note")]
    borrowed = [f"{r['ticker']} ({r['source_pf']})" for r in rk if r.get("source_pf")]
    if borrowed:
        no_rep = [x["ticker"] for x in rows if not x["rk"]]
        notes.append(f"<br><b>This list has no reports of its own</b> — rows are borrowed from the portfolio that covers each name: {', '.join(borrowed)}." +
                     (f" No report in any portfolio for: {', '.join(no_rep)}." if no_rep else ""))
    basis = (f"<p class='hint'><b>Data basis.</b> Consensus EPS / revenue / sell-side targets from the IBES & Refinitiv reports dated {', '.join(rep_dates)}; "
             f"<b>prices re-set to the latest Yahoo quote ({asof})</b> for every name (so P/E 27, upside and bear % move with the market while the estimates stay at report date). "
             f"Tags: <sup>S</sup> consensus as reported · <sup>X</sup> extrapolated (no FY+2 consensus: 2028E = 2027E + 0.75 × (2027E − 2026E)) · <sup>D</sup> analyst-adjusted (see driver). "
             f"Exit P/E haircut ×0.8 applied to: {', '.join(hair) or 'none'}. " + (" ".join(notes) if notes else "") + "</p>")

    def t1row(r, i):
        return [(i, ""), (tl(r["ticker"]), ""), (esc(r.get("company"), 26), "font-size:12px"), (f"{r['price']:,.2f}" + ("" if r["price_note"] == "override" else " <small>(report)</small>"), ""),
                (eps_str(r), "white-space:nowrap"), (pc(r["g"]), ""), (mu(r["pe27"]), ""), (mn(r.get("peg27")), ""), (mu(r["exit_pe"]) + (" ✂" if r["haircut"] else ""), ""),
                (f"<b>{mn(r['target27'])}</b> ({pc(r['up27'])})", up_col(r["up27"])), (f"{mn(r.get('bear27'))} ({pc(r.get('bear_up'))})", up_col(r.get("bear_up")) if r.get("bear_up") is not None else ""),
                (f"{mn(r.get('pt_mean'))} ({pc(r.get('pt_up'))})" if r.get("pt_mean") else "n/m", ""), (f"{mn(r['rev27']/1e9 if r.get('rev27') and r['rev27']>1e7 else r.get('rev27'),2)}B ({pc(r.get('rev27_yoy'))})" if r.get("rev27") else "n/a", ""),
                (pc(r.get("debt_to_capital")) if r.get("debt_to_capital") is not None else "n/m", ""), (pc(r.get("eps27_chg90")) if r.get("eps27_chg90") is not None else "n/m", ""),
                (r.get("n28") or r.get("n27") or "—", ""), (f"<span title='{H.escape(r.get('driver') or '')}'>{esc(r.get('driver'), 120)}</span>", "font-size:12px")]
    hdr1 = ["#", "Ticker", "Company", "Px", "EPS 26E / 27E / 28E", "g 27→28", "P/E 27", "PEG", "Exit P/E", "Target end-2027 (upside)", "Bear end-2027", "Sell-side PT", "Rev 27E (YoY)", "D/C", "27E rev 90d", "# brokers", "Driver / risk"]
    out = [basis, f"<h3>Tier 1 — rankable on earnings ({len(t1)})</h3><p class='hint'>Target end-2027 = 2028E × exit P/E; exit P/E = min(g, 30x) if g ≥ 30% else max(0.8 g, 10x). Bear = (IBES 2028E − 1 SD) × trough P/E. Sorted by upside.</p>",
           table_html(hdr1, [t1row(r, i) for i, r in enumerate(t1, 1)], "sortable")]
    if dem: out += [f"<h3>Demoted — bear case worse than −30% ({len(dem)})</h3>", table_html(hdr1, [t1row(r, i) for i, r in enumerate(dem, 1)], "sortable")]
    if an: out += [f"<h3>ANOMALY — target mechanically inflated ({len(an)})</h3><p class='hint'>Exit multiple (30x cap / 10x floor) is ≥2× the name's structural 5-yr forward P/E, or the consensus is 2–3 brokers with wide dispersion, or EPS is a commodity / FX / cyclical deck. Ranked for completeness, excluded from the upside rank.</p>",
                   table_html(hdr1, [t1row(r, i) for i, r in enumerate(an, 1)], "sortable")]
    # multiples / durability
    mrows = [[(tl(r["ticker"]), ""), (f"{r['price']:,.2f}", ""), (mu(r.get("pe_fwd_now")) if r.get("pe_fwd_now") else "n/a", ""), (esc(r.get("pe_fwd_now_basis") or ""), "font-size:11px"), (mu(r["pe27"]), ""),
              (mu(r["exit_pe"]), ""), (mu(r.get("pe_fwd_5y")) if r.get("pe_fwd_5y") else "n/a", ""),
              (f"{r['exit_pe']/r['pe_fwd_5y']:.2f}x" if r.get("pe_fwd_5y") else "—", "color:crimson;font-weight:bold" if r.get("pe_fwd_5y") and r["exit_pe"] / r["pe_fwd_5y"] >= 2 else ""),
              (mn(r.get("eps29")), ""), (pc(r.get("g29")), ""), (f"{mn(r.get('target29'))} ({pc(r.get('up29'))})", up_col(r.get("up29"))), (pc(r.get("cagr_26_29")), ""),
              (f"{mn(r.get('target27_held'))} ({pc(r.get('up27_held'))})", up_col(r.get("up27_held")))] for r in t1 + dem + an]
    out += ["<h3>Multiples & durability (all Tier-1 names)</h3><p class='hint'>Re-rating check: current forward P/E at today's price vs the 2027 exit P/E vs the 5-yr average forward P/E (ratio ≥ 2 in red = the target needs a doubling of the multiple). "
            "Durability: 2029E = 2028E × (1 + 0.75 g), exit P/E held. 'Multiple held' = 2028E × today's 2027 P/E (what you get if nothing re-rates).</p>",
            table_html(["Ticker", "Px", "Fwd P/E now", "Basis", "P/E 27", "Exit P/E 27", "5-yr avg fwd P/E", "Exit ÷ 5-yr", "EPS 29E", "g 28→29", "Target end-2029", "EPS CAGR 26–29", "Target 27 multiple held"], mrows, "sortable small")]
    if t2:
        out += [f"<h3>Tier 2 — not rankable on earnings ({len(t2)})</h3><p class='hint'>2027E or 2028E ≤ 0, or first-profit year with >50x P/E and no FY+2 consensus, or forced (earnings not underwritable). Ordered by sell-side target upside.</p>",
                table_html(["Ticker", "Company", "Px", "EPS 26E / 27E / 28E", "Sell-side PT (upside)", "Rev 27E (YoY)", "Driver / risk"],
                           [[(tl(r["ticker"]), ""), (esc(r.get("company"), 28), "font-size:12px"), (f"{r['price']:,.2f}", ""), (eps_str(r), "white-space:nowrap"),
                             (f"{mn(r.get('pt_mean'))} ({pc(r.get('pt_up'))})" if r.get("pt_mean") else "n/m", ""), (f"{mn(r['rev27']/1e9,2)}B ({pc(r.get('rev27_yoy'))})" if r.get("rev27") and r["rev27"] > 1e7 else "n/a", ""),
                             (esc(r.get("driver"), 160), "font-size:12px")] for r in t2], "sortable")]
    if missing_scan: out.append(f"<p class='hint'>No Yahoo price history for: {', '.join(missing_scan)} (oversold metrics unavailable).</p>")
    out.append("<h3>What to watch</h3>" + table_html(["Ticker", "Next earnings", "Target basis", "Haircut reason"],
        [[(tl(r["ticker"]), ""), (esc(r.get("next_earnings") or "n/a"), ""), ({"S": "IBES 2028E consensus", "X": "extrapolated 2028E", "D": "analyst-adjusted 2028E"}.get(r["tag"]["eps28"], "—"), "color:#b35c00;font-weight:bold" if r["tag"]["eps28"] == "X" else ""),
          (esc(r.get("haircut_reason") or "—"), "font-size:12px")] for r in t1 + dem], "small sortable"))
    out.append("<p class='hint'><i>Model output from consensus data and fixed rules — not personalized investment advice.</i></p>")
    return "".join(out)


def stock_target_html(r, pf):
    if not r: return f"<p>No IBES / Refinitiv report for this ticker in the <b>{pf['name']}</b> portfolio.</p>"
    g = lambda k, f=mn: f(r.get(k)) if r.get(k) is not None else "—"
    kv = [("Price used", f"{r['price']:,.2f} ({'Yahoo ' + str(r.get('price_asof')) if r['price_note']=='override' else 'report ' + str(r['price_note'])})"), ("Report price / date", f"{mn(r['report_price'])} / {r.get('report_date')}"),
          ("Status", status(r)), ("EPS 26E / 27E / 28E", eps_str(r)), ("2028E range (low–high, SD)", f"{g('eps28_low')} – {g('eps28_high')} (SD {g('sd28')})"),
          ("Growth 27→28 (g)", pc(r.get("g"))), ("P/E on 2027E today", mu(r.get("pe27")) if r.get("pe27") else "—"), ("PEG (P/E 27 ÷ g)", g("peg27")),
          ("Exit P/E end-2027", (mu(r["exit_pe"]) + (f" (haircut ×0.8 from {mu(r['exit_pe_raw'])})" if r["haircut"] else "")) if r.get("exit_pe") else "—"),
          ("<b>Target end-2027</b>", f"<b>{mn(r.get('target27'))} ({pc(r.get('up27'))})</b>" if r.get("target27") else "—"),
          ("Bear end-2027", f"{mn(r.get('bear27'))} ({pc(r.get('bear_up'))}) — {r.get('bear_basis') or ''} × trough {mu(r.get('trough_pe')) if r.get('trough_pe') else ''}" if r.get("bear27") else "— (bear EPS ≤ 0)" if r.get("tier") == 1 else "—"),
          ("Target 2027 if multiple held", f"{mn(r.get('target27_held'))} ({pc(r.get('up27_held'))})" if r.get("target27_held") else "—"),
          ("2029E / target end-2029", f"{g('eps29')} / {mn(r.get('target29'))} ({pc(r.get('up29'))})" if r.get("target29") else "—"), ("EPS CAGR 26–29", pc(r.get("cagr_26_29"))),
          ("Fwd P/E now (basis)", f"{mu(r['pe_fwd_now'])} ({r.get('pe_fwd_now_basis')})" if r.get("pe_fwd_now") else "n/a"), ("5-yr avg fwd P/E", mu(r.get("pe_fwd_5y")) if r.get("pe_fwd_5y") else "n/a"),
          ("Sell-side mean PT", f"{mn(r.get('pt_mean'))} ({pc(r.get('pt_up'))}, {r.get('pt_n') or '?'} analysts)" if r.get("pt_mean") else "n/m"), ("Recommendation", f"{r.get('rec') or '—'} ({r.get('rec_n') or '?'})"),
          ("Rev 26E → 27E", f"{mn((r.get('rev26') or 0)/1e9,2)}B → {mn((r.get('rev27') or 0)/1e9,2)}B ({pc(r.get('rev27_yoy'))})" if r.get("rev27") and r["rev27"] > 1e7 else "n/a"),
          ("27E EPS revision 90d / brokers", f"{pc(r.get('eps27_chg90'))} / {r.get('n28') or r.get('n27') or '—'}"), ("Debt/capital · net margin", f"{pc(r.get('debt_to_capital'))} · {pc(r.get('net_margin'))}"),
          ("Beta · worst month (60m)", f"{g('beta')} · {pc(r.get('worst_month'))}"), ("Next earnings", r.get("next_earnings") or "n/a"),
          ("Lead product % / LOE", f"{r.get('lead_product_pct') if r.get('lead_product_pct') is not None else '—'} / {r.get('loe_year') or '—'}")]
    tbl = table_html(["Metric", "Value"], [[(k, "font-weight:600;width:260px"), (v, "")] for k, v in kv], "small")
    txt = ""
    if r.get("driver"): txt += f"<h4>Driver / risk</h4><p>{H.escape(r['driver'])}</p>"
    if r.get("haircut_reason"): txt += f"<h4>Haircut reason</h4><p>{H.escape(r['haircut_reason'])}</p>"
    for k in ("split_note", "currency_note"):
        if r.get(k): txt += f"<p class='hint'>⚠ {H.escape(r[k])}</p>"
    if r.get("business_summary"): txt += f"<h4>Business</h4><p class='hint'>{H.escape(r['business_summary'][:900])}</p>"
    if r.get("earnings_highlights"): txt += f"<h4>Earnings highlights (report)</h4><p class='hint'>{H.escape(str(r['earnings_highlights'])[:900])}</p>"
    src = f" (report borrowed from the <b>{r['source_pf']}</b> portfolio run)" if r.get("source_pf") else ""
    return f"<p>Price-target analysis from the <b>{pf['name']}</b> run{src} (Biotech Ranker engine rules, prices as of today).</p><div class='row'><div class='col'>{tbl}</div><div class='col'>{txt}</div></div>"


# ----------------------------------------------------------------------------- portfolio page
def portfolio_variant(pf, tickers, rk_rows, scan, asof, failed):
    sub = dict(pf, tickers=tickers)
    rows, n_up, n_os = combined_rows(sub, rk_rows, scan)
    res = [scan[t] for t in tickers if t in scan]; res.sort(key=lambda r: (-r["n_flags"], r["avg_pctile"], -r["n_turn"]))
    missing = [t for t in tickers if t not in scan]
    v = {"order": [r["ticker"] for r in res], "combined": combined_html(sub, rows, n_up, n_os, asof), "targets": targets_html(sub, rows, asof, missing),
         "figs": {"scatter": fj(fig_scatter(rows, f"{pf['name']} — upside to end-2027 target vs oversold depth"))}}
    if res:
        names = {x["ticker"]: name_of(x["ticker"], x["rk"]) for x in rows}
        v["ranking"] = d1.ranking_html(res, names) + (f"<p class='hint'>No Yahoo price history for: {', '.join(missing)}.</p>" if missing else "")
        v["figs"]["heat"] = fj(fig_heatmap(res)); v["figs"]["flags"] = fj(fig_flags(res))
        v["method"] = d1.methodology_html(res)
    else:
        v["ranking"] = v["method"] = "<p>No price history available.</p>"
    v["summary"] = {"n": len(tickers), "n_up": n_up, "n_os": n_os, "deep": sum(r["n_flags"] >= 7 for r in res),
                    "top": [(x["ticker"], x["c_rank"]) for x in rows if x["both"]][:5], "top_os": [r["ticker"] for r in res[:5]],
                    "top_up": [x["ticker"] for x in sorted([x for x in rows if x.get("up_rank")], key=lambda x: x["up_rank"])[:5]]}
    return v, rows


def build():
    portfolios = json.load(open(os.path.join(CACHE, "portfolios.json")))
    rk = pickle.load(open(os.path.join(CACHE, "rankings.pkl"), "rb"))
    scan = pickle.load(open(os.path.join(CACHE, "scan_results.pkl"), "rb"))
    failed = json.load(open(os.path.join(CACHE, "scan_failed.json"))) if os.path.exists(os.path.join(CACHE, "scan_failed.json")) else {}
    prices = json.load(open(os.path.join(CACHE, "prices.json")))
    asof = max(prices["asof"].values())
    for d in ("data/p", "data/s"): os.makedirs(os.path.join(SITE, d), exist_ok=True)
    meta = []
    for pf in portfolios:
        rows_all = rk[pf["id"]]["rows"] if pf["id"] in rk else None
        if rows_all is None:
            # no report run for this list: borrow each ticker's row from the first other portfolio that covers it
            rows_all = []
            for t in pf["tickers"]:
                for o in portfolios:
                    if o["id"] in rk and t in o["tickers"]:
                        src = next((r for r in rk[o["id"]]["rows"] if r["ticker"] == t), None)
                        if src: rows_all.append(dict(src, source_pf=o["short"])); break
            rows_all = rows_all or None
        ai = sorted(t for t in pf["tickers"] if (rows_all and any(r["ticker"] == t and r["ai_infra"] for r in rows_all)))
        variants = {}
        variants["all"], rows = portfolio_variant(pf, pf["tickers"], rows_all, scan, asof, failed)
        if pf["ai_filter"] and ai:
            variants["noai"], _ = portfolio_variant(pf, [t for t in pf["tickers"] if t not in ai], rows_all, scan, asof, failed)
        rkmap = {r["ticker"]: r for r in (rows_all or [])}
        page = {"id": pf["id"], "name": pf["name"], "desc": pf["desc"], "tickers": pf["tickers"], "ai": ai, "ai_filter": bool(pf["ai_filter"] and ai),
                "variants": variants, "target_html": {t: stock_target_html(rkmap.get(t), pf) for t in pf["tickers"]},
                "flags": {t: scan[t]["n_flags"] for t in pf["tickers"] if t in scan}}
        json.dump(page, open(os.path.join(SITE, "data/p", f"{pf['id']}.json"), "w", encoding="utf-8"), default=str)
        meta.append({"id": pf["id"], "name": pf["name"], "short": pf["short"], "desc": pf["desc"], "ai_filter": page["ai_filter"], "n_ai": len(ai), "summary": variants["all"]["summary"]})
        print(pf["id"], "written;", len(pf["tickers"]), "tickers,", len(ai), "AI-tagged")
    # per-stock pages (shared)
    allt = sorted({t for pf in portfolios for t in pf["tickers"]})
    membership = {t: [pf["id"] for pf in portfolios if t in pf["tickers"]] for t in allt}
    for t in allt:
        if t not in scan: continue
        r = scan[t]
        s = {"html": d1.stock_blocks(r), "figs": {"main": fj(fig_ticker(r)), "pct": fj(fig_pctile_bars(r))}, "in": membership[t]}
        json.dump(s, open(os.path.join(SITE, "data/s", f"{t}.json"), "w", encoding="utf-8"), default=str)
    json.dump({"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "asof": asof, "portfolios": meta, "membership": membership,
               "failed": failed, "all": allt}, open(os.path.join(SITE, "data/portfolios.json"), "w", encoding="utf-8"))
    from plotly.offline import get_plotlyjs
    with open(os.path.join(SITE, "plotly.min.js"), "w", encoding="utf-8") as f: f.write(get_plotlyjs())
    write_shell(meta, asof)
    print("site built:", SITE)


# ----------------------------------------------------------------------------- shell
def write_shell(meta, asof):
    css = """
    body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f9;color:#222}
    header{background:#1f2f44;color:#fff;padding:8px 24px;display:flex;align-items:center;gap:18px;flex-wrap:wrap;position:sticky;top:0;z-index:10}
    header h1{font-size:18px;margin:0} header .meta{font-size:12px;color:#c9d3df}
    header select{font-size:14px;padding:4px 8px;border-radius:6px;border:none} header label{font-size:13px}
    nav.tabs{display:flex;gap:4px;background:#2b3e55;padding:6px 24px 0;position:sticky;top:44px;z-index:9;flex-wrap:wrap}
    nav.tabs button{background:#3b5273;color:#dde;border:none;padding:9px 16px;border-radius:8px 8px 0 0;cursor:pointer;font-size:14px}
    nav.tabs button.active{background:#f4f6f9;color:#1f2f44;font-weight:600} nav.tabs button:disabled{opacity:.4;cursor:default}
    .wrap{max-width:1600px;margin:0 auto;padding:14px 24px 40px} .pane{display:none} .pane.active{display:block}
    .toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:6px 0 12px;background:#fff;padding:10px 14px;border-radius:8px;border:1px solid #e1e5ea}
    .toolbar select{font-size:15px;padding:5px 8px} .toolbar button{padding:5px 12px;cursor:pointer;border:1px solid #bbb;background:#fff;border-radius:6px}
    #back{background:#2b7bba;color:#fff;border-color:#2b7bba;display:none} #back.show{display:inline-block}
    .aitog{margin-left:auto;background:#fff3cd;padding:5px 10px;border-radius:6px;border:1px solid #e0c97a;font-size:13px} .aitog.off{display:none}
    nav.sub{display:flex;gap:4px;margin:4px 0 10px;flex-wrap:wrap} nav.sub button{background:#e3e8ef;border:none;padding:7px 14px;border-radius:6px;cursor:pointer}
    nav.sub button.active{background:#2b7bba;color:#fff;font-weight:600}
    .tbl{border-collapse:collapse;width:100%;margin:8px 0 16px;font-size:13px;background:#fff} .tbl th{background:#2b3e55;color:#fff;padding:6px 8px;text-align:left;position:sticky;top:88px}
    .tbl td{padding:5px 8px;border-bottom:1px solid #e5e5e5} .tbl.small{font-size:12px} .tbl tr:hover td{filter:brightness(.97)} .tbl sup{font-size:9px;color:#b35c00}
    .tbl.sortable th{cursor:pointer} .tbl.sortable th:hover{background:#3b5273} .tbl.sortable th.asc::after{content:' ▲'} .tbl.sortable th.desc::after{content:' ▼'}
    .row{display:flex;flex-wrap:wrap;gap:16px} .col{flex:1 1 440px} .ep{background:#fff;border:1px solid #e3e3e3;border-radius:6px;padding:8px 12px;flex:1 1 440px;max-width:720px}
    a{color:#2b7bba} a.tl{text-decoration:none} a.tl:hover{text-decoration:underline} .hint{color:#555;font-size:13px} h3{margin-top:28px} h4{margin:18px 0 6px}
    .fig{background:#fff;border:1px solid #e1e5ea;border-radius:8px;margin:10px 0}
    .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px} .card{background:#fff;border:1px solid #dde;border-radius:10px;padding:14px 18px;cursor:pointer;transition:box-shadow .15s}
    .card:hover{box-shadow:0 4px 14px rgba(0,0,0,.12)} .card h3{margin:0 0 6px} .card.sel{border:2px solid #2b7bba} .card .st{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;margin:8px 0} .card .st b{font-size:18px;display:block}
    .card .lk{font-size:12px;color:#555} .badge{display:inline-block;background:#fff3cd;border:1px solid #e0c97a;border-radius:4px;padding:1px 6px;font-size:11px;margin-left:6px}
    #loading{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#1f2f44;color:#fff;padding:14px 26px;border-radius:10px;display:none;z-index:50}
    """
    js = r"""
const META = __META__;
const TABS = {portfolios:'Portfolios', combined:'Combined ranking', targets:'Price targets', ranking:'Oversold ranking', heat:'Heat-map', stock:'Stock detail', method:'Methodology'};
const SUBS = {readings:'Readings', charts:'Charts', history:'Episode history', lows:'Swing lows', target:'Price target'};
let S = {pf:'tipranks', tab:'portfolios', ticker:null, sub:'readings', noai:false}; let stack=[]; const drawn={}; const P={}, ST={};
const $ = id=>document.getElementById(id);
function plot(id, fig){ const el=$(id); if(!el||!fig) return null; Plotly.react(el, fig.data, fig.layout, {displaylogo:false, responsive:true}); return el; }
async function fetchJ(u){ $('loading').style.display='block'; try{ const r=await fetch(u); if(!r.ok) throw new Error(u); return await r.json(); } finally{ $('loading').style.display='none'; } }
async function pf(){ if(!P[S.pf]) P[S.pf]=await fetchJ(`data/p/${S.pf}.json`); return P[S.pf]; }
async function stock(t){ if(!ST[t]) { try{ ST[t]=await fetchJ(`data/s/${t}.json`); } catch(e){ ST[t]=null; } } return ST[t]; }
function variant(p){ return (S.noai && p.variants.noai) ? p.variants.noai : p.variants.all; }
function visible(p){ return S.noai && p.ai_filter ? p.tickers.filter(t=>!p.ai.includes(t)) : p.tickers; }
function cards(){
  $('cards').innerHTML = META.portfolios.map(m=>{const s=m.summary; return `<div class='card ${m.id===S.pf?'sel':''}' data-pf='${m.id}'><h3>${m.name}${m.ai_filter?`<span class='badge'>AI chip/infra filter · ${m.n_ai} tagged</span>`:''}</h3>
    <div class='hint'>${m.desc}</div><div class='st'><span><b>${s.n}</b>stocks</span><span><b>${s.n_up}</b>rankable targets</span><span><b>${s.deep}</b>≥7 oversold flags</span><span><b>${s.n_os}</b>with price history</span></div>
    <div class='lk'>Best on both: <b>${s.top.map(x=>x[0]).join(', ')||'—'}</b><br>Most oversold: ${s.top_os.join(', ')||'—'}<br>Highest target upside: ${s.top_up.join(', ')||'—'}</div>
    <div style='margin-top:8px'><button class='open' data-pf='${m.id}' data-tab='combined'>Open combined ranking ▶</button> <button class='open' data-pf='${m.id}' data-tab='targets'>Price targets</button> <button class='open' data-pf='${m.id}' data-tab='ranking'>Oversold</button></div></div>`;}).join('');
}
function sortable(root){ root.querySelectorAll('table.sortable').forEach(tb=>{ tb.querySelectorAll('th').forEach((th,i)=>{ th.onclick=()=>{ const tbody=tb.tBodies[0]; const rows=[...tbody.rows]; const asc=!th.classList.contains('asc');
  tb.querySelectorAll('th').forEach(h=>h.classList.remove('asc','desc')); th.classList.add(asc?'asc':'desc');
  const val=r=>{ const s=r.cells[i].textContent.replace(/[,%$x]/g,'').trim(); const m=s.match(/^[-+]?\d+(\.\d+)?/); return m?parseFloat(m[0]):(s==='—'||s==='n/r'||s==='n/a'||s===''?(asc?1e12:-1e12):s.toLowerCase()); };
  rows.sort((a,b)=>{ const x=val(a), y=val(b); if(typeof x==='number'&&typeof y==='number') return asc?x-y:y-x; return asc?String(x).localeCompare(String(y)):String(y).localeCompare(String(x)); }); rows.forEach(r=>tbody.appendChild(r)); }; }); }); }
async function render(){
  const p = await pf(); const v = variant(p); const m = META.portfolios.find(x=>x.id===S.pf);
  document.querySelectorAll('nav.tabs button').forEach(b=>b.classList.toggle('active', b.dataset.tab===S.tab));
  document.querySelectorAll('.pane').forEach(x=>x.classList.toggle('active', x.id==='pane-'+S.tab));
  $('pfsel').value=S.pf; $('pfname').textContent=m.name; $('aitog').classList.toggle('off', !p.ai_filter); $('aichk').checked=S.noai;
  const back=$('back'); back.classList.toggle('show', stack.length>0);
  if(stack.length){ const q=stack[stack.length-1]; back.textContent='← Back to '+TABS[q.tab]+(q.tab==='stock'?' ('+q.ticker+')':'')+(q.pf!==S.pf?' · '+META.portfolios.find(x=>x.id===q.pf).short:''); }
  const key=S.pf+(S.noai?'/noai':'');
  if(S.tab==='portfolios') cards();
  if(S.tab==='combined' && drawn.combined!==key){ $('combined-body').innerHTML=v.combined; sortable($('combined-body')); const el=plot('fig-scatter', v.figs.scatter); if(el) el.on('plotly_click', e=>go('stock',{ticker:e.points[0].text, sub:'readings'})); drawn.combined=key; }
  if(S.tab==='targets' && drawn.targets!==key){ $('targets-body').innerHTML=v.targets; sortable($('targets-body')); drawn.targets=key; }
  if(S.tab==='ranking' && drawn.ranking!==key){ $('ranking-body').innerHTML=v.ranking; sortable($('ranking-body')); const el=plot('fig-flags', v.figs.flags); if(el) el.on('plotly_click', e=>go('stock',{ticker:e.points[0].x})); drawn.ranking=key; }
  if(S.tab==='heat' && drawn.heat!==key){ const el=plot('fig-heat', v.figs.heat); if(el) el.on('plotly_click', e=>go('stock',{ticker:e.points[0].y})); drawn.heat=key; }
  if(S.tab==='method' && drawn.method!==key){ $('method-body').innerHTML=v.method; drawn.method=key; }
  if(S.tab==='stock'){
    const list = v.order.concat(visible(p).filter(t=>!v.order.includes(t)));
    if(!S.ticker || !p.tickers.includes(S.ticker)) S.ticker=list[0];
    $('tsel').innerHTML=list.map(t=>`<option value='${t}'>${t}${p.flags[t]!==undefined?' — '+p.flags[t]+'/10':' — no price data'}</option>`).join(''); $('tsel').value=S.ticker;
    document.querySelectorAll('nav.sub button').forEach(b=>b.classList.toggle('active', b.dataset.sub===S.sub));
    const others=(META.membership[S.ticker]||[]).filter(x=>x!==S.pf).map(x=>`<a href='#' class='pl' data-pf='${x}'>${META.portfolios.find(y=>y.id===x).short}</a>`).join(', ');
    $('stock-title').innerHTML=S.ticker+(others?` <small style='font-size:12px;font-weight:normal'>also in: ${others}</small>`:'');
    const body=$('stock-body');
    if(S.sub==='target'){ body.innerHTML=p.target_html[S.ticker]||'<p>No report.</p>'; }
    else { const s=await stock(S.ticker); if(!s){ body.innerHTML='<p>No Yahoo price history for this ticker — oversold analysis unavailable. See the Price target sub-tab.</p>'; }
      else if(S.sub==='charts'){ body.innerHTML="<div id='fig-main' class='fig'></div>"; plot('fig-main', s.figs.main); }
      else { body.innerHTML=s.html[S.sub]; if(S.sub==='readings') plot('fig-pct', s.figs.pct); } }
  }
  location.hash = [S.pf, S.tab].concat(S.tab==='stock'?[S.ticker,S.sub]:[]).join('/')+(S.noai?'?noai':''); lastHash=location.hash;
  window.scrollTo(0,0);
}
function go(tab, opts={}, push=true){ if(push){ stack.push({...S}); if(stack.length>30) stack.shift(); } S={...S, tab, ...opts}; render(); }
function back(){ const q=stack.pop(); if(q){ S=q; render(); } }
async function step(d){ const p=await pf(); const v=variant(p); const list=v.order.concat(visible(p).filter(t=>!v.order.includes(t))); const i=list.indexOf(S.ticker); S.ticker=list[(i+d+list.length)%list.length]; render(); }
document.addEventListener('click', e=>{ const a=e.target.closest('a.tl'); if(a){ e.preventDefault(); go('stock',{ticker:a.dataset.t, sub:'readings'}); return; }
  const pl=e.target.closest('a.pl'); if(pl){ e.preventDefault(); go('stock',{pf:pl.dataset.pf, sub:'target'}); return; }
  const o=e.target.closest('button.open'); if(o){ e.stopPropagation(); go(o.dataset.tab,{pf:o.dataset.pf, noai:false}); return; }
  const c=e.target.closest('.card'); if(c){ go('combined',{pf:c.dataset.pf, noai:false}); } });
document.querySelectorAll('nav.tabs button').forEach(b=>b.onclick=()=>go(b.dataset.tab));
document.querySelectorAll('nav.sub button').forEach(b=>b.onclick=()=>{ S.sub=b.dataset.sub; render(); });
$('tsel').onchange=e=>{ S.ticker=e.target.value; render(); };
$('pfsel').onchange=e=>{ go(S.tab==='portfolios'?'combined':S.tab, {pf:e.target.value, noai:false}); };
$('aichk').onchange=e=>{ S.noai=e.target.checked; render(); };
$('back').onclick=back;
let lastHash=null;
(function init(){ $('pfsel').innerHTML=META.portfolios.map(m=>`<option value='${m.id}'>${m.name}</option>`).join('');
  fromHash(); render(); })();
function fromHash(){ const h=decodeURIComponent(location.hash.replace('#','')); if(!h) return; const [path,q]=h.split('?'); const p=path.split('/');
  if(META.portfolios.some(x=>x.id===p[0])){ S.pf=p[0]; S.tab=TABS[p[1]]?p[1]:'combined'; if(p[1]==='stock'){ S.ticker=p[2]||null; S.sub=SUBS[p[3]]?p[3]:'readings'; } S.noai=q==='noai'; } }
window.addEventListener('hashchange', ()=>{ if(location.hash!==lastHash){ fromHash(); render(); } });
"""
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>Stock Portfolios — Oversold × Price-Target Dashboard</title><style>{css}</style><script src='plotly.min.js'></script></head><body>
<header><h1>Oversold × Price-target dashboard</h1><label>Portfolio <select id='pfsel'></select></label><span class='meta'>Prices as of {asof} · built {datetime.now():%Y-%m-%d %H:%M} ET · Yahoo Finance + IBES/Refinitiv consensus · today's bars intraday/partial</span></header>
<nav class='tabs'><button data-tab='portfolios'>1 · Portfolios</button><button data-tab='combined'>2 · Combined ranking</button><button data-tab='targets'>3 · Price targets</button><button data-tab='ranking'>4 · Oversold ranking</button><button data-tab='heat'>5 · Heat-map</button><button data-tab='stock'>6 · Stock detail</button><button data-tab='method'>7 · Methodology</button></nav>
<div class='wrap'>
<div class='toolbar'><button id='back'>← Back</button><span class='hint'>Portfolio: <b id='pfname'></b></span><label class='aitog' id='aitog'><input type='checkbox' id='aichk'> Hide AI chip / infrastructure stocks</label></div>
<div id='loading'>loading…</div>
<div id='pane-portfolios' class='pane'><h2>Choose a portfolio</h2><p class='hint'>Each card is one stock universe. Every other tab shows the selected portfolio only (change it any time from the header). Cards summarise the two ranking dimensions: upside to the engine's end-2027 price target (IBES/Refinitiv consensus) and oversold depth (10-test technical checklist vs the stock's own 4-year history).</p><div class='cards' id='cards'></div></div>
<div id='pane-combined' class='pane'><h2>Combined ranking — end-2027 target upside × oversold depth</h2><div id='combined-body'></div></div>
<div id='pane-targets' class='pane'><h2>Price-target analysis (Biotech Ranker engine, re-priced today)</h2><div id='targets-body'></div></div>
<div id='pane-ranking' class='pane'><h2>Oversold ranking – most oversold to least</h2><div id='ranking-body'></div><div id='fig-flags' class='fig'></div></div>
<div id='pane-heat' class='pane'><h2>Percentile heat-map</h2><p class='hint'>Each cell = today's indicator reading as a percentile of its own {PCTL_YEARS}-yr history (0 = most oversold ever). ✅ flag firing now · 🟡 fired in recent window. Click a cell to drill into the stock.</p><div id='fig-heat' class='fig'></div></div>
<div id='pane-stock' class='pane'>
  <div class='toolbar'><label>Ticker <select id='tsel'></select></label><button onclick='step(-1)'>◀ prev</button><button onclick='step(1)'>next ▶</button><h2 style='margin:0 0 0 12px' id='stock-title'></h2></div>
  <nav class='sub'><button data-sub='readings'>Readings & flags</button><button data-sub='charts'>Charts</button><button data-sub='history'>Episode history</button><button data-sub='lows'>Swing lows vs today</button><button data-sub='target'>Price target</button></nav>
  <div id='stock-body'></div></div>
<div id='pane-method' class='pane'><h2>Methodology & data audit</h2>
<h3>Combined ranking</h3><ul><li><b>Upside rank</b>: Tier-1 names (profitable 2027E and 2028E, not ANOMALY) ordered by upside to the end-2027 target. Target = 2028E EPS × exit P/E; exit P/E = min(g, 30x) if g ≥ 30% else max(0.8 g, 10x), ×0.8 haircut when a loss-of-exclusivity / direct competitor lands inside ~2 yrs. Bear = (IBES 2028E − 1 SD) × trough P/E; demoted (still ranked) if bear < −30%.</li>
<li><b>Oversold rank</b>: order by oversold flags (/10), then average indicator percentile, then turn signals — identical to the Oversold ranking tab.</li><li><b>Combined #</b> = sum of the two ranks, lowest first. Only names rankable on both dimensions get a combined number.</li>
<li><b>ANOMALY</b> (carried over from the original runs): exit multiple ≥2× the name's 5-yr forward P/E, 2–3 broker consensus with wide dispersion, or commodity / FX / cyclical EPS deck — the 30x-cap / 10x-floor rule mechanically inflates the target, so these are shown but excluded from the upside rank.</li>
<li><b>Prices</b>: every ticker re-priced with the latest Yahoo quote; consensus estimates remain as of the report dates (Oct 22 – Nov 6, 2026). Stock splits after the report date are adjusted (SNDK 8:1); HK-line reports (BABA, TCOM) keep their report price because EPS is in local currency.</li>
<li><b>AI chip / infrastructure filter</b>: semis & semi-cap, memory, AI servers / networking / optics, data-centre power, cooling & REITs, neoclouds / miners-turned-hosts. Hyperscalers and software users of AI are <i>not</i> tagged. Toggling re-ranks the remaining names.</li></ul>
<div id='method-body'></div></div>
</div><script>{js.replace('__META__', json.dumps({'portfolios': meta, 'membership': json.load(open(os.path.join(SITE,'data/portfolios.json')))['membership']}).replace('</', '<\\/'))}</script></body></html>"""
    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as f: f.write(doc)


if __name__ == "__main__":
    build()
