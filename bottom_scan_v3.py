"""
Oversold checklist scanner v3  –  clean-history percentiles + interactive HTML dashboard.

Data hygiene
------------
* Downloads DOWNLOAD_YEARS (6) years of adjusted daily OHLCV from Yahoo Finance.
* ALL indicators are computed from raw OHLCV only (never from another indicator), except
  W4 which is *by design* the percentile rank of weekly RSI.
* The first BURN_IN_YEARS (2) are discarded after indicator computation so every indicator
  (200-d SMA, 252-d rolling high, Wilder RSI recursion, 40-wk SMA ...) is fully warmed.
* Percentiles are therefore computed over a clean PCTL_YEARS (4) window of complete data.
* Episode tables and swing-low tables cover the last HIST_YEARS (2).

Outputs (in --out):  dashboard.html, oversold_report.md, oversold_summary.csv, data/*.csv
"""
import argparse, os, sys, warnings, json, html
from datetime import datetime
import numpy as np, pandas as pd, yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

warnings.filterwarnings("ignore")

DEFAULT_TICKERS = ["NOW", "DDOG", "DASH", "SNOW", "MDB", "META", "GOOGL", "CORT", "AXSM", "PL",
                   "RDDT", "TEAM", "JPM", "BLK", "MELI", "DAL", "UAL", "SPX", "NDX"]
SYMBOL_MAP = {"SPX": "^GSPC", "NDX": "^NDX", "VIX": "^VIX", "DJI": "^DJI", "RUT": "^RUT"}
DOWNLOAD_YEARS, BURN_IN_YEARS, PCTL_YEARS, HIST_YEARS = 6, 2, 4, 2
RECENT_D, RECENT_W = 10, 2

# ---------------------------------------------------------------- indicators (price-only inputs)
def rsi(c, n=14):
    d = c.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    al = l.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    return 100 - 100/(1 + ag/al.replace(0, np.nan))

def macd(c, f=12, s=26, sig=9):
    line = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
    sg = line.ewm(span=sig, adjust=False).mean()
    return line, sg, line - sg

def stoch(h, l, c, k=14, d=3):
    ll, hh = l.rolling(k).min(), h.rolling(k).max()
    pk = 100*(c-ll)/(hh-ll).replace(0, np.nan)
    return pk, pk.rolling(d).mean()

def add_ind(df, freq):
    c, h, l = df["Close"], df["High"], df["Low"]
    long_ma, mid_ma = (200, 50) if freq == "D" else (40, 10)
    yr = 252 if freq == "D" else 52
    df["RSI14"] = rsi(c)
    df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd(c)
    df["StochK"], df["StochD"] = stoch(h, l, c)
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["BB_up"], df["BB_lo"] = ma + 2*sd, ma - 2*sd
    df["BB_pctB"] = (c - df["BB_lo"]) / (df["BB_up"] - df["BB_lo"]).replace(0, np.nan)
    df["SMA_mid"] = c.rolling(mid_ma).mean()
    df["SMA_long"] = c.rolling(long_ma).mean()
    df["Dist_long_pct"] = (c/df["SMA_long"]-1)*100
    df["Drawdown_1y_pct"] = (c/c.rolling(yr).max()-1)*100          # full 252/52 bars required
    df["Drawdown_ATH_pct"] = (c/c.cummax()-1)*100
    df["ROC_3m_pct"] = c.pct_change(63 if freq == "D" else 13)*100
    return df

def download(sym, years):
    df = yf.download(sym, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
    if df.empty: raise RuntimeError(f"no data for {sym}")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def to_weekly(d):
    return d.resample("W-FRI").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna(subset=["Close"])

# ---------------------------------------------------------------- swing lows
def swing_lows(df, window, min_decline, since):
    low = df["Low"]
    is_min = low == low.rolling(2*window+1, center=True).min()
    deep = df["Drawdown_1y_pct"] <= -min_decline
    b = df[is_min & deep & (df.index >= since)]
    keep = []
    for ts, row in b.iterrows():
        if keep and (ts-keep[-1][0]).days < (window if window > 20 else window*7)*1.5:
            if row["Low"] < keep[-1][1]["Low"]: keep[-1] = (ts, row)
        else: keep.append((ts, row))
    return pd.DataFrame([r for _, r in keep], index=[t for t, _ in keep])

# ---------------------------------------------------------------- tests
def build_tests(d_bot, w_bot):
    med = lambda b, col, fb: (b[col].median() if (not b.empty and b[col].notna().any()) else fb)
    return [
        ("D1", "Daily RSI-14 ≤ 30",                        "D", "RSI14",        30.0,  "{:.1f}"),
        ("D2", "Daily Stochastic %K(14,3) ≤ 20",           "D", "StochK",       20.0,  "{:.1f}"),
        ("D3", "Daily close below lower Bollinger (%B ≤ 0)", "D", "BB_pctB",    0.0,   "{:.2f}"),
        ("D4", "1-yr drawdown ≥ median at prior swing lows", "D", "Drawdown_1y_pct", med(d_bot, "Drawdown_1y_pct", -20.0), "{:.1f}%"),
        ("D5", "Below 200-d SMA ≥ median at prior swing lows", "D", "Dist_long_pct", min(med(d_bot, "Dist_long_pct", -10.0), -5.0), "{:.1f}%"),
        ("W1", "Weekly RSI-14 ≤ 35",                       "W", "RSI14",        35.0,  "{:.1f}"),
        ("W2", "Weekly Stochastic %K(14,3) ≤ 20",          "W", "StochK",       20.0,  "{:.1f}"),
        ("W3", "Weekly Bollinger %B ≤ 0.10",               "W", "BB_pctB",      0.10,  "{:.2f}"),
        ("W4", f"Weekly RSI-14 in bottom 5% of {PCTL_YEARS}-yr history", "W", "RSI14_pctile", 5.0, "{:.0f}"),
        ("W5", "13-week rate of change ≤ −20%",            "W", "ROC_3m_pct",  -20.0,  "{:.1f}%"),
    ]

def episodes(series, thr, gap, since):
    s = series[series.index >= since]; hit = s[s <= thr]; eps = []
    for ts, val in hit.items():
        if eps and ts - eps[-1]["end"] <= gap:
            eps[-1]["end"] = ts
            if val < eps[-1]["min"]: eps[-1]["min"], eps[-1]["min_date"] = val, ts
        else: eps.append({"start": ts, "end": ts, "min": val, "min_date": ts})
    return eps

def fwd_ret(close, ts, bars):
    i = close.index.get_indexer([ts])[0]
    return np.nan if (i < 0 or i+bars >= len(close)) else (close.iloc[i+bars]/close.iloc[i]-1)*100

def fmt(x, f="{:.1f}"):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x)

def pctile(series, value):
    """% of the clean history window with a reading LOWER (more oversold) than `value`. 0 = most extreme ever."""
    s = series.dropna()
    return np.nan if (s.empty or pd.isna(value)) else 100*(s < value).mean()

# ---------------------------------------------------------------- per ticker
def analyze(t, sym):
    raw = download(sym, DOWNLOAD_YEARS)
    d_full = add_ind(raw.copy(), "D")
    w_full = add_ind(to_weekly(raw), "W")
    cutoff = d_full.index[-1] - pd.DateOffset(years=PCTL_YEARS)
    d, w = d_full[d_full.index >= cutoff].copy(), w_full[w_full.index >= cutoff].copy()
    # data-completeness audit
    audit = {"first_raw_bar": raw.index[0].date(), "raw_bars": len(raw),
             "window_start": d.index[0].date(), "window_bars_d": len(d), "window_bars_w": len(w),
             "nan_in_window": int(d[["RSI14","StochK","BB_pctB","Drawdown_1y_pct","Dist_long_pct"]].isna().sum().sum()
                                 + w[["RSI14","StochK","BB_pctB","Dist_long_pct","ROC_3m_pct"]].isna().sum().sum())}
    w["RSI14_pctile"] = w["RSI14"].rank(pct=True)*100       # rank within clean window
    since = d.index[-1] - pd.DateOffset(years=HIST_YEARS)
    d_bot = swing_lows(d, 30, 15, since); w_bot = swing_lows(w, 8, 15, since)
    d_bot = d_bot[d_bot.index < d.index[-1] - pd.Timedelta(days=45)]
    w_bot = w_bot[w_bot.index < w.index[-1] - pd.Timedelta(days=60)]
    tests = build_tests(d_bot, w_bot)
    res = []
    for tid, label, frame, col, thr, f in tests:
        df = d if frame == "D" else w
        rn = RECENT_D if frame == "D" else RECENT_W
        cur, rmin = df[col].iloc[-1], df[col].tail(rn).min()
        eps = episodes(df[col].iloc[:-rn], thr, pd.Timedelta(days=15 if frame == "D" else 35), since)
        b1, b3 = (21, 63) if frame == "D" else (4, 13)
        ep_rows = [{"start": e["start"], "end": e["end"], "min": e["min"], "px": df["Close"].loc[e["min_date"]],
                    "fwd1m": fwd_ret(df["Close"], e["end"], b1), "fwd3m": fwd_ret(df["Close"], e["end"], b3)} for e in eps]
        bot = d_bot if frame == "D" else w_bot
        bv = bot[col] if (not bot.empty and col in bot) else pd.Series(dtype=float)
        res.append({"id": tid, "label": label, "frame": frame, "col": col, "thr": thr, "fmt": f, "cur": cur,
                    "recent_min": rmin, "flag": bool(cur <= thr or rmin <= thr), "flag_now": bool(cur <= thr),
                    "pctile": pctile(df[col], cur), "pctile_recent": pctile(df[col], rmin),
                    "episodes": ep_rows, "bot_med": bv.median() if len(bv) else np.nan, "bot_min": bv.min() if len(bv) else np.nan})
    tail = d.tail(120); half = len(tail)//2
    lo1, lo2 = tail["Low"].iloc[:half], tail["Low"].iloc[half:]
    div = bool(lo2.min() < lo1.min() and tail["RSI14"].loc[lo2.idxmin()] > tail["RSI14"].loc[lo1.idxmin()])
    turns = [
        ("T1", "Daily MACD histogram rising (3 bars)", bool(d["MACD_hist"].iloc[-1] > d["MACD_hist"].iloc[-3]), f"{d['MACD_hist'].iloc[-3]:.2f} → {d['MACD_hist'].iloc[-1]:.2f}"),
        ("T2", "Weekly MACD histogram rising (3 bars)", bool(w["MACD_hist"].iloc[-1] > w["MACD_hist"].iloc[-3]), f"{w['MACD_hist'].iloc[-3]:.2f} → {w['MACD_hist'].iloc[-1]:.2f}"),
        ("T3", "Bullish RSI divergence (~6 mo)", div, f"lows {lo1.min():.2f}→{lo2.min():.2f}, RSI {tail['RSI14'].loc[lo1.idxmin()]:.0f}→{tail['RSI14'].loc[lo2.idxmin()]:.0f}"),
        ("T4", "Daily RSI hooked back above 30", bool(d["RSI14"].tail(RECENT_D).min() <= 30 and d["RSI14"].iloc[-1] > 30 and d["RSI14"].iloc[-1] > d["RSI14"].iloc[-2]),
             f"10d low {d['RSI14'].tail(RECENT_D).min():.0f}, now {d['RSI14'].iloc[-1]:.0f}"),
    ]
    return {"ticker": t, "asof": d.index[-1].date(), "price": d["Close"].iloc[-1], "dd1y": d["Drawdown_1y_pct"].iloc[-1],
            "ddath": d["Drawdown_ATH_pct"].iloc[-1], "tests": res, "turns": turns, "n_flags": sum(r["flag"] for r in res),
            "n_turn": sum(x[2] for x in turns), "d_bot": d_bot, "w_bot": w_bot, "d": d, "w": w, "audit": audit,
            "rsi_w_pctile": res[5]["pctile"], "avg_pctile": float(np.nanmean([r["pctile"] for r in res]))}

def tier(r):
    n, k = r["n_flags"], r["n_turn"]
    if n >= 7: return "Tier 1 – Deeply oversold, turn signals present" if k >= 2 else "Tier 2 – Deeply oversold, no confirmed turn yet"
    if n >= 4: return "Tier 3 – Partially oversold"
    if n >= 2: return "Tier 4 – Mild pullback"
    return "Tier 5 – Not oversold"

# ---------------------------------------------------------------- markdown report
def write_md(results, path):
    L = [f"# Oversold Checklist Scan – {datetime.now():%Y-%m-%d %H:%M} ET", "",
         f"Data: Yahoo Finance adjusted daily OHLCV, {DOWNLOAD_YEARS} yrs downloaded, first {BURN_IN_YEARS} yrs used only as indicator warm-up. "
         f"Percentiles = share of the last {PCTL_YEARS} yrs (fully-warmed data) with a *more oversold* reading than today (0 = most extreme). "
         f"Episode & swing-low history: {HIST_YEARS} yrs. **Today's bars are partial (intraday).**", "",
         "## Ranking – most oversold to least", "",
         "| # | Ticker | Flags /10 | Turn /4 | Avg pctile | Tier | Firing | Px | DD 1y | DD ATH | RSI-D (pct) | RSI-W (pct) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        t0, t5 = r["tests"][0], r["tests"][5]
        L.append(f"| {i} | **{r['ticker']}** | **{r['n_flags']}** | {r['n_turn']} | {r['avg_pctile']:.0f} | {tier(r)} | {' '.join(x['id'] for x in r['tests'] if x['flag']) or '—'} "
                 f"| {r['price']:.2f} | {r['dd1y']:.0f}% | {r['ddath']:.0f}% | {t0['cur']:.0f} ({t0['pctile']:.0f}) | {t5['cur']:.0f} ({t5['pctile']:.0f}) |")
    L += ["", "## Percentile matrix (today's reading vs its own 4-yr history; ✅ = flag firing)", "",
          "| Ticker | " + " | ".join(x["id"] for x in results[0]["tests"]) + " | Flags |", "|---|" + "---|"*(len(results[0]["tests"])+1)]
    for r in results:
        L.append(f"| {r['ticker']} | " + " | ".join(f"{fmt(x['pctile'],'{:.0f}')}{' ✅' if x['flag_now'] else (' 🟡' if x['flag'] else '')}" for x in r["tests"]) + f" | {r['n_flags']} |")
    L += ["", "## Per-stock detail", ""]
    for r in results:
        a = r["audit"]
        L += [f"### {r['ticker']} — {r['n_flags']}/10 flags, {r['n_turn']}/4 turn signals — {tier(r)}", "",
              f"Price {r['price']:.2f} · {r['dd1y']:.1f}% from 1-yr high · {r['ddath']:.1f}% from ATH · as of {r['asof']}  ",
              f"Data audit: raw from {a['first_raw_bar']} ({a['raw_bars']} bars); percentile window from {a['window_start']} "
              f"({a['window_bars_d']} daily / {a['window_bars_w']} weekly bars); missing indicator values in window: {a['nan_in_window']}", "",
              "| ID | Test | Threshold | Now | Pctile now | Recent low | Pctile recent | Flag | Median @ prior lows | Min @ prior lows |", "|---|---|---|---|---|---|---|---|---|---|"]
        for x in r["tests"]:
            f = x["fmt"]
            L.append(f"| {x['id']} | {x['label']} | ≤ {fmt(x['thr'],f)} | {fmt(x['cur'],f)} | **{fmt(x['pctile'],'{:.0f}')}** | {fmt(x['recent_min'],f)} | {fmt(x['pctile_recent'],'{:.0f}')} "
                     f"| {'✅' if x['flag_now'] else ('🟡' if x['flag'] else '—')} | {fmt(x['bot_med'],f)} | {fmt(x['bot_min'],f)} |")
        L += ["", "| Turn | Signal | Status | Detail |", "|---|---|---|---|"] + [f"| {a_} | {b} | {'✅' if c else '—'} | {dd} |" for a_, b, c, dd in r["turns"]]
        L += ["", f"**History ({HIST_YEARS} yrs) – prior episodes at the same level and what price did next**", ""]
        for x in r["tests"]:
            if not x["episodes"]:
                L.append(f"- {x['id']}: no prior episode ≤ {fmt(x['thr'],x['fmt'])}" + (" → **today is unprecedented in window**" if x["flag"] else "")); continue
            L += [f"- {x['id']} (≤ {fmt(x['thr'],x['fmt'])}) – {len(x['episodes'])} episode(s):", "", "  | Episode | Extreme | Price | +1 mo | +3 mo |", "  |---|---|---|---|---|"]
            for e in x["episodes"]:
                rng = f"{e['start']:%Y-%m-%d}" + ("" if e["start"] == e["end"] else f" → {e['end']:%Y-%m-%d}")
                L.append(f"  | {rng} | {fmt(e['min'],x['fmt'])} | {e['px']:.2f} | {fmt(e['fwd1m'],'{:+.1f}%')} | {fmt(e['fwd3m'],'{:+.1f}%')} |")
            L.append("")
        L.append("---\n")
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(L))

# ---------------------------------------------------------------- dashboard
def pct_color(p):
    if p is None or np.isnan(p): return "#eee"
    p = max(0, min(100, p))
    # 0 → deep red (extreme oversold), 50 → light grey, 100 → green
    if p <= 50:
        t = p/50; return f"rgb({int(200+55*t)},{int(40+190*t)},{int(40+190*t)})"
    t = (p-50)/50; return f"rgb({int(255-155*t)},{int(230-30*t)},{int(230-130*t)})"

def table_html(headers, rows, cls=""):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td{(' style=\"'+st+'\"') if st else ''}>{v}</td>" for v, st in row) + "</tr>" for row in rows)
    return f"<table class='tbl {cls}'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"

def fig_heatmap(results):
    ids = [x["id"] for x in results[0]["tests"]]
    z = [[x["pctile"] for x in r["tests"]] for r in results]
    txt = [[f"{fmt(x['pctile'],'{:.0f}')}{'<br>✅' if x['flag_now'] else ('<br>🟡' if x['flag'] else '')}" for x in r["tests"]] for r in results]
    hover = [[f"{r['ticker']} {x['id']}: {x['label']}<br>Now {fmt(x['cur'],x['fmt'])} (thr ≤ {fmt(x['thr'],x['fmt'])})<br>Percentile {fmt(x['pctile'],'{:.0f}')}" for x in r["tests"]] for r in results]
    fig = go.Figure(go.Heatmap(z=z, x=ids, y=[r["ticker"] for r in results], text=txt, texttemplate="%{text}", hovertext=hover, hoverinfo="text",
                               colorscale=[[0, "#c82828"], [0.25, "#f28c8c"], [0.5, "#f2f2f2"], [1, "#4a9a5a"]], zmin=0, zmax=100,
                               colorbar=dict(title="Percentile<br>(0 = most<br>oversold)")))
    fig.update_layout(title=f"Today's reading as a percentile of its own {PCTL_YEARS}-yr history (✅ flag firing now, 🟡 fired within recent window)",
                      height=40*len(results)+160, yaxis=dict(autorange="reversed"), margin=dict(l=60, r=20, t=70, b=40))
    return fig

def fig_flags(results):
    fig = go.Figure()
    fig.add_bar(x=[r["ticker"] for r in results], y=[r["n_flags"] for r in results], name="Oversold flags (/10)",
                marker_color=[("#c82828" if r["n_flags"] >= 7 else "#e8a33c" if r["n_flags"] >= 4 else "#9aa") for r in results],
                text=[r["n_flags"] for r in results], textposition="outside")
    fig.add_bar(x=[r["ticker"] for r in results], y=[r["n_turn"] for r in results], name="Turn signals (/4)", marker_color="#2b7bba", text=[r["n_turn"] for r in results], textposition="outside")
    fig.update_layout(barmode="group", title="Oversold flags vs turn signals (sorted most → least oversold)", height=380, yaxis=dict(range=[0, 11]), margin=dict(t=60, b=40))
    return fig

def fig_ticker(r):
    d, w = r["d"], r["w"]
    since = d.index[-1] - pd.DateOffset(years=HIST_YEARS)
    dd, ww = d[d.index >= since], w[w.index >= since]
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
                        subplot_titles=("Price, 50/200-d SMA, Bollinger(20,2), ▼ = historical swing lows", "Daily RSI-14 (≤30 oversold)",
                                        "Weekly RSI-14 (≤35 oversold)", "Daily Stochastic %K (≤20)", "Drawdown from 1-yr high (dashed = median at prior lows)"))
    fig.add_trace(go.Scatter(x=dd.index, y=dd["BB_up"], line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd["BB_lo"], line=dict(width=0), fill="tonexty", fillcolor="rgba(120,120,200,0.12)", name="Bollinger", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd["Close"], name="Close", line=dict(color="#222", width=1.4)), 1, 1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd["SMA_mid"], name="SMA50", line=dict(color="#e8a33c", width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd["SMA_long"], name="SMA200", line=dict(color="#2b7bba", width=1.2)), 1, 1)
    if not r["d_bot"].empty:
        fig.add_trace(go.Scatter(x=r["d_bot"].index, y=r["d_bot"]["Low"]*0.985, mode="markers", name="Swing low", marker=dict(symbol="triangle-up", size=12, color="#c82828"),
                                 hovertext=[f"{ts:%Y-%m-%d} low {b['Low']:.2f}<br>RSI-D {b['RSI14']:.0f}, DD {b['Drawdown_1y_pct']:.0f}%, vs200d {b['Dist_long_pct']:.0f}%" for ts, b in r["d_bot"].iterrows()], hoverinfo="text"), 1, 1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd["RSI14"], name="RSI-D", line=dict(color="#7b3fa0", width=1.2)), 2, 1)
    fig.add_hline(y=30, line=dict(color="#c82828", dash="dot"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ww.index, y=ww["RSI14"], name="RSI-W", line=dict(color="#7b3fa0", width=1.6), mode="lines+markers", marker=dict(size=4)), 3, 1)
    fig.add_hline(y=35, line=dict(color="#c82828", dash="dot"), row=3, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd["StochK"], name="Stoch-D", line=dict(color="#2b7bba", width=1)), 4, 1)
    fig.add_hline(y=20, line=dict(color="#c82828", dash="dot"), row=4, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd["Drawdown_1y_pct"], name="DD 1y", fill="tozeroy", line=dict(color="#c82828", width=1), fillcolor="rgba(200,40,40,0.15)"), 5, 1)
    thr = r["tests"][3]["thr"]
    fig.add_hline(y=thr, line=dict(color="#555", dash="dash"), row=5, col=1)
    fig.update_layout(height=1000, margin=dict(l=50, r=20, t=50, b=30), legend=dict(orientation="h", y=1.02, x=0), hovermode="x unified")
    fig.update_yaxes(range=[0, 100], row=2, col=1); fig.update_yaxes(range=[0, 100], row=3, col=1); fig.update_yaxes(range=[0, 100], row=4, col=1)
    return fig

def fig_pctile_bars(r):
    ids = [x["id"] for x in r["tests"]]
    fig = go.Figure()
    fig.add_bar(y=ids, x=[x["pctile"] for x in r["tests"]], orientation="h", name="Now",
                marker_color=[pct_color(x["pctile"]) for x in r["tests"]], marker_line=dict(color=["#c82828" if x["flag"] else "#999" for x in r["tests"]], width=2),
                text=[f"{fmt(x['pctile'],'{:.0f}')}th · now {fmt(x['cur'],x['fmt'])} (thr ≤ {fmt(x['thr'],x['fmt'])}) {'✅' if x['flag_now'] else ('🟡' if x['flag'] else '')}" for x in r["tests"]], textposition="outside", cliponaxis=False)
    fig.add_vline(x=5, line=dict(color="#c82828", dash="dot"), annotation_text="5th", annotation_position="top")
    fig.add_vline(x=20, line=dict(color="#e8a33c", dash="dot"), annotation_text="20th", annotation_position="top")
    fig.update_layout(title=f"Where today's reading sits in its own {PCTL_YEARS}-yr distribution (0 = most oversold ever)", height=420, xaxis=dict(range=[0, 160], title="percentile"), yaxis=dict(autorange="reversed"), margin=dict(l=40, r=20, t=60, b=40), showlegend=False)
    return fig

def write_dashboard(results, path):
    pio.templates.default = "plotly_white"
    parts = []
    def fig_div(fig): return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displaylogo": False})
    # --- overview
    rank_rows = []
    for i, r in enumerate(results, 1):
        t0, t5 = r["tests"][0], r["tests"][5]
        rank_rows.append([(i, ""), (f"<a href='#{r['ticker']}'><b>{r['ticker']}</b></a>", ""), (f"<b>{r['n_flags']}</b>", f"background:{'#f6c9c9' if r['n_flags']>=7 else '#fbe7c6' if r['n_flags']>=4 else '#eee'}"),
                          (r["n_turn"], ""), (f"{r['avg_pctile']:.0f}", f"background:{pct_color(r['avg_pctile'])}"), (tier(r), ""),
                          (" ".join(x["id"] for x in r["tests"] if x["flag"]) or "—", "font-family:monospace"), (f"{r['price']:.2f}", ""),
                          (f"{r['dd1y']:.0f}%", ""), (f"{r['ddath']:.0f}%", ""), (f"{t0['cur']:.0f} <small>({t0['pctile']:.0f}th)</small>", f"background:{pct_color(t0['pctile'])}"),
                          (f"{t5['cur']:.0f} <small>({t5['pctile']:.0f}th)</small>", f"background:{pct_color(t5['pctile'])}")])
    parts.append("<h2>1 · Ranking – most oversold to least</h2>" + table_html(["#", "Ticker", "Flags /10", "Turn /4", "Avg pctile", "Tier", "Flags firing", "Price", "DD 1-yr", "DD ATH", "RSI-D (pctile)", "RSI-W (pctile)"], rank_rows))
    parts.append(fig_div(fig_flags(results)))
    parts.append("<h2>2 · Percentile heat-map</h2><p>Each cell: today's indicator reading expressed as the % of the last "
                 f"{PCTL_YEARS} years when the reading was <i>more</i> oversold. Red = rare / extreme; green = normal.</p>" + fig_div(fig_heatmap(results)))
    tests0 = results[0]["tests"]
    parts.append("<h2>3 · Checklist definitions</h2>" + table_html(["ID", "Oversold test", "Threshold", "Frame", "Recent window"],
                 [[(x["id"], "font-family:monospace"), (x["label"], ""), ("stock-relative (median at own prior swing lows)" if x["id"] in ("D4", "D5") else f"≤ {fmt(x['thr'],x['fmt'])}", ""),
                   ("Daily" if x["frame"] == "D" else "Weekly", ""), (f"{RECENT_D} sessions" if x["frame"] == "D" else f"{RECENT_W} weeks", "")] for x in tests0]) +
                 "<p>Turn signals (bonus): T1 daily MACD-histogram rising 3 bars · T2 weekly MACD-histogram rising · T3 bullish RSI divergence (lower price low, higher RSI low over ~6 mo) · T4 daily RSI hooked back above 30.<br>"
                 "Tiers: ≥7 flags → Tier 1 (≥2 turn signals) / Tier 2; 4–6 → Tier 3; 2–3 → Tier 4; ≤1 → Tier 5.</p>")
    # --- per ticker
    parts.append("<h2>4 · Per-stock detail</h2><p class='nav'>" + " · ".join(f"<a href='#{r['ticker']}'>{r['ticker']}</a>" for r in results) + "</p>")
    for r in results:
        a = r["audit"]
        sec = [f"<section id='{r['ticker']}'><h3>{r['ticker']} — {r['n_flags']}/10 oversold flags · {r['n_turn']}/4 turn signals · {tier(r)}</h3>",
               f"<p>Price <b>{r['price']:.2f}</b> · {r['dd1y']:.1f}% from 1-yr high · {r['ddath']:.1f}% from ATH · as of {r['asof']}<br>"
               f"<small>Data audit: raw history from {a['first_raw_bar']} ({a['raw_bars']} bars); percentile window from {a['window_start']} "
               f"({a['window_bars_d']} daily / {a['window_bars_w']} weekly bars); missing indicator values inside window: {a['nan_in_window']}</small></p>"]
        rows = []
        for x in r["tests"]:
            f = x["fmt"]
            rows.append([(x["id"], "font-family:monospace"), (x["label"], ""), (f"≤ {fmt(x['thr'],f)}", ""), (f"<b>{fmt(x['cur'],f)}</b>", ""),
                         (f"<b>{fmt(x['pctile'],'{:.0f}')}</b>", f"background:{pct_color(x['pctile'])}"), (fmt(x["recent_min"], f), ""), (fmt(x["pctile_recent"], "{:.0f}"), f"background:{pct_color(x['pctile_recent'])}"),
                         ("✅ now" if x["flag_now"] else ("🟡 recent" if x["flag"] else "—"), ""), (fmt(x["bot_med"], f), ""), (fmt(x["bot_min"], f), "")])
        sec.append(table_html(["ID", "Test", "Threshold", "Now", "Pctile now", "Recent low", "Pctile recent", "Flag", "Median @ prior lows", "Min @ prior lows"], rows))
        sec.append(table_html(["Turn", "Signal", "Status", "Detail"], [[(t_, "font-family:monospace"), (lab, ""), ("✅" if ok else "—", ""), (det, "")] for t_, lab, ok, det in r["turns"]], "small"))
        sec.append("<div class='row'><div class='col'>" + fig_div(fig_pctile_bars(r)) + "</div></div>")
        sec.append(fig_div(fig_ticker(r)))
        sec.append(f"<h4>History ({HIST_YEARS} yrs): prior episodes at today's thresholds and what price did next</h4><div class='row'>")
        for x in r["tests"]:
            if not x["episodes"]:
                sec.append(f"<div class='col ep'><b>{x['id']}</b> (≤ {fmt(x['thr'],x['fmt'])}): no prior episode" + (" — <b>today is unprecedented in window</b>" if x["flag"] else "") + "</div>"); continue
            ep_rows = [[(f"{e['start']:%Y-%m-%d}" + ("" if e["start"] == e["end"] else f" → {e['end']:%Y-%m-%d}"), ""), (fmt(e["min"], x["fmt"]), ""), (f"{e['px']:.2f}", ""),
                        (fmt(e["fwd1m"], "{:+.1f}%"), f"color:{'green' if (e['fwd1m'] or 0) > 0 else 'crimson'}"), (fmt(e["fwd3m"], "{:+.1f}%"), f"color:{'green' if (e['fwd3m'] or 0) > 0 else 'crimson'}")] for e in x["episodes"]]
            sec.append(f"<div class='col ep'><b>{x['id']}</b> (≤ {fmt(x['thr'],x['fmt'])}) — {len(x['episodes'])} episode(s); today {fmt(x['cur'],x['fmt'])}" + table_html(["Episode", "Extreme", "Price", "+1 mo", "+3 mo"], ep_rows, "small") + "</div>")
        sec.append("</div>")
        for name, bot, cols in (("daily", r["d_bot"], ["RSI14", "StochK", "BB_pctB", "Drawdown_1y_pct", "Dist_long_pct"]), ("weekly", r["w_bot"], ["RSI14", "StochK", "BB_pctB", "RSI14_pctile", "ROC_3m_pct"])):
            hdr = {"RSI14": "RSI-14", "StochK": "Stoch %K", "BB_pctB": "BB %B", "Drawdown_1y_pct": "DD 1-yr", "Dist_long_pct": "vs 200d" if name == "daily" else "vs 40w", "RSI14_pctile": "RSI pctile", "ROC_3m_pct": "13w ROC"}
            sec.append(f"<h4>Readings at historical {name} swing lows (≥15% off 1-yr high, past {HIST_YEARS} yrs)</h4>")
            if bot.empty: sec.append("<p><i>none</i></p>"); continue
            rows = [[(f"{ts:%Y-%m-%d}", ""), (f"{b['Low']:.2f}", "")] + [(fmt(b.get(c, np.nan), "{:.2f}" if c == "BB_pctB" else ("{:.1f}%" if ("pct" in c and c != "RSI14_pctile") else "{:.1f}")), "") for c in cols] for ts, b in bot.iterrows()]
            sec.append(table_html(["Date", "Low"] + [hdr[c] for c in cols], rows, "small"))
        sec.append("<p><a href='#top'>↑ top</a></p></section>")
        parts.append("\n".join(sec))
    css = """
    body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#fafafa;color:#222}
    .wrap{max-width:1400px;margin:0 auto;padding:16px 24px}
    h1{margin:.2em 0} h2{border-bottom:2px solid #ddd;padding-bottom:4px;margin-top:40px} h3{margin-top:36px;background:#eef2f7;padding:8px 12px;border-left:5px solid #2b7bba}
    .tbl{border-collapse:collapse;width:100%;margin:8px 0 16px;font-size:13.5px;background:#fff} .tbl th{background:#2b3e55;color:#fff;padding:6px 8px;text-align:left;position:sticky;top:0}
    .tbl td{padding:5px 8px;border-bottom:1px solid #e5e5e5} .tbl.small{font-size:12.5px} .tbl tr:hover td{filter:brightness(.97)}
    .row{display:flex;flex-wrap:wrap;gap:16px} .col{flex:1 1 440px} .ep{background:#fff;border:1px solid #e3e3e3;border-radius:6px;padding:8px 12px;flex:1 1 420px;max-width:680px}
    .nav a{margin-right:4px} a{color:#2b7bba} section{padding-bottom:16px;border-bottom:1px dashed #ccc} .meta{color:#555;font-size:13.5px}
    """
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>Oversold Checklist Dashboard</title><style>{css}</style>
<script>{pio.get_plotlyjs() if hasattr(pio,'get_plotlyjs') else __import__('plotly.offline',fromlist=['get_plotlyjs']).get_plotlyjs()}</script></head><body><div class='wrap' id='top'>
<h1>Oversold Checklist Dashboard</h1><p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} ET · Source: Yahoo Finance adjusted OHLCV · {DOWNLOAD_YEARS} yrs downloaded, {BURN_IN_YEARS} yrs burn-in, percentiles over the last {PCTL_YEARS} yrs of fully-warmed indicators · all indicators computed from raw price/volume only · <b>today's bars are intraday/partial</b>.</p>
{''.join(parts)}</div></body></html>"""
    with open(path, "w", encoding="utf-8") as f: f.write(doc)

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    results = []
    for t in a.tickers:
        t = t.upper()
        try: r = analyze(t, SYMBOL_MAP.get(t, t))
        except Exception as e: print(f"[{t}] failed: {e}", file=sys.stderr); continue
        results.append(r)
        r["d"].to_csv(os.path.join(a.out, "data", f"{t}_daily.csv")); r["w"].to_csv(os.path.join(a.out, "data", f"{t}_weekly.csv"))
        print(f"[{t:5}] flags={r['n_flags']:2}/10 turn={r['n_turn']}/4 avg_pctile={r['avg_pctile']:4.0f} window_from={r['audit']['window_start']} nan={r['audit']['nan_in_window']}  " + " ".join(x['id'] for x in r['tests'] if x['flag']))
    results.sort(key=lambda r: (-r["n_flags"], r["avg_pctile"], -r["n_turn"]))
    write_md(results, os.path.join(a.out, "oversold_report.md"))
    import dashboard
    dashboard.write_dashboard(results, os.path.join(a.out, "dashboard.html"))
    rows = []
    for r in results:
        row = {"ticker": r["ticker"], "asof": r["asof"], "price": round(r["price"], 2), "flags": r["n_flags"], "turn_signals": r["n_turn"], "avg_pctile": round(r["avg_pctile"], 1), "tier": tier(r), "dd_1y_pct": round(r["dd1y"], 1), "dd_ath_pct": round(r["ddath"], 1)}
        for x in r["tests"]: row[f"{x['id']}_now"] = round(x["cur"], 2); row[f"{x['id']}_pctile"] = round(x["pctile"], 1); row[f"{x['id']}_flag"] = x["flag"]
        for tid, _, ok, _ in r["turns"]: row[tid] = ok
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "oversold_summary.csv"), index=False)
    print(f"\nWrote dashboard.html, oversold_report.md, oversold_summary.csv in {a.out}")

if __name__ == "__main__":
    main()
