"""
Multi-portfolio driver.

1. Defines the six portfolios (original oversold list + five IBES/Refinitiv report sets) and the
   AI chip / infrastructure tags used by the dashboard filter.
2. Pulls today's prices from Yahoo and re-runs the Biotech Ranker engine (rank.py) for every
   portfolio with a `price` override for every ticker -> runs/<name>-<date>/ranking.json
3. Runs the oversold checklist (bottom_scan_v3.analyze) on the union of all tickers (cached).

Run:  python portfolios.py            (full)      python portfolios.py --skip-scan   (re-rank only)
"""
import argparse, json, os, pickle, subprocess, sys, time
from datetime import datetime
import pandas as pd, yfinance as yf
import bottom_scan_v3 as bs

ENGINE = r"C:\Users\tride\OneDrive\Laptop - DELL 2024\Stocks\Biotech Ranker"
RUNS = os.path.join(ENGINE, "runs")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
TODAY = datetime.now().strftime("%Y-%m-%d")

# AI chip / infrastructure classification (semis, semi-cap, AI servers/networking/optics, data-centre power/cooling/REIT,
# neoclouds/bitcoin-miners-turned-AI-hosts). Hyperscalers / software users of AI are NOT tagged.
AI_INFRA = {
    # TipRanks Buy portfolio
    "NVDA", "AMD", "AVGO", "MRVL", "TSM", "ASML", "SNPS", "DELL", "VRT", "CORZ", "ORCL",
    # E*TRADE screener (as tagged in that run's report)
    "SMCI", "MTZ", "SEI", "TTMI", "SNX", "TSEM", "SIMO", "IFNNY", "AAON", "MXL", "SMTC", "KLIC", "STM", "DIOD", "HIMX",
    "MKSI", "TER", "UCTT", "VICR", "SLAB", "CECO", "FIX", "STRL", "BABA",
    # Mega/Large-cap screener
    "MCHP", "FSLR", "CMI", "AMT",
    # AI stock ranker = whole universe
    "ALAB", "AMAT", "ANET", "AXTI", "BE", "CBRS", "CEG", "CIEN", "CLS", "COHR", "CRDO", "CRWV", "DY", "EQIX", "ETN", "FLEX",
    "FN", "GNRC", "HUBB", "IREN", "JCI", "LITE", "LRCX", "MOD", "MPWR", "MTSI", "MU", "NBIS", "NVT", "OKLO", "POET", "PWR",
    "SKHY", "SNDK", "STX", "TT", "VST", "WDC",
    # US growth screener (11/22): semis, semi-cap/materials, networking/servers/optics, connectors, data-centre power
    "INTC", "TXN", "ADI", "KLAC", "NXPI", "ON", "GFS", "ENTG", "Q", "GLW", "APH", "TEL", "JBL", "HPE", "CSCO", "GEV", "NRG", "KEYS",
}
# Yahoo symbols that differ from the report ticker
YSYM = {"SKHY": "000660.KS", "IFNNY": "IFNNY", "FWONK": "FWONK", "BABA": "BABA", "COMP": "^IXIC"}
# Report rows whose EPS/price are not in USD per ADR -> keep report price (override would mix currencies)
NO_PRICE_OVERRIDE = {("etrade", "BABA"), ("ai", "SKHY"), ("screener", "TCOM"),
                    ("growth", "TD"), ("growth", "BMO"), ("growth", "BNS"), ("growth", "CM"), ("growth", "GFL"), ("growth", "BAP")}
# Stock splits after the report date: report EPS / prices / targets are divided by this factor
SPLIT_ADJ = {("ai", "SNDK"): 8.0}


def split_adjust(d, k):
    """Return a copy of a data.json entry with every per-share figure divided by k."""
    import copy
    d = copy.deepcopy(d); d["price"] = d["price"] / k
    for y in d.get("eps_annual", {}).values():
        for f in ("mean", "sd", "high", "low", "prev_mean", "mean_30d", "mean_90d"):
            if y.get(f) is not None: y[f] /= k
    ib = d.get("ibes") or {}
    if ib.get("eps_fwd4q"): ib["eps_fwd4q"] /= k
    pt = (d.get("refinitiv") or {}).get("price_target") or {}
    for f in ("mean", "high", "low", "median"):
        if pt.get(f) is not None: pt[f] /= k
    d["report_prices"] = {kk: v / k for kk, v in (d.get("report_prices") or {}).items() if v}
    d["split_note"] = f"{k:g}-for-1 split after report date; report per-share figures divided by {k:g}"
    return d

PORTFOLIOS = [
    {"id": "orig", "name": "Original watch list", "short": "Watch list",
     "desc": "The 17 single names from the first request. No reports of its own → price-target rows are borrowed from the portfolio that covers each name.",
     "tickers": [t for t in bs.DEFAULT_TICKERS if t not in ("SPX", "NDX")], "run": None, "ai_filter": False},
    {"id": "indices", "name": "Market indices", "short": "Indices",
     "desc": "S&P 500 (SPX), Nasdaq-100 (NDX), Nasdaq Composite (COMP), Dow Jones Industrial Average (DJI). Oversold analysis only — no earnings-based targets for indices.",
     "tickers": ["SPX", "NDX", "COMP", "DJI"], "run": None, "ai_filter": False},
    {"id": "tipranks", "name": "TipRanks Buy portfolio", "short": "TipRanks Buy",
     "desc": "46 names from the TipRanks Buy portfolio reports (IBES/Refinitiv, Oct-2026). AI chip/infrastructure filter available.",
     "run": "tipranks", "data": "data_new.json", "overrides": "overrides.json", "ai_filter": True},
    {"id": "ai", "name": "AI stock ranker", "short": "AI stocks",
     "desc": "41 AI-infrastructure names (semis, memory, optics, networking, power, neoclouds). Reports dated 10/22–10/26/26; haircuts for cyclical-peak names carried over from the 11/05 run.",
     "run": "ai-stocks-1105", "data": "data.json", "overrides": "overrides.json", "ai_filter": False},
    {"id": "biotech", "name": "Commercial biotech", "short": "Biotech",
     "desc": "15 commercial-stage biotechs. LOE / competitor haircuts and D-adjustments carried over from the original run.",
     "run": "", "data": "data.json", "overrides": "overrides.json", "ai_filter": False},
    {"id": "etrade", "name": "E*TRADE screener (Oct-2026)", "short": "E*TRADE screener",
     "desc": "49 names from the E*TRADE >$2B screener shortlist (reports 10/26–10/29/26). ANOMALY tags (30x-cap / 10x-floor inflation) carried over. AI chip/infrastructure filter available.",
     "run": "etrade-oct2026", "data": "data.json", "overrides": "overrides.json", "ai_filter": True},
    {"id": "screener", "name": "Mega/Large-cap Strong Buy screener", "short": "Mega/Large-cap",
     "desc": "84 names from the TipRanks Mega/Large-cap Strong Buy screener (reports 11/05–11/06/26; 21 names had no report). ANOMALY tags carried over.",
     "run": "tipranks-rest", "data": "data.json", "overrides": "overrides.json", "ai_filter": True},
    {"id": "growth", "name": "US growth screener (3-5yr EPS growth >12%)", "short": "US growth screener",
     "desc": "155 US-listed large/mid-cap names from the 11/22/26 growth screen (projected 3-5 yr EPS growth > 12%) with IBES/Refinitiv reports dated 11/22/26 (VYLR: no report available). AI chip/infrastructure filter available.",
     "run": "growth-1122", "data": "data.json", "overrides": "overrides.json", "ai_filter": True},
]


def load_json(p):
    with open(p, encoding="utf-8") as f: return json.load(f)


def ysym(t):
    return YSYM.get(t) or bs.SYMBOL_MAP.get(t, t)


def fetch_prices(tickers):
    """Latest Yahoo price (live/intraday bar if the session is open) for every ticker."""
    out, asof = {}, {}
    syms = sorted({ysym(t) for t in tickers})
    df = yf.download(syms, period="5d", interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=True)
    for t in tickers:
        s = ysym(t)
        try:
            c = df[s]["Close"].dropna() if isinstance(df.columns, pd.MultiIndex) else df["Close"].dropna()
            if len(c): out[t] = float(c.iloc[-1]); asof[t] = str(c.index[-1].date())
        except Exception: pass
    return out, asof


def rerun_rankings(prices, price_asof):
    results = {}
    for p in PORTFOLIOS:
        if p["run"] is None: continue
        src = os.path.join(RUNS, p["run"]); data = load_json(os.path.join(src, p["data"]))
        ov = load_json(os.path.join(src, p["overrides"]))
        out = os.path.join(RUNS, f"{p['run'].strip() or 'biotech'}-{TODAY.replace('-', '')[4:]}"); os.makedirs(out, exist_ok=True)
        ov2 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in ov.items()}
        ov2["_comment"] = (f"Re-run {TODAY}: every ticker's price overridden with the latest Yahoo price ({TODAY}); "
                           f"haircuts / anomaly judgments unchanged from the original run. " + str(ov.get("_comment", "")))
        missing = []
        for t in data:
            if (p["id"], t) in SPLIT_ADJ: data[t] = split_adjust(data[t], SPLIT_ADJ[(p["id"], t)])
            if (p["id"], t) in NO_PRICE_OVERRIDE: continue
            if t in prices: ov2.setdefault(t, {})["price"] = round(prices[t], 2)
            else: missing.append(t)
        json.dump(ov2, open(os.path.join(out, "overrides.json"), "w", encoding="utf-8"), indent=1)
        data_p = os.path.join(out, "data.json"); json.dump(data, open(data_p, "w", encoding="utf-8"))
        cmd = [sys.executable, os.path.join(ENGINE, "rank.py"), data_p, "--overrides", os.path.join(out, "overrides.json"),
               "--top", "400", "--md", os.path.join(out, "ranking.md"), "--json", os.path.join(out, "ranking.json")]
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode: print(r.stdout[-2000:], r.stderr[-3000:]); raise SystemExit(f"rank.py failed for {p['id']}")
        rk = load_json(os.path.join(out, "ranking.json"))
        for row in rk["rows"]:
            row["anomaly"] = (row.get("driver") or "").upper().startswith("ANOMALY")
            row["ai_infra"] = row["ticker"] in AI_INFRA
            row["price_asof"] = price_asof.get(row["ticker"]) if row["price_note"] == "override" else row["price_note"]
            row["report_price"] = data[row["ticker"]]["price"]; row["report_date"] = data[row["ticker"]].get("price_date")
            row["split_note"] = data[row["ticker"]].get("split_note")
            row["currency_note"] = ("Report is the local-currency line (HK/CAD/PEN); ratios valid, price kept at report level — not re-priced"
                                    if (p["id"], row["ticker"]) in NO_PRICE_OVERRIDE else None)
        p["tickers"] = [row["ticker"] for row in rk["rows"]]
        results[p["id"]] = {"rows": rk["rows"], "missing_price": missing, "dir": out}
        print(f"{p['id']:10s} {len(rk['rows'])} ranked, price override missing for {missing}")
    return results


def run_scan(tickers, refresh=False):
    cache = os.path.join(CACHE, "scan_results.pkl")
    res = {} if refresh or not os.path.exists(cache) else pickle.load(open(cache, "rb"))
    failed = {}
    todo = [t for t in tickers if t not in res]
    for i, t in enumerate(todo, 1):
        try:
            res[t] = bs.analyze(t, ysym(t)); print(f"[{i}/{len(todo)}] {t}: {res[t]['n_flags']}/10", flush=True)
        except Exception as e:
            failed[t] = str(e)[:120]; print(f"[{i}/{len(todo)}] {t}: FAILED {failed[t]}", flush=True)
        if i % 10 == 0: pickle.dump(res, open(cache, "wb"))
    pickle.dump(res, open(cache, "wb"))
    return res, failed


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-scan", action="store_true"); ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-rerank", action="store_true"); a = ap.parse_args()
    # collect tickers
    for p in PORTFOLIOS:
        if p["run"] is not None: p["tickers"] = sorted(load_json(os.path.join(RUNS, p["run"], p["data"])))
    allt = sorted({t for p in PORTFOLIOS for t in p["tickers"]})
    print(f"{len(allt)} unique tickers across {len(PORTFOLIOS)} portfolios")
    prices, asof = fetch_prices(allt)
    print(f"prices fetched for {len(prices)}/{len(allt)}; missing: {sorted(set(allt)-set(prices))}")
    json.dump({"prices": prices, "asof": asof, "fetched": datetime.now().isoformat(timespec='minutes')}, open(os.path.join(CACHE, "prices.json"), "w"), indent=1)
    if not a.no_rerank:
        rk = rerun_rankings(prices, asof)
        pickle.dump(rk, open(os.path.join(CACHE, "rankings.pkl"), "wb"))
    if not a.skip_scan:
        res, failed = run_scan(allt, a.refresh)
        json.dump(failed, open(os.path.join(CACHE, "scan_failed.json"), "w"), indent=1)
        print(f"scan done: {len(res)} ok, {len(failed)} failed: {failed}")
    json.dump(PORTFOLIOS, open(os.path.join(CACHE, "portfolios.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
