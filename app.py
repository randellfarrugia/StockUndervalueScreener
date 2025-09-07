import re, time, random
from typing import Optional, Dict, List

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import streamlit.components.v1 as components

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Stock Undervaluation Screener",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Sidebar: appearance & chart controls
# =========================
st.sidebar.header("Appearance")
APP_THEME = st.sidebar.selectbox("App theme", ["Light", "Dark"], index=0)
chart_interval = st.sidebar.selectbox("Chart interval", ["1D", "1W", "1M"], index=0)

if APP_THEME == "Dark":
    st.markdown("""
    <style>
      html, body, [data-testid="stAppViewContainer"] { background:#0f1116 !important; color:#e6e6e6 !important; }
      .stMarkdown, .stText, .stCaption, .stDataFrame { color:#e6e6e6 !important; }
      div[data-testid="stHeader"] { background: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

# =========================
# Title + caption
# =========================
st.markdown("# Stock Undervaluation Screener")
st.caption("Screen and score stocks using Finviz data with quick ratios and an interactive chart.")
st.divider()

# =========================
# Scoring thresholds
# =========================
st.sidebar.header("Scoring thresholds")
PE_CAP         = st.sidebar.slider("P/E cap", 0.0, 30.0, 5.0, 0.5)
FWD_PE_CAP     = st.sidebar.slider("Forward P/E cap", 0.0, 50.0, 10.0, 0.5)  # separate slider
EV_EBITDA_CAP  = st.sidebar.slider("EV/EBITDA cap", 3.0, 12.0, 6.0, 0.5)
EV_SALES_CAP   = st.sidebar.slider("EV/Sales cap", 0.3, 6.0, 2.0, 0.1)
PB_CAP         = st.sidebar.slider("P/B cap", 0.5, 3.0, 1.0, 0.1)

VALUE_YIELD_FLOOR    = st.sidebar.slider("Value yield floor (%) [FCF/MC or EBIT/EV]", 0.0, 20.0, 8.0, 0.5) / 100.0
DEBT_EQUITY_CAP      = st.sidebar.slider("Debt/Equity cap", 0.1, 3.0, 1.0, 0.1)
DIVIDEND_YIELD_FLOOR = st.sidebar.slider("Dividend yield floor (%)", 0.0, 8.0, 2.0, 0.25) / 100.0
RECOM_BUY_CEILING    = st.sidebar.slider("Analyst Recom buy ceiling (≤ is Buy)", 1.0, 4.0, 2.5, 0.1)

# =========================
# Rate limiting & diagnostics
# =========================
st.sidebar.header("Rate limiting")
RATE_DELAY  = st.sidebar.slider("Delay between Finviz calls (sec)", 0.0, 3.0, 1.2, 0.1)
MAX_RETRIES = st.sidebar.slider("Max retries on 429/5xx", 0, 5, 3, 1)
FETCH_DETAILS = st.sidebar.checkbox("Fetch Balance Sheet & Cash Flow (more accurate metrics)", value=True)

st.sidebar.header("Diagnostics")
DEBUG = st.sidebar.checkbox("Debug mode (show API calls & parsed fields)", value=False)
if st.sidebar.button("Clear cached data"):
    st.cache_data.clear()
    st.session_state["debug_msgs"] = []
    st.success("Cache cleared.")

# Collect debug lines in session; render at bottom only if DEBUG is True
if "debug_msgs" not in st.session_state:
    st.session_state["debug_msgs"] = []
def debug_log(msg: str):
    if DEBUG:
        st.session_state["debug_msgs"].append(msg)

# =========================
# Helpers
# =========================
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finviz.com/"}

def safe_float(x) -> Optional[float]:
    try:
        if x in (None, "", "-", "—"): return None
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip().replace(",", "")
        return float(s)
    except Exception:
        return None

def pct_str(x: Optional[float]) -> str:
    return f"{100*x:.1f}%" if isinstance(x, (int,float)) and x is not None else "—"

def human(x, d: int = 2) -> str:
    if x is None: return "—"
    try: return f"{x:.{d}f}"
    except Exception: return str(x)

def normalize_symbol(q: str) -> str:
    return "".join(ch for ch in q.strip() if ch.isalnum() or ch in ".-").upper()

def to_tradingview_symbol(sym: str) -> str:
    if ":" in sym: return sym
    m = re.match(r"^([A-Z0-9\-]+)\.([A-Z]{1,3})$", sym)
    if m:
        base, suf = m.group(1), m.group(2)
        mapping = {
            "TO":"TSX","V":"TSXV","CN":"CSE",
            "PA":"EURONEXT","AS":"EURONEXT","BR":"EURONEXT",
            "OL":"OSL","ST":"OMXSTO","CO":"OMXCOP","HE":"OMXHEX",
            "L":"LSE","DE":"XETR","F":"FWB","SW":"SIX",
            "MC":"BME","MI":"MIL","HK":"HKEX",
            "AX":"ASX","NZ":"NZX","T":"TSE","SA":"BMFBOVESPA",
        }
        return f"{mapping.get(suf, '')}:{base}" if mapping.get(suf) else base
    return sym

def recom_label(v: Optional[float]) -> str:
    """Map Finviz Recom numeric (1-5) to text + emoji."""
    if v is None: return "—"
    try:
        v = float(v)
    except Exception:
        return str(v)
    if v <= 1.5:   txt, emoji = "Strong Buy", "🟢🟢"
    elif v <= 2.5: txt, emoji = "Buy", "🟢"
    elif v <= 3.5: txt, emoji = "Hold", "🟡"
    elif v <= 4.5: txt, emoji = "Sell", "🔴"
    else:          txt, emoji = "Strong Sell", "⛔️"
    return f"{v:.1f} ({txt}) {emoji}"

# =========================
# Finviz Statement API (JSON) with throttle/backoff
# =========================
BASE = "https://finviz.com/api/statement.ashx"
_last_call_ts = 0.0
def _throttle():
    global _last_call_ts
    now = time.time()
    wait = max(0.0, RATE_DELAY - (now - _last_call_ts))
    if wait > 0: time.sleep(wait)
    _last_call_ts = time.time()

@st.cache_data(show_spinner=False, ttl=3600)  # stickier cache
def get_statement(symbol: str, s: Optional[str] = None, debug: bool = False) -> Optional[Dict]:
    params = {"t": symbol}
    if s: params["s"] = s
    attempt = 0
    while True:
        _throttle()
        try:
            r = requests.get(BASE, params=params, timeout=12, headers=HEADERS)
            if debug:  # collect textual traces; do not write UI here
                debug_log(f"GET {r.status_code} {r.url}")
            if r.status_code == 429:
                if attempt >= MAX_RETRIES:
                    if debug: debug_log("429: max retries reached; aborting.")
                    return None
                time.sleep((2 ** attempt) + random.uniform(0, 0.5)); attempt += 1; continue
            r.raise_for_status()
            js = r.json()
            return js if isinstance(js, dict) and "data" in js else None
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status and 500 <= status < 600 and attempt < MAX_RETRIES:
                time.sleep((2 ** attempt) + random.uniform(0, 0.5)); attempt += 1; continue
            return None
        except Exception:
            return None

# =========================
# Finviz Snapshot (HTML) → Sector/Industry/FwdPE/Recom
# =========================
SNAPSHOT_URL = "https://finviz.com/quote.ashx"

@st.cache_data(show_spinner=False, ttl=3600)  # stickier cache
def fetch_snapshot(symbol: str, debug: bool = False) -> dict:
    out = {"sector": None, "industry": None, "fwd_pe": None, "recom": None}
    params = {"t": symbol}; attempt = 0
    while True:
        time.sleep(RATE_DELAY if attempt == 0 else (RATE_DELAY + (2 ** attempt) + random.uniform(0, 0.5)))
        try:
            r = requests.get(SNAPSHOT_URL, params=params, timeout=12, headers=HEADERS)
            if debug:
                debug_log(f"GET {r.status_code} {r.url}")
            if r.status_code == 429:
                if attempt >= MAX_RETRIES:
                    if debug: debug_log("429: max retries reached (snapshot); aborting.")
                    return out
                attempt += 1; continue
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            table = soup.select_one("table.snapshot-table2, table.snapshot-table")
            if not table: return out
            cells = [td.get_text(strip=True) for td in table.select("td")]
            pairs = {cells[i].rstrip(":"): cells[i+1] for i in range(0, len(cells)-1, 2)}
            def to_float(s: str) -> Optional[float]:
                try: return float(s.replace(",", ""))
                except: return None
            out["sector"]  = pairs.get("Sector") or None
            out["industry"]= pairs.get("Industry") or None
            out["fwd_pe"]  = to_float(pairs.get("Forward P/E") or pairs.get("ForwardPE") or "")
            out["recom"]   = to_float(pairs.get("Recom") or "")
            return out
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status and 500 <= status < 600 and attempt < MAX_RETRIES:
                attempt += 1; continue
            return out
        except Exception:
            return out

# =========================
# TradingView interactive chart
# =========================
def render_tv_chart(symbol: str, theme: str = "Light", interval: str = "1D"):
    tv_symbol = to_tradingview_symbol(symbol)
    tv_theme = "light" if theme.lower().startswith("l") else "dark"
    interval_map = {"1D": "D", "1W": "W", "1M": "M"}
    tv_interval = interval_map.get(interval, "D")
    height = 520 if tv_interval == "D" else 500
    widget_id = f"tv_{re.sub(r'[^A-Za-z0-9]', '', tv_symbol)}"
    html = f"""
    <div class="tradingview-widget-container" style="height:{height}px;">
      <div id="{widget_id}" style="height:{height}px;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{tv_symbol}",
          "interval": "{tv_interval}",
          "timezone": "Etc/UTC",
          "theme": "{tv_theme}",
          "style": "1",
          "locale": "en",
          "hide_top_toolbar": false,
          "hide_legend": false,
          "allow_symbol_change": true,
          "container_id": "{widget_id}"
        }});
      </script>
    </div>
    """
    components.html(html, height=height, scrolling=False)

# =========================
# Key helpers for Finviz statement
# =========================
def find_key(data: Dict[str, List[str]], candidates: List[str]) -> Optional[str]:
    keys = list(data.keys()); lower_map = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in lower_map: return lower_map[c.lower()]
    for c in candidates:
        for k in keys:
            if c.lower() in k.lower(): return k
    return None

def get_ttm_value(js: Dict, key_candidates: List[str]) -> Optional[float]:
    data = js.get("data", {}); period = data.get("Period", [])
    k = find_key(data, key_candidates)
    if not k: return None
    arr = data.get(k, [])
    if not arr: return None
    idx = 0
    for i, p in enumerate(period):
        if str(p).strip().upper() == "TTM": idx = i; break
    return safe_float(arr[idx])

def get_latest_annual(js: Dict, key_candidates: List[str]) -> Optional[float]:
    data = js.get("data", {})
    k = find_key(data, key_candidates)
    if not k: return None
    arr = data.get(k, [])
    return safe_float(arr[0]) if arr else None

# =========================
# Build metrics (Finviz statement + snapshot)
# =========================
def build_metrics(symbol: str) -> Optional[Dict]:
    sym = normalize_symbol(symbol)

    base = get_statement(sym, None, debug=DEBUG)
    if not base: return None

    bal = cfa = None
    if FETCH_DETAILS:
        bal = get_statement(sym, "BA", debug=DEBUG)
        cfa = get_statement(sym, "CA", debug=DEBUG)

    snap = fetch_snapshot(sym, debug=DEBUG)
    sector = snap.get("sector") or ""
    industry = snap.get("industry") or ""
    fwd_pe = snap.get("fwd_pe")
    recom  = snap.get("recom")

    mktcap = get_ttm_value(base, ["Market Capitalization"]) or get_latest_annual(base, ["Market Capitalization"])
    shares  = get_ttm_value(base, ["Shares Outstanding"]) or get_latest_annual(base, ["Shares Outstanding"])
    pe      = get_ttm_value(base, ["Price To Earnings Ratio","P/E","Price/Earnings"])
    ebitda  = get_ttm_value(base, ["EBITDA"]) or get_latest_annual(base, ["EBITDA"])
    revenue = get_ttm_value(base, ["Total Revenue"]) or get_latest_annual(base, ["Total Revenue"])
    op_inc  = get_ttm_value(base, ["Operating Income"]) or get_latest_annual(base, ["Operating Income"])

    total_equity = total_debt = cash_like = None
    if bal:
        total_equity = get_latest_annual(bal, ["Total Stockholder Equity","Total Stockholders Equity","Total Shareholder Equity","Total Equity"])
        total_debt = (get_latest_annual(bal, ["Total Debt","Total debt"])
                      or ((get_latest_annual(bal, ["Long-Term Debt"]) or 0.0) + (get_latest_annual(bal, ["Short-Term Debt","Short Term Debt"]) or 0.0)))
        cash_like = get_latest_annual(bal, ["Cash and Cash Equivalents","Cash & Equivalents","Cash & Short Term Investments","Cash"])

    ocf = capex = dividends_paid = None
    if cfa:
        ocf = get_latest_annual(cfa, ["Operating Cash Flow","Net Cash Provided by Operating Activities","Total Cash From Operating Activities"])
        capex = get_latest_annual(cfa, ["Capital Expenditures","Purchase of Property, Plant & Equipment","Capital Expenditure"])
        dividends_paid = get_latest_annual(cfa, ["Dividends Paid","Cash Dividends Paid"])

    price = (mktcap / shares) if (mktcap and shares and shares != 0) else None
    pb    = (mktcap / total_equity) if (mktcap and total_equity and total_equity != 0) else None
    ev    = (mktcap + (total_debt or 0.0) - (cash_like or 0.0)) if mktcap is not None else None
    ev_ebitda = (ev / ebitda) if (ev is not None and ebitda not in (None, 0)) else None
    ev_sales  = (ev / revenue) if (ev is not None and revenue not in (None, 0)) else None

    # Treat zero values correctly; guard equity==0
    fcf_yield = ((ocf - capex) / mktcap) if (mktcap and ocf is not None and capex is not None) else None
    ebit_ev_yield = (op_inc / ev) if (op_inc is not None and ev not in (None, 0)) else None
    div_yield = (abs(dividends_paid) / mktcap) if (mktcap and dividends_paid is not None) else None
    debt_eq   = (total_debt / total_equity) if (total_debt is not None and total_equity not in (None, 0)) else None

    return {
        "source": "Finviz (statement + snapshot)",
        "name": sym,
        "sector": sector,
        "industry": industry,
        "price": price,
        "pe": pe,
        "fwd_pe": fwd_pe,
        "pb": pb,
        "ev_ebitda": ev_ebitda,
        "ev_sales": ev_sales,
        "fcf_yield": fcf_yield,
        "ebit_ev_yield": ebit_ev_yield,
        "div_yield": div_yield,
        "debt_eq": debt_eq,
        "recom": recom,
    }

# =========================
# Scoring
# =========================
def score_metrics(m: Dict):
    r1_pass = (m["pe"] is not None and m["pe"] > 0 and m["pe"] <= PE_CAP)
    r1_metric = f"PE: {human(m['pe'])}"; r1_thresh = f"≤ {human(PE_CAP)}"

    if m.get("fwd_pe") is not None and m["fwd_pe"] > 0:
        r2_pass   = m["fwd_pe"] <= FWD_PE_CAP
        r2_metric = f"Fwd P/E: {human(m['fwd_pe'])}"; r2_thresh = f"≤ {human(FWD_PE_CAP)}"
    else:
        r2_pass   = (m["ev_sales"] is not None and m["ev_sales"] <= EV_SALES_CAP)
        r2_metric = f"EV/Sales: {human(m['ev_sales'])}"; r2_thresh = f"≤ {human(EV_SALES_CAP)}"

    r3_pass = (m["ev_ebitda"] is not None and m["ev_ebitda"] <= EV_EBITDA_CAP)
    r4_pass = (m["pb"] is not None and m["pb"] > 0 and m["pb"] <= PB_CAP)

    value_yield = m["fcf_yield"] if m["fcf_yield"] is not None else m["ebit_ev_yield"]
    r5_pass = (value_yield is not None and value_yield >= VALUE_YIELD_FLOOR)

    r6_pass = (m["debt_eq"] is not None and m["debt_eq"] <= DEBT_EQUITY_CAP)

    has_div = (m["div_yield"] is not None and m["div_yield"] >= DIVIDEND_YIELD_FLOOR)
    has_buy = (m.get("recom") is not None and m["recom"] <= RECOM_BUY_CEILING)
    r7_pass = has_div or has_buy

    # Build Rule 7 metric string, but only include items that exist
    r7_parts = []
    if m.get("div_yield") is not None:
        r7_parts.append(f"Div: {pct_str(m['div_yield'])}")
    if m.get("recom") is not None:
        r7_parts.append(f"Recom: {recom_label(m.get('recom'))}")
    r7_metric = " | ".join(r7_parts) if r7_parts else "—"

    total = sum([r1_pass, r2_pass, r3_pass, r4_pass, r5_pass, r6_pass, r7_pass])
    breakdown = [
        {"Rule":"1) Low P/E","Pass":r1_pass,"Metric":r1_metric,"Thresh":r1_thresh},
        {"Rule":"2) Fwd P/E (or EV/Sales if missing)","Pass":r2_pass,"Metric":r2_metric,"Thresh":r2_thresh},
        {"Rule":"3) EV/EBITDA cheap","Pass":r3_pass,"Metric":f"EV/EBITDA: {human(m['ev_ebitda'])}","Thresh":f"≤ {human(EV_EBITDA_CAP)}"},
        {"Rule":"4) P/B cheap","Pass":r4_pass,"Metric":f"P/B: {human(m['pb'])}","Thresh":f"≤ {human(PB_CAP)}"},
        {"Rule":"5) Value yield strong","Pass":r5_pass,"Metric":f"{'FCF/MktCap' if m['fcf_yield'] is not None else 'EBIT/EV'}: {pct_str(value_yield)}","Thresh":f"≥ {pct_str(VALUE_YIELD_FLOOR)}"},
        {"Rule":"6) Balance-sheet safe","Pass":r6_pass,"Metric":f"D/E: {human(m['debt_eq'])}","Thresh":f"≤ {human(DEBT_EQUITY_CAP)}"},
        {"Rule":"7) Income/Analyst Support","Pass":r7_pass,"Metric":r7_metric,"Thresh":f"Div≥{pct_str(DIVIDEND_YIELD_FLOOR)} OR Recom≤{human(RECOM_BUY_CEILING)}"},
    ]
    return total, breakdown

def render_breakdown(breakdown: List[Dict]):
    lines = []
    for row in breakdown:
        tick = "✅" if row["Pass"] else "❌"
        lines.append(f"- {tick} **{row['Rule']}** — {row['Metric']}  *(target: {row['Thresh']})*")
    st.markdown("\n".join(lines))

# =========================
# SINGLE TICKER — inline input & button via form
# =========================
st.markdown("### Single Ticker")
st.caption("Enter a symbol and click **Run Score**")

with st.form("single_search", clear_on_submit=False):
    c1, c2 = st.columns([8, 2])
    with c1:
        ticker = st.text_input(
            label="Ticker",
            value="CIVI",
            placeholder="e.g., CIVI, CTRA, MOS, SFM, AF.PA, OBE.TO",
            label_visibility="collapsed"
        )
    with c2:
        submit = st.form_submit_button("Run Score", use_container_width=True)

# Results directly below the controls
chart_col, score_col = st.columns([5, 5])
if submit:
    symbol = normalize_symbol(ticker)
    with st.spinner(f"Scoring {symbol}…"):
        metrics = build_metrics(symbol)

    with chart_col:
        st.markdown("### Price chart")
        render_tv_chart(symbol, theme=APP_THEME, interval=chart_interval)

    with score_col:
        if metrics:
            st.markdown("### Score")
            st.write(
                f"**Symbol**: {symbol}  |  **Sector**: {metrics.get('sector') or '—'}  |  **Industry**: {metrics.get('industry') or '—'}  |  "
                f"**Price (est.)**: {human(metrics.get('price'))}  |  **Source**: {metrics.get('source','—')}"
            )
            total, breakdown = score_metrics(metrics)
            st.markdown(f"#### Total Score: **{total} / 7**")
            render_breakdown(breakdown)

            # ---- Split display for Income & Analysts (only show if present)
            if (metrics.get("div_yield") is not None) or (metrics.get("recom") is not None):
                st.markdown("#### Income & Analysts")
                if metrics.get("div_yield") is not None:
                    st.write(f"Dividend yield: **{pct_str(metrics['div_yield'])}**")
                if metrics.get("recom") is not None:
                    st.write(f"Analyst rating: **{recom_label(metrics['recom'])}**")
        else:
            st.error(f"No data from Finviz for '{symbol}'. If non-US, try the exact exchange suffix (e.g., AF.PA, YAR.OL, OBE.TO).")

st.divider()

# =========================
# BATCH MODE — spinner + progress + numbering starts at 1
# =========================
st.markdown("### Batch Mode")
batch_txt = st.text_area("Symbols (comma/space/newline separated):", "CIVI, CTRA, MOS, SFM, CCJ")
run_batch = st.button("Run Batch")

if run_batch:
    raw = re.split(r"[,\s]+", batch_txt.strip())
    syms = [normalize_symbol(s) for s in raw if s]
    rows: List[Dict] = []
    with st.spinner("Running batch scoring…"):
        progress = st.progress(0)
        n = max(len(syms), 1)
        for i, s in enumerate(syms):
            time.sleep(RATE_DELAY)
            m = build_metrics(s)
            if m:
                score, _ = score_metrics(m)
                rows.append({
                    "Symbol": s, "Score": score, "Sector": m.get("sector") or "",
                    "Price(est.)": m.get("price"),
                    "PE": m.get("pe"),
                    "FwdPE": m.get("fwd_pe"),
                    "EV/Sales": m.get("ev_sales"),
                    "EV/EBITDA": m.get("ev_ebitda"),
                    "P/B": m.get("pb"),
                    "ValueYield(%)": (100*(m.get("fcf_yield") if m.get("fcf_yield") is not None else m.get("ebit_ev_yield")) if (m.get("fcf_yield") is not None or m.get("ebit_ev_yield") is not None) else None),
                    "D/E": m.get("debt_eq"),
                    "DivYield(%)": (100*m.get("div_yield") if m.get("div_yield") is not None else None),
                    "Recom": recom_label(m.get("recom")),  # numeric + text + emoji
                })
            else:
                rows.append({"Symbol": s, "Score": None, "Sector": "",
                             "Price(est.)": None, "PE": None, "FwdPE": None,
                             "EV/Sales": None, "EV/EBITDA": None, "P/B": None,
                             "ValueYield(%)": None, "D/E": None, "DivYield(%)": None, "Recom": None})
            progress.progress(int((i+1) / n * 100))
        progress.empty()

    if rows:
        outdf = pd.DataFrame(rows).sort_values(["Score","Symbol"], ascending=[False, True]).reset_index(drop=True)
        outdf.index = outdf.index + 1
        outdf.index.name = "#"

        # Bold top-scoring row(s)
        try:
            max_score = pd.to_numeric(outdf["Score"], errors="coerce").max()
            def _hl(row, max_score=max_score):
                return ["font-weight: 700;" if row.get("Score", None) == max_score else "" for _ in row]
            styled = outdf.style.apply(_hl, axis=1)
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(outdf, use_container_width=True)

        # include index in CSV so numbering persists
        csv = outdf.to_csv(index=True).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="scores.csv", mime="text/csv")

# =========================
# Footer — GitHub link with icon
# =========================
st.markdown("""
<hr style="margin:2rem 0 0.75rem 0; border:none; border-top:1px solid rgba(128,128,128,.25);" />
<div style="display:flex;justify-content:center;align-items:center;gap:.5rem;opacity:.8;">
  <a href="https://github.com/randellfarrugia" target="_blank" style="text-decoration:none;color:inherit;display:flex;align-items:center;gap:.4rem;">
    <svg height="18" viewBox="0 0 16 16" width="18" aria-hidden="true" style="vertical-align:middle;">
      <path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
      0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01
      1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
      0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.68 7.68 0 0 1 2-.27c.68 0 1.36.09
      2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87
      3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013
      8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
    </svg>
    <span>github.com/randellfarrugia</span>
  </a>
</div>
""", unsafe_allow_html=True)

# =========================
# Bottom-of-page debug (only if DEBUG is ON)
# =========================
if DEBUG and st.session_state["debug_msgs"]:
    st.markdown("#### Debug")
    st.code("\n".join(st.session_state["debug_msgs"]))
else:
    # ensure no stale logs show up when toggling off
    st.session_state["debug_msgs"] = []
