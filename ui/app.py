import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import streamlit as st


import streamlit as st
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "DATABASE_URL" in st.secrets:
    os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]


API_BASE = (
    st.secrets.get("MP_API_BASE")
    if "MP_API_BASE" in st.secrets
    else os.getenv("MP_API_BASE", "https://marketpulse-6qis.onrender.com")
)
WATCHLIST_PATH = Path("data/watchlist.txt")

# --------------------------
# Page + theme
# --------------------------
st.set_page_config(page_title="MarketPulse", page_icon="📈", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;600&display=swap');

:root {
  --bg: #0b0f14;
  --panel: #111722;
  --ink: #e6eef7;
  --muted: #9db0c6;
  --accent: #39c6ff;
}

html, body, [class^="css"]  {
  background-color: var(--bg) !important;
  color: var(--ink) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial;
}

/* Pull content down a bit so header never clips, but keep it tight */
.block-container { padding-top: 1.1rem; }

/* Header */
h1.marketpulse {
  font-family: Rajdhani, Inter, sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  margin: 0.4rem 0 0.2rem 0; /* <-- top margin prevents clipping */
  background: linear-gradient(90deg, #7df9ff, #39c6ff 40%, #00ffa3);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 2.35rem !important;
}
div.subtitle { color: var(--muted); margin-bottom: 0.5rem; }

/* Sidebar */
div[data-testid="stSidebar"] { background: #111722; }

/* Tabs underline */
div.stTabs [role="tablist"] { border-bottom: 1px solid #1e2836; margin-bottom: 0.4rem; }
div.stTabs [role="tab"] { color: #9db0c6; }
div.stTabs [aria-selected="true"] { color: var(--ink); font-weight: 600; }

/* Headline bubble card */
.news-card {
  width: 100%;
  border-radius: 14px;
  padding: 12px 14px;
  border: 1px solid rgba(255,255,255,0.08);
  margin: 8px 0 10px;
  transition: transform .05s ease, border-color .15s ease, box-shadow .15s ease;
}
.news-card:hover { transform: translateY(-1px); box-shadow: 0 0 0 1px rgba(255,255,255,.06) inset; }
.news-title a { color: #eaf6ff; font-weight: 600; text-decoration: none; }
.news-title a:hover { text-decoration: underline; }
.news-meta { color: #9db0c6; font-size: 0.82rem; }

/* right-side summarize button cell */
div.sum-cell { display: flex; justify-content: flex-end; align-items: center; height: 100%; }

/* Answer block */
.answer {
  background: #0f1520;
  border: 1px solid #1f2a37;
  border-radius: 10px;
  padding: 0.9rem;
  white-space: pre-wrap;
  margin-top: .3rem;
}

/* Chips (example questions) */
.chips { display:flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.chip {
  background: #142030;
  border: 1px solid #1f2e42;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.85rem;
  color: var(--ink);
  cursor: pointer;
}
.chip:hover { border-color: #2a3b55; }

/* Disclaimer box */
.disclaimer {
  font-size: 0.78rem;
  color: #9db0c6;
  border: 1px solid #1f2a37;
  background: #0f1520;
  border-radius: 8px;
  padding: 8px 10px;
}

/* small helper text under sliders */
.smallhint { color: #9db0c6; font-size: 0.85rem; margin-top: -4px; }

/* Summarize button styling */
button[data-testid="baseButton-secondary"] {
  font-size: 0.75rem !important;
  padding: 0.25rem 0.5rem !important;
}
</style>
"""



st.markdown(CSS, unsafe_allow_html=True)

# --------------------------
# Helpers
# --------------------------
def load_watchlist(path: Path) -> List[str]:
    if path.exists():
        tickers = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().upper()
                if s and not s.startswith("#"):
                    tickers.append(s)
        if tickers:
            return tickers
    return ["NVDA", "MSFT", "AAPL", "GOOG", "AMZN", "META", "AVGO", "TSM", "TSLA"]

def api_get_headlines(ticker: str, days: int, limit: int = 100) -> pd.DataFrame:
    url = f"{API_BASE}/headlines"
    r = requests.get(url, params={"ticker": ticker, "days": days, "limit": limit}, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())

def api_post_ask(query: str, ticker: Optional[str], days: int, topk: int = 6) -> Dict[str, Any]:
    url = f"{API_BASE}/ask"
    r = requests.post(url, json={"query": query, "ticker": ticker, "days": days, "topk": topk}, timeout=120)
    r.raise_for_status()
    return r.json()

def api_post_summarize(article_id: int) -> str:
    url = f"{API_BASE}/summarize"
    r = requests.post(url, json={"article_id": article_id}, timeout=120)
    r.raise_for_status()
    return r.json().get("summary", "")

def intensity_hex(label: str, p_neg: float, p_neu: float, p_pos: float) -> str:
    label = (label or "neutral").lower()
    m = max(p for p in [p_neg or 0.0, p_neu or 0.0, p_pos or 0.0])
    t = max(0.2, min(0.95, m))  # 0.2..0.95
    if label == "negative":
        r = int(255 * t); g = int(28 * (1 - t)); b = int(36 * (1 - t))
    elif label == "positive":
        r = int(18 * (1 - t)); g = int(240 * t); b = int(83 * (0.4 + 0.6 * t))
    else:
        v = int(40 + 120 * t); r = g = b = v
    return f"#{r:02x}{g:02x}{b:02x}"

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def conf_pct(p_neg: Optional[float], p_neu: Optional[float], p_pos: Optional[float]) -> int:
    m = max(p for p in [p_neg or 0, p_neu or 0, p_pos or 0])
    return int(round(m * 100))

# --------------------------
# Header
# --------------------------
st.markdown('<h1 class="marketpulse">MarketPulse</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">News triage · sentiment · cited answers</div>', unsafe_allow_html=True)

# --------------------------
# Sidebar
# --------------------------
with st.sidebar:
    st.markdown("### Settings")
    days = st.slider("Lookback (days)", min_value=1, max_value=30, value=7)
    limit = st.slider("Headlines per ticker", min_value=10, max_value=300, value=120, step=10)
    st.caption("Tip: the deeper the lookback, the more context the retriever can use.")

# --------------------------
# Layout
# --------------------------
col_left, col_right = st.columns([2.2, 1.0], gap="large")
tickers = load_watchlist(WATCHLIST_PATH)

# --------------------------
# LEFT: Headlines 
# --------------------------
with col_left:
    tabs = st.tabs(tickers)
    for idx, t in enumerate(tickers):
        with tabs[idx]:
            st.caption(f"Most recent for **{t}** (last {days}d)")
            try:
                df = api_get_headlines(t, days=days, limit=limit)
            except Exception as e:
                st.error(f"Unable to load headlines for {t}: {e}")
                continue

            if df.empty:
                st.info("No headlines in this window.")
                continue

            for _, row in df.iterrows():
                label = (row.get("sentiment_label") or "neutral").lower()
                pneg, pneu, ppos = row.get("p_neg") or 0.0, row.get("p_neu") or 0.0, row.get("p_pos") or 0.0
                bg_hex = intensity_hex(label, pneg, pneu, ppos)
                fill = hex_to_rgba(bg_hex, 0.16)
                border = hex_to_rgba(bg_hex, 0.55)

                head = (row.get("headline") or "").strip()
                url  = row.get("url") or "#"
                src  = row.get("source") or "source"
                date = row.get("published_at") or ""
                aid  = int(row.get("article_id"))
                conf = conf_pct(pneg, pneu, ppos)

                left, right = st.columns([0.80, 0.20])
                with left:
                    st.markdown(
                        f"""
                        <div class="news-card" style="background:{fill}; border-color:{border}">
                          <div class="news-title"><a href="{url}" target="_blank">{head}</a></div>
                          <div class="news-meta">{src} · {date} &nbsp;&nbsp;<span style="color:#9db0c6;">({label.capitalize()} • {conf}% confidence)</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with right:
                    st.markdown('<div class="sum-cell">', unsafe_allow_html=True)
                    if st.button("Summarize", key=f"sum_{t}_{aid}"):
                        try:
                            summary = api_post_summarize(aid)
                            st.session_state[f"_summary_{aid}"] = summary
                        except Exception as e:
                            st.session_state[f"_summary_{aid}"] = f"Summarize failed: {e}"
                    st.markdown('</div>', unsafe_allow_html=True)

                if st.session_state.get(f"_summary_{aid}"):
                    st.markdown(f"<div class='answer'>{st.session_state[f'_summary_{aid}']}</div>", unsafe_allow_html=True)

# --------------------------
# RIGHT: Ask (PulseBot)
# --------------------------
with col_right:
    st.markdown("#### PulseBot")
    examples = [
        "Why is NVDA up today?",
        "Top 3 drivers for AMZN this week",
        "What moved semiconductors today?",
        "Why did TSLA drop after earnings?",
        "Summarize Wells Fargo’s note on AAPL",
    ]
    st.markdown('<div class="chips">' + "".join([f'<span class="chip" onclick="window.postMessage(\'setq::{e}\', \'*\')">{e}</span>' for e in examples]) + "</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <script>
        window.addEventListener('message', (ev) => {
          if (typeof ev.data === 'string' && ev.data.startsWith('setq::')) {
            const q = ev.data.slice(6);
            const target = window.parent.document.querySelector('textarea');
            if (target) { target.value = q; target.dispatchEvent(new Event('input', {bubbles: true})); }
          }
        });
        </script>
        """,
        unsafe_allow_html=True,
    )

    sel_ticker = st.selectbox("Ticker (optional)", options=["(auto)"] + tickers, index=0)
    q = st.text_area("Question", value="Why is NVDA up today?", height=90)

    topk = st.slider("Number of source articles", 3, 8, 5)
    st.markdown('<div class="smallhint">Higher = more coverage (slower), lower = snappier.</div>', unsafe_allow_html=True)

    if st.button("Ask", type="primary"):
        tt = None if sel_ticker == "(auto)" else sel_ticker
        try:
            resp = api_post_ask(q, tt, days=days, topk=topk)
            st.markdown("**Answer**")
            st.markdown(f"<div class='answer'>{resp.get('answer','')}</div>", unsafe_allow_html=True)

            if resp.get("hits"):
                st.markdown("**Sources used**")
                for h in resp["hits"]:
                    st.write(f"- [{h['headline']}]({h['url']}) — {h.get('source','')} · {h.get('published_at','')}")
        except Exception as e:
            st.error(f"Ask failed: {e}")

    st.markdown("<div class='disclaimer'>This is not financial advice. It’s automated news analysis for information only.</div>", unsafe_allow_html=True)

st.caption("Tip: bubble color = sentiment × confidence. Deeper green/red = stronger model confidence.")
