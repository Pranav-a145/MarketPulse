import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, urlunparse
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import feedparser
import pandas as pd

from sqlalchemy import text
from db.conn import get_engine

# ---------------- General publisher feeds ----------------
FINANCIAL_RSS_FEEDS = {
    # Major wires and market desks
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "reuters_markets": "https://feeds.reuters.com/reuters/USMarketsNews",
    "cnbc_markets": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "bloomberg_markets": "https://feeds.bloomberg.com/markets/news.rss",
    "wsj_markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",

    # Additional free-heavy sources to boost volume
    "investing_com": "https://www.investing.com/rss/news.rss",
    "motley_fool": "https://www.fool.com/feeds/index.aspx",
    "zacks": "https://www.zacks.com/feeds/most_recent_articles.rss",
    "marketbeat": "https://www.marketbeat.com/headlines/rss/",
    "benzinga": "https://www.benzinga.com/feed",

    # Tech-business that often carries FAAMG news
    "techcrunch": "https://techcrunch.com/feed/",
    "ars_technica": "https://feeds.arstechnica.com/arstechnica/index",
    "fortune": "https://fortune.com/feed/",
    "business_insider": "https://www.businessinsider.com/sai/rss",
}

# ---------------- Tunables ----------------
SLEEP_BETWEEN_FEEDS = 0.35
DAYS_FILTER = 7

MAX_PER_TICKER = 150
MIN_PER_TICKER = 50
STRICT_THRESHOLD = 2.5
RELAXED_THRESHOLD = 1.5

DOMAIN_BLACKLIST = {
}

BAD_TITLE_PATTERNS = [
    r"\bInc\. 5000\b",
    r"\bindustry roundup\b",
    r"\bpress release\b",
    r"\bawards?\b",
    r"\brand(s|ed)? ranks?\b",
]

# ---------------- Watchlist ----------------
def load_watchlist(path="data/watchlist.txt"):
    tickers = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().upper()
                if s and not s.startswith("#"):
                    tickers.append(s)
    except FileNotFoundError:
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META", "JPM", "BAC", "GS", "AVGO", "TSM"]
    return tickers

# ---------------- HTTP session ----------------
def get_session():
    s = requests.Session()
    try:
        r = Retry(total=3, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "HEAD"], backoff_factor=0.6)
    except TypeError:
        r = Retry(total=3, status_forcelist=[429, 500, 502, 503, 504], method_whitelist=["GET", "HEAD"], backoff_factor=0.6)
    a = HTTPAdapter(max_retries=r)
    s.mount("http://", a)
    s.mount("https://", a)
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, text/xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    return s

def domain_of(url: str) -> str:
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

def canonical_url(u: str) -> str:
    try:
        p = urlparse(u)
        return urlunparse((p.scheme, p.netloc.lower(), p.path, "", "", ""))
    except Exception:
        return u

# ---------------- Company dictionaries ----------------
COMPANY_PRIMARY = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "GOOGL": "Alphabet",   # Class A
    "GOOG": "Alphabet",    # Class C
    "AMZN": "Amazon",
    "META": "Meta",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "GS":  "Goldman Sachs",
    "AVGO": "Broadcom",
    "TSM": "TSMC",
    "WMT": "Walmart",
    "ORCL": "Oracle",
    "NFLX": "Netflix",
    "XOM": "Exxon Mobil",
    "COST": "Costco Wholesale",
    "PLTR": "Palantir",
    "JNJ": "Johnson & Johnson",
    "BA":  "Boeing",
}

COMPANY_ALIASES = {
    "AAPL": ["Apple Inc", "iPhone", "iPad", "MacBook", "Vision Pro", "Apple Watch"],
    "MSFT": ["Microsoft Corporation", "Windows", "Azure", "Office", "Copilot", "Bing", "LinkedIn", "GitHub", "Xbox"],
    "NVDA": ["NVIDIA Corporation", "GeForce", "H100", "A100", "Blackwell", "RTX GPU", "CUDA"],
    "GOOGL": ["Google", "Alphabet Inc", "Android", "Chrome", "YouTube", "DeepMind", "Gemini"],
    "GOOG":  ["Google", "Alphabet Inc", "Android", "Chrome", "YouTube", "DeepMind", "Gemini"],
    "AMZN": ["Amazon.com", "AWS", "Alexa", "Prime Video", "Kindle", "Whole Foods", "Amazon Fresh"],
    "META": ["Meta Platforms", "Facebook", "Instagram", "WhatsApp", "Quest", "Threads", "Reality Labs"],
    "TSLA": ["Tesla Inc", "Elon Musk", "Model 3", "Model Y", "Cybertruck", "Gigafactory", "Full Self-Driving"],
    "JPM": ["JPMorgan", "JPMorgan Chase & Co", "Chase", "Jamie Dimon"],
    "BAC": ["Bank of America Corp", "BofA", "Merrill Lynch"],
    "GS":  ["Goldman Sachs Group", "Goldman", "Marcus"],
    "AVGO": ["Broadcom Inc", "VMware", "VMW"],
    "TSM": ["Taiwan Semiconductor", "Taiwan Semi", "TSMC"],
    "WMT": ["Walmart Inc", "Sam's Club", "Walmart U.S.", "Walmart International"],
    "ORCL": ["Oracle Corp", "Oracle Corporation", "OCI", "Cerner", "MySQL HeatWave", "Fusion Cloud"],
    "NFLX": ["Netflix Inc", "ad-supported tier", "password sharing crackdown"],
    "XOM": ["ExxonMobil", "Exxon", "Exxon Mobil Corp", "Permian", "Guyana"],
    "COST": ["Costco", "Costco Wholesale Corp", "Kirkland Signature", "membership fee"],
    "PLTR": ["Palantir Technologies", "Gotham", "Foundry", "Apollo"],
    "JNJ": ["Johnson & Johnson", "J&J", "Janssen", "MedTech"],
    "BA":  ["Boeing Co", "737 MAX", "787 Dreamliner", "777X", "Boeing Commercial Airplanes"],
}

# ---------------- Per-symbol feeds ----------------
def per_symbol_feed_urls(ticker: str):
    t = ticker.upper()
    urls = []
    urls.append(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US")
    urls.append(f"https://seekingalpha.com/api/sa/combined/{t}.xml")
    urls.append(f"https://www.fool.com/feeds/index.aspx?ticker={t}")
    urls.append(f"https://www.marketbeat.com/stocks/NASDAQ/{t}/feed/")
    urls.append(f"https://www.marketbeat.com/stocks/NYSE/{t}/feed/")
    return urls

# ---------------- Scoring ----------------
def relevance_score(title: str, description: str, ticker: str) -> float:
    if not title:
        return 0.0
    title_l = title.lower()
    desc_l = (description or "").lower()

    t = ticker.lower()
    primary = COMPANY_PRIMARY.get(ticker, "").lower()
    aliases = [a.lower() for a in COMPANY_ALIASES.get(ticker, [])]

    score = 0.0

    if re.search(rf"\b{re.escape(t)}\b", title_l):
        score += 3.0
    if re.search(rf"\${re.escape(t)}\b", title_l):
        score += 3.0
    if primary and primary in title_l:
        score += 2.0
    score += sum(1.0 for a in aliases if a in title_l)

    if primary and title_l.startswith(primary):
        score += 0.5
    if primary and (" " + primary + " ") in title_l[:40]:
        score += 0.5

    if primary and primary in desc_l:
        score += 0.25
    score += sum(0.25 for a in aliases if a in desc_l)

    return score

def likely_noise(title: str) -> bool:
    t = (title or "").lower()
    for pat in BAD_TITLE_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False

def fetch_general_feeds(tickers, debug=False):
    out = []
    for name, url in FINANCIAL_RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            if debug:
                print(f"\n-- {name} -- entries={len(feed.entries)}")
            for e in feed.entries:
                title = (e.get("title") or "").strip()
                link  = (e.get("link")  or "").strip()
                desc  = (e.get("description") or "").strip()
                if not title or not link:
                    continue

                # recency
                dt_str = ""
                pub = e.get("published_parsed")
                if pub:
                    dt = datetime(*pub[:6], tzinfo=timezone.utc)
                    if DAYS_FILTER is not None and dt < (datetime.now(timezone.utc) - timedelta(days=DAYS_FILTER)):
                        continue
                    dt_str = dt.strftime("%Y-%m-%d")

                src = domain_of(link)
                if src in DOMAIN_BLACKLIST:
                    continue

                # score per ticker using headline focus
                for t in tickers:
                    sc = relevance_score(title, desc, t)
                    if sc <= 0:
                        continue
                    out.append({
                        "ticker": t,
                        "headline": title,
                        "url": link,
                        "published_at": dt_str,
                        "source": src,
                        "text": desc[:500] if desc else "",
                        "feed_source": name,
                        "score": sc,
                        "noise": likely_noise(title),
                    })
        except Exception as ex:
            if debug:
                print(f"[{name}] fetch error: {ex}")
        time.sleep(SLEEP_BETWEEN_FEEDS)
    return out

def fetch_ticker_specific_feeds(ticker: str, debug=False):
    rows = []
    for url in per_symbol_feed_urls(ticker):
        try:
            feed = feedparser.parse(url)
            if debug:
                print(f"[{ticker}] per-symbol feed {url} entries={len(feed.entries)}")
            for e in feed.entries[:30]:
                title = (e.get("title") or "").strip()
                link  = (e.get("link")  or "").strip()
                desc  = (e.get("description") or "").strip()
                if not title or not link:
                    continue
                pub = e.get("published_parsed")
                dt  = datetime(*pub[:6], tzinfo=timezone.utc).strftime("%Y-%m-%d") if pub else ""
                sc  = relevance_score(title, desc, ticker)
                rows.append({
                    "ticker": ticker,
                    "headline": title,
                    "url": link,
                    "published_at": dt,
                    "source": domain_of(link),
                    "text": desc[:500] if desc else "",
                    "feed_source": "symbol_feed",
                    "score": sc,
                    "noise": likely_noise(title),
                })
        except Exception as ex:
            if debug:
                print(f"[{ticker}] symbol feed error: {ex}")
        time.sleep(SLEEP_BETWEEN_FEEDS)
    return rows

def select_per_ticker(candidates, debug=False):
    by_ticker = defaultdict(list)
    for row in candidates:
        by_ticker[row["ticker"]].append(row)

    selected = []
    for tkr, rows in by_ticker.items():
        seen = set()
        deduped = []
        for r in rows:
            key = (canonical_url(r["url"]), r["headline"].strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)

        strict = [r for r in deduped if r["score"] >= STRICT_THRESHOLD and not r["noise"]]
        strict.sort(key=lambda r: (r["score"], r.get("published_at", "")), reverse=True)

        if len(strict) >= MIN_PER_TICKER:
            chosen = strict[:MAX_PER_TICKER]
        else:
            relaxed_pool = [r for r in deduped if r["score"] >= RELAXED_THRESHOLD]
            relaxed_pool.sort(key=lambda r: (r["score"], r.get("published_at", "")), reverse=True)
            chosen = []
            seen2 = set()
            for r in strict + relaxed_pool:
                key2 = (canonical_url(r["url"]), r["headline"].strip().lower())
                if key2 in seen2:
                    continue
                seen2.add(key2)
                chosen.append(r)
                if len(chosen) >= MAX_PER_TICKER:
                    break

        if debug:
            print(f"{tkr}: {len(chosen)} selected (strict >= {STRICT_THRESHOLD}, relaxed >= {RELAXED_THRESHOLD})")
        selected.extend(chosen)

    return selected

# --- helper: normalize published_at to timestamptz strings for Postgres
def normalize_ts(s: str) -> str:
    """
    Convert 'YYYY-MM-DD' or empty -> ISO8601 UTC string.
    If empty/missing, default to today's date at 00:00:00Z to satisfy NOT NULL.
    """
    try:
        s = (s or "").strip()
        if not s:
            dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            return dt.isoformat()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return dt.isoformat()
        # fallback: try parsing anything else
        dt = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(dt):
            dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return dt.to_pydatetime().isoformat()
    except Exception:
        dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return dt.isoformat()

def upsert_articles(df: pd.DataFrame) -> int:
    """
    Inserts rows into articles with dedupe on (url).
    Counts inserted rows by before/after like your original behavior.
    """
    engine = get_engine()
    # pre-count
    with engine.connect() as c:
        before = c.execute(text("SELECT COUNT(*) FROM articles")).scalar()

    rows = [
        {
            "ticker": r.ticker,
            "headline": r.headline,
            "url": r.url,
            "published_at": normalize_ts(r.published_at),
            "source": r.source,
            "text": r.text,
        }
        for r in df.itertuples(index=False)
    ]

    ins_sql = text("""
        INSERT INTO articles (ticker, headline, url, published_at, source, text)
        VALUES (:ticker, :headline, :url, :published_at, :source, :text)
        ON CONFLICT (url) DO NOTHING
    """)

    # executemany (driver runs once per dict)
    with engine.begin() as c:
        c.execute(ins_sql, rows)

    with engine.connect() as c:
        after = c.execute(text("SELECT COUNT(*) FROM articles")).scalar()

    return (after - before)

def main(debug=True, save_csv=True):
    tickers = load_watchlist()
    print(f"Collecting for {len(tickers)} tickers: {', '.join(tickers)}")

    gen_candidates = fetch_general_feeds(tickers, debug=debug)

    sym_candidates = []
    for t in tickers:
        sym_candidates.extend(fetch_ticker_specific_feeds(t, debug=debug))

    candidates = gen_candidates + sym_candidates
    if not candidates:
        print("No articles collected.")
        return

    chosen = select_per_ticker(candidates, debug=debug)

    df = pd.DataFrame(chosen)[["ticker", "headline", "url", "published_at", "source", "text"]]

    if save_csv:
        df.to_csv("data/headlines_clean.csv", index=False, encoding="utf-8")
        print("Saved CSV -> data/headlines_clean.csv")

    inserted = upsert_articles(df)
    print(f"DB UPSERT: inserted {inserted} new rows into 'articles'.")

    # sample preview (Postgres)
    engine = get_engine()
    with engine.connect() as c:
        sample = list(c.execute(text(
            "SELECT id, ticker, substring(headline from 1 for 80) AS h, source, published_at "
            "FROM articles ORDER BY id DESC LIMIT 6"
        )))
    print("Newest rows:")
    for row in sample:
        print(" ", tuple(row))

if __name__ == "__main__":
    main(debug=True, save_csv=True)
