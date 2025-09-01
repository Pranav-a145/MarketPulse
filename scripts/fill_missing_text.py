import time, re, sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import trafilatura

DB_PATH = "marketpulse.db"

ALLOW_MINSIZE = 400          
LOOKBACK_DAYS_DEFAULT = 14    
BATCH_LIMIT_DEFAULT = 1000   
SLEEP_BETWEEN = 0.25         

def clean_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def fetch_text(url: str) -> str:
    try:
        html = trafilatura.fetch_url(url, no_ssl=True)
        if not html: return ""
        txt = trafilatura.extract(html, include_comments=False, include_links=False, favor_recall=True)
        return clean_text(txt)
    except Exception:
        return ""

def select_targets(conn, lookback_days, limit):
    return conn.execute(
        """
        SELECT id, url FROM articles
        WHERE (text IS NULL OR length(trim(text)) = 0)
          AND datetime(published_at) >= datetime('now', ?)
        ORDER BY published_at DESC
        LIMIT ?
        """,
        (f"-{lookback_days} day", limit),
    ).fetchall()

def update_text(conn, article_id, text):
    conn.execute("UPDATE articles SET text = ? WHERE id = ?", (text, article_id))

def main(lookback_days=LOOKBACK_DAYS_DEFAULT, limit=BATCH_LIMIT_DEFAULT):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        targets = select_targets(conn, lookback_days, limit)
        print(f"Found {len(targets)} articles needing text (lookback={lookback_days}d, limit={limit})")

        filled = short = fail = 0
        for aid, url in targets:
            txt = fetch_text(url)
            if not txt:
                fail += 1
            elif len(txt) < ALLOW_MINSIZE:
                short += 1
            else:
                # keep only first 1–3 paragraphs ~ up to ~1500 chars
                trimmed = txt[:1500]
                update_text(conn, aid, trimmed)
                filled += 1
            if (filled + short + fail) % 10 == 0:
                conn.commit()
            time.sleep(SLEEP_BETWEEN)
        conn.commit()

    print(f"filled={filled} | short(<{ALLOW_MINSIZE})={short} | fail={fail}")

if __name__ == "__main__":
    main()
