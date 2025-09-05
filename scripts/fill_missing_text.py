import time, re
from pathlib import Path
from datetime import datetime, timedelta
import trafilatura

from sqlalchemy import text
from db.conn import get_engine

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
        if not html:
            return ""
        txt = trafilatura.extract(
            html,
            include_comments=False,
            include_links=False,
            favor_recall=True
        )
        return clean_text(txt)
    except Exception:
        return ""

def select_targets(engine, lookback_days, limit):
    """
    Postgres version of:
      WHERE (text IS NULL OR length(trim(text)) = 0)
        AND datetime(published_at) >= datetime('now', ?)
    """
    sql = text("""
        SELECT id, url
        FROM articles
        WHERE (text IS NULL OR length(btrim(text)) = 0)
          AND published_at >= (now() - (:days || ' days')::interval)
        ORDER BY published_at DESC
        LIMIT :lim
    """)
    with engine.connect() as c:
        rows = c.execute(sql, {"days": str(int(lookback_days)), "lim": int(limit)}).all()
    return rows

def update_text(engine, article_id, txt):
    sql = text("UPDATE articles SET text = :t WHERE id = :id")
    with engine.begin() as c:
        c.execute(sql, {"t": txt, "id": int(article_id)})

def main(lookback_days=LOOKBACK_DAYS_DEFAULT, limit=BATCH_LIMIT_DEFAULT):
    engine = get_engine()

    targets = select_targets(engine, lookback_days, limit)
    print(f"Found {len(targets)} articles needing text (lookback={lookback_days}d, limit={limit})")

    filled = short = fail = 0
    processed = 0

    for aid, url in targets:
        txt = fetch_text(url)
        if not txt:
            fail += 1
        elif len(txt) < ALLOW_MINSIZE:
            short += 1
        else:
            trimmed = txt[:1500]  
            update_text(engine, aid, trimmed)
            filled += 1

        processed += 1
        if processed % 10 == 0:
           
            pass

        time.sleep(SLEEP_BETWEEN)

    print(f"filled={filled} | short(<{ALLOW_MINSIZE})={short} | fail={fail}")

if __name__ == "__main__":
    main()
