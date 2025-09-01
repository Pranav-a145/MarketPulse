

import os
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "marketpulse.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Tuning knobs
LOOKBACK_DAYS = None            
MAX_ARTICLES_PER_RUN = 1000    

# Chunking
MAX_CHARS = 600               
OVERLAP = 100                  
INCLUDE_HEADLINE = True        
HEADLINE_EVERY_CHUNK = False  

MIN_CHUNK_LEN = 40


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def ensure_embeddings_table(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            chunk_id INTEGER NOT NULL,
            vector BLOB NOT NULL,
            dim INTEGER NOT NULL,
            text_snippet TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE CASCADE
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_embeddings_article ON embeddings(article_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_embeddings_article_chunk ON embeddings(article_id, chunk_id);")


def select_articles_to_embed(
    conn: sqlite3.Connection,
    lookback_days: int | None = LOOKBACK_DAYS,
    limit: int = MAX_ARTICLES_PER_RUN
) -> List[Tuple[int, str, str, str]]:
    """
    Returns rows: (article_id, headline, body, published_at).
    Headline-only articles are allowed. We skip articles that already have any embeddings.
    """
    clauses = ["NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.article_id = a.id)"]
    params: list = []

    if lookback_days is not None:
        clauses.append("datetime(a.published_at) >= datetime('now', ?)")
        params.append(f"-{lookback_days} day")

    where_sql = " AND ".join(clauses)

    sql = f"""
        SELECT a.id,
               COALESCE(NULLIF(TRIM(a.headline), ''), '') AS headline,
               COALESCE(NULLIF(TRIM(a.text), ''), '')     AS body,
               a.published_at
        FROM articles a
        WHERE {where_sql}
        ORDER BY a.published_at DESC, a.id DESC
        LIMIT ?
    """
    params.append(limit)
    return conn.execute(sql, params).fetchall()


# sentence boundary split
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')


def chunk_text(headline: str, body: str) -> List[str]:
    """
    Split article text into chunks near MAX_CHARS, on sentence boundaries.
    Fallback to a headline-only chunk when the body is empty.
    Optionally prepend headline to the first or all chunks.
    """
    headline = (headline or "").strip()
    text = (body or "").strip()

    # Headline-only fallback when no body text exists
    if not text:
        return [headline] if headline else []

    sentences = _SENT_SPLIT.split(text)
    chunks: List[str] = []
    current = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # flush to a new chunk if adding s would exceed MAX_CHARS
        if current and (len(current) + 1 + len(s) > MAX_CHARS):
            chunks.append(current)
            if OVERLAP > 0 and len(current) > OVERLAP:
                current = current[-OVERLAP:] + " " + s
            else:
                current = s
        else:
            current = (current + " " + s).strip()

    if current:
        chunks.append(current)

    # Prepend headline
    if INCLUDE_HEADLINE and headline and chunks:
        if HEADLINE_EVERY_CHUNK:
            chunks = [f"{headline} — {c}" for c in chunks]
        else:
            chunks[0] = f"{headline} — {chunks[0]}"

    if text:
        chunks = [c for c in chunks if len(c) >= MIN_CHUNK_LEN]

    return chunks


def insert_embeddings(conn: sqlite3.Connection, article_id: int, vectors: np.ndarray, snippets: List[str]):
    """
    Insert each chunk embedding as a row in embeddings table.
    """
    conn.execute("BEGIN IMMEDIATE;")
    try:
        rows = [
            (article_id, i, vec.astype(np.float32).tobytes(), int(vec.shape[0]), snippets[i])
            for i, vec in enumerate(vectors)
        ]
        conn.executemany(
            """
            INSERT INTO embeddings (article_id, chunk_id, vector, dim, text_snippet)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def main():
    print("Connecting to DB...")
    conn = connect_db()
    ensure_embeddings_table(conn)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    to_embed = select_articles_to_embed(conn)
    if not to_embed:
        print("No articles need embeddings. You are up to date.")
        conn.close()
        return

    print(f"Found {len(to_embed)} articles to embed (lookback={LOOKBACK_DAYS}).")
    total_chunks = 0
    embedded_articles = 0

    for (article_id, headline, body, published_at) in to_embed:
        chunks = chunk_text(headline, body)
        if not chunks:
            continue

        vectors = model.encode(
            chunks,
            normalize_embeddings=True,        
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        insert_embeddings(conn, article_id, vectors, chunks)

        total_chunks += len(chunks)
        embedded_articles += 1

        if embedded_articles % 20 == 0:
            print(f"  Embedded {embedded_articles} articles, {total_chunks} chunks so far")

    conn.close()
    print(f"Done. Embedded {embedded_articles} articles → {total_chunks} chunks.")


if __name__ == "__main__":
    main()
