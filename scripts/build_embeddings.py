import os
import re
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# NEW: Postgres via SQLAlchemy
from sqlalchemy import text
from db.conn import get_engine

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

def ensure_embeddings_table(engine):
    """Creates the embeddings table if it doesn't exist."""
    sql = text("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
            chunk_id INTEGER NOT NULL,
            vector BYTEA NOT NULL,
            dim INTEGER NOT NULL,
            text_snippet TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """)
    idx1 = text("CREATE INDEX IF NOT EXISTS ix_embeddings_article ON embeddings(article_id);")
    idx2 = text("CREATE INDEX IF NOT EXISTS ix_embeddings_article_chunk ON embeddings(article_id, chunk_id);")
    with engine.begin() as c:
        c.execute(sql)
        c.execute(idx1)
        c.execute(idx2)

def select_articles_to_embed(engine, lookback_days=LOOKBACK_DAYS, limit=MAX_ARTICLES_PER_RUN) -> List[Tuple[int, str, str, str]]:
    """
    Returns rows: (article_id, headline, body, published_at).
    Skips articles that already have embeddings.
    """
    sql = """
        SELECT a.id,
               COALESCE(NULLIF(TRIM(a.headline), ''), '') AS headline,
               COALESCE(NULLIF(TRIM(a.text), ''), '')     AS body,
               a.published_at
        FROM articles a
        WHERE NOT EXISTS (
            SELECT 1 FROM embeddings e WHERE e.article_id = a.id
        )
    """
    params = {}

    if lookback_days is not None:
        sql += " AND a.published_at >= (now() - (:days || ' days')::interval) "
        params["days"] = str(int(lookback_days))

    sql += " ORDER BY a.published_at DESC, a.id DESC LIMIT :lim"
    params["lim"] = int(limit)

    with engine.connect() as c:
        rows = c.execute(text(sql), params).all()
    return rows

# sentence boundary split
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')

def chunk_text(headline: str, body: str) -> List[str]:
    """
    Split article text into chunks near MAX_CHARS, on sentence boundaries.
    Fallback to a headline-only chunk when the body is empty.
    """
    headline = (headline or "").strip()
    text = (body or "").strip()

    if not text:
        return [headline] if headline else []

    sentences = _SENT_SPLIT.split(text)
    chunks: List[str] = []
    current = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue
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

    if INCLUDE_HEADLINE and headline and chunks:
        if HEADLINE_EVERY_CHUNK:
            chunks = [f"{headline} — {c}" for c in chunks]
        else:
            chunks[0] = f"{headline} — {chunks[0]}"

    if text:
        chunks = [c for c in chunks if len(c) >= MIN_CHUNK_LEN]

    return chunks

def insert_embeddings(engine, article_id: int, vectors: np.ndarray, snippets: List[str]):
    """
    Insert each chunk embedding into the embeddings table.
    """
    sql = text("""
        INSERT INTO embeddings (article_id, chunk_id, vector, dim, text_snippet)
        VALUES (:article_id, :chunk_id, :vector, :dim, :snippet)
    """)
    payload = [
        {
            "article_id": article_id,
            "chunk_id": i,
            "vector": vec.astype(np.float32).tobytes(),
            "dim": int(vec.shape[0]),
            "snippet": snippets[i]
        }
        for i, vec in enumerate(vectors)
    ]
    with engine.begin() as c:
        c.execute(sql, payload)

def main():
    print("Connecting to DB…")
    engine = get_engine()
    ensure_embeddings_table(engine)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    to_embed = select_articles_to_embed(engine)
    if not to_embed:
        print("No articles need embeddings. You are up to date.")
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

        insert_embeddings(engine, article_id, vectors, chunks)

        total_chunks += len(chunks)
        embedded_articles += 1

        if embedded_articles % 20 == 0:
            print(f"  Embedded {embedded_articles} articles, {total_chunks} chunks so far")

    print(f"Done. Embedded {embedded_articles} articles → {total_chunks} chunks.")

if __name__ == "__main__":
    main()
