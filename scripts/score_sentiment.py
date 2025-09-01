

import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DB_PATH = "marketpulse.db"
MODEL_DIR = "models/my_sentiment_model"  
USE_ARTICLE_TEXT_IF_AVAILABLE = True      
MIN_TEXT_CHARS = 120                     
LOOKBACK_DAYS = None                      
MAX_TO_SCORE = 5000                       

def connect_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def fetch_unscored_articles(conn) -> List[Tuple]:
    where = []
    params = []
    if LOOKBACK_DAYS is not None:
        where.append("datetime(published_at) >= datetime('now', ?)")
        params.append(f"-{LOOKBACK_DAYS} day")
    where_sql = "WHERE " + " AND ".join(where) if where else ""
    sql = f"""
      SELECT a.id, a.ticker, a.headline, a.text, a.published_at
      FROM articles a
      LEFT JOIN sentiment s ON s.article_id = a.id
      {where_sql} AND s.article_id IS NULL
      ORDER BY a.published_at DESC, a.id DESC
      LIMIT ?
    """ if where_sql else """
      SELECT a.id, a.ticker, a.headline, a.text, a.published_at
      FROM articles a
      LEFT JOIN sentiment s ON s.article_id = a.id
      WHERE s.article_id IS NULL
      ORDER BY a.published_at DESC, a.id DESC
      LIMIT ?
    """
    params.append(MAX_TO_SCORE)
    return conn.execute(sql, params).fetchall()

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    id2label = {int(k): v for k, v in model.config.id2label.items()} if hasattr(model.config, "id2label") else {}
    idx_neg = idx_neu = idx_pos = None
    for i, lbl in id2label.items():
        l = str(lbl).lower()
        if "neg" in l: idx_neg = i
        elif "neu" in l: idx_neu = i
        elif "pos" in l: idx_pos = i
    if idx_neg is None or idx_neu is None or idx_pos is None:
        idx_neg, idx_neu, idx_pos = 0, 1, 2

    return tokenizer, model, device, (idx_neg, idx_neu, idx_pos)

@torch.inference_mode()
def predict_batch(texts: List[str], tokenizer, model, device, idx_map):
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
    ).to(device)
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    idx_neg, idx_neu, idx_pos = idx_map
    labels_idx = probs.argmax(axis=1)
    id2label = getattr(model.config, "id2label", None)
    if id2label:
        label_names = [str(id2label[int(i)]).lower() for i in labels_idx]
    else:
        tmp = {0: "negative", 1: "neutral", 2: "positive"}
        label_names = [tmp.get(int(i), str(int(i))) for i in labels_idx]
    p_triplets = [(float(p[idx_neg]), float(p[idx_neu]), float(p[idx_pos])) for p in probs]
    return label_names, p_triplets

def choose_text(headline: str, body: str) -> Tuple[str, str]:
    """
    Returns (text_to_score, source_used) where source_used ∈ {'text','headline'}
    """
    if USE_ARTICLE_TEXT_IF_AVAILABLE and body and len(body.strip()) >= MIN_TEXT_CHARS:
        return body.strip(), "text"
    return headline.strip(), "headline"

def insert_sentiment_rows(conn, rows_to_insert):
    """
    rows_to_insert: list of tuples (article_id, label, p_neg, p_neu, p_pos)
    created_at has default in schema; we don't set it explicitly.
    """
    conn.execute("BEGIN IMMEDIATE;")
    try:
        conn.executemany(
            """
            INSERT OR IGNORE INTO sentiment (article_id, label, p_neg, p_neu, p_pos)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

def main():
    print("Loading model…")
    tokenizer, model, device, idx_map = load_model()

    print("Connecting to DB…")
    conn = connect_db()

    print("Selecting unscored articles…")
    records = fetch_unscored_articles(conn)
    if not records:
        print("No unscored articles found. You're up to date.")
        conn.close()
        return

    print(f"Found {len(records)} to score.")
    article_ids, tickers, headlines, bodies, pubs = zip(*records)

    inputs = []
    used_src = []
    for h, b in zip(headlines, bodies):
        txt, src = choose_text(h or "", b or "")
        inputs.append(txt if txt else (h or ""))
        used_src.append(src)

    labels, prob_triplets = [], []
    for i in range(0, len(inputs), BATCH_SIZE):
        batch_texts = inputs[i : i + BATCH_SIZE]
        l, p = predict_batch(batch_texts, tokenizer, model, device, idx_map)
        labels.extend(l)
        prob_triplets.extend(p)

    rows_db = []
    for aid, lab, probs in zip(article_ids, labels, prob_triplets):
        p_neg, p_neu, p_pos = probs
        rows_db.append((int(aid), lab, p_neg, p_neu, p_pos))

    print("Inserting into sentiment table…")
    insert_sentiment_rows(conn, rows_db)

    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = f"data/sentiment_scores_{now}.csv"
    df_out = pd.DataFrame({
        "article_id": article_ids,
        "ticker": tickers,
        "headline": headlines,
        "used": used_src,                 
        "label": labels,
        "p_neg": [p[0] for p in prob_triplets],
        "p_neu": [p[1] for p in prob_triplets],
        "p_pos": [p[2] for p in prob_triplets],
        "published_at": pubs,
    })
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved CSV → {out_path}")

    summary = (
        df_out["label"]
        .value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
        .to_string(index=False)
    )
    print("\nSummary (newly scored):")
    print(summary)

    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()
