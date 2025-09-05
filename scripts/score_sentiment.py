import os
from datetime import datetime, timezone
from typing import List, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sqlalchemy import text
from db.conn import get_engine

MODEL_REPO = os.getenv("SENTIMENT_MODEL_REPO", "pranava145/my_sentiment_model")

USE_ARTICLE_TEXT_IF_AVAILABLE = True
MIN_TEXT_CHARS = 120
LOOKBACK_DAYS = None          
MAX_TO_SCORE = 5000
BATCH_SIZE = 32

def connect_db():
    """Return an SQLAlchemy engine connected via DATABASE_URL."""
    return get_engine()

def fetch_unscored_articles(engine) -> List[Tuple]:
    """
    Returns rows: (id, ticker, headline, text, published_at)
    Equivalent to your SQLite query, but in Postgres.
    """
    base_sql = """
      SELECT a.id, a.ticker, a.headline, a.text, a.published_at
      FROM articles a
      LEFT JOIN sentiment s ON s.article_id = a.id
      WHERE s.article_id IS NULL
    """
    params = {}

    if LOOKBACK_DAYS is not None:
        base_sql += " AND a.published_at >= (now() - (:days || ' days')::interval) "
        params["days"] = str(int(LOOKBACK_DAYS))

    base_sql += " ORDER BY a.published_at DESC NULLS LAST, a.id DESC LIMIT :limit "
    params["limit"] = int(MAX_TO_SCORE)

    with engine.connect() as c:
        rows = c.execute(text(base_sql), params).all()
    return rows

def load_model():
    """
    Load the sentiment model from a (private) Hugging Face repo.
    Auth order:
      1) env HF_TOKEN or HUGGINGFACE_HUB_TOKEN
      2) cached login from `huggingface-cli login` (dev only)
    """
    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or None
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO, token=hf_token)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_auth_token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO, use_auth_token=hf_token)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    id2label = {int(k): v for k, v in getattr(model.config, "id2label", {}).items()}
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
    return (headline or "").strip(), "headline"

def insert_sentiment_rows(engine, rows_to_insert):
    """
    rows_to_insert: list of tuples (article_id, label, p_neg, p_neu, p_pos)
    created_at has default now() in Postgres schema.
    Preserves your original "INSERT OR IGNORE" behavior via ON CONFLICT DO NOTHING.
    """
    sql = text("""
        INSERT INTO sentiment (article_id, label, p_neg, p_neu, p_pos)
        VALUES (:article_id, :label, :p_neg, :p_neu, :p_pos)
        ON CONFLICT (article_id) DO NOTHING
    """)
    payload = [
        {
            "article_id": int(aid),
            "label": lab,
            "p_neg": float(pn),
            "p_neu": float(pu),
            "p_pos": float(pp),
        }
        for (aid, lab, pn, pu, pp) in rows_to_insert
    ]
    with engine.begin() as c:
        c.execute(sql, payload)

def main():
    print("Loading model…")
    tokenizer, model, device, idx_map = load_model()

    print("Connecting to DB…")
    engine = connect_db()

    print("Selecting unscored articles…")
    records = fetch_unscored_articles(engine)
    if not records:
        print("No unscored articles found. You're up to date.")
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
    insert_sentiment_rows(engine, rows_db)

    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = f"data/sentiment_scores_{now}.csv"
    os.makedirs("data", exist_ok=True)
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
    print("Done.")

if __name__ == "__main__":
    main()
