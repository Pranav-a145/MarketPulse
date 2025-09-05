from datetime import datetime, timezone
from math import exp
from typing import List, Optional, Dict, Any
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from sqlalchemy import text
from db.conn import get_engine

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

W_SEMANTIC = 0.80
W_RECENCY  = 0.20
RECENCY_HALF_LIFE_DAYS = 7.0
SEMANTIC_MIN = 0.25

CONF_BAND = 0.60
NEG_WEIGHT = 0.20
POS_WEIGHT = 0.20

DEFAULT_CANDIDATES = 3000
DEFAULT_TOPK = 5

NEG_Q = re.compile(r"\b(down|drop|fell|falling|slump|red|sell[- ]?off|plunge|decline|lower|bear|why.*down)\b", re.I)
POS_Q = re.compile(r"\b(up|rally|rose|jump|green|surge|soar|gain|higher|bull|why.*up)\b", re.I)

NEG_ARTICLE_HINTS = re.compile(
    r"\b(downgrade|price target cut|lawsuit|recall|miss(ed)?|guidance cut|probe|investigation|FTC|DOJ|SEC|slump|falls|declines|weak|warning)\b",
    re.I,
)
POS_ARTICLE_HINTS = re.compile(
    r"\b(upgrade|raised guidance|beats|beat estimates|record high|strong|outperform|buy rating|surge|soar|wins|contract|partnership)\b",
    re.I,
)

def _to_vec(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(bytes(blob), dtype=np.float32, count=dim)

def _age_days(published_at) -> float:
    """
    Accepts either a datetime (from Postgres) or a 'YYYY-MM-DD' string.
    """
    if published_at is None:
        return 365.0
    try:
        if isinstance(published_at, datetime):
            dt = published_at.astimezone(timezone.utc)
        else:
            dt = datetime.strptime(str(published_at), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return 365.0
    now = datetime.now(timezone.utc)
    return max(0.0, (now - dt).total_seconds() / 86400.0)

def _recency_score(days: float) -> float:
    return exp(-days / RECENCY_HALF_LIFE_DAYS)

def _intent_from_query(q: str) -> str:
    if NEG_Q.search(q):
        return "neg"
    if POS_Q.search(q):
        return "pos"
    return "neutral"

def _safe_date_to_string(date_obj) -> str:
    """Convert datetime object to string safely"""
    if date_obj is None:
        return ""
    if isinstance(date_obj, datetime):
        return date_obj.strftime("%Y-%m-%d")
    return str(date_obj)

class Retriever:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.engine = get_engine()

    def embed_query(self, text: str) -> np.ndarray:
        v = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return v[0].astype(np.float32)

    def _fetch_candidates(
        self,
        ticker: Optional[str],
        days: Optional[int],
        limit: int,
        intent: str
    ):
        """
        Fetch candidate chunks joined with sentiment (if available).
        Includes headline-only embeddings (no filter on a.text).
        """
        where = ["1=1"]
        params: Dict[str, Any] = {}

        if ticker:
            where.append("a.ticker = :ticker")
            params["ticker"] = ticker.upper()

        if days is not None and days > 0:
            where.append("a.published_at >= (now() - (:days || ' days')::interval)")
            params["days"] = str(int(days))

        label_filter = ""
        if intent == "neg":
            label_filter = " AND (s.label IN ('negative','neutral') OR s.label IS NULL)"
        elif intent == "pos":
            label_filter = " AND (s.label IN ('positive','neutral') OR s.label IS NULL)"

        sql = f"""
        SELECT
            e.id AS emb_id, e.article_id, e.chunk_id, e.vector, e.dim, e.text_snippet,
            a.ticker, a.headline, a.url, a.source, a.published_at,
            COALESCE(s.label, 'neutral') AS s_label,
            COALESCE(s.p_neg, 0.0) AS p_neg,
            COALESCE(s.p_neu, 0.0) AS p_neu,
            COALESCE(s.p_pos, 0.0) AS p_pos
        FROM embeddings e
        JOIN articles a ON a.id = e.article_id
        LEFT JOIN sentiment s ON s.article_id = a.id
        WHERE {" AND ".join(where)}
        {label_filter}
        ORDER BY a.published_at DESC, e.article_id DESC, e.chunk_id ASC
        LIMIT :limit
        """
        params["limit"] = int(limit)

        with self.engine.connect() as c:
            rows = c.execute(text(sql), params).mappings().all()
        return rows

    def search(
        self,
        query: str,
        *,
        ticker: Optional[str] = None,
        days: int = 7,
        candidate_limit: int = DEFAULT_CANDIDATES,
        topk: int = DEFAULT_TOPK
    ) -> List[Dict[str, Any]]:
        """
        Returns top-k *articles* with fields:
        {score, semantic, recency, sent_boost, age_days, snippet, ticker, headline, url, source, published_at, article_id}
        """
        intent = _intent_from_query(query)
        qvec = self.embed_query(query)

        rows = self._fetch_candidates(ticker, days, candidate_limit, intent)
        if not rows:
            return []

        dims = [r["dim"] for r in rows]
        dim0 = dims[0]
        if not all(d == dim0 for d in dims):
            rows = [r for r in rows if r["dim"] == dim0]
            if not rows:
                return []

        mat = np.stack([_to_vec(r["vector"], dim0) for r in rows])  # (N, D)
        sims = mat.dot(qvec)  # (N,)

        per_article: Dict[int, Dict[str, Any]] = {}

        for r, sem in zip(rows, sims):
            sem = float(sem)
            if sem < SEMANTIC_MIN:
                continue

            age = _age_days(r["published_at"])
            rec = _recency_score(age)

            p_neg, p_neu, p_pos = float(r["p_neg"]), float(r["p_neu"]), float(r["p_pos"])
            confident = max(p_neg, p_neu, p_pos) >= CONF_BAND
            sent_boost = 0.0
            if confident:
                if intent == "neg":
                    sent_boost = NEG_WEIGHT * p_neg
                elif intent == "pos":
                    sent_boost = POS_WEIGHT * p_pos

            text_for_hints = f"{r['headline'] or ''} {r['text_snippet'] or ''}"
            if intent == "neg" and NEG_ARTICLE_HINTS.search(text_for_hints):
                sent_boost += 0.03
            if intent == "pos" and POS_ARTICLE_HINTS.search(text_for_hints):
                sent_boost += 0.03

            base = W_SEMANTIC * sem + W_RECENCY * rec
            final = base + sent_boost

            aid = int(r["article_id"])
            prev = per_article.get(aid)
            if (prev is None) or (final > prev["score"]):
                per_article[aid] = {
                    "score": float(final),
                    "semantic": float(sem),
                    "recency": float(rec),
                    "sent_boost": float(sent_boost),
                    "age_days": float(age),
                    "snippet": r["text_snippet"],
                    "ticker": r["ticker"],
                    "headline": r["headline"],
                    "url": r["url"],
                    "source": r["source"],
                    "published_at": _safe_date_to_string(r["published_at"]),  # FIXED: Convert datetime to string
                    "article_id": aid,
                }

        results = sorted(per_article.values(), key=lambda x: x["score"], reverse=True)
        return results[:topk]

if __name__ == "__main__":
    import argparse, textwrap
    ap = argparse.ArgumentParser(description="Sentiment-aware retriever")
    ap.add_argument("query", type=str, help="e.g., 'why is TSLA down today?'")
    ap.add_argument("--ticker", type=str, default=None)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    args = ap.parse_args()

    r = Retriever(MODEL_NAME)
    hits = r.search(
        args.query,
        ticker=args.ticker,
        days=args.days,
        candidate_limit=args.candidates,
        topk=args.topk
    )
    if not hits:
        print("No results.")
    else:
        for i, h in enumerate(hits, 1):
            print(f"\n[{i}] score={h['score']:.3f} (sem={h['semantic']:.3f}, rec={h['recency']:.3f}, boost={h['sent_boost']:.3f}, age={h['age_days']:.1f}d)")
            print(f"    {h['ticker']} | {h['published_at']} | {h['source']}")
            print(f"    {h['headline']}")
            print("    Snippet:", textwrap.shorten(h['snippet'] or '', width=200, placeholder='…'))
            print(f"    URL: {h['url']}")
