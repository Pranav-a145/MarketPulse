# api/app.py  — Postgres + SQLAlchemy version
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import text
from db.conn import get_engine  # uses DATABASE_URL env var
from scripts.retriever import Retriever

app = FastAPI(title="MarketPulse API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Models ----
class Headline(BaseModel):
    article_id: int
    ticker: str
    headline: str
    url: str
    source: Optional[str] = None
    published_at: Optional[str] = None
    sentiment_label: Optional[str] = None
    p_neg: Optional[float] = None
    p_neu: Optional[float] = None
    p_pos: Optional[float] = None

class AskRequest(BaseModel):
    query: str = Field(..., example="why is TSLA down today?")
    ticker: Optional[str] = Field(None, example="TSLA")
    days: int = Field(7, ge=1, le=30)
    topk: int = Field(8, ge=1, le=10)

class Hit(BaseModel):
    score: float
    semantic: float
    recency: float
    sent_boost: float
    age_days: float
    ticker: str
    headline: str
    url: str
    source: Optional[str]
    published_at: Optional[str]
    article_id: int

class AskResponse(BaseModel):
    answer: str
    hits: List[Hit]

class SummarizeRequest(BaseModel):
    article_id: int

class SummarizeResponse(BaseModel):
    summary: str

# ---- Helpers ----
def _lazy_answer_query(*, query: str, ticker: Optional[str], days: int, topk: int) -> str:
    # Import only when needed to avoid slow startup
    from scripts.rag_answer import answer_query
    return answer_query(query, ticker=ticker, days=days, topk=topk)

def _lazy_summarize_article(article_id: int) -> str:
    from scripts.rag_answer import summarize_article
    return summarize_article(article_id)

# ---- Routes ----
@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/headlines", response_model=List[Headline])
def get_headlines(
    ticker: str = Query(..., min_length=1),
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Recent articles for a ticker within the last N days. Joined with the *latest* sentiment row per article.
    Works whether published_at is stored as TEXT 'YYYY-MM-DD' or as DATE.
    """
    eng = get_engine()
    sql = text("""
        WITH latest AS (
          SELECT article_id, MAX(created_at) AS created_at
          FROM sentiment
          GROUP BY article_id
        )
        SELECT
          a.id AS article_id,
          a.ticker,
          a.headline,
          a.url,
          a.source,
          a.published_at,
          s.label AS sentiment_label,
          s.p_neg,
          s.p_neu,
          s.p_pos
        FROM articles a
        LEFT JOIN latest ls
          ON ls.article_id = a.id
        LEFT JOIN sentiment s
          ON s.article_id = ls.article_id
         AND s.created_at = ls.created_at
        WHERE a.ticker = :ticker
          AND COALESCE(
                CASE
                  WHEN a.published_at IS NULL OR a.published_at = '' THEN NULL
                  ELSE to_date(a.published_at, 'YYYY-MM-DD')
                END,
                -- if it's already a date, this CASE will be NULL; fallback to cast
                NULLIF(a.published_at::text, '')
              ) >= (CURRENT_DATE - (:days || ' days')::interval)
        ORDER BY a.published_at DESC, a.id DESC
        LIMIT :limit
    """)
    # Note: above WHERE handles TEXT date; if your column is DATE already, it still works.

    with eng.connect() as c:
        rows = c.execute(sql, {
            "ticker": ticker.upper(),
            "days": int(days),
            "limit": int(limit),
        }).mappings().all()

    return [
        Headline(
            article_id=r["article_id"],
            ticker=r["ticker"],
            headline=r["headline"],
            url=r["url"],
            source=r["source"],
            published_at=str(r["published_at"]) if r["published_at"] is not None else None,
            sentiment_label=r["sentiment_label"],
            p_neg=r["p_neg"],
            p_neu=r["p_neu"],
            p_pos=r["p_pos"],
        )
        for r in rows
    ]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Sentiment-aware retrieval + RAG answer.
    """
    r = Retriever()
    hits = r.search(
        req.query, ticker=req.ticker, days=req.days, topk=req.topk, candidate_limit=3000
    )

    answer = _lazy_answer_query(query=req.query, ticker=req.ticker, days=req.days, topk=req.topk)

    hits_out = [
        Hit(
            score=h["score"],
            semantic=h["semantic"],
            recency=h["recency"],
            sent_boost=h.get("sent_boost", 0.0),
            age_days=h["age_days"],
            ticker=h["ticker"],
            headline=h["headline"],
            url=h["url"],
            source=h["source"],
            published_at=h["published_at"],
            article_id=h["article_id"],
        )
        for h in hits
    ]
    return AskResponse(answer=answer, hits=hits_out)

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    """
    Summarize a single article (used by the 'Summarize' button next to a headline).
    """
    text_out = _lazy_summarize_article(req.article_id)
    if not text_out:
        raise HTTPException(status_code=404, detail="No summary available.")
    return SummarizeResponse(summary=text_out)
