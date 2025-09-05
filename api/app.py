from typing import List, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import text
from db.conn import get_engine  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MarketPulse API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def _lazy_answer_query(*, query: str, ticker: Optional[str], days: int, topk: int) -> str:
    try:
        from scripts.rag_answer import answer_query
        return answer_query(query, ticker=ticker, days=days, topk=topk)
    except ImportError as e:
        logger.error(f"Failed to import rag_answer: {e}")
        return f"Unable to generate answer due to missing dependencies. Query was: {query}"
    except Exception as e:
        logger.error(f"Error in answer_query: {e}")
        return f"An error occurred while generating the answer: {str(e)}"

def _lazy_summarize_article(article_id: int) -> str:
    try:
        from scripts.rag_answer import summarize_article
        return summarize_article(article_id)
    except ImportError as e:
        logger.error(f"Failed to import summarize_article: {e}")
        return "Unable to summarize article due to missing dependencies."
    except Exception as e:
        logger.error(f"Error in summarize_article: {e}")
        return f"An error occurred while summarizing: {str(e)}"

@app.get("/")
def root():
    return {"message": "MarketPulse API is running", "docs": "/docs"}

@app.get("/health")
def health():
    try:
        eng = get_engine()
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        return {"ok": True, "time": datetime.utcnow().isoformat() + "Z", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"ok": False, "time": datetime.utcnow().isoformat() + "Z", "database": "error", "error": str(e)}

@app.get("/headlines", response_model=List[Headline])
def get_headlines(
    ticker: str = Query(..., min_length=1),
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Recent articles for a ticker within the last N days. Joined with the *latest* sentiment row per article.
    Fixed to handle empty strings and NULL values in published_at properly.
    """
    try:
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
              AND a.published_at IS NOT NULL 
              AND TRIM(a.published_at::text) != ''
              AND (
                CASE 
                  WHEN a.published_at::text ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' THEN
                    to_date(a.published_at::text, 'YYYY-MM-DD') >= (CURRENT_DATE - (:days || ' days')::interval)
                  ELSE
                    a.published_at::date >= (CURRENT_DATE - (:days || ' days')::interval)
                END
              )
            ORDER BY 
              CASE 
                WHEN a.published_at::text ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' THEN
                  to_date(a.published_at::text, 'YYYY-MM-DD')
                ELSE
                  a.published_at::date
              END DESC, 
              a.id DESC
            LIMIT :limit
        """)

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
    
    except Exception as e:
        logger.error(f"Error in get_headlines: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Sentiment-aware retrieval + RAG answer.
    """
    try:
        try:
            from scripts.retriever import Retriever
            r = Retriever()
            hits = r.search(
                req.query, ticker=req.ticker, days=req.days, topk=req.topk, candidate_limit=3000
            )
        except ImportError as e:
            logger.error(f"Failed to import Retriever: {e}")
            hits = []
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            hits = []

        answer = _lazy_answer_query(query=req.query, ticker=req.ticker, days=req.days, topk=req.topk)

        hits_out = [
            Hit(
                score=h.get("score", 0.0),
                semantic=h.get("semantic", 0.0),
                recency=h.get("recency", 0.0),
                sent_boost=h.get("sent_boost", 0.0),
                age_days=h.get("age_days", 0.0),
                ticker=h.get("ticker", ""),
                headline=h.get("headline", ""),
                url=h.get("url", ""),
                source=h.get("source"),
                published_at=h.get("published_at"),
                article_id=h.get("article_id", 0),
            )
            for h in hits
        ]
        return AskResponse(answer=answer, hits=hits_out)
    
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        return AskResponse(
            answer=f"I apologize, but I encountered an error while processing your question: {req.query}. Please try again later.",
            hits=[]
        )

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    """
    Summarize a single article (used by the 'Summarize' button next to a headline).
    """
    try:
        text_out = _lazy_summarize_article(req.article_id)
        if not text_out or text_out.strip() == "":
            raise HTTPException(status_code=404, detail="No summary available for this article.")
        return SummarizeResponse(summary=text_out)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}

@app.get("/test")
def test():
    return {"message": "API is working", "timestamp": datetime.utcnow().isoformat()}
