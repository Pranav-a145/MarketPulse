from typing import List, Dict, Any, Optional, Tuple
import re
import os
import json
import textwrap
import hashlib
import requests
from urllib.parse import urlparse, urlunparse

# DB: Postgres via SQLAlchemy
from sqlalchemy import text
from db.conn import get_engine

# Retriever (already Postgres-compatible)
try:
    from scripts.retriever import Retriever
except ImportError:
    from retriever import Retriever

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in env or Streamlit secrets")

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_URL = "https://api.openai.com/v1/responses"
API_TIMEOUT = 30
MAX_OUTPUT_TOKENS_BULLETS = 500  # unchanged

MAX_SOURCES = 5
CHUNKS_PER_ARTICLE = 3
TARGET_WORDS = (180, 220)

SUM_CHUNKS_PER_ARTICLE = 8
SUMMARY_PROMPT_VERSION = "v1"
# NOTE: your original file references MAX_OUTPUT_TOKENS_SUMMARY later.
# I am not defining it here because you asked for DB-only changes.

# ---------- cache tables (Postgres) ----------
def _ensure_cache_tables(engine):
    with engine.begin() as c:
        c.execute(text("""
            CREATE TABLE IF NOT EXISTS answers_cache (
                key_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                ticker TEXT,
                article_ids TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """))
        c.execute(text("CREATE INDEX IF NOT EXISTS ix_answers_cache_created ON answers_cache(created_at)"))
        c.execute(text("""
            CREATE TABLE IF NOT EXISTS summaries_cache (
                key_hash TEXT PRIMARY KEY,
                article_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """))
        c.execute(text("CREATE INDEX IF NOT EXISTS ix_summaries_cache_created ON summaries_cache(created_at)"))

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _canonical_url(u: str) -> str:
    try:
        p = urlparse(u or "")
        return urlunparse((p.scheme, p.netloc.lower(), p.path, "", "", ""))
    except Exception:
        return u or ""

def _word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))

def _get_joined_snippet(engine, article_id: int, limit: int) -> str:
    sql = text("""
        SELECT text_snippet FROM embeddings
        WHERE article_id = :aid
        ORDER BY chunk_id ASC
        LIMIT :lim
    """)
    parts = []
    with engine.connect() as c:
        rows = c.execute(sql, {"aid": int(article_id), "lim": int(limit)}).all()
    for r in rows:
        sn = (r[0] or "").replace("\n", " ").strip()
        if sn:
            parts.append(sn)
    return " ".join(parts).strip()

def _make_answer_cache_key(query: str, ticker: Optional[str], article_ids: List[int], evidences: List[str]) -> Tuple[str, str, str]:
    qn = re.sub(r"\s+", " ", (query or "").strip().lower())
    tk = (ticker or "").strip().upper()
    ids_sorted = sorted([int(x) for x in article_ids])
    ev_digest = _sha256("|||".join([re.sub(r"\s+", " ", e.strip()) for e in evidences]))
    material = json.dumps({"q": qn, "t": tk, "ids": ids_sorted, "ev": ev_digest}, sort_keys=True)
    key = _sha256(material)
    return key, ",".join(map(str, ids_sorted)), ev_digest

def _answer_cache_get(engine, key_hash: str) -> Optional[str]:
    with engine.connect() as c:
        row = c.execute(text("SELECT answer FROM answers_cache WHERE key_hash = :k"), {"k": key_hash}).fetchone()
    return row[0] if row else None

def _answer_cache_put(engine, key_hash: str, query: str, ticker: Optional[str],
                      article_ids_csv: str, evidence_hash: str, answer: str):
    sql = text("""
        INSERT INTO answers_cache (key_hash, query, ticker, article_ids, evidence_hash, answer)
        VALUES (:k, :q, :t, :ids, :ev, :ans)
        ON CONFLICT (key_hash) DO UPDATE SET
          query = EXCLUDED.query,
          ticker = EXCLUDED.ticker,
          article_ids = EXCLUDED.article_ids,
          evidence_hash = EXCLUDED.evidence_hash,
          answer = EXCLUDED.answer
    """)
    with get_engine().begin() as c:
        c.execute(sql, {"k": key_hash, "q": query, "t": (ticker or ""), "ids": article_ids_csv, "ev": evidence_hash, "ans": answer})

def _make_summary_cache_key(article_id: int, model: str, prompt_version: str, headline: str, content: str) -> Tuple[str, str]:
    content_norm = re.sub(r"\s+", " ", f"{headline or ''} || {content or ''}".strip())
    content_hash = _sha256(content_norm)
    material = json.dumps(
        {"aid": int(article_id), "model": model, "pv": prompt_version, "ch": content_hash},
        sort_keys=True,
    )
    key = _sha256(material)
    return key, content_hash

def _summary_cache_get(engine, key_hash: str) -> Optional[str]:
    with engine.connect() as c:
        row = c.execute(text("SELECT summary FROM summaries_cache WHERE key_hash = :k"), {"k": key_hash}).fetchone()
    return row[0] if row else None

def _summary_cache_put(engine, key_hash: str, article_id: int, model: str,
                       prompt_version: str, content_hash: str, summary: str):
    sql = text("""
        INSERT INTO summaries_cache (key_hash, article_id, model, prompt_version, content_hash, summary)
        VALUES (:k, :aid, :m, :pv, :ch, :s)
        ON CONFLICT (key_hash) DO UPDATE SET
          article_id = EXCLUDED.article_id,
          model = EXCLUDED.model,
          prompt_version = EXCLUDED.prompt_version,
          content_hash = EXCLUDED.content_hash,
          summary = EXCLUDED.summary
    """)
    with get_engine().begin() as c:
        c.execute(sql, {"k": key_hash, "aid": int(article_id), "m": model, "pv": prompt_version, "ch": content_hash, "s": summary})

def _format_sources_for_prompt(items: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, h in enumerate(items, 1):
        u = _canonical_url(h.get("url") or "")
        date = (h.get("published_at") or "").strip() or "n/a"
        src = (h.get("source") or "").strip() or urlparse(u).netloc
        head = (h.get("headline") or "").strip()
        ev = (h.get("_evidence") or "").strip()
        blocks.append(f"[{i}] {src} - {date}\nHeadline: {head}\nEvidence: {ev}")
    return "\n\n".join(blocks)

def _build_public_sources_list(items: List[Dict[str, Any]]) -> str:
    lines = []
    for i, h in enumerate(items, 1):
        u = _canonical_url(h.get("url") or "")
        date = (h.get("published_at") or "").strip() or "n/a"
        head = (h.get("headline") or "").strip()
        lines.append(f"[{i}] {head} - {date} - {u}")
    return "\n".join(lines)

def _build_bullets_prompt(query: str, ticker: Optional[str], items: List[Dict[str, Any]]) -> str:
    lo, hi = TARGET_WORDS
    src_block = _format_sources_for_prompt(items)
    tk = (ticker or "N/A").upper()
    return (
        f"Query: {query}\n"
        f"Ticker: {tk}\n\n"
        "Sources (use only these and cite with bracketed numbers that map to each source):\n"
        f"{src_block}\n\n"
        "Write a few short bullets totaling about "
        f"{(lo+hi)//2} words (acceptable range {lo}-{hi} words) that answer the Query using only the facts in Sources. "
        "Each sentence must end with bracket citations like [1] or [2-3] that fully support that sentence. "
        "Do NOT introduce any numbers, percentages, dates, quarters (e.g., Q1/Q2), or price targets unless they appear verbatim in Sources. "
        "If a figure is not present in Sources, stay qualitative. "
        "Return only the bullets, no preamble and no sources list."
        "Keep 3–5 bullets max. Avoid repeating the same source number in multiple bullets unless it adds a distinct fact."
    )

def _build_summary_prompt(headline: str, content: str) -> str:
    return (
        "Summarize the following article content for investors in 2 to 4 sentences. "
        "Use only the content provided. Do not add entities or numbers not present.\n\n"
        f"Headline: {headline}\n\nContent:\n{content}"
    )

_RX_DOLLAR  = re.compile(r'\$\s?\d[\d,]*(?:\.\d+)?')
_RX_PERCENT = re.compile(r'\d+(?:\.\d+)?\s?%')
_RX_BIGINT  = re.compile(r'\b\d{1,3}(?:,\d{3})+\b')
_RX_YEAR    = re.compile(r'\b20\d{2}\b')
_RX_QUARTER = re.compile(r'\bQ[1-4]\b', re.I)

def _normalize_num_token(tok: str) -> str:
    t = tok.strip()
    t = t.replace(" ", "")
    if t.startswith("$"):
        t = "$" + t[1:].replace(",", "")
    else:
        t = t.replace(",", "")
    return t

def _supported_in_evidence(token: str, evidence_text_norm: str) -> bool:
    t = _normalize_num_token(token)
    if not t:
        return True
    return (t in evidence_text_norm)

def _remove_unsupported_numbers(text: str, evidence_text: str) -> str:
    ev_norm = evidence_text.replace(" ", "").replace(",", "")
    def strip_tokens(rx: re.Pattern, s: str) -> str:
        out = []
        last = 0
        for m in rx.finditer(s):
            tok = m.group(0)
            if _supported_in_evidence(tok, ev_norm):
                continue
            out.append(s[last:m.start()])
            last = m.end()
        out.append(s[last:])
        return "".join(out)
    out = text
    out = strip_tokens(_RX_DOLLAR, out)
    out = strip_tokens(_RX_PERCENT, out)
    out = strip_tokens(_RX_BIGINT, out)
    out = strip_tokens(_RX_YEAR, out)
    def strip_quarters(s: str) -> str:
        return re.sub(_RX_QUARTER, lambda m: "" if not _supported_in_evidence(m.group(0), ev_norm) else m.group(0), s)
    out = strip_quarters(out)
    out = re.sub(r'\s{2,}', ' ', out).strip()
    out = re.sub(r'\s+(\[\d+(?:-\d+)?\])', r' \1', out)
    return out

def _extract_bullet_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets = [ln if ln.startswith("-") else f"- {ln}" for ln in lines if ln.startswith("-") or len(lines) <= 4]
    return bullets or [f"- {text.strip()}"]

def _enforce_numbers_from_evidence(bullets_text: str, evidence_text: str) -> str:
    bullets = _extract_bullet_lines(bullets_text)
    clean = []
    for b in bullets:
        cleaned = _remove_unsupported_numbers(b, evidence_text)
        core = re.sub(r'\[[\d\-]+\]', '', cleaned).strip(" -–—:;,.")
        if core:
            clean.append(cleaned)
    return "\n".join(clean) if clean else bullets_text

def _openai_respond(prompt: str, *, max_output_tokens: int) -> str:
    headers = {
        # NOTE: use your OPENAI_API_KEY env var
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ],
        "text": {"format": {"type": "text"}},
        "max_output_tokens": max_output_tokens,
    }
    r = requests.post(OPENAI_URL, headers=headers, json=body, timeout=API_TIMEOUT)
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text}
        raise RuntimeError(f"OpenAI {r.status_code}: {err}")
    data = r.json()

    if os.environ.get("RAG_DEBUG") == "1":
        try:
            print("[DEBUG] raw OpenAI JSON:", json.dumps(data, indent=2)[:1500])
        except Exception:
            pass

    txt = (data.get("output_text") or "").strip()
    if txt:
        return txt

    def _collect_from_output(d):
        texts = []
        items = d.get("output") or d.get("response") or []
        if isinstance(items, dict):
            items = [items]
        for it in items:
            if not isinstance(it, dict):
                continue
            if it.get("type") == "message":
                for part in (it.get("content") or []):
                    if isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str) and t.strip():
                            texts.append(t.strip())
        return texts

    parts = _collect_from_output(data)
    if parts:
        return "\n".join(parts).strip()

    def _collect_any_text(x):
        out = []
        if isinstance(x, dict):
            if x.get("type") in ("output_text", "text", "message", "summary_text"):
                t = x.get("text")
                if isinstance(t, str) and t.strip():
                    out.append(t)
            for k in ("output", "response", "content", "choices", "message", "messages", "items", "reasoning"):
                if k in x:
                    out.extend(_collect_any_text(x[k]))
        elif isinstance(x, list):
            for i in x:
                out.extend(_collect_any_text(i))
        return out

    parts = [p for p in _collect_any_text(data) if isinstance(p, str) and p.strip()]
    return "\n".join(parts).strip()

def answer_query(query: str, ticker: Optional[str], days: int = 7, topk: int = 5, debug: bool = False) -> str:
    if debug:
        os.environ["RAG_DEBUG"] = "1"
    r = Retriever()
    hits = r.search(query, ticker=ticker, days=days, topk=topk, candidate_limit=2000)
    if not hits:
        return "No relevant articles found in the selected window."

    seen, top = set(), []
    for h in hits:
        key = (int(h["article_id"]), _canonical_url(h.get("url") or ""))
        if key in seen:
            continue
        seen.add(key)
        top.append(h)
        if len(top) >= MAX_SOURCES:
            break

    engine = get_engine()

    article_ids = []
    for h in top:
        aid = int(h["article_id"])
        article_ids.append(aid)
        ev = _get_joined_snippet(engine, aid, limit=CHUNKS_PER_ARTICLE)
        if not ev:
            ev = (h.get("snippet") or "").strip()
        ev = re.sub(r"\s+", " ", ev).strip()
        h["_evidence"] = ev

    evidences = [h["_evidence"] for h in top]
    key_hash, ids_csv, ev_hash = _make_answer_cache_key(query, ticker, article_ids, evidences)

    _ensure_cache_tables(engine)
    cached = _answer_cache_get(engine, key_hash)
    if cached:
        if debug:
            print("[DEBUG] Answer cache hit")
        return cached

    prompt = _build_bullets_prompt(query, ticker, top)
    if debug:
        print("\n[DEBUG] Using sources:")
        for i, h in enumerate(top, 1):
            print(f" {i}. {h['published_at']} | {h['source']} | {h['headline']}")
        print("\n[DEBUG] Prompt preview:")
        print(textwrap.shorten(prompt, width=500, placeholder=" ..."))

    evidence_concat = " ".join(
        [f"{(h.get('headline') or '').strip()} || {(h.get('_evidence') or '').strip()}" for h in top]
    )

    fallback_used = False
    try:
        bullets = _openai_respond(prompt, max_output_tokens=MAX_OUTPUT_TOKENS_BULLETS)
        if not bullets.strip():
            if debug:
                print("[DEBUG] Empty model response; using extractive fallback")
            fallback_used = True
        else:
            bullets = _enforce_numbers_from_evidence(bullets, evidence_concat)
    except Exception as e:
        if debug:
            print(f"[DEBUG] OpenAI call failed: {e}")
        fallback_used = True

    if fallback_used:
        fallback_lines = []
        for i, h in enumerate(top, 1):
            head = (h.get("headline") or "").strip()
            date = (h.get("published_at") or "n/a").strip()
            ev = h.get("_evidence") or head
            sent = textwrap.shorten(ev, width=200, placeholder="...")
            fallback_lines.append(f"- {date} - {head}. {sent} [{i}]")
        bullets = "\n".join(fallback_lines)

    wc = _word_count(bullets)
    if wc > TARGET_WORDS[1] + 60:
        bullets = textwrap.shorten(bullets, width=1600, placeholder="...")

    sources_list = _build_public_sources_list(top)
    final = f"{bullets}\n\nSources:\n{sources_list}"

    if not fallback_used:
        _ensure_cache_tables(engine)
        _answer_cache_put(engine, key_hash, query, ticker, ids_csv, ev_hash, final)
    else:
        if debug:
            print("[DEBUG] Skipping cache write due to fallback output")

    return final

def summarize_article(article_id: int, debug: bool = False) -> str:
    if debug:
        os.environ["RAG_DEBUG"] = "1"

    engine = get_engine()
    with engine.connect() as c:
        row = c.execute(
            text("SELECT a.id, a.ticker, a.headline, a.url, a.source, a.published_at FROM articles a WHERE a.id = :aid"),
            {"aid": int(article_id)},
        ).mappings().fetchone()
        if not row:
            return f"Article {article_id} not found."
        chunks = c.execute(
            text("SELECT text_snippet FROM embeddings WHERE article_id = :aid ORDER BY chunk_id ASC LIMIT :lim"),
            {"aid": int(article_id), "lim": int(SUM_CHUNKS_PER_ARTICLE)},
        ).all()

    headline = (row["headline"] or "").strip()
    url = _canonical_url(row["url"] or "")
    date = (row["published_at"] or "n/a")
    if hasattr(date, "isoformat"):
        date = date.strftime("%Y-%m-%d")
    src = (row["source"] or "").strip() or urlparse(url).netloc
    body_parts = [headline] + [((c[0] or "").replace("\n", " ").strip()) for c in chunks]
    content = re.sub(r"\s+", " ", " ".join([p for p in body_parts if p])).strip()

    key_hash, content_hash = _make_summary_cache_key(article_id, OPENAI_MODEL, SUMMARY_PROMPT_VERSION, headline, content)

    _ensure_cache_tables(engine)
    cached = _summary_cache_get(engine, key_hash)
    if cached:
        if debug:
            print("[DEBUG] Summary cache hit")
        summary = cached
    else:
        prompt = _build_summary_prompt(headline, content)
        if debug:
            print("[DEBUG] Summary prompt preview:")
            print(textwrap.shorten(prompt, width=500, placeholder=" ..."))
        try:
            # NOTE: your original code uses MAX_OUTPUT_TOKENS_SUMMARY here.
            summary = _openai_respond(prompt, max_output_tokens=MAX_OUTPUT_TOKENS_SUMMARY)  # noqa: F821
            if not summary.strip():
                if debug:
                    print("[DEBUG] Empty model response on summarize; using extractive fallback")
                summary = textwrap.shorten(content or headline, width=600, placeholder="...")
        except Exception as e:
            if debug:
                print(f"[DEBUG] OpenAI call failed: {e}")
            summary = textwrap.shorten(content or headline, width=600, placeholder="...")

        _summary_cache_put(engine, key_hash, int(article_id), OPENAI_MODEL, SUMMARY_PROMPT_VERSION, content_hash, summary)

    cite = f"{src} - {date} - {url}"
    return f"- {date} - {summary}\n\nSource:\n{cite}"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RAG answerer using OpenAI gpt-4o-mini via Responses API with caching (answers + summaries) and numeric post-filter")
    sub = ap.add_subparsers(dest="cmd")

    ap_q = sub.add_parser("ask", help="Ask a question (RAG bullets)")
    ap_q.add_argument("query", type=str)
    ap_q.add_argument("--ticker", type=str, default=None)
    ap_q.add_argument("--days", type=int, default=7)
    ap_q.add_argument("--topk", type=int, default=5)
    ap_q.add_argument("--debug", action="store_true")

    ap_s = sub.add_parser("summarize", help="Summarize a specific article by id")
    ap_s.add_argument("article_id", type=int)
    ap_s.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    if args.cmd == "ask":
        print(
            answer_query(
                args.query,
                ticker=args.ticker,
                days=args.days,
                topk=args.topk,
                debug=args.debug,
            )
        )
    elif args.cmd == "summarize":
        print(summarize_article(args.article_id, debug=args.debug))
    else:
        ap.print_help()
