-- Postgres schema (no PRAGMA)

CREATE TABLE IF NOT EXISTS articles (
  id SERIAL PRIMARY KEY,
  ticker TEXT NOT NULL,
  headline TEXT NOT NULL,
  url TEXT NOT NULL,
  published_at TIMESTAMPTZ NOT NULL,
  source TEXT,
  text TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_articles_url ON articles(url);
CREATE INDEX IF NOT EXISTS ix_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS ix_articles_ticker ON articles(ticker);

CREATE TABLE IF NOT EXISTS sentiment (
  id SERIAL PRIMARY KEY,
  article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
  label TEXT NOT NULL,
  p_neg DOUBLE PRECISION,
  p_neu DOUBLE PRECISION,
  p_pos DOUBLE PRECISION,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_sentiment_article ON sentiment(article_id);
