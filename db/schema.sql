PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS articles (
  id INTEGER PRIMARY KEY,
  ticker TEXT NOT NULL,
  headline TEXT NOT NULL,
  url TEXT NOT NULL,
  published_at TEXT NOT NULL,
  source TEXT,
  text TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_articles_url ON articles(url);

CREATE INDEX IF NOT EXISTS ix_articles_published_at ON articles(published_at);

CREATE INDEX IF NOT EXISTS ix_articles_ticker ON articles(ticker);

CREATE TABLE IF NOT EXISTS sentiment (
  id INTEGER PRIMARY KEY,
  article_id INTEGER NOT NULL,
  label TEXT NOT NULL,
  p_neg REAL,
  p_neu REAL,
  p_pos REAL,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(article_id) REFERENCES articles(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_sentiment_article ON sentiment(article_id);
