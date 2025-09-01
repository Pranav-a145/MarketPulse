import sqlite3, pathlib

DB_PATH = "marketpulse.db"
SCHEMA = pathlib.Path("db/schema.sql").read_text()

conn = sqlite3.connect(DB_PATH)
conn.executescript(SCHEMA)
conn.close()

print(f"Created {DB_PATH} with tables 'articles' and 'sentiment'.")
