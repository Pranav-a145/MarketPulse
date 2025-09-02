from sqlalchemy import text
from db.conn import get_engine

sql = open("db/schema_postgres.sql", "r", encoding="utf-8").read()

with get_engine().begin() as conn:
    for stmt in sql.split(";"):
        s = stmt.strip()
        if s:
            conn.execute(text(s))
print("Schema applied.")
