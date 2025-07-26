import re
import sqlite3
from typing import List

def _sanitize_query(q: str) -> str:
    cleaned: str = re.sub(r"[^\w]+", " ", q)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words: List[str] = cleaned.split()
    return " OR ".join(words)


def _bm25_candidates(con: sqlite3.Connection, query: str, k: int):
    safe_q = _sanitize_query(query)
    rows = con.execute(
        "SELECT docs.id, bm25(fts_d) FROM fts_d JOIN docs ON docs.id=fts_d.rowid "
        "WHERE fts_d MATCH ? ORDER BY bm25(fts_d) LIMIT ?",
        (safe_q, k),
    ).fetchall()
    return {rid: i for i, (rid, _s) in enumerate(rows, 1)}, {rid: _s for rid, _s in rows}