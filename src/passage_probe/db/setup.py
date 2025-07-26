import sqlite3

import sqlite_vec  # type: ignore

from ..config import DB_PATH, EMBED_DIM


def ensure_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.execute("PRAGMA journal_mode=WAL;")

    # per‑file
    con.execute("""
        CREATE TABLE IF NOT EXISTS docs(
            id   INTEGER PRIMARY KEY,
            path TEXT UNIQUE
        );
    """)

    con.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_d
        USING fts5(fulltext, content='docs', content_rowid='id', tokenize='porter');
    """)

    # per‑passage
    con.execute("""
        CREATE TABLE IF NOT EXISTS passages(
            id    INTEGER PRIMARY KEY,
            doc_id INTEGER,
            chunk  INTEGER,
            passage TEXT,
            FOREIGN KEY(doc_id) REFERENCES docs(id),
            UNIQUE(doc_id, chunk)
        );
    """)

    con.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec
        USING vec0(embedding float[{EMBED_DIM}]);
    """)

    con.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_p
        USING fts5(passage, content='passages', content_rowid='id', tokenize='porter');
    """)

    return con