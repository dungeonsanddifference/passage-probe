from __future__ import annotations

import sys
import sqlite3
import struct
from typing import List, Tuple, Sequence

from sentence_transformers import SentenceTransformer  # type: ignore
import sqlite_vec  # type: ignore
from tqdm import tqdm # type: ignore

from .config import *

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def serialize_f32(vec: Sequence[float]) -> bytes:
    """Pack a float list/ndarray into raw little‑endian bytes (f32)."""
    return struct.pack(f"{len(vec)}f", *vec)


def chunk_text(text: str, max_len: int = CHUNK_LEN, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split *text* into overlapping chunks of ≈max_len characters.

    Very simple: step = max_len - overlap; slice on character boundaries.
    Guaranteed at least one chunk (even if text < max_len).
    """
    if len(text) <= max_len:
        return [text]

    step = max_len - overlap
    return [text[i : i + max_len] for i in range(0, len(text), step)]

# ---------------------------------------------------------------------------
# SQLite setup
# ---------------------------------------------------------------------------

def ensure_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.execute("PRAGMA journal_mode=WAL;")

    # Each chunk is a separate row, keyed by (path, chunk_idx)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS passages(
            id      INTEGER PRIMARY KEY,
            path    TEXT,
            chunk   INTEGER,
            passage TEXT,
            UNIQUE(path, chunk)
        );
        """
    )

    con.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec
        USING vec0(embedding float[{EMBED_DIM}]);
        """
    )

    return con

# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def _is_blacklisted(path: Path) -> bool:
    if path.suffix.lower() in BLACKLIST_EXT:
        return True
    if any(part in BLACKLIST_DIRS for part in path.parts):
        return True
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return True
    except Exception:
        return True
    return False


def load_files() -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for p in ROOT_DIR.rglob("*"):
        if p.is_file() and not _is_blacklisted(p):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rows.append((str(p), txt))
    return rows


def index_directory(con: sqlite3.Connection) -> None:
    existing_paths = {row[0] for row in con.execute("SELECT DISTINCT path FROM passages")}
    new_files = [(path, txt) for path, txt in load_files() if path not in existing_paths]
    if not new_files:
        return

    print(f"Indexing {len(new_files)} files in {ROOT_DIR} …")
    model = SentenceTransformer(MODEL_NAME)

    for path, full_text in tqdm(new_files, unit="file"):
        chunks = chunk_text(full_text)
        embeddings = model.encode(chunks, batch_size=32, normalize_embeddings=True)
        for idx, (chunk_text_str, emb_vec) in enumerate(zip(chunks, embeddings)):
            cur = con.execute(
                "INSERT OR IGNORE INTO passages(path, chunk, passage) VALUES (?,?,?)",
                (path, idx, chunk_text_str),
            )
            rowid = cur.lastrowid
            if rowid:
                con.execute(
                    "INSERT INTO vec(rowid, embedding) VALUES (?, ?)",
                    (rowid, serialize_f32(emb_vec)),
                )
    con.commit()

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def semantic_search(con: sqlite3.Connection, query: str, k: int = TOP_K):
    model = SentenceTransformer(MODEL_NAME)
    q_vec = serialize_f32(model.encode(query, normalize_embeddings=True))

    cur = con.execute(
        """
        SELECT rowid, distance
        FROM vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (q_vec, k),
    )
    for rowid, dist in cur.fetchall():
        path, chunk_idx, passage = con.execute(
            "SELECT path, chunk, passage FROM passages WHERE id=?", (rowid,)
        ).fetchone()
        preview = passage.strip().replace("\n", " ")[:300]
        yield f"{path}#chunk{chunk_idx}", preview, dist

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    if not ROOT_DIR.exists():
        print(f"Error: root_dir {ROOT_DIR} not found; check settings.toml.")
        sys.exit(1)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = ensure_db()
    index_directory(con)

    try:
        while True:
            query = input("\nQuery (blank to quit) > ").strip()
            if not query:
                break
            print("\nTop results:\n")
            for i, (ref, snippet, dist) in enumerate(semantic_search(con, query), 1):
                print(f"[{i}] {ref} (dist={dist:.4f})\n    {snippet}…\n")
    finally:
        con.close()


if __name__ == "__main__":
    main()
