from __future__ import annotations

import re
import sys
import sqlite3
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Iterable

from sentence_transformers import SentenceTransformer  # type: ignore
import sqlite_vec  # type: ignore
from tqdm import tqdm # type: ignore

from .config import *

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def serialize_f32(vec: Sequence[float]) -> bytes:
    """Pack a float list/ndarray into raw little-endian bytes (f32)."""
    return struct.pack(f"{len(vec)}f", *vec)


def chunk_text(text: str, max_len: int = CHUNK_LEN, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks of ≈max_len characters.

    Very simple: step = max_len - overlap; slice on character boundaries.
    Guaranteed at least one chunk (even if text < max_len).
    """
    if len(text) <= max_len:
        return [text]

    step = max_len - overlap
    return [text[i : i + max_len] for i in range(0, len(text), step)]


def passages_for_file(path_str: str, content: str) -> List[str]:
    """Return list of passages for *path* based on extension rules."""
    ext = Path(path_str).suffix.lower()
    if ext in LINE_BY_LINE_EXT:
        # One passage per non‑empty line
        return [ln for ln in content.splitlines() if ln.strip()]
    return chunk_text(content)

# ---------------------------------------------------------------------------
# SQLite setup
# ---------------------------------------------------------------------------

def ensure_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.execute("PRAGMA journal_mode=WAL;")

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

    # Full‑text index (external content so we keep a single source of truth)
    con.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS fts
        USING fts5(
            passage,
            content='passages',
            content_rowid='id',
            tokenize='porter'
        );
        """
    )

    # If fts is empty but passages already exists, populate in one go
    if con.execute("SELECT count(*) FROM fts").fetchone()[0] == 0 and \
       con.execute("SELECT count(*) FROM passages").fetchone()[0] > 0:
        print("Building BM25 index (one‑time)…")
        con.execute("INSERT INTO fts(fts) VALUES('rebuild');")
        con.commit()

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
    to_index = [(p, t) for p, t in load_files() if p not in existing_paths]
    if not to_index:
        return

    print(f"Indexing {len(to_index)} new files …")
    model = SentenceTransformer(MODEL_NAME)

    for path, txt in tqdm(to_index, unit="file"):
        passages = passages_for_file(path, txt)
        embeddings = model.encode(passages, batch_size=32, normalize_embeddings=True)
        for idx, (passage, vec) in enumerate(zip(passages, embeddings)):
            cur = con.execute(
                "INSERT OR IGNORE INTO passages(path, chunk, passage) VALUES (?,?,?)",
                (path, idx, passage),
            )
            rowid = cur.lastrowid
            if rowid:
                con.execute("INSERT INTO vec(rowid, embedding) VALUES (?, ?)", (rowid, serialize_f32(vec)))
                con.execute("INSERT INTO fts(rowid, passage) VALUES (?, ?)", (rowid, passage))
    con.commit()

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def semantic_search(con: sqlite3.Connection, query: str, k: int = TOP_K):
    """
    Perform a semantic search for the given query against stored embeddings.
    """
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


def _vector_candidates(con: sqlite3.Connection, q_vec: bytes, k: int):
    rows = con.execute(
        "SELECT rowid, distance FROM vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (q_vec, k),
    ).fetchall()
    return {rid: i for i, (rid, _d) in enumerate(rows, 1)}, {rid: _d for rid, _d in rows}


def _sanitize_query(q: str) -> str:
    cleaned: str = re.sub(r"[^\w]+", " ", q)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words: List[str] = cleaned.split()
    return " OR ".join(words)


def _bm25_candidates(con: sqlite3.Connection, query: str, k: int):
    safe_q = _sanitize_query(query)
    rows = con.execute(
        "SELECT rowid, bm25(fts) FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
        (safe_q, k),
    ).fetchall()
    return {rid: i for i, (rid, _s) in enumerate(rows, 1)}, {rid: _s for rid, _s in rows}


def _rrf_score(ranks: Iterable[int], k: int = RRF_K) -> float:
    return sum(1.0 / (k + r) for r in ranks)


def hybrid_search(con: sqlite3.Connection, query: str, top_k: int = TOP_K) -> List[Tuple[str, str, float]]:
    """
    Perform a hybrid semantic + lexical search over stored passages.

    This function combines dense embedding similarity (via a SentenceTransformer)
    with BM25 lexical matching, then fuses their rankings using Reciprocal Rank
    Fusion (RRF). It selects the highest-scoring passage chunk per document path,
    and returns the top_k results sorted by fused score.
    """
    model = SentenceTransformer(MODEL_NAME)
    q_vec = serialize_f32(model.encode(query, normalize_embeddings=True))

    vec_rank, _ = _vector_candidates(con, q_vec, POOL_SIZE)
    bm_rank, _ = _bm25_candidates(con, query, POOL_SIZE)

    # RRF fusion
    fused_scores = {
        rid: _rrf_score([vec_rank.get(rid, 10**9), bm_rank.get(rid, 10**9)])
        for rid in (set(vec_rank) | set(bm_rank))
    }

    # Pick highest‑scoring passage per document
    best_per_doc: Dict[str, Tuple[int, float]] = {}  # path → (rowid, score)
    for rid, score in fused_scores.items():
        path, chunk_idx = con.execute("SELECT path, chunk FROM passages WHERE id=?", (rid,)).fetchone()
        if (path not in best_per_doc) or (score > best_per_doc[path][1]):
            best_per_doc[path] = (rid, score)

    # Sort these winners by score
    winners = sorted(best_per_doc.values(), key=lambda t: t[1], reverse=True)[:top_k]

    results = []
    for rid, score in winners:
        path, idx, passage = con.execute(
            "SELECT path, chunk, passage FROM passages WHERE id=?", (rid,)
        ).fetchone()
        preview = passage.strip().replace("\n", " ")[:300]
        results.append((f"{path}#chunk{idx}", preview, score))
    return results


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
            q = input("\nQuery (blank to quit) > ").strip()
            if not q:
                break
            print("\nTop results (hybrid RRF):\n")
            for i, (ref, snippet, score) in enumerate(hybrid_search(con, q), 1):
                print(f"[{i}] {ref} (rrf={score:.4f})\n    {snippet}…\n")
    finally:
        con.close()


if __name__ == "__main__":
    main()
