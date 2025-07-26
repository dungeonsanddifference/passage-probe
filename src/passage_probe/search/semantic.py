import sqlite3

from sentence_transformers import SentenceTransformer  # type: ignore

from ..config import MODEL_NAME, TOP_K
from ..utils import serialize_f32


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