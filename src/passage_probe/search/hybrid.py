import sqlite3
from typing import List, Tuple, Iterable, Dict

from sentence_transformers import SentenceTransformer  # type: ignore

from ..config import MODEL_NAME, TOP_K, POOL_SIZE, RRF_K
from ..utils import serialize_f32
from .semantic import _vector_candidates
from .lexical import _bm25_candidates

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
    doc_rank, _ = _bm25_candidates(con, query, POOL_SIZE)

    # compute fused score per passage rowid → then keep best per doc
    fused_scores: Dict[int,float] = {}
    for rid, v_rank in vec_rank.items():
        doc_id = con.execute("SELECT doc_id FROM passages WHERE id=?", (rid,)).fetchone()[0]
        fused_scores[rid] = _rrf_score([v_rank, doc_rank.get(doc_id, 10**9)])

    # best passage per doc
    best: Dict[int, Tuple[int,float]] = {}  # doc_id → (rid, score)
    for rid, score in fused_scores.items():
        doc_id = con.execute("SELECT doc_id FROM passages WHERE id=?", (rid,)).fetchone()[0]
        if (doc_id not in best) or score > best[doc_id][1]:
            best[doc_id] = (rid, score)

    # Sort these winners by score
    winners = sorted(best.values(), key=lambda t: t[1], reverse=True)[:top_k]

    results = []
    for rid, score in winners:
        path, chunk, passage = con.execute(
            "SELECT docs.path, passages.chunk, passages.passage "
            "FROM passages JOIN docs ON docs.id=passages.doc_id WHERE passages.id=?",
            (rid,),
        ).fetchone()
        snippet = passage.replace("\n"," ")[:300]
        results.append((f"{path}#chunk{chunk}", snippet, score))
    return results