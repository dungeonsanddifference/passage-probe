import sqlite3
from typing import List, Tuple

from tqdm import tqdm # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

from .config import ROOT_DIR, MODEL_NAME
from .utils import is_blacklisted, passages_for_file, serialize_f32


def load_files() -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for p in ROOT_DIR.rglob("*"):
        if p.is_file() and not is_blacklisted(p):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rows.append((str(p), txt))
    return rows


def index_directory(con: sqlite3.Connection):
    seen = {row[0] for row in con.execute("SELECT path FROM docs")}
    to_index = [(p,t) for p,t in load_files() if p not in seen]
    if not to_index:
        return

    print(f"Indexing {len(to_index)} new files …")
    model = SentenceTransformer(MODEL_NAME)

    for path, full_text in tqdm(to_index, unit="file"):
        # insert into docs → get doc_id
        cur = con.execute("INSERT INTO docs(path) VALUES (?)", (path,))
        doc_id = cur.lastrowid
        con.execute("INSERT INTO fts_d(rowid, fulltext) VALUES (?,?)", (doc_id, full_text))

        passages = passages_for_file(path, full_text)
        embs = model.encode(passages, batch_size=32, normalize_embeddings=True)
        for idx, (passage, vec) in enumerate(zip(passages, embs)):
            cur2 = con.execute(
                "INSERT INTO passages(doc_id, chunk, passage) VALUES (?,?,?)",
                (doc_id, idx, passage),
            )
            rid = cur2.lastrowid
            con.execute("INSERT INTO vec(rowid, embedding) VALUES (?, ?)", (rid, serialize_f32(vec)))
            con.execute("INSERT INTO fts_p(rowid, passage) VALUES (?, ?)", (rid, passage))
    con.commit()