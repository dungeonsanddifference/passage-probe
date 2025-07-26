import sys

from .config import ROOT_DIR, DB_PATH
from .db.setup import ensure_db
from .indexer import index_directory
from .search.hybrid import hybrid_search


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
