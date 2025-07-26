import sys
import argparse

from .config import ROOT_DIR, DB_PATH
from .db.setup import ensure_db
from .indexer import index_directory
from .search.hybrid import hybrid_search


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Semantic and lexical command line search"
    )
    p.add_argument(
        "--reindex",
        action="store_true",
        help="delete the current index DB and rebuild from scratch",
    )
    p.add_argument(
        "--skip-index",
        action="store_true",
        help="skip any indexing work; open the existing DB read-only"
    )
    p.add_argument(
        "-q", "--query",
        metavar="TEXT",
        help="run a single query and exit",
    )
    # p.add_argument(
    #     "--scope",
    #     metavar="SUBPATH",
    #     help=("Only return results whose file path starts with this "
    #           "sub-directory (relative to ROOT_DIR or absolute)."),
    # )
    return p.parse_args()


def main() -> None:
    args = _parse_cli()

    if args.reindex and args.skip_index:
        print("Error: --reindex and --no-index are mutually exclusive.")
        sys.exit(1)

    if not ROOT_DIR.exists():
        print(f"Error: root_dir {ROOT_DIR} not found; check settings.toml.")
        sys.exit(1)
    
    # scope_prefix: str | None = None
    # if args.scope:
    #     scope_prefix = str((ROOT_DIR / args.scope).expanduser().resolve())
    #     if not scope_prefix.startswith(str(ROOT_DIR)):
    #         print("Scope must be inside the root directory!")
    #         sys.exit(1)
    
    if args.reindex and DB_PATH.exists():
        DB_PATH.unlink()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = ensure_db()
    
    if not args.skip_index:
        index_directory(con)
    
    try:
        if args.query is not None:
            print("\nTop results (hybrid RRF):\n")
            for i, (ref, snippet, score) in enumerate(hybrid_search(con, args.query), 1):
                print(f"[{i}] {ref} (rrf={score:.4f})\n    {snippet}…\n")
        else:
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
