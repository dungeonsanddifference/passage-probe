import sys
import argparse

from rich.console import Console

from .config import ROOT_DIR, DB_PATH
from .db.setup import ensure_db
from .indexer import index_directory
from .search.hybrid import hybrid_search
from .tui import PassageProbe


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
    console = Console()

    with console.status("[bold green]Starting...") as status:
        args = _parse_cli()

        if args.reindex and args.skip_index:
            console.print("[bold red]Error: --reindex and --no-index are mutually exclusive.")
            sys.exit(1)

        if not ROOT_DIR.exists():
            console.print(f"[bold red]Error: root_dir {ROOT_DIR} not found; check settings.toml.")
            sys.exit(1)
        
        # scope_prefix: str | None = None
        # if args.scope:
        #     scope_prefix = str((ROOT_DIR / args.scope).expanduser().resolve())
        #     if not scope_prefix.startswith(str(ROOT_DIR)):
        #         print("Scope must be inside the root directory!")
        #         sys.exit(1)
        

        status.update("Connecting to db...")
        if args.reindex and DB_PATH.exists():
            DB_PATH.unlink()

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        con = ensure_db()

    if args.query is not None:
        try:
            if not args.skip_index:
                status.update("Indexing...")
                index_directory(con)
            print("\nTop results (hybrid RRF):\n")
            for i, (ref, snippet, score) in enumerate(hybrid_search(con, args.query), 1):
                console.print(f"[bold purple][{i}] {ref} (rrf={score:.4f})\n    [gray66]{snippet}…\n")
        finally:
            con.close()
            sys.exit(0)

    app = PassageProbe(con, do_index=not args.skip_index)
    app.run()


if __name__ == "__main__":
    main()
