from typing import Tuple, Iterable

from textual.app import App, ComposeResult
from textual import on, work
from textual.containers import Vertical, ScrollableContainer, Grid
from textual.widgets import (
    Input, Header, Footer, ProgressBar, Collapsible, Static,
    Label, Button
)
from textual.reactive import reactive
from textual.screen import ModalScreen
from sentence_transformers import SentenceTransformer

from .db.setup import ensure_db
from .indexer import load_files,  passages_for_file
from .search.hybrid import hybrid_search
from .utils import serialize_f32
from .config import *


def _format_hits(hits: Iterable[Tuple[str, str, float]]) -> list[Collapsible]:
    """Return Collapsible widgets (header passed via `title` kwarg)."""
    out: list[Collapsible] = []
    for ref, snippet, score in hits:
        title = f"{ref}   [dim](rrf {score:.3f})[/]"
        out.append(Collapsible(Static(snippet), title=title, collapsed=True))
    return out

# ----- modal dialogs --------------------------------------------------------

class QuitScreen(ModalScreen):
    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Are you sure you want to quit?", id="question"),
            Button("Quit", variant="error", id="quit"),
            Button("Cancel", variant="primary", id="cancel"),
            id="dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()
        else:
            self.app.pop_screen()


class ConfirmIndex(ModalScreen):
    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Delete index and rebuild?", id="question"),
            Button("Yes", variant="error", id="yes"),
            Button("No", variant="primary", id="no"),
            id="dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "yes":
            self.app.pop_screen()
            self.app.action_confirm_reindex()
        else:
            self.app.pop_screen()

# ----- main app -------------------------------------------------------------

class PassageProbe(App):
    CSS_PATH = "modal.tcss"
    BINDINGS = [
        ("ctrl+c", "quit_dialog", "Quit"),
        ("ctrl+r", "reindex_dialog", "Re-index"),
    ]

    querying: reactive[bool] = reactive(False)

    def __init__(self, initial_con, do_index: bool):
        super().__init__()
        self.con = initial_con
        self.model = SentenceTransformer(MODEL_NAME)
        self.do_index = do_index

    # ----- compose ----------------
    def compose(self) -> ComposeResult:
        yield Header()
        self.input = Input(placeholder="Type query and ⏎")
        self.progress = ProgressBar(show_percentage=True)
        self.results = ScrollableContainer()
        yield Vertical(self.input, self.progress, self.results)
        yield Footer()

    def on_mount(self):
        self.theme = "dracula"
        self.set_focus(self.input)
        self.progress.display = False
        if self.do_index:
            self._index_worker()

    # -------------- indexing --------------
    def _init_progress(self, total: int):
        self.progress.display = True
        self.progress.update(total=total)
        self.progress.progress = 0

    @work(thread=True, exclusive=True)
    def _index_worker(self):
        """Incremental index with per-file progress updates (no extra param)."""
        con = ensure_db()

        existing = {row[0] for row in con.execute("SELECT path FROM docs")}
        files = [(p, t) for p, t in load_files() if p not in existing]
        total = len(files)
        self.call_from_thread(lambda: self._init_progress(total))

        if not files:
            self.call_from_thread(lambda: self._toggle_progress(False))
            return

        model = SentenceTransformer(MODEL_NAME)

        # reimplementation of `.indexer.index_directory` to show progress
        for path, full_text in files:
            cur = con.execute("INSERT INTO docs(path) VALUES (?)", (path,))
            doc_id = cur.lastrowid
            con.execute("INSERT INTO fts_d(rowid, fulltext) VALUES (?,?)",
                        (doc_id, full_text))

            passages = passages_for_file(path, full_text)
            embs = model.encode(passages, batch_size=32, normalize_embeddings=True)
            for idx, (passage, vec) in enumerate(zip(passages, embs)):
                rid = con.execute(
                    "INSERT INTO passages(doc_id, chunk, passage) VALUES (?,?,?)",
                    (doc_id, idx, passage),
                ).lastrowid
                con.execute("INSERT INTO vec(rowid, embedding) VALUES (?,?)",
                            (rid, serialize_f32(vec)))
                con.execute("INSERT INTO fts_p(rowid, passage) VALUES (?,?)",
                            (rid, passage))
            con.commit()
            self.call_from_thread(self.progress.advance)

        con.close()
        self.call_from_thread(lambda: self._toggle_progress(False))

    def _toggle_progress(self, show: bool):
        if not show:
            self.progress.display = False
            self.progress.progress = 0

    # ---------------- querying ---------------
    @on(Input.Submitted)
    def _on_submit(self, ev: Input.Submitted):
        q = ev.value.strip()
        if q:
            self._query_worker(q)

    @work(thread=True, exclusive=True)
    def _query_worker(self, q: str):
        self.call_from_thread(lambda: self._placeholder(True))
        con = ensure_db()
        hits = hybrid_search(con, q, TOP_K)
        con.close()
        self.call_from_thread(lambda: self._render_hits(hits))

    def _placeholder(self, busy: bool):
        self.input.placeholder = "Searching…" if busy else "Type query and ⏎ …"

    def _render_hits(self, hits):
        self._placeholder(False)
        self.results.remove_children()
        self.results.mount(*_format_hits(hits))
        if hits:
            self.results.query_one(Collapsible).collapsed = False

    # ---------------- actions ----------------
    def action_quit_dialog(self):
        self.push_screen(QuitScreen())

    def action_reindex_dialog(self):
        self.push_screen(ConfirmIndex())

    def action_confirm_reindex(self):
        # called by ConfirmIndex on positive response
        self.con.close()
        if DB_PATH.exists():
            DB_PATH.unlink()
        self.con = ensure_db()
        self._index_worker()
