# PassageProbe

*[Boot.dev](boot.dev) Hackathon 2025 submission*

One frustration I've encountered frequently in large organizations is how challenging it can be to find relevant documentation using traditional keyword search. Shared vocabulary often leads to results that are related only superficially, missing the deeper context and meaning being sought. Driven by this challenge, I've developed an interest in semantic search that prioritizes meaning over simple keyword matching.

This Python application attempts to perform efficient semantic file search by combining dense vector embedding similarity with BM25 lexical matching by fusing their rankings using Reciprocal Rank Fusion (RRF). It selects the highest‑scoring passage chunk per document, and returns the top_k results sorted by fused score.

---

## Features

- **Hybrid Sematic + Lexical Retrieval:** Fuses semantic and lexical results with Reciprocal Rank Fusion (RRF) for relevance-balanced search results.
- **Incremental Indexing:** Indexes only new files upon each run.
- **Customizable Parameters:** Configure the embedding model, chunk sizes, overlap, and file filtering.
- **Single-file DB:** Uses sqlite; easy to copy, back‑up, or inspect with the SQLite CLI.
- **Token‑window Chunking:** Split files by line or overlapping windows.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/dungeonsanddifference/passage-probe.git
cd passage-probe
```

### 2. Setup Environment with `uv`

> [!NOTE]
> Instrucions for `uv` are provided, as that is currently the preferred Boot.dev Python package installer as of June 2025. You should be able to achieve the results with `pip` and `venv` if that is your preferred approach.

Ensure you have `uv` installed. If not, install via your package manager of choice or:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Set up a virtual environment and dependencies with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install .
```

## Usage

```bash
# Launch the interactive TUI (indexes new files on startup by default)
passage-probe

# Skip indexing and open existing DB in read‑only mode
passage-probe --skip-index

# Force a full rebuild of the index database
passage-probe --reindex

# Execute a one‑off query and exit
passage-probe -q "your search text"

# Execute a one‑off query, limit to top 5 results, and exit
passage-probe -q "your search text" -k 5
```

### Interactive TUI

By default, the application runs an interactive TUI powered by [`textual`](https://github.com/Textualize/textual). Type your query in the prompt and run by pressing `Enter`. Results will be displayed in descending match order and previews can be expanded.

![image](/example.png "")

The db can be re-indexed during a session with `ctrl + r` and the application can be quit with `ctrl + c` or `ctrl + q`.

### CLI Search

If `-q` or `--query` arguments are passed, a single query is fetched and printed.

```bash
$ passage-probe -q "villainous monarch" -k 1

Top results (hybrid RRF):

[1] /path/to/docs/semantic_search.md#chunk3 (dist=0.1234)
    Semantic indexing refers to indexing documents based on meaning rather than exact keywords. It leverages vector embeddings to represent texts semantically…
```

## Configuration

Adjust indexing and search parameters in [`settings.toml`](./settings.toml)

### Performance Knobs

* `POOL_SIZE`: bigger ⇒ better recall, slower query. Try 20‑100.
* `RRF_K`: smaller ⇒ BM25/vec top ranks dominate; larger flattens influence.

---

## Extending & Integration

* **API Integration:** The core functions (`index_directory`, `hybrid_search`, etc.) can be imported and embedded into other Python apps or services
* **Custom Chunkers:** Modify or extend `passages_for_file` to support additional file types or chunking logic.

## Enhancements

As this was a weekend hackathon project, there's still significant room for improvement in the following areas:

* Extend UI (e.g., add fields for model, top_k)
* Snippet formatting (e.g., style tables, markdown)
* RFF parameter tuning
* Query scope filtering

It's worth noting that there may be some flaws in the implementation of vector embeddings and BM25. I went into this with a cursory understanding of both, and thus this warants a deeper dive.
