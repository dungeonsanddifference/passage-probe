# PassageProbe

*[Boot.dev](boot.dev) Hackathon 2025 submission*

One frustration I've encountered frequently in large organizations is how challenging it can be to find relevant documentation using traditional keyword search. Shared vocabulary often leads to results that are related only superficially, missing the deeper context and meaning I'm actually seeking. Driven by this challenge, I've developed an interest in semantic search that prioritizes meaning over simple keyword matching.

This Python script attempts to perform efficient semantic file search by combining dense vector embedding similarity with BM25 lexical matching by fusing their rankings using Reciprocal Rank Fusion (RRF). It selects the highest‑scoring passage chunk per document path, and returns the top_k results sorted by fused score.

---

## Features

- **Hybrid Sematic + Lexical Retrieval:** Fuses semantic and lexical results with Reciprocal Rank Fusion (RRF) for robust, relevance-balanced search results.
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

Run indexing and interactive semantic search with:

```bash
uv run passage-probe
```

### Interactive Search

* Enter a natural language query when prompted.
* Results displayed by semantic relevance.

Example:

```
Query (blank to quit) > What is semantic indexing?

Top results:

[1] /path/to/docs/semantic_search.md#chunk3 (dist=0.1234)
    Semantic indexing refers to indexing documents based on meaning rather than exact keywords. It leverages vector embeddings to represent texts semantically…

[2] /path/to/articles/nlp_overview.txt#chunk0 (dist=0.1567)
    In natural language processing, semantic indexing is essential for retrieving relevant documents based on meaning instead of keyword frequency…
```

Press `Enter` without typing anything to exit.

## Configuration

Adjust indexing and search parameters in [`settings.toml`](./settings.toml)

### Performance Knobs

* `POOL_SIZE`: bigger ⇒ better recall, slower query. Try 20‑100.
* `RRF_K`: smaller ⇒ BM25/vec top ranks dominate; larger flattens influence.

---

## Extending & Integration

* **API Integration:** The core functions (`index_directory`, `hybrid_search`, etc.) can be imported and embedded into other Python apps or services

* **Custom Chunkers:** Modify or extend `passages_for_file` to support additional file types or chunking logic.
