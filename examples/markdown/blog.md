# Beyond Keywords: Modern Search Algorithms Explained

*Written by Alex • July 26, 2025*

Searching is one of the most fundamental problems in computer science. Whether you are looking for a value in an array, the fastest route on a map, or the most relevant document in a knowledge base, you are running a **search algorithm**.

This post walks through the evolution of search—from the simplest linear scan to modern hybrid approaches that combine dense vector similarity with lexical ranking. Feel free to skip around or dig into the code snippets!

---

## 1 · Classic Data‑Structure Search

### 1.1 Linear Search

*Best when the data are small or unsorted.*

```python
for i, item in enumerate(arr):
    if item == target:
        return i
return -1
```

* **Time Complexity:** O(*n*)
* **Space Complexity:** O(1)

### 1.2 Binary Search

*Requires the data to be **sorted**.*

```python
from typing import List

def binary_search(arr: List[int], target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```

* **Time Complexity:** O(log *n*)
* **Space Complexity:** O(1)

### 1.3 Hash‑Table Lookup

| Operation       | Average Time | Worst Case |
| --------------- | ------------ | ---------- |
| insert / search | O(1)         | O(*n*)     |

Collisions are handled via chaining or open addressing.

### 1.4 Tree‑Based Search (BST, AVL, Red‑Black)

Self‑balancing trees guarantee O(log *n*) search even after many inserts/deletes.

---

## 2 · Graph Search

When the "data" are nodes connected by edges, you need a graph algorithm.

| Algorithm                  | Use‑Case                              | Optimal?          | Cost              |
| -------------------------- | ------------------------------------- | ----------------- | ----------------- |
| Breadth‑First Search (BFS) | Unweighted shortest path              | ✔                 | O(*V* + *E*)      |
| Depth‑First Search (DFS)   | Cycle detection, topological sort     | —                 | O(*V* + *E*)      |
| Dijkstra                   | Weighted shortest path (non‑negative) | ✔                 | O(*E* log *V*)    |
| A\*                        | Heuristic pathfinding                 | ✔ (if admissible) | O(*E*) on average |

```python
import heapq

def dijkstra(graph, src):
    dist = {v: float('inf') for v in graph}
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
```

---

## 3 · Information‑Retrieval Search

### 3.1 TF‑IDF

The classic vector‑space model that weighs terms by frequency and rarity.

### 3.2 BM25

A probabilistic ranking function that improves on TF‑IDF by normalizing for document length. Widely used in search engines and available in SQLite FTS5.

```sql
SELECT docid, bm25(fts) AS score
FROM   fts
WHERE  fts MATCH 'neural network'
ORDER  BY score
LIMIT  10;
```

---

## 4 · Semantic Search with Embeddings

Vector embeddings map text into a high‑dimensional space where cosine similarity approximates semantic closeness.

1. **Embed** passages with a model like `sentence-transformers/all-MiniLM-L6-v2`.
2. **Store** them in a vector index (e.g., FAISS, `sqlite-vec`).
3. **Query** by embedding the user input and retrieving nearest neighbors.

Pros:

* Handles synonyms and paraphrases.

Cons:

* May ignore exact keywords ("H2O" vs "water").

---

## 5 · Hybrid Search & Reciprocal Rank Fusion (RRF)

Combines lexical (BM25) and semantic (vector) scores.

```python
def rrf_score(ranks, k=60):
    return sum(1/(k + r) for r in ranks)
```

Hybrid search often yields the best of both worlds—precise keyword matching and semantic generalization.

---

## 6 · Evaluating Search Quality

* **Precision @ *k*** — How many of the top *k* results are relevant?
* **Recall @ *k*** — How many relevant results did we retrieve out of all
  that exist?
* **MRR** (Mean Reciprocal Rank) — Useful when there is exactly one correct
  answer per query.
* **nDCG** (Normalized Discounted Cumulative Gain) — Accounts for graded
  relevance.

---

## 7 · Practical Tips

1. **Chunking matters**: overlap chunks to avoid splitting sentences.
2. **Stop‑word handling**: keep vs remove depends on the model.
3. **Caching embeddings**: speeds up re‑indexing.
4. **Monitor drift**: refresh embeddings when the domain language shifts.
5. **Guardrails**: apply filters (date, language, user permissions) before ranking.

---

## 8 · Further Reading

* Manning, C. D., Raghavan, P., & Schütze, H. *Introduction to Information Retrieval*.
* Jurafsky, D., & Martin, J. H. *Speech and Language Processing*.
* **SQLite FTS5** documentation [https://sqlite.org/fts5.html](https://sqlite.org/fts5.html)
* **FAISS**: Facebook AI Similarity Search [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
* Cooperman, A. *The RRF Keynote* (SIGIR 2023).

---

Thanks for reading! Feel free to share questions, corrections, or war stories about search in the comments below. 🚀
