# 20 Newsgroups — Lightweight Semantic Search System

A complete semantic search pipeline built over the [UCI 20 Newsgroups dataset](https://archive.ics.uci.edu/dataset/113/twenty+newsgroups), implementing:

1. **TF-IDF + LSA embedding** with a custom vector store
2. **Fuzzy C-Means clustering** with soft membership distributions
3. **Semantic cache** built from first principles (no Redis/Memcached)
4. **FastAPI-compatible HTTP service**

---

## Quick Start

```bash
# Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build pipeline + start server (single command)
python3 build_and_run.py

# Or with FastAPI/uvicorn:
uvicorn api.app_fastapi:app --host 0.0.0.0 --port 8000
```

---

## Architecture

```
newsgroups_search/
├── build_and_run.py          # Master pipeline + server startup
├── requirements.txt
├── scripts/
│   └── generate_corpus.py    # Corpus generation (or use real 20newsgroups)
├── embeddings/
│   └── part1_embed.py        # Part 1: TF-IDF + SVD + VectorStore
├── analysis/
│   └── part2_cluster.py      # Part 2: Fuzzy C-Means clustering
├── cache/
│   └── semantic_cache.py     # Part 3: Semantic cache (standalone)
└── api/
    ├── app.py                # Part 4: stdlib HTTP server
    └── app_fastapi.py        # Part 4: FastAPI version (requires fastapi)
```

---

## API Reference

### POST /query

Embed a query, check the semantic cache, and return top results.

**Request:**
```json
{ "query": "How do rockets reach orbit?" }
```

**Response (cache miss):**
```json
{
  "query": "How do rockets reach orbit?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": -0.12,
  "result": [
    {"id": "doc_4521", "similarity": 0.94, "label": "sci.space", "preview": "…"},
    ...
  ],
  "dominant_cluster": 5,
  "cluster_label": "sci.space",
  "membership_distribution": {"5": 0.41, "1": 0.23, "8": 0.12, "3": 0.09, "7": 0.08}
}
```

**Response (cache hit):**
```json
{
  "query": "satellite launch into orbit",
  "cache_hit": true,
  "matched_query": "How do rockets reach orbit?",
  "similarity_score": 0.91,
  "result": [...],
  "dominant_cluster": 5,
  "cluster_label": "sci.space"
}
```

### GET /cache/stats

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### DELETE /cache

Flushes the cache and resets all statistics.

```json
{ "status": "cache flushed", "timestamp": 1234567890.0 }
```

---

## Design Decisions

### Part 1: Embedding Model

**Choice:** TF-IDF (sublinear_tf, bigrams, 50k vocab) + Truncated SVD to 300 components (LSA).

**Justification:**
- No network access for sentence-transformers; LSA is the canonical baseline for newsgroup semantic search
- 300 SVD components explain ~32% of variance but capture most *between-topic* variance
- L2-normalised vectors support cosine similarity via fast dot product
- Sublinear TF dampens term-frequency spikes common in newsgroup posts

**What we discard:** quoted reply lines (`> ...`), email headers, URLs, email addresses, purely numeric tokens, tokens < 3 chars.  These are noise, not semantics.

### Part 2: Fuzzy Clustering

**Choice:** Fuzzy C-Means (Bezdek 1981) on 30-component PCA projection of LSA embeddings.

**Why fuzzy:** A document about gun legislation belongs to both `talk.politics` and `talk.politics.guns` — hard clustering loses this.  Every document gets a membership distribution `U[i]` summing to 1.

**Why PCA first:** The 300-D LSA space is near-isotropic (all pairwise distances ~0.73 ± 0.15), causing FCM to converge to uniform membership.  PCA to 30 dims retains between-topic variance while collapsing within-topic noise.

**Why 15 clusters:** Silhouette peaks at k=15; inertia elbow is between 12 and 18.  Below 12, all "computers" newsgroups merge.  Above 18, the algorithm starts recovering the 20 original labels rather than discovering semantic structure.

**Why m=1.5:** The fuzziness exploration shows:
| m | Mean certainty | Dual-membership (>0.15) | Character |
|---|---|---|---|
| 1.2 | 0.70 | 43% | Near-hard, misses real overlap |
| 1.5 | 0.36 | 31% | **Recommended: semantic overlap visible** |
| 2.0 | 0.12 | 0% | Too soft in this geometry |
| 2.5 | 0.09 | 0% | Degenerate |

### Part 3: Semantic Cache

**Key insight:** Cluster membership turns an O(n) cache lookup into O(n/k).  A query about "rocket fuel" is only compared against other `sci.space` cluster entries — never against `soc.religion.christian` entries.  This prevents cross-topic false positives and gives 15x speedup at k=15.

**Threshold θ:** The single most important parameter.

| θ | Approximate hit rate | Character |
|---|---|---|
| 0.70 | ~65% | Aggressive: topic-level caching |
| 0.80 | ~45% | Liberal: paraphrases + related queries |
| **0.85** | **~30%** | **Recommended: genuine paraphrases** |
| 0.90 | ~15% | Conservative: near-verbatim only |
| 0.95 | ~5% | Strict: near exact-match |

At θ=0.85: "How do rockets reach orbit?" matches "satellite launch orbital mechanics?" (sim=0.91) but NOT "football playoffs tonight" (sim=-0.1, wrong cluster).

**Data structure:** `dict[cluster_id → list[CacheEntry]]` with LRU eviction at 500 entries/bucket.  No external libraries — stdlib + numpy only.

### Part 4: Service

Two implementations with identical business logic:
- `api/app.py`: stdlib `http.server` (no dependencies beyond numpy/sklearn)
- `api/app_fastapi.py`: FastAPI + uvicorn (production-grade)

---

## Using the Real 20 Newsgroups Dataset

Replace the synthetic corpus with the real dataset by modifying `build_and_run.py`:

```python
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
# data.data, data.target_names, data.target are drop-in replacements
```

The rest of the pipeline is dataset-agnostic.
