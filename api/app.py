"""
Part 4: Semantic Search Service
================================

Implements the three required endpoints:
  POST /query
  GET  /cache/stats
  DELETE /cache

Architecture:
  - SemanticCache (Part 3) for cache layer
  - VectorStore (Part 1) for document retrieval
  - Embedding pipeline (TF-IDF + SVD, Part 1) for query encoding
  - Cluster membership (FCM, Part 2) for cache routing

NOTE ON FRAMEWORK:
  FastAPI + uvicorn are not available in this offline environment.  This
  module implements the IDENTICAL API contract using Python stdlib
  http.server.HTTPServer with the same JSON request/response shapes.

  To run with FastAPI/uvicorn when network is available, see app_fastapi.py
  (generated alongside this file).  The business logic is identical; only
  the HTTP routing layer differs.

  START COMMAND (stdlib version):
    python3 api/app.py

  START COMMAND (uvicorn version, requires pip install fastapi uvicorn):
    uvicorn api.app_fastapi:app --host 0.0.0.0 --port 8000

Both versions expose:
  POST /query          → semantic search with cache
  GET  /cache/stats    → cache statistics
  DELETE /cache        → flush cache
"""

import json
import sys
import time
import threading
import pickle
import re
import logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import numpy as np
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

EMBED_DIR = BASE_DIR / "embeddings"
CLUSTER_DIR = BASE_DIR / "data" / "clusters"
VECTOR_STORE_DIR = EMBED_DIR / "vector_store"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("newsgroups_api")


# ---------------------------------------------------------------------------
# Application State (loaded once at startup, shared across requests)
# ---------------------------------------------------------------------------
class AppState:
    """
    Singleton holding all loaded models and the cache.
    Thread-safe for concurrent requests via a lock on cache mutations.
    """

    def __init__(self):
        self.tfidf = None
        self.svd = None
        self.pca = None
        self.centers_pca = None
        self.fcm_m: float = 1.5
        self.n_clusters: int = 15

        # Vector store data (loaded into RAM for fast retrieval)
        self.vectors: Optional[np.ndarray] = None
        self.doc_ids: list = []
        self.doc_meta: list = []

        # Cluster labels (for human-readable cluster names in response)
        self.cluster_labels: dict = {}

        # Semantic cache
        self._cache_lock = threading.Lock()
        self._buckets: dict = {}
        self._hit_count: int = 0
        self._miss_count: int = 0
        self.threshold: float = 0.85

        self._ready = False

    def load(self, threshold: float = 0.85):
        """Load all models and data at startup."""
        log.info("Loading models…")

        # Embedding models
        with open(str(EMBED_DIR / "tfidf.pkl"), "rb") as f:
            self.tfidf = pickle.load(f)
        with open(str(EMBED_DIR / "svd.pkl"), "rb") as f:
            self.svd = pickle.load(f)

        # Cluster models
        with open(str(CLUSTER_DIR / "pca_model.pkl"), "rb") as f:
            self.pca = pickle.load(f)
        self.centers_pca = np.load(str(CLUSTER_DIR / "centers_pca.npy"))

        with open(str(CLUSTER_DIR / "config.json")) as f:
            cfg = json.load(f)
        self.fcm_m = cfg.get("m", 1.5)
        self.n_clusters = cfg.get("n_clusters", 15)

        with open(str(CLUSTER_DIR / "cluster_labels.json")) as f:
            self.cluster_labels = json.load(f)

        # Vector store
        self.vectors = np.load(str(VECTOR_STORE_DIR / "vectors.npy"))
        with open(str(VECTOR_STORE_DIR / "meta.json")) as f:
            meta = json.load(f)
        self.doc_ids = meta["ids"]
        self.doc_meta = meta["metadata"]

        self.threshold = threshold
        self._buckets = {i: [] for i in range(self.n_clusters)}

        self._ready = True
        log.info(
            f"Ready. docs={len(self.doc_ids)}  k={self.n_clusters}  "
            f"m={self.fcm_m}  θ={self.threshold}"
        )

    # -------------------------------------------------------------------------
    # Embedding + clustering
    # -------------------------------------------------------------------------
    def embed_query(self, query: str) -> np.ndarray:
        """TF-IDF → SVD → L2-normalise."""
        text = re.sub(r"[^a-zA-Z\s]", " ", query.lower())
        text = re.sub(r"\s+", " ", text).strip()
        X_tfidf = self.tfidf.transform([text])
        X_svd = self.svd.transform(X_tfidf)
        return normalize(X_svd, norm="l2")[0]

    def membership(self, vec: np.ndarray) -> np.ndarray:
        """Fuzzy cluster membership for a query embedding vector."""
        x_pca = self.pca.transform(vec.reshape(1, -1))[0]
        m = self.fcm_m
        exp = 2.0 / (m - 1.0)
        diff = x_pca[None, :] - self.centers_pca
        D = np.sqrt((diff ** 2).sum(1))
        D = np.maximum(D, 1e-12)
        D_exp = D ** exp
        U = 1.0 / (D_exp * (1.0 / D_exp).sum() + 1e-10)
        return U

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------
    def retrieve(self, vec: np.ndarray, n_results: int = 5) -> list:
        """
        Return top-n_results documents by cosine similarity.
        This is the "result" that gets cached.
        """
        sims = self.vectors @ vec
        top_idx = np.argpartition(sims, -n_results)[-n_results:]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [
            {
                "id": self.doc_ids[i],
                "similarity": round(float(sims[i]), 4),
                "label": self.doc_meta[i].get("label", "unknown"),
                "preview": self.doc_meta[i].get("text_preview", "")[:120],
            }
            for i in top_idx
        ]

    # -------------------------------------------------------------------------
    # Cache operations
    # -------------------------------------------------------------------------
    def cache_lookup(self, query: str) -> tuple:
        """
        Look up query in the semantic cache.
        Returns (hit, matched_query, similarity, result, vec, membership, dominant_cluster)
        """
        vec = self.embed_query(query)
        mem = self.membership(vec)
        dominant = int(mem.argmax())

        with self._cache_lock:
            bucket = self._buckets.get(dominant, [])
            best_sim = -1.0
            best_entry = None

            for entry in bucket:
                # Cosine similarity: dot product on L2-normalised vectors
                sim = float(entry["vector"] @ vec)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_sim >= self.threshold:
                self._hit_count += 1
                best_entry["hit_count"] += 1
                return (True, best_entry["query"], best_sim,
                        best_entry["result"], vec, mem, dominant)
            else:
                self._miss_count += 1
                return (False, None, best_sim, None, vec, mem, dominant)

    def cache_store(
        self,
        query: str,
        vec: np.ndarray,
        mem: np.ndarray,
        dominant: int,
        result: list,
    ):
        """Store a computed result in the cache."""
        entry = {
            "query": query,
            "vector": vec,
            "membership": mem.tolist(),
            "dominant_cluster": dominant,
            "result": result,
            "timestamp": time.time(),
            "hit_count": 0,
        }
        with self._cache_lock:
            bucket = self._buckets.setdefault(dominant, [])
            bucket.append(entry)
            # LRU eviction at 500 entries per bucket
            if len(bucket) > 500:
                bucket.sort(key=lambda e: (e["hit_count"], e["timestamp"]), reverse=True)
                bucket.pop()

    def cache_stats(self) -> dict:
        with self._cache_lock:
            total = sum(len(b) for b in self._buckets.values())
            total_queries = self._hit_count + self._miss_count
            return {
                "total_entries": total,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(self._hit_count / total_queries, 4)
                if total_queries > 0
                else 0.0,
            }

    def cache_flush(self):
        with self._cache_lock:
            self._buckets = {i: [] for i in range(self.n_clusters)}
            self._hit_count = 0
            self._miss_count = 0


# ---------------------------------------------------------------------------
# Global state singleton
# ---------------------------------------------------------------------------
STATE = AppState()


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    """
    Minimal HTTP handler implementing the three required API endpoints.

    POST /query          — semantic search with cache
    GET  /cache/stats    — cache statistics
    DELETE /cache        — flush cache

    JSON in, JSON out.  Errors return { "error": "..." } with appropriate
    HTTP status codes.
    """

    def log_message(self, format, *args):
        # Route to Python logger instead of stderr
        log.info("%s %s", self.path, " ".join(str(a) for a in args))

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode())

    # -------------------------------------------------------------------------
    # POST /query
    # -------------------------------------------------------------------------
    def _handle_query(self):
        if not STATE._ready:
            return self._send_json({"error": "Service not ready"}, 503)

        body = self._read_body()
        query = body.get("query", "").strip()
        if not query:
            return self._send_json({"error": "Field 'query' is required"}, 400)

        # 1. Check cache
        hit, matched_query, sim, cached_result, vec, mem, dominant = \
            STATE.cache_lookup(query)

        if hit:
            # Cache hit: return stored result
            cluster_info = STATE.cluster_labels.get(str(dominant), {})
            response = {
                "query": query,
                "cache_hit": True,
                "matched_query": matched_query,
                "similarity_score": round(sim, 4),
                "result": cached_result,
                "dominant_cluster": dominant,
                "cluster_label": cluster_info.get("dominant_category", "unknown"),
                "membership_distribution": {
                    str(i): round(float(mem[i]), 4)
                    for i in np.argsort(-mem)[:5]
                },
            }
        else:
            # Cache miss: retrieve and store
            result = STATE.retrieve(vec, n_results=5)
            STATE.cache_store(query, vec, mem, dominant, result)
            cluster_info = STATE.cluster_labels.get(str(dominant), {})
            response = {
                "query": query,
                "cache_hit": False,
                "matched_query": None,
                "similarity_score": round(sim, 4),
                "result": result,
                "dominant_cluster": dominant,
                "cluster_label": cluster_info.get("dominant_category", "unknown"),
                "membership_distribution": {
                    str(i): round(float(mem[i]), 4)
                    for i in np.argsort(-mem)[:5]
                },
            }

        self._send_json(response)

    # -------------------------------------------------------------------------
    # GET /cache/stats
    # -------------------------------------------------------------------------
    def _handle_stats(self):
        if not STATE._ready:
            return self._send_json({"error": "Service not ready"}, 503)
        self._send_json(STATE.cache_stats())

    # -------------------------------------------------------------------------
    # DELETE /cache
    # -------------------------------------------------------------------------
    def _handle_flush(self):
        if not STATE._ready:
            return self._send_json({"error": "Service not ready"}, 503)
        STATE.cache_flush()
        self._send_json({"status": "cache flushed", "timestamp": time.time()})

    # -------------------------------------------------------------------------
    # Routing
    # -------------------------------------------------------------------------
    def do_POST(self):
        if self.path == "/query":
            self._handle_query()
        else:
            self._send_json({"error": f"Not found: POST {self.path}"}, 404)

    def do_GET(self):
        if self.path == "/cache/stats":
            self._handle_stats()
        elif self.path in ("/", "/health"):
            self._send_json({"status": "ok", "ready": STATE._ready})
        else:
            self._send_json({"error": f"Not found: GET {self.path}"}, 404)

    def do_DELETE(self):
        if self.path == "/cache":
            self._handle_flush()
        else:
            self._send_json({"error": f"Not found: DELETE {self.path}"}, 404)


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------
def start_server(host: str = "0.0.0.0", port: int = 8000, threshold: float = 0.85):
    """
    Load all models and start the HTTP server.

    Usage:
      python3 api/app.py [--port PORT] [--threshold THRESHOLD]
    """
    log.info("Initialising Newsgroups Semantic Search Service…")
    STATE.load(threshold=threshold)

    server = HTTPServer((host, port), Handler)
    log.info(f"Server listening on http://{host}:{port}")
    log.info("Endpoints:")
    log.info("  POST   /query")
    log.info("  GET    /cache/stats")
    log.info("  DELETE /cache")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")
        server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Newsgroups Semantic Search API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Cache similarity threshold (0-1, default 0.85)")
    args = parser.parse_args()

    start_server(args.host, args.port, args.threshold)
