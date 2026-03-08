"""
app_fastapi.py — FastAPI version of the Newsgroups Semantic Search Service
===========================================================================

This module requires:
  pip install fastapi uvicorn pydantic

Start with:
  uvicorn api.app_fastapi:app --host 0.0.0.0 --port 8000

The business logic is IDENTICAL to api/app.py; only the routing layer differs.
The AppState singleton from app.py is reused directly.
"""

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import sys
import json
import time
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from api.app import AppState  # reuse all model/cache logic

# ---------------------------------------------------------------------------
# Application state (loaded at startup via lifespan)
# ---------------------------------------------------------------------------
STATE = AppState()

if FASTAPI_AVAILABLE:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load all models at startup (FastAPI lifespan context)."""
        STATE.load(threshold=0.85)
        yield
        # (cleanup on shutdown if needed)

    app = FastAPI(
        title="20 Newsgroups Semantic Search",
        description=(
            "Semantic search over the 20 Newsgroups corpus with fuzzy clustering "
            "and a cluster-partitioned semantic cache."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # -------------------------------------------------------------------------
    # Request/Response models
    # -------------------------------------------------------------------------
    class QueryRequest(BaseModel):
        query: str

    class QueryResponse(BaseModel):
        query: str
        cache_hit: bool
        matched_query: str | None
        similarity_score: float
        result: list
        dominant_cluster: int
        cluster_label: str
        membership_distribution: dict

    class CacheStats(BaseModel):
        total_entries: int
        hit_count: int
        miss_count: int
        hit_rate: float

    # -------------------------------------------------------------------------
    # Endpoints
    # -------------------------------------------------------------------------
    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(req: QueryRequest):
        """
        Embed the query, check the semantic cache, and return results.

        On a cache hit: returns the stored result with similarity_score.
        On a cache miss: retrieves from the vector store, stores in cache, returns.
        """
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        hit, matched_query, sim, cached_result, vec, mem, dominant = \
            STATE.cache_lookup(req.query)

        cluster_info = STATE.cluster_labels.get(str(dominant), {})
        cluster_label = cluster_info.get("dominant_category", "unknown")
        membership_dist = {
            str(i): round(float(mem[i]), 4)
            for i in np.argsort(-mem)[:5]
        }

        if hit:
            return QueryResponse(
                query=req.query,
                cache_hit=True,
                matched_query=matched_query,
                similarity_score=round(sim, 4),
                result=cached_result,
                dominant_cluster=dominant,
                cluster_label=cluster_label,
                membership_distribution=membership_dist,
            )
        else:
            result = STATE.retrieve(vec, n_results=5)
            STATE.cache_store(req.query, vec, mem, dominant, result)
            return QueryResponse(
                query=req.query,
                cache_hit=False,
                matched_query=None,
                similarity_score=round(sim, 4),
                result=result,
                dominant_cluster=dominant,
                cluster_label=cluster_label,
                membership_distribution=membership_dist,
            )

    @app.get("/cache/stats", response_model=CacheStats)
    async def cache_stats():
        """Return current cache statistics."""
        return CacheStats(**STATE.cache_stats())

    @app.delete("/cache")
    async def flush_cache():
        """Flush the entire cache and reset statistics."""
        STATE.cache_flush()
        return {"status": "cache flushed", "timestamp": time.time()}

    @app.get("/health")
    async def health():
        return {"status": "ok", "ready": STATE._ready}

else:
    # Graceful degradation: if FastAPI isn't installed, raise a helpful error
    print("FastAPI not available. Install with: pip install fastapi uvicorn")
    print("Falling back to stdlib server in api/app.py")
    raise ImportError(
        "FastAPI not installed. Run:\n"
        "  pip install fastapi uvicorn\n"
        "Or use the stdlib version:\n"
        "  python3 api/app.py"
    )
