#!/usr/bin/env python3
"""
build_and_run.py
================
Single-entry-point script that runs the full pipeline:
  1. Generate corpus
  2. Build TF-IDF embeddings + vector store  (Part 1)
  3. Fuzzy clustering                         (Part 2)
  4. Start the HTTP service                   (Part 4)
     (Part 3 cache is initialised inside the service)

Usage:
  # Full pipeline + start server (default port 8000)
  python3 build_and_run.py

  # Skip corpus/embedding rebuild (use cached outputs)
  python3 build_and_run.py --skip-build

  # Custom port + threshold
  python3 build_and_run.py --port 8080 --threshold 0.80

  # With FastAPI/uvicorn (if installed):
  uvicorn api.app_fastapi:app --host 0.0.0.0 --port 8000
"""

import argparse
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


def main():
    parser = argparse.ArgumentParser(description="20 Newsgroups Semantic Search")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip corpus generation and embedding (use existing data)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Cache similarity threshold (0-1)")
    args = parser.parse_args()

    embed_dir = BASE_DIR / "embeddings"
    cluster_dir = BASE_DIR / "data" / "clusters"

    if not args.skip_build:
        # Part 0: Generate corpus
        if not (BASE_DIR / "data" / "corpus.json").exists():
            print("\n[Step 0] Generating synthetic corpus…")
            from scripts.generate_corpus import generate_corpus
            generate_corpus(n_docs=18000, output_dir=str(BASE_DIR / "data"))
        else:
            print("[Step 0] Corpus already exists, skipping generation")

        # Part 1: Embeddings
        if not (embed_dir / "embeddings.npy").exists():
            print("\n[Step 1] Building TF-IDF embeddings + vector store…")
            from embeddings.part1_embed import run_part1
            run_part1()
        else:
            print("[Step 1] Embeddings already exist, skipping")

        # Part 2: Clustering
        if not (cluster_dir / "W_norm.npy").exists():
            print("\n[Step 2] Running fuzzy clustering…")
            from analysis.part2_cluster import run_part2
            run_part2()
        else:
            print("[Step 2] Cluster data already exists, skipping")
    else:
        print("Skipping build (--skip-build)")

    # Part 4: Start service
    print(f"\n[Step 4] Starting service on {args.host}:{args.port} (θ={args.threshold})")
    print(f"  POST   http://{args.host}:{args.port}/query")
    print(f"  GET    http://{args.host}:{args.port}/cache/stats")
    print(f"  DELETE http://{args.host}:{args.port}/cache")

    from api.app import start_server
    start_server(host=args.host, port=args.port, threshold=args.threshold)


if __name__ == "__main__":
    main()
  
