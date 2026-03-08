# Use official Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for numpy/scipy
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching — only reinstalls
# if requirements.txt changes, not on every code change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code into the container
COPY . .

# Create necessary directories
RUN mkdir -p data embeddings/vector_store data/clusters

# Run the full pipeline at build time (corpus + embeddings + clustering)
# so the container starts instantly with all data ready
RUN python -c "\
import sys; sys.path.insert(0, '.'); \
from scripts.generate_corpus import generate_corpus; \
from embeddings.part1_embed import run_part1; \
from analysis.part2_cluster import run_part2; \
generate_corpus(n_docs=18000, output_dir='data'); \
run_part1(); \
run_part2(); \
print('Pipeline complete')"

# Expose port 8000
EXPOSE 8000

# Start uvicorn server
# --host 0.0.0.0 makes it accessible outside the container
CMD ["uvicorn", "api.app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
