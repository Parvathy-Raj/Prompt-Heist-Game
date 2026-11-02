# LLM Red-Team Arena - Gradio app image (CPU only, ~1.5 GB)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    ARENA_DB_PATH=/app/data/arena.sqlite3 \
    ARENA_CACHE_DIR=/app/data/cache \
    ARENA_BACKEND=auto \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# CPU-only torch first (much smaller than the default CUDA wheels), then the rest
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
# Cache mount keeps finished downloads if the build is interrupted on a flaky connection
RUN --mount=type=cache,target=/root/.cache/pip env -u PIP_NO_CACHE_DIR pip install --timeout 60 --retries 10 -r requirements.txt

# Bake the embedding model into the image so the container needs no download at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .
RUN mkdir -p /app/data && useradd --create-home arena && chown -R arena /app
USER arena

EXPOSE 7860
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
