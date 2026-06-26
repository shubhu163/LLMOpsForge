# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LLMOPS_DATA_DIR=/app/data

WORKDIR /app

# System deps kept minimal; faiss-cpu/sentence-transformers are optional extras.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app

# Install core only by default to keep the image lean and the build offline-friendly.
# Add vectors/dashboard extras by overriding the build arg.
ARG EXTRAS="dev"
RUN pip install --upgrade pip && pip install -e ".[${EXTRAS}]"

COPY configs ./configs
COPY documents ./documents
COPY datasets ./datasets
COPY dashboard ./dashboard

RUN mkdir -p /app/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
