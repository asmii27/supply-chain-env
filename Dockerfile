# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

# Copy entire project
COPY . .

# HF Spaces runs as non-root
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Required env vars (set in HF Space secrets)
ENV PYTHONPATH=/app
ENV PORT=7860

# Expose for HF Spaces
EXPOSE 7860

# Start the API server (inference.py is run separately by judges)
CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
