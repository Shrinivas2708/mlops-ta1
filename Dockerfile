# ---------- Stage 1: builder (deps compiled here, discarded later) ----------
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements-serve.txt .
# CPU-only torch keeps the image small (no CUDA libs on an edge box)
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-serve.txt

# ---------- Stage 2: runtime (what actually ships to the edge) ----------
FROM python:3.11-slim
# opencv-python-headless needs no GUI libs; just libgomp for torch
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
WORKDIR /srv
COPY app/ app/
COPY src/ src/
COPY models/autoencoder.pt models/autoencoder.pt

# Non-root user: standard hardening for devices sitting on a factory network
RUN useradd -m runner && chown -R runner /srv
USER runner

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
