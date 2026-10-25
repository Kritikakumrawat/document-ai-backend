# Document AI backend: FastAPI + OpenCV + Tesseract OCR
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Tesseract OCR engine with English language data (no download needed at runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-lock.txt ./
RUN pip install -r requirements.txt -c requirements-lock.txt

COPY app ./app

# Run as an unprivileged user; storage/ is the only writable folder (mount a volume there).
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/storage/thumbnails \
    && chown -R appuser:appuser /app/storage
USER appuser

EXPOSE 8000
VOLUME ["/app/storage"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
