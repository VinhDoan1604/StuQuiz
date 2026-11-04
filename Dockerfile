# Backend StudyRAG (FastAPI). Build: docker build -t studyrag-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    TIKTOKEN_CACHE_DIR=/models/tiktoken

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng curl \
    && rm -rf /var/lib/apt/lists/*

# torch bản CPU trước để FlagEmbedding không kéo bản CUDA (~2GB)
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake sẵn file BPE của tiktoken (runtime không cần mạng để đếm token)
RUN mkdir -p /models/tiktoken /models/hf \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./
COPY scripts ./scripts

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Chạy migration rồi khởi động. 1 worker vì model BGE-M3/reranker nằm trong RAM
# của từng tiến trình; scale ngang bằng nhiều container. --proxy-headers để
# rate limit/log thấy IP thật khi đứng sau reverse proxy.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
