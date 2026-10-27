FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FISHKEEPING_DB_PATH=/app/data/fishkeeping.sqlite3 \
    FISHKEEPING_HOST=0.0.0.0 \
    FISHKEEPING_PORT=8765

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . \
    && groupadd --gid 10001 fishkeeping \
    && useradd --uid 10001 --gid 10001 --no-create-home fishkeeping \
    && mkdir -p /app/data \
    && chown -R 10001:10001 /app/data

USER 10001:10001
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/readyz', timeout=3).close()"]
CMD ["fishkeeping", "serve"]
