FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8780 \
    CHECKPOINT_BACKEND=sqlite \
    CHECKPOINT_SQLITE_PATH=/app/data/checkpoints.db \
    JOB_DB_PATH=/app/data/jobs.db

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY agent_framework ./agent_framework
COPY domains ./domains
COPY services ./services
COPY scripts ./scripts
COPY tests ./tests

RUN pip install -e ".[api,dev,supervisor,a2a]" && pip install -e domains/

RUN mkdir -p /app/data

EXPOSE 8780

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8780/health')" || exit 1

CMD ["python", "scripts/run_api.py"]
