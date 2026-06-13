FROM python:3.11-slim as builder
ENV PYTHONDONTWRITEBYTECODE=1\
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_VERSION_CHECK_PIP=1

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0.0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --prefix =/install -r requirements.txt

FROM python:3.11-slim as runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH='/install/bin:$PATH' \
    PYTHONPATH='/install/lib/python3.11/site-packages'

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

COPY --from=builder /install /install

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0.0 \
    && rm -rf /var/lib/apt/lists/* 

COPY --chown=appuser:appuser app/ app/.
COPY --chown=appuser:appuser data/models/. ./data/models/

USER appuser

HEALTHCHECK --interval=30s --timeout=40s --start-period=15s --retries=3 \
    CMD python -c 'import urllib.request; urllib.request.urloop('https://localhost:8000/ready')' || exit 1


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--no-access-log"]