FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build



RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv\
    gcc libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*




RUN mkdir -p /install && \
    cp -r /usr/local/lib/python3.11/site-packages /install/ && \
    cp -r /usr/local/lib /install/
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/install/lib/python3.11/site-packages" \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0
WORKDIR /app
ENV VIRTUAL_ENV=/opt/venv

RUN python -m venv $VIRTUAL_ENV

ENV VENV_PATH="$VIRTUAL_ENV/bin:$VENV_PATH"


RUN useradd -m -s /bin/bash appuser
COPY requirements.txt .
RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache uvicorn==0.24.0 &&\
    pip install --no-cache -r requirements.txt

COPY app/ ./app/
COPY ./data/models ./app/data/models
COPY ./app/main.py ./app/
RUN chown -R appuser:appuser /app
COPY --from=builder /install /install
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --chown=appuser:appuser ./app/main.py ./app/
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser ./data/models ./app/data/models



EXPOSE 8800
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]