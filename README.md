# NSFW Detector

`NSFW Detector` is a FastAPI service for classifying uploaded images as `nsfw` or `sfw`. It combines two independent detectors, aggregates their scores with a configurable strategy, and optionally uses Redis for response caching and rate limiting.

The project is designed as an HTTP API rather than a desktop tool or training pipeline. You send an image to the API, and it returns a normalized score, a final label, latency information, and model metadata.

## Table Of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Response Format](#response-format)
- [Detection Strategies](#detection-strategies)
- [Caching And Rate Limiting](#caching-and-rate-limiting)
- [Operational Notes](#operational-notes)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

## Overview

This service uses a hybrid moderation pipeline:

- `OpenNSFW2` produces an image-level NSFW probability.
- `NudeNet` detects explicit body regions and derives a severity-based score.
- A `HybridDetector` combines both outputs into one final decision.

This approach is useful when you want:

- a single API endpoint for NSFW image screening
- a score instead of only a yes/no result
- metadata from multiple models
- optional Redis-based caching for repeated uploads
- basic request throttling suitable for an API gateway or RapidAPI-style integration

## How It Works

At startup, the application:

1. Loads the NudeNet ONNX model from `data/models/nudenet/nudenet.onnx`.
2. Initializes the OpenNSFW2 detector.
3. Builds a `HybridDetector` with one of three aggregation strategies: `max`, `weighted`, or `voting`.
4. Connects to Redis for caching and rate-limiting support.

When a client uploads an image to `POST /v1/detect`, the service:

1. Reads the uploaded file into memory.
2. Validates that the file is an image and does not exceed the size limit.
3. Checks Redis for a cached result using a SHA-256 hash of the image bytes.
4. If no cache entry exists, runs both detectors.
5. Aggregates the detector outputs into a final score.
6. Returns the score, label, latency, and metadata.

## Features

- Hybrid image moderation using `OpenNSFW2` and `NudeNet`
- FastAPI application with interactive docs at `/docs`
- Health endpoint at `/health/ready`
- Support for `JPEG`, `PNG`, and `WEBP`
- Configurable file size threshold
- Configurable aggregation strategy and per-model weights
- Redis response caching
- Redis-based sliding-window rate limiting
- Optional API key check via request header
- Docker and Docker Compose support

## Project Structure

```text
.
├── app/
│   ├── api/v1/
│   │   ├── health.py
│   │   └── predict.py
│   ├── models/
│   │   ├── base.py
│   │   ├── hybrid.py
│   │   ├── nudenet_detector.py
│   │   ├── opennsfw2_detector.py
│   │   └── preprocessing.py
│   ├── service/
│   │   ├── cache.py
│   │   └── validation.py
│   ├── config.py
│   └── main.py
├── data/
│   └── models/
│       └── nudenet/
│           └── nudenet.onnx
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfile
├── requirements.txt
└── README.md
```

## Requirements

Before running the service, make sure you have:

- Python `3.11`
- `pip`
- Redis, if you want caching and rate limiting enabled
- Docker and Docker Compose, if you want containerized deployment
- Git LFS, if model assets are stored through LFS in your checkout

The Python dependencies are listed in `requirements.txt` and include:

- `fastapi`
- `uvicorn`
- `tensorflow`
- `opennsfw2`
- `nudenet`
- `onnxruntime`
- `redis`
- `Pillow`

## Quick Start

If you want to get the API running quickly on your machine:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Readiness check: [http://localhost:8000/health/ready](http://localhost:8000/health/ready)

If you are on Windows PowerShell, activation is typically:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Local Development

### 1. Clone The Repository

```bash
git clone <your-repo-url>
cd "NSFW Detector"
```

### 2. Pull Model Assets

If your checkout uses Git LFS for model files, make sure the ONNX model is present:

```bash
git lfs pull
```

Verify that this file exists:

```text
data/models/nudenet/nudenet.onnx
```

### 3. Create An Environment File

The app reads configuration from a root-level `.env` file when you run it locally.

Example:

```env
APP_NAME=NSFW Detector API
APP_ENV=development
APP_VERSION=1.0.0
API_PREFIX=/v1
DEBUG=false

NUDENET_MODEL_PATH=data/models/nudenet/nudenet.onnx
MODEL_PROVIDER=CPU
HYBRID_STRATEGY=weighted
NUDENET_WEIGHT=0.4
OPENNSFW2_WEIGHT=0.6
DEFAULT_THRESHOLD=0.5

REDIS_URL=redis://localhost:6379/0
CACHE_TTL=86400
RATE_LIMIT=60
RATE_LIMIT_WINDOW=60

RAPIDAPI_KEY_HEADER=x-rapidapi-key
ALLOWED_RAPIDAPI_KEY=
CORS_ORIGINS=["*"]
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start Redis

If you are running without Docker, start a local Redis instance yourself. A typical local URL is:

```text
redis://localhost:6379/0
```

If Redis is unavailable, the application is written to log the failure and continue without a healthy cache connection. In that mode, caching and rate limiting may not behave as expected.

### 6. Run The API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Test The Health Endpoint

```bash
curl http://localhost:8000/health/ready
```

Expected response:

```json
{
  "status": "ok",
  "detector": "hybrid_detector"
}
```

## Docker Deployment

The repository includes:

- `docker/Dockerfile`
- `docker/docker-compose.yml`

The Compose setup starts:

- the FastAPI API service
- a Redis container for cache and rate limiting

### Run With Docker Compose

From the repository root:

```bash
docker compose -f docker/docker-compose.yml up --build
```

The API should then be available at:

```text
http://localhost:8000
```

### Docker Environment File

The Compose file expects an `.env` file inside the `docker/` directory because it declares:

```yaml
env_file:
  - .env
```

That means you should typically create:

```text
docker/.env
```

Example `docker/.env`:

```env
APP_NAME=NSFW Detector API
APP_ENV=production
NUDENET_MODEL_PATH=data/models/nudenet/nudenet.onnx
HYBRID_STRATEGY=weighted
NUDENET_WEIGHT=0.4
OPENNSFW2_WEIGHT=0.6
REDIS_URL=redis://redis:6379/0
CACHE_TTL=86400
RATE_LIMIT=60
RATE_LIMIT_WINDOW=60
ALLOWED_RAPIDAPI_KEY=
```

### Notes About The Container Setup

- The application process listens on port `8000`.
- The Compose file publishes `8000:8000`.
- The Dockerfile exposes `8800`, but the application command still starts Uvicorn on port `8000`.

If you use the provided Compose file, traffic should still reach the API on `8000`.

## Configuration

The application configuration is defined in `app/config.py`.

### Core Settings

| Variable | Default | Description |
|---|---:|---|
| `APP_NAME` | `NSFW Detector API` | Display name for the FastAPI application |
| `APP_ENV` | `production` | Environment label |
| `APP_VERSION` | `1.0.0` | Application version |
| `API_PREFIX` | `/v1` | API route prefix |
| `DEBUG` | `false` | Enables debug-oriented behavior if used by surrounding tooling |

### Model Settings

| Variable | Default | Description |
|---|---:|---|
| `NUDENET_MODEL_PATH` | `data/models/nudenet/nudenet.onnx` | Path to the NudeNet ONNX model |
| `MODEL_PROVIDER` | `CPU` | Intended compute provider flag |
| `HYBRID_STRATEGY` | `weighted` | Score aggregation mode: `max`, `weighted`, or `voting` |
| `NUDENET_WEIGHT` | `0.4` | Weight used when strategy is `weighted` |
| `OPENNSFW2_WEIGHT` | `0.6` | Weight used when strategy is `weighted` |
| `DEFAULT_THRESHOLD` | `0.5` | Default boundary between `sfw` and `nsfw` |

### Cache And Rate Limiting

| Variable | Default | Description |
|---|---:|---|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `CACHE_TTL` | `86400` | Cache TTL in seconds |
| `RATE_LIMIT` | `60` | Allowed requests within the rate window |
| `RATE_LIMIT_WINDOW` | `60` | Sliding-window duration in seconds |

### API Security

| Variable | Default | Description |
|---|---:|---|
| `RAPIDAPI_KEY_HEADER` | `x-rapidapi-key` | Request header checked by middleware |
| `ALLOWED_RAPIDAPI_KEY` | empty | If set, the service requires the exact key |
| `CORS_ORIGINS` | `["*"]` | Allowed CORS origins |

## API Reference

### `GET /health/ready`

Simple readiness endpoint used to confirm that the app has started and a detector instance is available.

Example request:

```bash
curl http://localhost:8000/health/ready
```

Example response:

```json
{
  "status": "ok",
  "detector": "hybrid_detector"
}
```

### `POST /v1/detect`

Uploads an image for classification.

#### Request

- Method: `POST`
- Content type: `multipart/form-data`
- Form field: `file`
- Accepted image types: `jpeg`, `png`, `webp`
- Maximum file size: `10 MB`

#### Basic `curl` Request

```bash
curl -X POST "http://localhost:8000/v1/detect" \
  -H "accept: application/json" \
  -F "file=@sample.jpg"
```

#### Request With API Key Header

Use this form if `ALLOWED_RAPIDAPI_KEY` is configured:

```bash
curl -X POST "http://localhost:8000/v1/detect" \
  -H "x-rapidapi-key: your-secret-key" \
  -F "file=@sample.jpg"
```

## Usage Examples

### Python Example With `requests`

```python
import requests

url = "http://localhost:8000/v1/detect"

with open("sample.jpg", "rb") as f:
    response = requests.post(url, files={"file": ("sample.jpg", f, "image/jpeg")})

response.raise_for_status()
data = response.json()

print("Label:", data["label"])
print("Score:", data["score"])
print("Latency (ms):", data["latency_ms"])
print("Cached:", data["cached"])
```

### Python Example With API Key

```python
import requests

url = "http://localhost:8000/v1/detect"
headers = {"x-rapidapi-key": "your-secret-key"}

with open("sample.webp", "rb") as f:
    response = requests.post(
        url,
        headers=headers,
        files={"file": ("sample.webp", f, "image/webp")},
    )

response.raise_for_status()
print(response.json())
```

### JavaScript Example With `fetch`

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const response = await fetch("http://localhost:8000/v1/detect", {
  method: "POST",
  body: formData,
});

if (!response.ok) {
  throw new Error(`HTTP ${response.status}`);
}

const data = await response.json();
console.log(data);
```

### JavaScript Example With API Key

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const response = await fetch("http://localhost:8000/v1/detect", {
  method: "POST",
  headers: {
    "x-rapidapi-key": "your-secret-key",
  },
  body: formData,
});

const data = await response.json();
console.log(data);
```

## Response Format

A successful detection response includes:

- `score`: final normalized NSFW probability-like value in the range `0.0` to `1.0`
- `label`: final label, usually `nsfw` or `sfw`
- `latency_ms`: detector latency in milliseconds
- `cached`: whether a cached result was returned
- `meta`: per-model scores and strategy information
- detected body-zone metadata from NudeNet

Example response:

```json
{
  "score": 0.7824,
  "label": "nsfw",
  "latency_ms": 143.82,
  "cached": false,
  "meta": {
    "sources": {
      "nudenet_detector": 0.71,
      "opennsfw2_detector": 0.83
    },
    "strategy": "weighted"
  },
  "detecter_zones": [
    {
      "zone": "F_BREAST_EXPOSED",
      "confidence": 0.941
    },
    {
      "zone": "BUTTOCKS_EXPOSED",
      "confidence": 0.874
    }
  ]
}
```

### Response Semantics

#### `score`

The final score is the aggregated output from the hybrid detector.

- Closer to `0.0` means the image is more likely safe.
- Closer to `1.0` means the image is more likely explicit.

#### `label`

The label is derived by comparing the final score to the active threshold. In the current implementation, the route uses a threshold of `0.5`.

#### `meta.sources`

This object carries per-model scores used by the aggregator. It is useful when you want to:

- inspect disagreements between models
- tune aggregation weights
- log model behavior for monitoring

#### Zone Metadata

NudeNet can contribute zone-level information such as exposed body regions. This makes the response more interpretable than a pure single-number classifier.

## Detection Strategies

The hybrid detector supports three aggregation strategies.

### `weighted`

This is the default strategy. Each model score is multiplied by its configured weight and normalized by the sum of active weights.

Use this when:

- you trust one model more than the other
- you want a smoother combined score
- you want consistent control over sensitivity

Example:

- NudeNet score: `0.60`
- OpenNSFW2 score: `0.90`
- Weights: `0.4` and `0.6`

Result:

```text
(0.60 * 0.4 + 0.90 * 0.6) / (0.4 + 0.6) = 0.78
```

### `max`

This strategy uses the largest score returned by any detector.

Use this when:

- you want a more conservative moderation rule
- a single strong NSFW signal should dominate
- false negatives are more costly than false positives

Example:

```text
max(0.60, 0.90) = 0.90
```

### `voting`

This strategy counts how many detectors classify the image as NSFW at the threshold and divides that count by the number of detectors.

Use this when:

- you want a consensus-style rule
- you care more about agreement than raw score magnitude

Example with two detectors:

- Detector A says NSFW
- Detector B says SFW

Result:

```text
1 / 2 = 0.5
```

## Caching And Rate Limiting

### Response Caching

The cache service hashes the raw uploaded bytes with SHA-256 and uses that value as a Redis key. If the exact same file is submitted again before the TTL expires, the API can return the stored result without re-running inference.

Benefits:

- lower latency on repeated uploads
- lower model compute usage
- more predictable response times for duplicate content

### Sliding-Window Rate Limiting

The middleware also uses Redis for rate limiting. Each client is identified by:

- the API key header, if present
- otherwise the request IP address

The implementation stores timestamps in a Redis sorted set and removes entries that are older than the configured window.

This is useful for:

- public API endpoints
- RapidAPI-style packaging
- abuse reduction

## Operational Notes

### Interactive Docs

Once the server is running, FastAPI exposes Swagger UI at:

```text
http://localhost:8000/docs
```

This is the easiest way to:

- inspect request and response schemas
- try uploads from the browser
- validate headers and status codes

### Logging

The application logs:

- startup and shutdown events
- cache hits and misses
- detector failures
- inference errors
- rate-limit violations

### Image Preprocessing

Before model inference, the hybrid detector may resize large images so the longest side is capped to improve inference cost and memory behavior.

### Health Checks

The Docker image includes a health check hitting:

```text
http://localhost:8000/health/ready
```

## Troubleshooting

### The API Fails During Startup

Check the following:

- the ONNX model file exists at `data/models/nudenet/nudenet.onnx`
- all Python dependencies installed successfully
- TensorFlow and ONNX Runtime support your environment
- Redis is reachable if you expect cache connectivity

### `401 Invalid or missing x-rapidapi-key`

This means `ALLOWED_RAPIDAPI_KEY` is configured and the request did not include the correct header value.

Fix:

- send the `x-rapidapi-key` header
- or unset `ALLOWED_RAPIDAPI_KEY` for local development

### `429 Rate limit exceeded`

This means the request volume exceeded the configured sliding window.

Fix:

- increase `RATE_LIMIT`
- increase `RATE_LIMIT_WINDOW`
- retry after the current rate window passes

### `413 File too large`

The uploaded file is above the `10 MB` validation limit.

Fix:

- compress the image
- resize the image before upload
- adjust the validation logic in the service if your deployment requires larger files

### `415 Unsupported file type`

Only `jpeg`, `png`, and `webp` are accepted.

Fix:

- convert the image to a supported format
- verify that the uploaded bytes are actually an image

### Cache Does Not Work

Check:

- Redis is running
- `REDIS_URL` is correct
- the app logs show a successful Redis connection

If Redis cannot be reached, the service logs the failure and disables healthy cache behavior.

## Limitations

This repository is an inference API, not a full moderation platform. Keep these boundaries in mind:

- It only handles image uploads, not video, text, or multimodal moderation.
- The final label quality depends on the underlying models and threshold choices.
- Different moderation policies may require different weights or a stricter threshold.
- NudeNet zone detection is useful metadata, but it should not be treated as a legal or policy-grade explanation by itself.
- There is currently minimal repository documentation beyond the code itself, so operational conventions may need to be standardized for team use.

## Summary

`NSFW Detector` provides a practical starting point for building an image moderation API with:

- hybrid model inference
- configurable aggregation
- Redis-backed caching
- rate limiting
- FastAPI-based deployment

If you want to extend it further, common next steps are:

- add automated tests for the API contract
- add benchmark scripts for latency and throughput
- add batch inference endpoints
- add structured observability and metrics
