# app/main.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings
from models.vit_detector import VitClassifier
from models.nudenet_detector import NudeNetDetector
from models.hybrid import HybridDetector
from service.cache import CacheService
from api.v1 import predict, health

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

class RapidAPIMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. Auth
        if settings.ALLOWED_RAPIDAPI_KEY:
            api_key = request.headers.get(settings.RAPIDAPI_KEY_HEADER)
            if api_key != settings.ALLOWED_RAPIDAPI_KEY:
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing x-rapidapi-key"})
        
        # 2. Rate Limit
        if request.app.state.cache and request.app.state.cache._is_ready:
            client_id = request.headers.get(settings.RAPIDAPI_KEY_HEADER) or request.client.host
            allowed = await request.app.state.cache.check_rate_limit(
                identifier=client_id,
                limit=settings.RATE_LIMIT,
                window=settings.RATE_LIMIT_WINDOW
            )
            if not allowed:
                return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        
        return await call_next(request)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Initializing Hybrid NSFW Pipeline...")
    try:
        nudenet = NudeNetDetector(model_path=settings.NUDENET_MODEL_PATH)
        vit = VitClassifier(model_path=settings.VIT_MODEL_PATH, provider=settings.MODEL_PROVIDER)
        
        app.state.detector = HybridDetector(
            detectors=[nudenet, vit],
            strategy=settings.HYBRID_STRATEGY,
            weights={"nudenet_detector": settings.NUDENET_WEIGHT, "vit_classifier": settings.VIT_WEIGHT}
        )
        
        app.state.cache = CacheService(redis_url=settings.REDIS_URL, ttl=settings.CACHE_TTL)
        await app.state.cache.connect()
        logger.info("✅ Models loaded & Redis connected. Ready.")
    except Exception as e:
        logger.error(f"The Initialization failed: {e}", exc_info=True)
        raise
    yield
    if app.state.cache:
        await app.state.cache.disconnect()
    logger.info("The Shutdown complete.")

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="High-accuracy Hybrid NSFW Image Detection API. Optimized for RapidAPI.",
    docs_url="/docs",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RapidAPIMiddleware)

app.include_router(predict.router, prefix=settings.API_PREFIX)
app.include_router(health.router, prefix="/health")