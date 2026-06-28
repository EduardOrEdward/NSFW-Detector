# app/main.py
import logging,os,sys
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
#os.environ['TF_AUTOTUNE_THRESHOLD'] ='2'
#os.environ['ORT_LOG_SEVERITY_LEVEL'] ='3'
sys.path.append('/app')

import warnings
warnings.filterwarnings('ignore')
for name in ['tensorflow', 'keras', 'onnxruntime', 'transformers', 'absl', 'h5py']:
    logging.getLogger(name).setLevel(logging.CRITICAL)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.models.opennsfw2_detector import OpenNSFW2Detector
from app.models.nudenet_detector import NudeNetDetector
from app.models.hybrid import HybridDetector
from app.service.cache import CacheService
from app.api.v1 import predict, health

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
        vit = OpenNSFW2Detector()
        
        app.state.detector = HybridDetector(
            detectors=[nudenet, vit],
            strategy=settings.HYBRID_STRATEGY,
            weights={"nudenet_detector": settings.NUDENET_WEIGHT, "OpenNSFW2": settings.OPENNSFW2_WEIGHT}
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