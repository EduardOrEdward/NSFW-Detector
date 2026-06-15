import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


from config import settings
from models.nudenet_detector import NudeNetDetector
from models.vit_detector import VitClassifier
from models.hybrid import HybridDetector

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app:FastAPI):
    logger.info('Starting NSFW Detector API...')
    try:
        logger.info(f'Loading NudeNet from {settings.NUDENET_MODEL_PATH}')
        nudenet = NudeNetDetector(settings.NUDENET_MODEL_PATH)
        logger.info(f'Loading ViT from {settings.VIT_MODEL_PATH}')
        vit = VitClassifier(
            model_path=settings.VIT_MODEL_PATH,
            provider= settings.MODEL_PROVIDER
            )
        logger.info(f'Initializing HybridDetector with strategy: {settings.HYBRID_STRATEGY}')
        hybrid_detector = HybridDetector(
            detectors=[nudenet,vit],
            strategy=settings.HYBRID_STRATEGY,
            weights=settings.hybrid_weights
        )
        app.state.detector = hybrid_detector
        logger.info('Models loaded successfully')
        
    except Exception as e:
        logger.critical(f'Failed to load cause error acused: {e}')
    yield
    logger.info('Shutting down NSFW Detector API...')

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description='The production ready API NSFW Detector(NudeNet+ViT)',
    lifespan=lifespan,
    docs_url='/docs' if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

@app.middleware('http')
async def add_request_middleware(request:Request,call_next):
    request_id = str(uuid.uuid4())
    request.state.requst_id = request_id
    response = await call_next
    response.header['X-Request-ID'] = request_id
    return response

@app.get('/health',tags=['Health'])
async def readiness_check(request:Request):
    if hasattr(request.app.state,'detector'):
        return {
            'status':'ready',
            'models_loaded':True
        }
    else:
        return JSONResponse(status_code=503,content={'status':'not_ready','model_loaded':False})

from api.v1 import predict
app.include_router(predict.router,prefix='/')