import logging,time
#import time
from typing import Dict, Any,Optional, Literal
from fastapi import APIRouter,UploadFile,File,HTTPException,Request
from pydantic import BaseModel,Field
from app.config import settings
from app.service.validation import validate_image

router = APIRouter(tags=['Detection'])

logger = logging.getLogger(__name__)
class MetaInfo(BaseModel):
    strategy: str = Field(..., description="Aggregator algorithm (max/weighted/voting)")
    sources: Dict[str, float] = Field(..., description="Scores of the models separetly")

class DetectionSource(BaseModel):
    nudenet_detector:Optional[float] = Field(None,description='NudeNet score -> [0,1]')
    opennsfw2_detector:Optional[float] = Field(None,description='OpeNSFW2 score -> [0,1]')
    

class DetectionResponse(BaseModel):
    score:float = Field(...,ge=0.0,le=1.0,description='NSFW Probability -> [0,1]')
    label:str = Field(...,description='NSFW/SFW')
    latency_ms:float = Field(...,description='Model processing time in miliseconds')
    #strategy:str = Field(...,description='The strategy we use for hybrid model')
    cached:bool = Field(...,description='Cached or not')
    meta:MetaInfo = Field(...,description='Sources, zones, fallback info, etc.')
    detected_zones:Optional[list] = Field(None,description='The detected zones')
    
    #model_name:str = Field(...,description='Active detector pipeline name')
    
@router.post('/detect',response_model=DetectionResponse,summary='Checking the image if that NSFW or NOT',description='Loading the image(WEBP,PNG,JPEG, max size is 10MB) - returning the probability if the image is NSFW')
async def detect_nsfw(
    request:Request,
    file:UploadFile = File(...,description='Sent file')
) -> Dict[str,Any]:
    '''
    The Main endpoint:
    1.Reading file into memory
    2.Validates size and finename
    3.Checking Redis-cache
    4.Loading hybrid model(unless file cached)
    5.Saving results into cache
    '''
    start = time.perf_counter()
    request_id = getattr(request.state,'request_id','unknown')
    logger.info(f'[{request_id}] Received detection for request: {file.filename}')
    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.error(f'[{request_id}] Failed to read: {e}')
        raise HTTPException(status_code=400,detail='Failed to read uploaded file')
    validate_image(file_bytes=file_bytes,filename=file.filename or 'unknown')
    
    cache = getattr(request.app.state,'cache',None)
    if cache and cache.is_healthy:
        cached_result = await cache.get(file_bytes)
        if cached_result:
            logger.info(f'[{request_id}] Cache HIT. Returning cache result')
            cached_result['cached'] = True
            cached_result['latency']=round((time.perf_counter()-start)*1000,4)
            return cached_result
    logger.info(f'[{request_id}] Cache MISS. Running Hybrid model inference...')
    detector = getattr(request.app.state,'detector',None)
    if not detector:
        logger.critical(f'[{request_id}] Detector not initialized in app.state')
        raise HTTPException(
            status_code=503,
            detail='Detection service is not ready'
        )
    try:
        result = await detector.predict(file_bytes,threshold=0.5)
    except Exception as e:
        logger.error(f'[{request_id}] Inference failed: {e}',exc_info=True)
        raise HTTPException(
            status_code=500,
            detail='Internal error during image processing'
        )
    if cache and cache.is_healthy:
        import asyncio
        asyncio.create_task(cache.set(file_bytes,result))
    
    response_data = {
        'score':result['score'],
        'label':result['label'],
        'latency_ms':result['latency_ms'],
        'cached':cache is not None,
        'meta':{
            'sources':{
                'nudenet_detector':result['meta']['sources'].get('nudenet_detector') or 0.0,
                'opennsfw2_detector':result['meta']['sources'].get('vit_clasifier') or 0.0
            },
                'strategy':settings.HYBRID_STRATEGY
                },
        'detecter_zones':result['meta'].get('detected_zones',[])
    }
    logger.info(f"[{request_id}] Detection complete. Label: {response_data['label']}, Score: {response_data['score']}, Latency_ms: {response_data['latency_ms']}ms")
    return response_data

'''
def _get_detector(request:Request):
    """Инжектит гибридный детектор из app.state (загружается в lifespan)."""
    detector = getattr(request.app.state,'detector',None)
    if detector is None:
        raise HTTPException(status_code=503,detail='Model pipeline is not initialized')
    return detector
@router.post('/detect',response_model=DetectionResponse,tags=['Detection'])
async def detect_image(
    request:Request,
    file:UploadFile = File(...,description='Image to analyze (JPG, PNG, WEBP)'),
    threshold:float = Query(0.5,ge=0.0,le=1,description='NSFW Detection threshold'),
    detector=Depends(_get_detector)
):
    start_time = time.perf_counter()
    
    if file.content_type and file.content_type not in ALLOWED_MIME_NAMES:
        raise HTTPException(status_code=400,detail=f'Unsupported file type: {file.content_type}')
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400,detail='Empty file provided')
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413,detail=f'File exceed the {MAX_FILE_SIZE//1024}KB limit')
    try:
        result = await detector.predict(image_bytes=contents,threshold=threshold)
    except Exception as e:
        logger.error(f'Detection pipeline failed: {e}',exc_info=True)
        raise HTTPException(status_code=500,detail='Internal model inference error')
    endpoint_latency_ms = round((time.perf_counter()-start_time)*1000,2)
    
    response_data = {
        "score": result["score"],
        "label": result["label"],
        "latency_ms": result["latency_ms"],  
        "model_name": result["model_name"],
        "meta": result["meta"]
    }
    
    headers = {
        "X-Endpoint-Latency": str(endpoint_latency_ms),
        "X-Request-ID": request.headers.get("X-Request-ID", "unknown"),
    }
    return JSONResponse(content=response_data, headers=headers)
'''
