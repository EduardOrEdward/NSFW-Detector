import logging
import time
from typing import Dict, Any
from fastapi import APIRouter,UploadFile,File,HTTPException,Query,Depends,Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field

router = APIRouter()
logger = logging.getLogger(__name__)
ALLOWED_MIME_NAMES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10*1024 *1024

class DetectionResponse(BaseModel):
    score:float = Field(...,ge=0.0,le=1.0,description='NSFW Probability -> [0,1]')
    label:str = Field(...,description='NSFW/SFW')
    latency_ms:float = Field(...,description='Model processing time in miliseconds')
    model_name:str = Field(...,description='Active detector pipeline name')
    meta:Dict[str,Any] = Field(default_factory=dict,description='Sources, zones, fallback info, etc.')
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