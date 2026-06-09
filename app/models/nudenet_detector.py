import io
import time
import numpy as np
from PIL import Image
from base import BaseDetector 
import logging
from typing import Dict, Any, List
from nudenet import NudeDetector

logger = logging.getLogger(__name__)

ZONE_SEVERITY = {
    "F_GENITALIA_EXPOSED": 1.0, "M_GENITALIA_EXPOSED": 1.0,
    "BUTTOCKS_EXPOSED": 0.85, "F_BREAST_EXPOSED": 0.85,
    "ANUS_EXPOSED": 0.9, "ARMPITS_EXPOSED": 0.4,
    "BELLY_EXPOSED": 0.3, "FACE_FEMALE": 0.1,
    "MALE_BREAST_EXPOSED": 0.7, "BUTTOCKS": 0.6,
    "F_BUTT_EXPOSED": 0.85, "M_GENITALIA": 0.95
}

MAX_SEVERITY = max(ZONE_SEVERITY.values(),default=1.0)

class NudeNetDetector(BaseDetector):
    def __init__(self,model_path:str|None=None):
        self.model_path = model_path
        self._detector = NudeDetector()
        logger.info('Nude')
    @property
    def name(self)->str:
        return 'nudenet_detector'
    def predict(self,image_bytes:bytes,threshold:float=0.7) ->Dict[str,Any]:
        self._validate_inputs(image_bytes=image_bytes,threshold=threshold)
        start = time.perf_counter()
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_array = np.array(img)
            detections = self._detector.detect(img_array)
            if not detections:
                detections=[]
        except Exception as e:
            logger.error(f'NudeNet infrence failed: {e} ',exc_info=True)
            raise RuntimeError('Nudenet model failed to load: ') from e
        finally:
            end = time.perf_counter()
            latency_ms = round((end-start)*1000,2)
        max_severity_score = 0.0
        top_zone = 'none'
        zones:List[Dict] = []
        for det in detections:
            cls = det.get('class','UNKNOWN')
            score = det.get('score',0.0)
            severity = ZONE_SEVERITY.get(cls,0.5) * score
            if severity > max_severity_score:
                max_severity_score=severity
                top_zone=cls
        zones.append({'zone':cls,'confidence':round(score,3)})
        normalized_score = round(min(max_severity_score/MAX_SEVERITY,1.0),2)
        
        return {
            "score": normalized_score,
            "label": "nsfw" if normalized_score >= threshold else "sfw",
            "latency_ms": latency_ms,
            "model_name": self.name,
            "meta": {
                "detected_zones": zones,
                "top_zone": top_zone,
                "raw_max_severity": max_severity_score
            }
        }