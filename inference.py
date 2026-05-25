from .preprocessing import preprocess_image
import time
import numpy as np
import logging
from pathlib import Path
import onnxruntime as ort
from typing import Dict, List
from nudenet import NudeDetector
from PIL import Image
import io
logger = logging.getLogger(__name__)


ZONE_SEVERITY = {
    "F_GENITALIA_EXPOSED": 1.0, "M_GENITALIA_EXPOSED": 1.0,
    "BUTTOCKS_EXPOSED": 0.85, "F_BREAST_EXPOSED": 0.85,
    "ANUS_EXPOSED": 0.9, "ARMPITS_EXPOSED": 0.4,
    "BELLY_EXPOSED": 0.3, "FACE_FEMALE": 0.1
}

MAX_SEVERITY = max(ZONE_SEVERITY.values(),default=1.0)

class NudeNetDetector:
    def __init__(self,model_path:None|str=None):
        self.detector = NudeDetector(model_path=model_path)
    def predict(self,image_bytes:bytes,threshold:float= 0.4) -> Dict:
        start = time.perf_counter()
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            detection = self.detector.detect(np.array(img))
        except Exception as e:
            logger.error(f'Inference failed : {e}')
            raise RuntimeError('Model inference error') from e
        latency_ms = round((time.perf_counter() - start),4)
        max_severity_score = 0.0
        top_zone = 'None'
        zone:List[Dict]=[]
        for det in detection:
            cls = det.get('class','UNKOWN')
            score= det.get('score',0.0)
            severity = ZONE_SEVERITY.get(cls,0.5) * score
            if severity > max_severity_score:
                max_severity_score = severity
                top_zone = cls
            zone.append({'zone':cls,'confidence':round(score,3)})
        normalization_score = round(min(max_severity_score/MAX_SEVERITY,1.0),4)
        label = 'nsfw' if normalization_score >= 0.5 else 'sfw'
        return {
            'nsfw_score':normalization_score,
            'label':label,
            'threshold':threshold,
            'latency_ms':latency_ms,
            'detected_zones':zone,
            'model_version':'nudenet_v3'
        }
        