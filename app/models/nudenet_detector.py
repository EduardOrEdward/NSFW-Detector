# app/models/nudenet_detector.py
import time, logging, io
from typing import Any, Dict, List
import numpy as np
import onnxruntime as ort
from PIL import Image
from nudenet import NudeDetector
from base import BaseDetector

logger = logging.getLogger(__name__)

ZONE_SEVERITY = {
    "F_GENITALIA_EXPOSED": 1.0, "M_GENITALIA_EXPOSED": 1.0,
    "BUTTOCKS_EXPOSED": 0.85, "F_BREAST_EXPOSED": 0.85,
    "ANUS_EXPOSED": 0.9, "ARMPITS_EXPOSED": 0.4,
    "BELLY_EXPOSED": 0.3, "FACE_FEMALE": 0.1,
    "MALE_BREAST_EXPOSED": 0.7, "BUTTOCKS": 0.6,
    "F_BUTT_EXPOSED": 0.85, "M_GENITALIA": 0.95
}
MAX_SEVERITY = max(ZONE_SEVERITY.values(), default=1.0)

class NudeNetDetector(BaseDetector):
    def __init__(self, model_path: str | None = None):
        self._detector = NudeDetector(model_path=model_path)
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opt.inter_op_num_threads = 1 #Making sure  there won't unparalled operations
        sess_opt.intra_op_num_threads = 2 # 2 Threads for two models is quite balanced
        sess_opt.enable_cpu_mem_arena = False #Less alocation, better GC
        sess_opt.add_session_config_entry('session.intra_op.allow_spinning','0') #is more economic-friendly for CPU idle
        self._session = ort.InferenceSession(path_or_bytes=model_path,sess_options=sess_opt,providers=['OpenVINOExecutionProvider'])

    @property
    def name(self) -> str:
        return "nudenet_detector"

    def predict(self, image_bytes: bytes, threshold: float = 0.5,executor=None) -> Dict[str, Any]:
        self._validate_inputs(image_bytes, threshold)
        start = time.perf_counter()
        
        # 🔒 Безопасное получение детекций
        raw_detections = []
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            res = self._detector.detect(np.array(img),)
            raw_detections = res if isinstance(res, list) else []
        except Exception as e:
            logger.warning(f"NudeNet fallback (non-critical): {e}")
            raw_detections = []
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

        max_severity_score = 0.0
        top_zone = "none"
        zones: List[Dict] = []

        # ✅ Безопасная итерация. cls НИКОГДА не выйдет за пределы цикла
        for det in raw_detections:
            if not isinstance(det, dict): continue
            zone_cls = det.get("class", "UNKNOWN")
            score = float(det.get("score", 0.0))
            if not zone_cls or score <= 0: continue

            severity = ZONE_SEVERITY.get(zone_cls, 0.5) * score
            if severity > max_severity_score:
                max_severity_score = severity
                top_zone = zone_cls
            zones.append({"zone": zone_cls, "confidence": round(score, 3)})

        normalized_score = round(min(max_severity_score / MAX_SEVERITY, 1.0), 4)

        return {
            "score": normalized_score,
            "label": "nsfw" if normalized_score >= threshold else "sfw",
            "latency_ms": latency_ms,
            "model_name": self.name,
            "meta": {"detected_zones": zones, "top_zone": top_zone, "raw_max_severity": max_severity_score}
        }