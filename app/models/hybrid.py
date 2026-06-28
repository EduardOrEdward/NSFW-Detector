import asyncio,io
import logging
from typing import Dict, Any, List, Literal
from app.models.base import BaseDetector
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
logger = logging.getLogger(__name__)

MAX_INPUT_SIZE = 1080

class HybridDetector(BaseDetector):
    def __init__(self,
                 detectors:List[BaseDetector],
                 strategy:Literal['max','weighted','voting']='weighted',
                 weights: Dict[str,float]|None=None,
                 fallback_detector: BaseDetector|None = None):
        self.detectors = detectors
        self.strategy = strategy
        self.fallback_detector = fallback_detector
        self._executor = ThreadPoolExecutor(max_workers=len(detectors),thread_name_prefix='nsfw_inf')
        if weights:
            self.weights = weights
        else:
            self.weights =  {d.name:1.0/len(detectors) for d in detectors}
        
        expected_names = {d.name for d in detectors}
        if set(self.weights.keys()) != expected_names:
            raise ValueError(f"Weight keys {list(self.weights.keys())} must exactly match detector names {expected_names}")
    @property
    def name(self) ->str:
        return 'hybrid_detector'
    
    def _aggregate(self, results: List[Dict], threshold: float) -> float:
        if not results:
            return 0.0
            
        scores = {r["model_name"]: r["score"] for r in results}
        logger.debug(f"Aggregating scores: {scores} | Strategy: {self.strategy}")

        if self.strategy == "max":
            return max(scores.values())
            
        if self.strategy == "weighted":
            active_weights = [self.weights.get(n, 0.0) for n in scores.keys()]
            w_sum = sum(active_weights)
            if w_sum == 0: return max(scores.values())
            return sum(s * w for s, w in zip(scores.values(), active_weights)) / w_sum
            
        if self.strategy == "voting":
            nsfw_count = sum(1 for s in scores.values() if s >= threshold)
            return nsfw_count / len(scores)
            
        return max(scores.values())
    @staticmethod
    def _fast_resize(image_bytes:bytes) ->bytes:
        img = Image.open(io.BytesIO(image_bytes))
        if max(img.size) < MAX_INPUT_SIZE:
            return image_bytes
        else:
            img.thumbnail((MAX_INPUT_SIZE,MAX_INPUT_SIZE),Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf,format='JPEG',quality=85) #JPEG is easier for ONNX models
            return buf.getvalue()
    @staticmethod
    def _merge_zones(results:List[Dict]) -> List[Dict]:
        zones = []
        for r in results:
            meta = r.get('meta',{})
            if 'detected_zones' in meta:
                zones.extend(meta['detected_zones'])
        return zones
    async def predict(self,images_bytes:bytes,threshold:float=0.5) -> Dict[str, Any]:
        self._validate_inputs(image_bytes=images_bytes,threshold=threshold)
        start = time.perf_counter()
        optimized_bytes = await asyncio.to_thread(self._fast_resize,images_bytes)
        tasks = [asyncio.to_thread(det.predict,optimized_bytes,threshold,executor=self._executor) for det in self.detectors]
        results:List[Dict[str,Any]] = []
        errors:List[Dict[str,str]] = [] 
        
        #tasks = [asyncio.to_thread(det.predict,images_bytes,threshold) for det in self.detectors]
        raw_outputs = await asyncio.gather(*tasks,return_exceptions=True)
        
        for det, out in zip(self.detectors,raw_outputs):
            if isinstance(out, Exception):
                logger.warning(f"Detector {det.name} failed: {out}")
                errors.append({"detector": det.name, "error": str(out)})
            else:
                results.append(out)
        if not results:
            if self.fallback_detector:
                logger.critical('All primary detectors failed. Switched to fallback one')
                try:
                    fallback_res = await asyncio.to_thread(self.fallback_detector,images_bytes,threshold)
                    results.append(fallback_res)
                except Exception as e:
                    logger.error(f'Fallback detector also failed: {e}')
                    raise RuntimeError('ALL detectors failed')
            else:
                raise RuntimeError('ALL failed and there\'s no fallback detector provided ')
        final_score = self._aggregate(results,threshold)
        latency_ms = round((time.perf_counter()-start)*1000,2)
        
        return {
            "score": round(final_score, 4),
            "label": "nsfw" if final_score >= threshold else "sfw",
            "latency_ms": latency_ms,
            "model_name": self.name,
            "meta": {
                "strategy": self.strategy,
                "sources": {r["model_name"]: r["score"] for r in results},
                "errors": errors,
                "detected_zones": self._merge_zones(results)
            }
        }