import time
import logging
import io
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
from typing import Any, Dict
import onnxruntime as ort
from base import BaseDetector

logger = logging.getLogger(__name__)

class VitClassifier(BaseDetector):
    def __init__(self,model_path:str,provider:str='CPU'):
        self.model_path = model_path
        self._processer = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection_26",use_fast=True)
        if provider.upper() == "GPU" and 'CUBAExecutiveProvider' in ort.get_available_providers():
            providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            sess_opt = ort.SessionOptions()
            sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            providers = ["CPUExecutionProvider"]
            sess_opt = ort.SessionOptions()
            sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_opt.inter_op_num_threads = 1
            sess_opt.intra_op_num_threads=2
            sess_opt.enable_cpu_mem_arena = True
            sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self._session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=sess_opt
        )
        self._input_name = self._session.get_inputs()[0].name 
        self._output_name = self._session.get_outputs()[0].name
        dummy = np.zeros((1,3,224,224),dtype=np.float32)
        self._session.run(None,{self._input_name:dummy})
        logger.info(f"ViTClassifier initialized. Providers: {self._session.get_providers()}")
    @property
    def name(self) ->str:
        return "vit_classifier"
    def predict(self,image_bytes:bytes,threshold:float=0.5,) -> Dict[str,Any]:
        self._validate_inputs(image_bytes,threshold=threshold)
        start = time.perf_counter()
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB").load()
            img_pil = Image.fromarray(np.array(img))
            
            # 🔥 Fast-процессор требует return_tensors="pt", потом конвертируем в numpy
            inputs = self._processer(images=img_pil, return_tensors="pt")
            pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
            
            outputs = self._session.run([self._output_name], {self._input_name: pixel_values})
            logits = outputs[0][0]
            
            # Softmax
            shifted = logits - np.max(logits)
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            
            # Динамический выбор NSFW-классов
            if len(probs) == 2:
                nsfw_score = float(probs[1])
            else:
                nsfw_score = float(np.max(probs[3:]))
                
        except Exception as e:
            logger.error(f"ViT inference failed: {e}", exc_info=True)
            raise RuntimeError("ViT model inference error") from e
        latency_ms = round((time.perf_counter()-start)*1000,2)
        return {
            "score": round(nsfw_score, 4),
            "label": "nsfw" if nsfw_score >= threshold else "sfw",
            "latency_ms": latency_ms,
            "model_name": self.name,
            "meta": {
                "classes": ["draw", "hentai", "neutral", "porn", "sexy"],
                "probabilities": [round(float(p), 4) for p in probs.tolist()]
            }
        }
        # The ZESTIES and FREAKIES Output of today VS The HORNINESS Output of tomorrow...