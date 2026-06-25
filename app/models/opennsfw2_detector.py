import time
import logging
import io,os
import numpy as np
from PIL import Image
#from transformers import AutoImageProcessor
from typing import Any, Dict
import onnxruntime as ort
from base import BaseDetector
import opennsfw2
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

class OpenNSFW2Detector(BaseDetector):
    _model_loaded = False 
    def __init__(self, model_path:str|None=None)->None:
        if not OpenNSFW2Detector._model_loaded:
            try:
                import tensorflow as tf
                tf.config.threading.set_intra_op_parallelism_threads(2)
                tf.config.threading.set_inter_op_parallelism_threads(1)
                
                dummy = Image.new('RGB',(224,224),color=(128,128,128))
                _ = opennsfw2.predict_image(dummy)
                OpenNSFW2Detector._model_loaded=True
            except Exception as e:
                logger.warning('Can\'t load OpenNSFW2 model from the servers')
    
    
    
    
    
    
    '''
    def __init__(self,model_path:str|None=None,provider:str='CPU'):
        self.model_path = model_path
        #self._processer = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection_26",use_fast=True)
        
        providers = ['CPUExecutionProvider']
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opt.inter_op_num_threads = 1                           # 1 поток на операцию
        sess_opt.intra_op_num_threads = 2                           # 2 потока внутри op (баланс)
        sess_opt.enable_cpu_mem_arena = False
        
        
        
        if 'OpenVINOExecutionProvider' in ort.get_all_providers():
            providers = ['OpenVINOExecutionProvider']
            sess_opt = ort.SessionOptions()
            sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opt.inter_op_num_threads = 1
        else:
            providers = ['CPUExecutionProvider']
            sess_opt = ort.SessionOptions()
            sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opt.executive_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_opt.inter_op_num_threads=1
            sess_opt.intra_op_num_threads=max(2,os.cpu_count() or 2)
            sess_opt.enable_cpu_mem_arena=True
        
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
            sess_opt.enable_cpu_mem_arena = False
            sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opt.add_session_config_entry('session.intra_op.allow_spinning','0')
        '''
        
    @property
    def name(self) ->str:
        return "opennsfw2_detector"
    def predict(self,image_bytes:bytes,threshold:float=0.5,executor=None) -> Dict[str,Any]:
        self._validate_inputs(image_bytes,threshold=threshold)
        t0 = time.perf_counter()
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224,224))
            nsfw_score = opennsfw2.predict_image(img)
            
            # Динамический выбор NSFW-классов
            
            #logger.info(f"[{self.name}] Decode: {(t1-t0)*1000:.1f}ms | PP: {(t2-t1)*1000:.1f}ms | INF: {(t3-t2)*1000:.1f}ms")
        except Exception as e:
            logger.error(f"OpenNSFW2 inference failed: {type(e).__name__}", exc_info=True)
            raise RuntimeError("OpenNSFW2 model inference error") from e
        latency_ms = round((time.perf_counter()-t0)*1000,2)
        return {
            "score": round(nsfw_score, 4),
            "label": "nsfw" if nsfw_score >= threshold else "sfw",
            "latency_ms": latency_ms,
            "model_name": self.name,
            "meta": {
                #"classes": ["draw", "hentai", "neutral", "porn", "sexy"],
                'model_name': OpenNSFW2Detector.name
            }
        }
        # The ZESTIES and FREAKIES Output of today VS The HORNINESS Output of tomorrow...