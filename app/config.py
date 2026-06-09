from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )
    
    APP_NAME:str = 'NSFW Detector API'
    APP_VERSION:str = '1.0.0'
    DEBUG:bool = False
    NUDENET_MODEL_PATH:str='data/models/nudenet/nudenet.onnx'
    VIT_MODEL_PATH: str = "data/models/vit/quantized_model.onnx"
    MODEL_PROVIDER:Literal['CPU','GPU']='CPU'
    
    HYBRID_STRATEGY:Literal['weighted','max','voting'] = 'weighted'
    NUDENET_WEIGHT:float=0.6
    VIT_WEIGHT:float=0.4
    THRESHOLD:float = 0.5
    
    @property
    def hybrid_weights(self) -> dict[str,float]:
        '''Making the vocabulary of the weights, using BaseDetector'''
        return {
            'nudenet_detector':self.NUDENET_WEIGHT,
            'vit_detector':self.VIT_WEIGHT
        }

settings = Settings()
