from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

# app/config.py
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    APP_NAME: str = "NSFW Detector API"
    APP_ENV: str = "production"
    APP_VERSION:str = '1.0.0'
    API_PREFIX: str = "/"
    DEBUG:bool = False
    NUDENET_MODEL_PATH: str = "data/models/nudenet_v3.onnx"
    #VIT_MODEL_PATH: str = "data/models/quanited_model.onnx"
    MODEL_PROVIDER: Literal["CPU", "GPU"] = "CPU"
    HYBRID_STRATEGY: Literal["max", "weighted", "voting"] = "weighted"
    NUDENET_WEIGHT: float = 0.4
    OPENNSFW2_WEIGHT: float = 0.6
    DEFAULT_THRESHOLD: float = 0.5

    REDIS_URL: str = "redis://redis:6379/0"
    CACHE_TTL: int = 86400
    RATE_LIMIT: int = 60
    RATE_LIMIT_WINDOW: int = 60

    # RapidAPI / Security
    RAPIDAPI_KEY_HEADER: str = "x-rapidapi-key"
    ALLOWED_RAPIDAPI_KEY: str | None = None
    CORS_ORIGINS: list[str] = ["*"]

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def hybrid_weights(self) -> dict[str,float]:
        '''Making the vocabulary of the weights, using BaseDetector'''
        return {
            'nudenet_detector':self.NUDENET_WEIGHT,
            'vit_detector':self.OPENNSFW2_WEIGHT
        }

settings = Settings()
