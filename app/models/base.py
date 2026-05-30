from abc import abstractmethod,ABC
from typing import Any

class BaseDetector(ABC):
    @property
    @abstractmethod
    def name(str) ->str:
        """Уникальное имя модели для логирования, метрик и агрегации 
        (напр. 'nudenet_v3', 'vit_classifier')."""
        ...
    @abstractmethod
    def predict(self,image_bytes:bytes,threshold:float=0.8) ->dict[str,Any]:
        """
        Args:
            image_bytes: Сырые байты изображения (JPEG/PNG/WEBP и т.д.).
            threshold: Порог срабатывания [0.0, 1.0]. По умолчанию 0.5.

        Returns:
            dict со строгой структурой:
                - score (float): Итоговая вероятность NSFW [0.0, 1.0].
                - label (str): 'nsfw' или 'sfw' (рассчитывается по threshold).
                - latency_ms (float): Время инференса в миллисекундах.
                - model_name (str): Дублирует self.name.
                - meta (dict): Опциональные данные (зоны, логиты, bbox, raw outputs).

        Raises:
            ValueError: При пустых данных или threshold вне [0.0, 1.0].
            RuntimeError: При ошибке препроцессинга или ONNX-runtime.
        """
        ...
    @staticmethod
    def _validate_inputs(image_bytes:bytes,threshold:float=0.6) ->None:
        if not image_bytes:
            raise ValueError("image_bytes cannot be empty or None")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold should be in [0.0, 1.0], got {threshold}")
        