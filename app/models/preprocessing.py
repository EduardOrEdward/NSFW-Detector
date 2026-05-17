import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image_bytes: bytes, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Подготовка изображения для OpenNSFW2 ONNX модели.
    
    Модель ожидает входной тензор формы: [batch, height, width, channels]
    - batch: 1
    - height: 224
    - width: 224
    - channels: 3 (RGB)
    - значения: float32 в диапазоне [0, 1]
    
    Args:
        image_bytes: сырые байты изображения
        target_size: (height, width) - размер, до которого нужно изменить изображение
    
    Returns:
        np.ndarray: тензор формы (1, 224, 224, 3)
    """
    # Открываем изображение
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # PIL.resize() ожидает (width, height), а мы храним (height, width)
    height, width = target_size
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    
    # Конвертируем в numpy массив и нормализуем
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Добавляем batch dimension (без транспонирования!)
    batch_tensor = np.expand_dims(img_array, axis=0)
    
    # Логируем форму для отладки (можно убрать в production)
    logger.debug(f"Preprocessed tensor shape: {batch_tensor.shape}")
    
    # Валидация
    expected_shape = (1, height, width, 3)
    if batch_tensor.shape != expected_shape:
        raise ValueError(
            f"Неверная форма тензора: {batch_tensor.shape}, "
            f"ожидается {expected_shape}"
        )
    
    return batch_tensor