import io
from PIL import Image, ImageOps
import logging
from typing import Tuple
logger = logging.getLogger(__name__)

MAX_IMAGE_DEMENSION = 1024

def safe_load_image(image_bytes:bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if max(img.size) > MAX_IMAGE_DEMENSION:
            img.thumbnail((MAX_IMAGE_DEMENSION,MAX_IMAGE_DEMENSION),Image.Resampling.LANCZOS)
            logger.debug(f'The image had been resized to: {img.size}')
        return img
    except Exception as e:
        logger.error(f'During image loading we catched an error: {e}')
        raise ValueError('Invalid or corrupted data')