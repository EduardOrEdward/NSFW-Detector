import filetype
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 10 *1024 *1024 #10MB is max size
ALLOWED_MINE_TYPES = {'image/jpeg','image/png','image/webp'}

def validate_image(file_bytes:bytes,filename:str) ->None:
    '''
    Checking the file size and it's type
    '''
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        logger.warning(f'File too large: {filename} ({len(file_bytes)} bytes)')
        raise HTTPException(
            status_code=413,
            detail=f'File too large. Maximus size - {(MAX_FILE_SIZE_BYTES//1024)//1024}MB'
        )
    kind = filetype.guess(file_bytes)
    if kind is None:
        logger.warning(f'Cannot determinade filetype: {kind}')
        raise HTTPException(
            status_code=415,
            detail='Invaild file type'
        )
    if kind.mime not in ALLOWED_MINE_TYPES:
        logger.warning(f'Disallowed file type: {kind.mime} for {filename}')
        raise HTTPException(
            status_code= 415,
            detail=f'Unsupported file type: {kind.mime}. Allowed types: jpeg, png, webp'
        )
    