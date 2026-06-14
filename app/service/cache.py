import hashlib
import json
import logging
from typing import Dict, Any, Optional
import redis.asyncio as redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self, redis_url='redis://localhost:6379/0',ttl:int=86400):
        self.redis_url = redis_url
        self.ttl = ttl
        self._redis: Optional[redis.Redis] = None
    async def connect(self) -> None:
        '''Connection to Redis, activated once'''
        try:
            self._redis = redis.from_url(
                self.redis_url,
                encoding='utf-8',
                decode_response=True,
                health_check_interval=30,
                )
            await self._redis.ping()
            logger.info(f'Redis connected: {self.redis_url}')
        except RedisError as e:
            logger.error(f'Failed to connect to Redis, error accused: {e}. Caching is disabled')
            self._redis = None
        return
    async def disconnect(self)->None:
        '''Correct shut down the Redis'''
        if self._redis:
            await self._redis.close()
            logger.info('The Redis is succussfully closed!')
        return
    def _get_cache_key(self, image_bytes:bytes) -> str:
        '''Generating unique cache key by using SHA256'''
        file_hash = hashlib.sha256(image_bytes).hexdigest()
        return f'nsfw:cache:{file_hash}'
    async def get(self,image_bytes:bytes) -> Optional[Dict[str,Any]]:
        '''
        Trying to get dict of the cache
        If there's no cache - return None
        '''
        if not self._redis:
            return None
        key = self._get_cache_key(image_bytes)
        try:
            cached_data = await self._redis.get(key)
            if cached_data:
                logger.debug(f'Cache HIT for key: {key[:16]}...')
                result = json.loads(cached_data)
                result['cached'] = True
                return result
        except RedisError as e:
            logging.warning(f'Redis get error accused: {e}')
        except json.JSONDecodeError as e:
            logging.error(f'Failed to parse cached JSON: {e}')
        logger.debug(f'Cache miss for the key: {key[:16]}')
        return None
    async def set(self, image_bytes:bytes,prediction_result:Dict[str,Any]) -> None:
        '''Saving the cache with settled TTL'''
        if not self._redis:
            return None
        key = self._get_cache_key(image_bytes=image_bytes)
        data_to_save = {k:v for k,v in prediction_result.items() if k!='cached'}
        data_to_save['cached'] = False
        try:
            await self._redis.setex(
                key,
                self.ttl,
                json.dumps(data_to_save)
            )
            logger.debug(f'Cached set for the key: {key[:16]}... TTL settled for: {self.ttl}')
            return
        except RedisError as e:
            logging.warning(f'Redis set error: {e}')
        return 