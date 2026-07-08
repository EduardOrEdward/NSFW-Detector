import hashlib,time
import json
import logging
from typing import Dict, Any, Optional
import redis.asyncio as redis
from redis.exceptions import RedisError
from app.config import settings
logger = logging.getLogger(__name__)
from app.api.v1.predict import DetectionSource
class CacheService:
    def __init__(self, redis_url='redis://localhost:6379/0',ttl:int=86400):
        self.redis_url = redis_url
        self.ttl = ttl
        self._redis: Optional[redis.Redis] = None
        self._rate_limit_window=settings.RATE_LIMIT_WINDOW
        self._rate_limit = settings.RATE_LIMIT
    async def connect(self) -> None:
        '''Connection to Redis, activated once'''
        try:
            self._redis = redis.from_url(
                self.redis_url,
                encoding='utf-8',
                decode_responses=True,
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
        start = time.perf_counter()
        if not self._redis:
            return None
        key = self._get_cache_key(image_bytes)
        try:
            cached_data = await self._redis.get(key)
            if cached_data:
                logger.debug(f'Cache HIT for key: {key[:16]}...')
                result = json.loads(cached_data)
                result['cached'] = True
                result['latency_ms'] = round((time.perf_counter()-start)*1000,4)
                return {
                    'score':result['score'],
                    'label':result['label'],
                    'latency_ms':result['latency_ms'],
                    'cached':result['cached'],
                    'meta':{
                        'sources':{
                            'nudenet_detector':result['meta']['sources'].get('nudenet_detector') or 0.0,
                            'opennsfw2_detector':result['meta']['sources'].get('opennsfw2_detector') or 0.0
                        },
                        'strategy':settings.HYBRID_STRATEGY
                    },
                    'detecter_zones':result['meta'].get('detected_zones',[])
                }
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
    def _make_rate_key(self,identifier:str)->str:
        return f"identifier:{identifier}"
    async def check_rate_limit(self,identifier:str,limit=settings.RATE_LIMIT,window=settings.RATE_LIMIT_WINDOW)->bool:
        if not self._redis or not self._rate_limit:
            return True
        try:
            key = self._make_rate_key(identifier)
            now = time.time()
            window_start=now-window
            
            await self._redis.zremrangebyscore(key,0,window_start)
            current_count = await self._redis.zcard(key)
            if current_count >= self._rate_limit:
                logger.warning(f"Rate limit exceed for {identifier} ({current_count}/{self._rate_limit})")
                return False
            await self._redis.zadd(key,{f'{now}':now})
            await self._redis.expire(key,self._rate_limit+1)
            logger.debug(f"Rate limit OK for {identifier} ({current_count + 1}/{self._rate_limit})")
            return True
        except Exception as e:
            logger.error(f'Rate limit check failed: {e}')
            return True
    @property
    def is_healthy(self)->bool:
        return self._redis is not None