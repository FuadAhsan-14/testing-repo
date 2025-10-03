import redis.asyncio as redis
import json
import pickle

from typing import Any, Dict, List, Optional
from config.setting import env


class RedisManager:
    def __init__(self):
        self.client = redis.Redis(
            host=env.redis_host,
            port=env.redis_port,
            db=env.redis_db,
            password=env.redis_password,
            username=env.redis_username,
            decode_responses=True
        )
        
    def get_client(self) -> redis.Redis:
        """Get the underlying Redis client."""
        return self.client
    
    async def ping(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            return await self.client.ping()
        except Exception:
            return False
    
    async def close(self):
        """Close Redis connection."""
        await self.client.aclose()

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a key-value pair with automatic serialization.
        
        Args:
            key: Redis key
            value: Value to store (automatically serialized)
            ttl: Time to live in seconds
        """
        serialized_value = self._serialize(value)
        return await self.client.set(key, serialized_value, ex=ttl)
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value by key with automatic deserialization.
        
        Args:
            key: Redis key
            default: Default value if key doesn't exist
        """
        value = await self.client.get(key)
        if value is None:
            return default
        return self._deserialize(value)
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys at once."""
        values = await self.client.mget(keys)
        result = {}
        for key, value in zip(keys, values):
            result[key] = self._deserialize(value) if value else None
        return result
    
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple key-value pairs at once."""
        serialized_mapping = {k: self._serialize(v) for k, v in mapping.items()}
        
        pipe = self.client.pipeline()
        pipe.mset(serialized_mapping)
        
        if ttl:
            for key in mapping.keys():
                pipe.expire(key, ttl)
        
        results = await pipe.execute()
        return all(results)

    def _serialize(self, value: Any) -> str:
        """Serialize value for Redis storage."""
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value)
        elif isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value, default=str)
        else:
            return f"pickle:{pickle.dumps(value).hex()}"
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize value from Redis."""
        if value.startswith("pickle:"):
            hex_data = value[7:]  # Remove "pickle:" prefix
            return pickle.loads(bytes.fromhex(hex_data))
        else:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value  # Return as string if can't deserialize

redis_manager = RedisManager()
