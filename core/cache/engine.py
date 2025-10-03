# main.py
import time
import asyncio
import threading
from functools import wraps
from typing import Callable, Any, get_type_hints

from fastapi import FastAPI, Request
from uvicorn import run

class Cache:
    """
    A thread-safe, in-memory cache with both fixed and sliding Time-To-Live (TTL) support.
    
    This cache is implemented as a singleton to ensure a single instance
    is used across the application.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Cache, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.cache = {}
            self.lock = threading.Lock()
            self.initialized = True

    def set(self, key: str, value: Any, ttl: int):
        """
        Sets a value in the cache with a specific key and TTL.

        Args:
            key (str): The key to store the value under.
            value (Any): The value to be cached.
            ttl (int): The time-to-live for the cache entry in seconds.
        """
        with self.lock:
            expires_at = time.time() + ttl
            self.cache[key] = (value, expires_at, ttl)

    def get(self, key: str, sliding: bool = False) -> Any:
        """
        Retrieves a value from the cache, optionally refreshing its TTL (sliding expiration).

        Args:
            key (str): The key of the value to retrieve.
            sliding (bool): If True, resets the TTL on a cache hit.

        Returns:
            Any: The cached value if the key exists and has not expired, otherwise None.
        """
        with self.lock:
            if key in self.cache:
                value, expires_at, ttl = self.cache[key]
                
                if time.time() < expires_at:
                    if sliding:
                        # Refresh the TTL by setting a new expiration time
                        new_expires_at = time.time() + ttl
                        self.cache[key] = (value, new_expires_at, ttl)
                    return value
                else:
                    del self.cache[key]
            
            return None

    def clear(self):
        """Clears all items from the cache."""
        with self.lock:
            self.cache.clear()

cache_singleton = Cache()

def cache(ttl: int, sliding: bool = False):
    """
    A unified decorator to cache the result of functions or FastAPI endpoints.

    Args:
        ttl (int): The time-to-live for the cache entry in seconds.
        sliding (bool): If True, the TTL is refreshed on every cache hit.
    """
    def decorator(func: Callable):
        # --- Logic to determine if it's an endpoint or a regular function ---
        is_endpoint = False
        try:
            # Inspect the function signature to see if it expects a 'request' argument
            hints = get_type_hints(func)
            if "request" in hints and hints["request"] == Request:
                 is_endpoint = True
        except Exception:
            # Fallback for functions without type hints
            pass

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = ""
            if is_endpoint and args and isinstance(args[0], Request):
                request = args[0]
                sorted_params = sorted(request.query_params.items())
                query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
                cache_key = f"{func.__module__}:{func.__name__}:{query_string}"
            else:
                sorted_kwargs = sorted(kwargs.items())
                key_args = str(args) + str(sorted_kwargs)
                cache_key = f"{func.__module__}:{func.__name__}:{key_args}"

            cached_result = cache_singleton.get(cache_key, sliding=sliding)
            if cached_result is not None:
                return cached_result

            result = await func(*args, **kwargs)
            cache_singleton.set(cache_key, result, ttl)
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            sorted_kwargs = sorted(kwargs.items())
            key_args = str(args) + str(sorted_kwargs)
            cache_key = f"{func.__module__}:{func.__name__}:{key_args}"

            cached_result = cache_singleton.get(cache_key, sliding=sliding)
            if cached_result is not None:
                return cached_result

            result = func(*args, **kwargs)
            cache_singleton.set(cache_key, result, ttl)
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator
