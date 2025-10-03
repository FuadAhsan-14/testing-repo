import time
import uuid
import logging
import asyncio
import redis.asyncio as redis
from config.setting import env

class LockError(Exception):
    """Custom exception for lock-related errors."""
    pass

class AsyncRedisDistributedLock:
    """
    A robust distributed lock implementation using Redis.

    This lock is designed to be:
    1.  **Mutually Exclusive**: Only one client can hold the lock at any given time.
    2.  **Deadlock-Free**: Uses a timeout (TTL) on the lock key, so if a client crashes,
        the lock is eventually released automatically.
    3.  **Safe**: Prevents a client from mistakenly releasing a lock held by another client.
        This is achieved by storing a unique owner ID in the lock's value.
    """
    # Lua script to safely release the lock.
    # It ensures that we only delete the key if it still exists AND its value
    # matches the unique owner ID of this lock instance. This is atomic.
    LUA_RELEASE_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """

    def __init__(self,
                 redis_client: redis.Redis,
                 lock_key: str,
                 timeout_seconds: int = 5):
        """
        Initializes the distributed lock.

        Args:
            redis_client: An active redis.Redis client instance.
            lock_key: The name of the lock key to use in Redis.
            timeout_seconds: The time in seconds for the lock to live before it
                             auto-expires. This prevents permanent deadlocks.
        """
        if not isinstance(redis_client, redis.Redis):
            raise TypeError("redis_client must be a valid redis.Redis instance.")
        if not lock_key:
            raise ValueError("lock_key cannot be empty.")
        
        self.redis_client = redis_client
        self.lock_key = f"lock_{env.app_name}:{lock_key}"
        self.timeout_ms = int(timeout_seconds * 1000)
        # Unique ID for this specific lock instance to prevent releasing another's lock.
        self.owner_id = uuid.uuid4().hex
        self._is_locked = False

    async def acquire(self, blocking: bool = True, blocking_timeout_seconds: int = -1) -> bool:
        """
        Tries to acquire the lock.

        Args:
            blocking: If True, the call will block until the lock is acquired
                      or the `blocking_timeout_seconds` is reached.
            blocking_timeout_seconds: Maximum time in seconds to wait for the lock.
                                      A value of -1 means wait indefinitely.

        Returns:
            True if the lock was acquired, False otherwise.
        """
        start_time = time.monotonic()
        
        while True:
            # The `set` command with `nx=True` and `px=timeout` is ATOMIC.
            # It sets the key with our unique owner_id only if it does not exist (nx=True)
            # and sets the expiration in milliseconds (px=timeout_ms) in a single operation.
            if await self.redis_client.set(self.lock_key, self.owner_id, nx=True, px=self.timeout_ms):
                self._is_locked = True
                logging.info(f"Lock '{self.lock_key}' acquired.")
                return True

            if not blocking:
                return False

            # Check if the blocking timeout has been exceeded
            elapsed = time.monotonic() - start_time
            if blocking_timeout_seconds != -1 and elapsed >= blocking_timeout_seconds:
                logging.warning(f"Failed to acquire lock '{self.lock_key}' within {blocking_timeout_seconds}s.")
                return False

            # Wait a short while before retrying
            await asyncio.sleep(0.1)

    async def release(self) -> bool:
        """
        Releases the lock using a safe Lua script.
        
        Returns:
            True if the lock was released by this instance, False otherwise.
        """
        if not self._is_locked:
            return False
            
        try:
            # Execute the Lua script to ensure atomicity.
            # It only deletes the key if the owner_id matches.
            result = await self.redis_client.eval(self.LUA_RELEASE_SCRIPT, 1, self.lock_key, self.owner_id)
            if result == 1:
                self._is_locked = False
                return True
            else:
                # This can happen if our lock expired and another client acquired it.
                self._is_locked = False
                return False
        except redis.exceptions.RedisError as e:
            self._is_locked = False
            return False

    async def __aenter__(self):
        """Context manager entry. Acquires the lock."""
        if not await self.acquire(blocking=True):
            raise LockError(f"Failed to acquire lock: {self.lock_key}")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Context manager exit. Releases the lock."""
        await self.release()
