from config.redis import redis_client
from core.distribution_lock import AsyncRedisDistributedLock, LockError
import asyncio

class SimulateLockController:
    def __init__(self):
        self.redis = redis_client
    
    async def input_data(self, text):
        try:
            async with AsyncRedisDistributedLock(
                self.redis, 
                "input_data", 
                timeout_seconds=30
                ):
                await asyncio.sleep(20)
                return f"Processed: {text}"
            
        except LockError as e:
            print(f"Process {text} could not acquire the lock: {e}")

simulateLockController = SimulateLockController()
