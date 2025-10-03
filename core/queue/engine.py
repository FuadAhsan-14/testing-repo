import asyncio
import json
import time
import uuid
import inspect
from typing import Callable
from concurrent.futures import ThreadPoolExecutor
import redis.asyncio as redis
from .manager import QueueManager
from config.setting import env

class FunctionQueue:
    """
    Implementation of FunctionQueue optimized based on Redis best practices.
    - Uses "Reliable Queue" pattern with BRPOPLPUSH to prevent data loss.
    - Uses asyncio.Semaphore for clean and safe concurrency management.
    - Includes task status tracking for completed, queued, or processing tasks.
    
    This class implements a reliable task queue using Redis as the backend storage.
    Tasks are processed asynchronously with controlled concurrency.

    Attributes:
        redis (redis.Redis): Redis client instance for queue operations
        function_name (str): Name of the function/task type handled by this queue
        queue_name (str): Name of the main task queue in Redis
        processing_queue_name (str): Name of the processing queue in Redis
        max_concurrent (int): Maximum number of tasks that can be processed concurrently
        executor (ThreadPoolExecutor): Thread pool for executing synchronous functions
        running (bool): Flag indicating if the worker is running
        semaphore (asyncio.Semaphore): Semaphore for controlling concurrent task execution
        result_ttl (int): Time in seconds to store task results before they expire.

    Example:
        redis_client = redis.Redis()
        queue = FunctionQueue(redis_client, "my_task", max_concurrent=5)
        
        # Enqueue a task
        task_id = await queue.enqueue(args=[1, 2], kwargs={"x": "y"})
        
        # Check status
        status = await queue.get_task_status(task_id)
        print(status)  # -> {'status': 'queued', 'position': 1}
        
        # Start worker
        await queue.start_worker(my_task_function)
    """
    def __init__(
        self,
        function_name: str,
        redis_client: redis.Redis,
        process_function: Callable,
        max_concurrent: int = 10,
        result_ttl: int = 86400
    ):
        self.redis = redis_client
        self.function_name = function_name
        self.queue_name = f"queue_{env.app_name}:{function_name}"
        self.processing_queue_name = f"processing_{env.app_name}:{function_name}"
        self.max_concurrent = max_concurrent
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.running = False
        self.result_ttl = result_ttl
        
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.worker_task: asyncio.Task | None = None
        
        QueueManager.register_queue(self, process_function)

    async def enqueue(self, args: list = None, kwargs: dict = None, task_id: str = None) -> str:
        """Add a task to the main queue.

        Args:
            args (list, optional): Positional arguments for the task. Defaults to None.
            kwargs (dict, optional): Keyword arguments for the task. Defaults to None. 
            task_id (str, optional): Custom task ID. If not provided, a UUID will be generated.

        Returns:
            str: The task ID that can be used to track the task.
        """        
        if task_id is None:
            task_id = str(uuid.uuid4())

        task = {
            'id': task_id,
            'function_name': self.function_name,
            'args': args or [],
            'kwargs': kwargs or {},
            'created_at': time.time()
        }

        await self.redis.lpush(self.queue_name, json.dumps(task))
        return task_id

    async def _process_task(self, task_json: str, func: Callable):
        task_data = json.loads(task_json)
        task_id = task_data['id']
        args = task_data.get('args', [])
        kwargs = task_data.get('kwargs', {})
        result_key = f"result:{self.function_name}:{task_id}"

        try:
            if inspect.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.executor, func, *args, **kwargs)

            result_data = {'status': 'complete', 'completed_at': time.time()}
            await self.redis.set(result_key, json.dumps(result_data), ex=self.result_ttl)

        except Exception as e:
            error_data = {'status': 'failed', 'error': str(e), 'failed_at': time.time()}
            await self.redis.set(result_key, json.dumps(error_data), ex=self.result_ttl)
        
        finally:
            await self.redis.lrem(self.processing_queue_name, 1, task_json)
            self.semaphore.release()

    async def _worker_loop(self, func: Callable, timeout: int = 1):
        self.running = True

        while self.running:
            try:
                await self.semaphore.acquire()
                task_json = await self.redis.brpoplpush(self.queue_name, self.processing_queue_name, timeout=timeout)

                if task_json is None:
                    self.semaphore.release()
                    continue

                asyncio.create_task(self._process_task(task_json, func))

            except asyncio.CancelledError:
                break
            
            except Exception:
                if self.semaphore.locked():
                    self.semaphore.release()
                await asyncio.sleep(1)

    def start_worker(self, func: Callable):
        """
        Start a worker that implements the Reliable Queue pattern and concurrency management.

        Args:
            func (Callable): The function to be executed for each task
        """
        if self.worker_task and not self.worker_task.done():
            return
        self.worker_task = asyncio.create_task(self._worker_loop(func))

    async def stop_worker(self):
        """
        Menghentikan worker secara graceful.
        """
        if not self.worker_task or self.worker_task.done():
            return

        self.running = False
        self.worker_task.cancel()
        try:
            await asyncio.wait_for(self.worker_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        finally:
            self.executor.shutdown(wait=True)
            self.worker_task = None

    async def get_stats(self) -> dict:
        """Get queue statistics.
        
        Returns:
            dict: Dictionary containing queue statistics including:
                - function_name: Name of the function
                - queue_size: Number of tasks waiting in queue
                - processing_count: Number of tasks currently being processed
                - max_concurrent: Maximum number of concurrent tasks allowed
                - available_slots: Number of available slots for new tasks
        """        
        queue_size = await self.redis.llen(self.queue_name)
        processing_count = await self.redis.llen(self.processing_queue_name)
        return {
            'function_name': self.function_name,
            'queue_size': queue_size,
            'processing_count': processing_count,
            'max_concurrent': self.max_concurrent,
            'available_slots': self.max_concurrent - processing_count,
        }

    async def get_task_status(self, task_id: str) -> dict:
        """Checks the status of a task by its ID.

        The method checks in the following order:
        1. Completed/Failed results (by checking for a 'result:...' key).
        2. The processing queue.
        3. The main waiting queue.

        Args:
            task_id (str): The UUID of the task to check.

        Returns:
            dict: A dictionary containing the task's status. 
                  Possible statuses are 'complete', 'failed', 'processing', 
                  'queued', or 'not_found'. For 'queued' status, the
                  position in the queue is also returned.
        """
        result_key = f"result:{self.function_name}:{task_id}"
        result = await self.redis.get(result_key)
        if result:
            return json.loads(result)

        async def find_in_queue(queue_name: str):
            items = await self.redis.lrange(queue_name, 0, -1)
            total_items = len(items) 
            for i, item_json in enumerate(items):
                try:
                    task_data = json.loads(item_json)
                    if task_data.get('id') == task_id:
                        return total_items - i
                except json.JSONDecodeError:
                    continue # Skip corrupted data
            return None

        position = await find_in_queue(self.processing_queue_name)
        if position is not None:
            return {'status': 'processing'}

        position = await find_in_queue(self.queue_name)
        if position is not None:
            return {'status': 'queued', 'tasks_ahead': position}

        return {'status': 'not_found'}

