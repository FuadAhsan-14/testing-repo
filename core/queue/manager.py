import asyncio
from typing import Callable, List, Tuple

class QueueManager:
    """
    A class-based manager for handling the lifecycle of all FunctionQueue instances.
    It uses class attributes and methods to act as a global, stateful manager
    without needing to be instantiated.
    """
    # The state is now stored directly on the class.
    _queues: List[Tuple[object, Callable]] = []

    @classmethod
    def register_queue(cls, queue_instance: object, process_function: Callable):
        """Register a FunctionQueue instance."""
        cls._queues.append((queue_instance, process_function))

    @classmethod
    def init(cls):
        """Start all registered workers as background tasks."""
        if not cls._queues:
            return
        for queue, func in cls._queues:
            queue.start_worker(func)

    @classmethod
    async def close(cls):
        """Gracefully stop all registered workers."""
        if not cls._queues:
            return
        stop_tasks = [queue.stop_worker() for queue, _ in cls._queues]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
