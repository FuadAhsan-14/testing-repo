import asyncio
from config.redis import redis_manager
from core.queue import FunctionQueue
from app.schemas import SimulateItem
from app.generative import manager

class SimulateQueueController:
    """
    Manages a background task queue for PDF processing using asyncio.

    This controller handles adding tasks to a Redis-backed queue and manages
    a background worker implemented as an asyncio.Task to process those tasks.
    """
    def __init__(self):
        self.pdf_queue = FunctionQueue(
            function_name="pdf_queue",
            redis_client=redis_manager.get_client(), 
            process_function=self.process_pdf,
            max_concurrent=3,
            result_ttl=86400
        )
        self.info = "SimulateQueueController initialized with FunctionQueue."
        self.count = 0
        self.llm = manager.gemini_model_flash_lite25()

    async def add_queue(self, user_session_id: str, input_item: SimulateItem) -> str:
        """
        Adds a new PDF processing task to the queue.
        This is now an async function to correctly call the async enqueue method.
        """
        task_id = await self.pdf_queue.enqueue(
            kwargs={
                "user_session_id": user_session_id,
                "input_text": input_item.text
            }
        )
        return task_id

    async def process_pdf(self, user_session_id: str, input_text: str) -> None:
        """
        The actual task logic for processing a PDF. This is an async function.
        It simulates a long-running I/O-bound operation.
        """
        print(f"Starting PDF processing for user {user_session_id}...")
        print(self.info)
        self.count += 1
        print(f"Current task count: {self.count}")
        await asyncio.sleep(60)
        print(f"PDF processing completed for user {user_session_id}.")

    async def get_queue_stats(self) -> dict:
        """
        Gets the current statistics from the FunctionQueue.
        This is now an async function to correctly call the async get_stats method.
        """
        return await self.pdf_queue.get_stats()
    
    async def get_task_status(self, task_id: str) -> dict:
        """
        Gets the status of a specific task by its ID.
        This is now an async function to correctly call the async get_task_status method.
        """
        return await self.pdf_queue.get_task_status(task_id)

simulateQueueController = SimulateQueueController()
