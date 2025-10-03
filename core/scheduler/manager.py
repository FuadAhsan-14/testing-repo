from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from config.setting import env
import importlib
import logging
from typing import Optional

SCHEDULED_TASKS = []

def scheduled_task(cron_expression: str):
    """Decorator to mark functions as scheduled tasks
    
    Args:
        cron_expression: Cron expression like "*/5 * * * * *" for seconds
    """
    def decorator(func):
        SCHEDULED_TASKS.append({
            'func': func,
            'cron': cron_expression,
            'id': func.__name__
        })
        return func
    return decorator

class SchedulerManager:
    """
    A static, class-based manager for the APScheduler.

    This class handles the initialization, task registration, and lifecycle
    (init/close) of the application's scheduler without needing an instance.
    It scans a specified module for tasks decorated with @scheduled_task.
    """
    _scheduler: Optional[AsyncIOScheduler] = None

    @classmethod
    def _initialize(cls):
        """
        Initializes the AsyncIOScheduler instance if it hasn't been already.
        This is called internally by the start() method to ensure lazy initialization.
        """
        if cls._scheduler is None:
            aps_logger = logging.getLogger('apscheduler')
            aps_logger.setLevel(logging.WARNING)
            cls._scheduler = AsyncIOScheduler(timezone=env.scheduler_timezone)

    @classmethod
    def _scan_for_tasks(cls, module_name: str = "routes.cron"):
        """
        Imports the specified module to trigger the @scheduled_task decorators,
        then registers the collected tasks with the scheduler.
        """
        try:
            # Importing the module executes the code inside, including the decorators
            importlib.import_module(module_name)
            for task_info in SCHEDULED_TASKS:
                cls._register_task(task_info)
        except Exception as e:
            raise e

    @classmethod
    def _register_task(cls, task_info: dict):
        """Adds a single discovered task to the scheduler instance."""
        func = task_info['func']
        cron = task_info['cron']
        task_id = task_info['id']

        try:
            trigger = cls._parse_cron(cron)
            cls._scheduler.add_job(
                func,
                trigger,
                id=task_id,
                replace_existing=True # Overwrite job if one with the same ID exists
            )
        except Exception as e:
            raise e 

    @staticmethod
    def _parse_cron(cron_expression: str) -> CronTrigger:
        """
        Parses a 5- or 6-part cron string into an APScheduler CronTrigger.
        This is a static method as it does not depend on any class or instance state.
        """
        parts = cron_expression.split()

        if len(parts) == 5: # Standard cron (minute, hour, day, month, day_of_week)
            return CronTrigger.from_crontab(cron_expression)

        elif len(parts) == 6: # Cron with seconds (second, minute, hour, day, month, day_of_week)
            second, minute, hour, day, month, day_of_week = parts
            return CronTrigger(
                second=second,
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week
            )
        else:
            raise ValueError(f"Invalid cron expression format: '{cron_expression}'")

    @classmethod
    async def init(cls):
        """
        The main entry point to start the scheduler.
        It initializes, scans for tasks, and starts the scheduling loop.
        """
        cls._initialize()
        cls._scan_for_tasks()
        if cls._scheduler and not cls._scheduler.running:
            cls._scheduler.start()

    @classmethod
    async def close(cls):
        """Gracefully shuts down the scheduler if it is running."""
        if cls._scheduler and cls._scheduler.running:
            cls._scheduler.shutdown(wait=True)
