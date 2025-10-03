import os
import contextlib
from fastapi import FastAPI
from config.setting import env
from contextlib import asynccontextmanager
from fastapi_limiter import FastAPILimiter
from config.ratelimit import custom_callback, service_name_identifier, redis_connection
from routes.mcp import mcp as mcp_server
from config.phoenix import Phoenix
from core.scheduler import SchedulerManager
from core.queue import QueueManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # os.environ["LANGSMITH_API_KEY"] = env.langsmith_api_key
    # os.environ["LANGSMITH_ENDPOINT"] = env.langsmith_endpoint
    # os.environ["LANGSMITH_TRACING_V2"] = env.langsmith_tracing
    # os.environ["LANGSMITH_PROJECT"] = env.langsmith_project
    # async with contextlib.AsyncExitStack() as stack:
        # await stack.enter_async_context(mcp_server.session_manager.run())
        
    Phoenix.init()
    QueueManager.init()
    await SchedulerManager.init()
    await FastAPILimiter.init(
        redis_connection,
        identifier=service_name_identifier,
        http_callback=custom_callback,
    )

    yield
    await FastAPILimiter.close()
    await SchedulerManager.close()
    await QueueManager.close() 

app = FastAPI(lifespan=lifespan)
