import asyncio
from app import schemas
from core.cache import cache

@cache(ttl=30, sliding=False)
async def cache(req: schemas.SimulateItem):
    await asyncio.sleep(20)
    return f"Cache {req.text}"
