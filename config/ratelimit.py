from math import ceil
from fastapi import HTTPException, Request, Response
from fastapi import status
import redis.asyncio as redis
from config.setting import env

redis_connection = redis.Redis(
    host=env.redis_ratelimit_host, 
    port=env.redis_ratelimit_port, 
    db=env.redis_ratelimit_db, 
    encoding="utf-8", 
    decode_responses=True
) 

async def service_name_identifier(request: Request):
    return request.client.host

async def custom_callback(request: Request, response: Response, pexpire: int):
    """
    default callback when too many requests
    :param request:
    :param pexpire: The remaining milliseconds
    :param response:
    :return:
    """
    expire = ceil(pexpire / 1000)

    raise HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS,
        {
            "msg": "Too many requests",
            "data": None
        },
        headers={"Retry-After": str(expire)},
    )
