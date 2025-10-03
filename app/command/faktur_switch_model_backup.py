import os
import sys
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.ratelimit import redis_connection


async def set_model(faktur_model_backup):
    try:
        await redis_connection.set("faktur_model", faktur_model_backup)
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    asyncio.run(set_model("gpt_4.1"))