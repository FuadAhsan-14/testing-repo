import os
import sys
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.ratelimit import redis_connection


async def set_model(primary):
    try:
        #await redis_connection.set("anomaly_model", anomaly_model)
        #await redis_connection.set("primary_model", primary_model)
        #await redis_connection.set("quantity_model", quantity_model)
        #await redis_connection.set("ordinary_box_model", ordinary_box_model)
        #await redis_connection.set("bned_model", bned_model)
        await redis_connection.set("object_model", primary)
        print("Models updated to primary successfully")
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    #asyncio.run(set_model("claude_3.5", "claude_3.5", "claude_3.5", "claude_3.7", "gpt_4.1"))
    asyncio.run(set_model("primary"))

