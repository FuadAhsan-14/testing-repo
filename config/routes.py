# import os
# import importlib
from fastapi import Depends, Security
from config.setting import env
from routes.ws import v1 as ws_v1
from routes.api import v1 as api_v1, v2 as api_v2
from routes.mcp import mcp as mcp_server
from fastapi_limiter.depends import RateLimiter
from app.middleware import CorsMiddleware, SignatureMiddleware, JwtMiddleware


def setup_routes(app):
    jwtMiddleware = JwtMiddleware(algo="HS256")

    api_dependencies = [
        Depends(RateLimiter(times=60, seconds=60)),
        # Security(CorsMiddleware.validate),
        # Security(jwtMiddleware.validate_token),
        # Security(SignatureMiddleware.validate_signature)
    ]
    app.include_router(
        api_v1.router,
        prefix="/v1",
        tags=["api_v1"],
        dependencies = api_dependencies
    )
    app.include_router(
        api_v2.router,
        prefix="/v2",
        tags=["api_v2"],
        dependencies = api_dependencies
    )
    # app.include_router(
    #     ws_v1.router,
    #     prefix="/ws/v1",
    #     tags=["ws_v1"]
    # )
    
    @app.get("/health-check", dependencies=[Depends(RateLimiter(times=60, seconds=60))])
    async def read_health():
        return {"status": "OK"}
        
    @app.get("/", dependencies=[Depends(RateLimiter(times=60, seconds=60))])
    async def read_root():
        return {
            "app_env": env.app_env,
            "app_name": env.app_name,
            "app_version": env.app_version,
        }

    app.mount("/tools", mcp_server.streamable_http_app())
