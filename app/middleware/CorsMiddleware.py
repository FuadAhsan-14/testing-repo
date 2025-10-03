
from typing import Annotated
from config.setting import env
from fastapi import Header, HTTPException, status
from typing_extensions import Annotated


class CorsMiddleware:
    
    @staticmethod
    async def validate(origin: Annotated[str, Header()] = None):
        try:
            if origin not in env.allowed_origins.split(","):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail= {
                        "msg": "Not allowed by cors",
                        "data": None,
                        "error": None
                    }
                )
            return True
        except Exception as e:
            raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail= {
                        "msg": "Not allowed by cors",
                        "data": None,
                        "error": e
                    },
                    headers={"Authorization": "Bearer"},
                )

