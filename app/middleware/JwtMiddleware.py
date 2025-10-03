from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Annotated
from config.setting import env
import base64

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class CredentialError(Exception):
    pass

class JwtMiddleware:
    def __init__(self, algo: str = "HS256"):
        match algo.upper():
            case 'RS256':
                self.key = base64.b64decode(env.jwt_rs_public_key)
            case 'HS256' | None:
                self.key = base64.b64decode(env.jwt_hs_secret)
            case _:
                raise CredentialError(f"Unsupported JWT Algorithm: {algo}.")
            
        self.algorithm = algo

    async def validate_token(
        self, 
        token: Annotated[str, Depends(oauth2_scheme)],
        ):
        try:
            payload = jwt.decode(token, self.key, algorithms=[self.algorithm])
            if (
                payload.get("sub") not in env.allowed_jwt.split(',') or 
                payload.get("exp") is None or
                payload.get("service_name").upper() != env.app_name.upper() or
                payload.get("deployment_enviroment").upper() != env.app_env.upper()
            ):
                raise CredentialError("Could not validate credentials")
            return True
        
        except (JWTError, CredentialError) as e:
            raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail= {
                        "msg": "Could not validate credentials" if not isinstance(e, CredentialError) else str(e),
                        "data": None,
                        "error": e if not isinstance(e, CredentialError) else None
                    },
                    headers={"Authorization": "Bearer"},
                )
