from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from backend.config import settings

_bearer_scheme = HTTPBearer()

_VALID_API_KEYS = {"dev-key-change-in-production"}


def create_access_token(api_key: str) -> str:
    """Issue a signed JWT for a validated API key."""
    expiry = datetime.utcnow() + timedelta(minutes=settings.jwt_expiry_minutes)
    return jwt.encode(
        {"sub": api_key, "exp": expiry},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def validate_api_key(api_key: str) -> bool:
    """Return True if the provided API key is recognized."""
    return api_key in _VALID_API_KEYS


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> str:
    """Decode and validate a Bearer JWT, returning the subject claim."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        subject: str = payload.get("sub", "")
        if not subject:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return subject
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
