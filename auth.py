from secrets import compare_digest

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from settings import APP_SETTINGS

api_key_header = APIKeyHeader(
    name=APP_SETTINGS.api_key_header,
    auto_error=False,
)

async def verify_api_key(api_key: str | None = Security(api_key_header)):
    if APP_SETTINGS.api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "api_key_not_configured",
                "message": "Backend API-key authentication is not configured.",
            },
        )
    if not api_key or not compare_digest(api_key, APP_SETTINGS.api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "invalid_api_key",
                "message": "Invalid or missing API key.",
            },
        )
    return api_key
