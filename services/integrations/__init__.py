"""
services/integrations
---------------------
External platform integrations (Google Meet, etc.).
"""

from services.integrations.base import (
    ApiError,
    IntegrationError,
    NotConnected,
    RateLimited,
    TokenRevoked,
)
from services.integrations.crypto import (
    decrypt_token,
    encrypt_token,
)
from services.integrations.google_meet import (
    DEFAULT_SCOPES,
    PROVIDER_SLUG,
    exchange_code_for_tokens,
    get_authorization_url,
    get_valid_access_token,
    refresh_access_token,
    revoke_token,
)

__all__ = [
    # Exceptions
    "IntegrationError",
    "NotConnected",
    "TokenRevoked",
    "RateLimited",
    "ApiError",
    # Crypto
    "encrypt_token",
    "decrypt_token",
    # Google Meet OAuth
    "PROVIDER_SLUG",
    "DEFAULT_SCOPES",
    "get_authorization_url",
    "exchange_code_for_tokens",
    "refresh_access_token",
    "revoke_token",
    "get_valid_access_token",
]
