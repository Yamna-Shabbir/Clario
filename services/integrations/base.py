"""
services/integrations/base.py
------------------------------
Base exception classes and interfaces for external platform integrations.
"""


class IntegrationError(Exception):
    """Base exception for all integration and OAuth errors."""
    pass


class NotConnected(IntegrationError):
    """Raised when an operation requires an active integration connection that does not exist."""
    pass


class TokenRevoked(IntegrationError):
    """
    Raised when an OAuth token has been revoked, expired, or rejected with 'invalid_grant'.
    The stored connection should be cleared and the user prompted to reconnect.
    """
    pass


class RateLimited(IntegrationError):
    """Raised when an external API rate limit is exceeded (HTTP 429)."""
    pass


class ApiError(IntegrationError):
    """Raised when an external API returns an unexpected error or bad response."""
    pass

