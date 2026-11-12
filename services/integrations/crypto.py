"""
services/integrations/crypto.py
--------------------------------
Fernet symmetric encryption helpers for sensitive OAuth tokens.
Tokens are encrypted before being persisted in SQLite and decrypted in memory
when needed for API calls.

Security & privacy rules:
- Fails with clear, secret-free error messages if the encryption key is missing or invalid.
- Never logs, prints, or exposes raw keys, plaintext tokens, or ciphertexts.
"""

import os
from typing import Optional, Union
from cryptography.fernet import Fernet, InvalidToken

try:
    import config
except ImportError:
    config = None


def _resolve_fernet(key: Optional[Union[str, bytes]] = None) -> Fernet:
    """
    Resolve and validate a Fernet instance using the provided key, config, or environment.

    Raises
    ------
    ValueError
        If the key is missing or invalid. Error messages are strictly generic and
        never expose key contents.
    """
    resolved: Optional[Union[str, bytes]] = key

    if resolved is None:
        env_val = os.environ.get("CLARIO_TOKEN_ENCRYPTION_KEY")
        if env_val:
            resolved = env_val
        elif config is not None:
            resolved = getattr(config, "TOKEN_ENCRYPTION_KEY", "")

    if not resolved:
        raise ValueError(
            "Token encryption key is missing. Set CLARIO_TOKEN_ENCRYPTION_KEY in your environment."
        )

    if isinstance(resolved, str):
        key_bytes = resolved.strip().encode("utf-8")
    elif isinstance(resolved, bytes):
        key_bytes = resolved.strip()
    else:
        raise ValueError(
            "Token encryption key is invalid. It must be a 32-byte URL-safe base64-encoded string."
        )

    try:
        return Fernet(key_bytes)
    except Exception:
        raise ValueError(
            "Token encryption key is invalid. It must be a 32-byte URL-safe base64-encoded string."
        ) from None


def encrypt_token(plaintext: str, key: Optional[Union[str, bytes]] = None) -> str:
    """
    Encrypt a plaintext token using Fernet symmetric encryption.

    Parameters
    ----------
    plaintext : Plaintext token string to encrypt.
    key       : Optional 32-byte url-safe base64 Fernet key. Falls back to
                CLARIO_TOKEN_ENCRYPTION_KEY env var / config.

    Returns
    -------
    str : URL-safe base64 ciphertext string.
    """
    if plaintext is None:
        raise ValueError("Cannot encrypt None: plaintext string required.")

    fernet = _resolve_fernet(key)
    ciphertext_bytes = fernet.encrypt(plaintext.encode("utf-8"))
    return ciphertext_bytes.decode("utf-8")


def decrypt_token(ciphertext: str, key: Optional[Union[str, bytes]] = None) -> str:
    """
    Decrypt a Fernet ciphertext back to plaintext.

    Parameters
    ----------
    ciphertext : Ciphertext string to decrypt.
    key        : Optional 32-byte url-safe base64 Fernet key. Falls back to
                 CLARIO_TOKEN_ENCRYPTION_KEY env var / config.

    Returns
    -------
    str : Decrypted plaintext string.
    """
    if not ciphertext:
        raise ValueError("Cannot decrypt empty or None ciphertext.")

    fernet = _resolve_fernet(key)
    try:
        plaintext_bytes = fernet.decrypt(ciphertext.encode("utf-8"))
        return plaintext_bytes.decode("utf-8")
    except InvalidToken:
        raise ValueError("Token decryption failed: invalid or corrupted ciphertext.") from None
    except Exception:
        raise ValueError("Token decryption failed: ciphertext could not be decoded.") from None
