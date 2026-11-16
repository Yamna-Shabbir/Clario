"""
Optional chat completion for the assistant.

Clario does not load a chat model of its own. When
``CLARIO_ASSISTANT_API_URL`` is set, that OpenAI-compatible endpoint may
rephrase notes that were already retrieved. The key is never written to logs.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


class AssistantProviderError(RuntimeError):
    """The configured chat endpoint did not return an answer."""


def assistant_configured() -> bool:
    """True when an external chat endpoint is configured."""
    import config

    return bool((getattr(config, "ASSISTANT_API_URL", "") or "").strip())


def complete(messages: Sequence[Dict[str, str]], *, timeout: float = 30) -> str:
    """
    Send a chat completion and return the assistant text.

    Raises
    ------
    AssistantProviderError
        The endpoint is unset, unreachable, or returned an unexpected body.
    """
    import config

    url = (getattr(config, "ASSISTANT_API_URL", "") or "").strip().rstrip("/")
    if not url:
        raise AssistantProviderError("No assistant model is configured.")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"
    key = (getattr(config, "ASSISTANT_API_KEY", "") or "").strip()
    model = (getattr(config, "ASSISTANT_MODEL", "") or "").strip() or "gpt-4o-mini"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        import requests

        response = requests.post(
            url,
            json={
                "model": model,
                "temperature": 0,
                "messages": list(messages),
            },
            headers=headers,
            timeout=timeout,
        )
    except Exception as exc:
        raise AssistantProviderError("The assistant model could not be reached.") from exc
    if response.status_code != 200:
        raise AssistantProviderError("The assistant model returned an error.")
    try:
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise AssistantProviderError("The assistant model returned an unexpected response.") from exc
    if not isinstance(text, str) or not text.strip():
        raise AssistantProviderError("The assistant model returned an empty response.")
    return text.strip()


def keeps_to_notes(reply: str, notes: str, question: str) -> bool:
    """True when the reply adds no content word outside the notes and question."""
    from services.assistant.knowledge import tokenize

    allowed = set(tokenize(notes)) | set(tokenize(question)) | {
        "clario", "information", "documentation", "meeting", "meetings",
        "available", "account",
    }
    extra = [token for token in tokenize(reply) if token not in allowed]
    return not extra
