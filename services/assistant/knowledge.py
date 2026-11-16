"""
Search Clario's written documentation.

Chunks are scored by word overlap. A passage is returned only when the
question's distinctive words appear in that passage.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "your", "you", "are",
    "was", "were", "have", "has", "had", "not", "but", "can", "how", "what",
    "when", "where", "who", "why", "does", "did", "into", "about", "them",
    "then", "than", "its", "our", "any", "all", "just", "will", "would",
    "could", "should", "there", "their", "they", "she", "her", "his", "him",
    "use", "using", "used", "please", "tell", "show", "give", "need", "want",
}

_SOURCES = (
    "README.md",
    "docs/assistant-help.md",
    "docs/integrations.md",
    "docs/deploy-render.md",
)


def tokenize(text: str) -> List[str]:
    """Lowercase words, with stopwords and one-character tokens removed."""
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return [word for word in words if len(word) > 1 and word not in _STOP]


def _covers(token: str, chunk_tokens: Sequence[str]) -> bool:
    if token in chunk_tokens:
        return True
    if len(token) < 5:
        return False
    stem = token[:5]
    return any(len(other) >= 5 and (other.startswith(stem) or token.startswith(other[:5])) for other in chunk_tokens)


def chunk_markdown(text: str, source: str) -> List[Dict[str, str]]:
    """Split a markdown document on headings."""
    sections: List[Dict[str, str]] = []
    title = os.path.basename(source)
    body: List[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if len(content) < 40:
            return
        sections.append({"source": source, "title": title, "text": content[:1600]})

    for line in (text or "").splitlines():
        if line.startswith("#"):
            flush()
            body = []
            title = line.lstrip("#").strip() or title
            continue
        body.append(line)
    flush()
    return sections


def load_documents(root: Optional[str] = None) -> List[Dict[str, str]]:
    """Read the user-facing documents that ship with Clario."""
    from config import ROOT_DIR

    base = root or ROOT_DIR
    chunks: List[Dict[str, str]] = []
    for relative in _SOURCES:
        path = os.path.join(base, relative.replace("/", os.sep))
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            chunks.extend(chunk_markdown(handle.read(), relative))
    return chunks


def search_help(
    question: str,
    *,
    documents: Optional[Iterable[Dict[str, str]]] = None,
    limit: int = 3,
) -> List[Dict[str, str]]:
    """
    Return documentation passages that contain the question's distinctive words.

    Passages that miss a word of five or more characters are left out.
    """
    query = tokenize(question)
    if not query:
        return []
    distinctive = [token for token in query if len(token) >= 5]
    ranked: List[Dict[str, str]] = []
    for chunk in documents if documents is not None else load_documents():
        tokens = tokenize(f"{chunk.get('title', '')} {chunk.get('text', '')}")
        if distinctive and not all(_covers(token, tokens) for token in distinctive):
            continue
        overlap = [token for token in query if _covers(token, tokens)]
        if not overlap:
            continue
        score = len(overlap) / len(query)
        if score < 0.34:
            continue
        ranked.append({
            "source": chunk.get("source") or "",
            "title": chunk.get("title") or "",
            "text": (chunk.get("text") or "").strip(),
            "score": f"{score:.3f}",
        })
    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    return ranked[: max(1, limit)]
