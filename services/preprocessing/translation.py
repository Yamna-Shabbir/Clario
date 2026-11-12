"""
Optional Marian MT (Urdu → English) for pasted transcripts and cleanup.
Loaded lazily; not used when Whisper already returned English.
"""

from __future__ import annotations

import os
import re
import threading
from typing import List, Optional, Tuple

_mt_bundle: Optional[Tuple[object, object]] = None
_mt_lock = threading.Lock()

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟۔])\s+|\n+")


def _chunk_sentences(text: str, *, max_chars: int = 400) -> List[str]:
    """Split text into chunks Marian can handle without truncation."""
    parts: List[str] = []
    for block in _SENTENCE_SPLIT.split(text.strip()):
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            parts.append(block)
            continue
        words = block.split()
        current: List[str] = []
        length = 0
        for word in words:
            extra = len(word) + (1 if current else 0)
            if current and length + extra > max_chars:
                parts.append(" ".join(current))
                current = [word]
                length = len(word)
            else:
                current.append(word)
                length += extra
        if current:
            parts.append(" ".join(current))
    return parts or [text.strip()]


def _get_mt_model(model_name: str):
    """Load Marian without the SentencePiece extension (blocked on this PC)."""
    global _mt_bundle
    if _mt_bundle is not None:
        return _mt_bundle
    with _mt_lock:
        if _mt_bundle is not None:
            return _mt_bundle
        import json
        import sys

        from huggingface_hub import hf_hub_download
        from tokenizers import SentencePieceUnigramTokenizer

        from services.native_compat import install

        install()
        schema_dir = os.path.dirname(os.path.abspath(__file__))
        if schema_dir not in sys.path:
            sys.path.insert(0, schema_dir)

        import torch
        from transformers import MarianMTModel

        print(f"[translation] Loading {model_name} …")
        spm_path = hf_hub_download(model_name, "source.spm")
        vocab_path = hf_hub_download(model_name, "vocab.json")
        with open(vocab_path, encoding="utf-8") as handle:
            vocab = json.load(handle)
        encoder = SentencePieceUnigramTokenizer.from_spm(spm_path)
        model = MarianMTModel.from_pretrained(model_name)
        model.eval()
        # The checkpoint's max_length is 512. Leaving it set makes every
        # generate(max_new_tokens=...) print a warning, which flooded the disk.
        model.generation_config.max_length = 96
        model.generation_config.num_beams = 4
        inverse = {int(idx): token for token, idx in vocab.items()}
        _mt_bundle = (encoder, vocab, inverse, model, torch)
        print("[translation] Marian MT ready.")
        return _mt_bundle


def _decode_ids(token_ids, inverse, eos_id: int, pad_id: int) -> str:
    words = []
    for token_id in token_ids:
        if token_id in (eos_id, pad_id):
            continue
        words.append(inverse.get(int(token_id), ""))
    return "".join(words).replace("\u2581", " ").strip()


def _is_degenerate(line: str) -> bool:
    if not line:
        return True
    if line.count(".") >= 6:
        return True
    words = [word.lower() for word in line.split()]
    if len(words) < 6:
        return False
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return max(counts.values()) >= 4


def _translate_piece(piece: str, encoder, vocab, inverse, model, torch) -> str:
    from services.preprocessing.language import arabic_script_ratio

    cleaned = piece.strip()
    if not cleaned or arabic_script_ratio(cleaned) < 0.2:
        return cleaned
    token_ids = [vocab[token] for token in encoder.encode(cleaned).tokens if token in vocab]
    if not token_ids:
        return cleaned
    token_ids.append(0)
    pad_id = int(vocab.get("<pad>", 62024))
    with torch.no_grad():
        generated = model.generate(
            torch.tensor([token_ids]),
            max_length=96,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    line = _decode_ids(generated[0].tolist(), inverse, 0, pad_id)
    if not _is_degenerate(line):
        return line
    with torch.no_grad():
        generated = model.generate(
            torch.tensor([token_ids]),
            max_length=64,
            num_beams=1,
            repetition_penalty=1.3,
        )
    line = _decode_ids(generated[0].tolist(), inverse, 0, pad_id)
    if _is_degenerate(line):
        return cleaned
    return line


def translate_urdu_text_to_english(text: str, *, model_name: str | None = None) -> str:
    """
    Translate Urdu (Arabic script) text to English using Marian MT.

    Tokenization uses the Rust ``tokenizers`` library. The SentencePiece
    extension is not imported, because Windows Application Control blocks it.
    """
    from config import ENABLE_TEXT_TRANSLATION, TRANSLATION_MODEL

    if not ENABLE_TEXT_TRANSLATION:
        return text
    name = model_name or TRANSLATION_MODEL
    from services.preprocessing.urdu_norm import (
        attribute_speakers,
        render_urdu_line,
        speaker_names,
    )

    names = speaker_names(text)
    rendered: List[Optional[str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            rendered.append("")
            continue
        rendered.append(render_urdu_line(raw_line, names))

    if any(line is None for line in rendered):
        encoder, vocab, inverse, model, torch = _get_mt_model(name)
        filled: List[str] = []
        for raw_line, line in zip(text.splitlines(), rendered):
            if line is not None:
                filled.append(line)
                continue
            parts = [
                _translate_piece(piece, encoder, vocab, inverse, model, torch)
                for piece in _chunk_sentences(raw_line.strip())
            ]
            filled.append(" ".join(part for part in parts if part).strip())
        translated = "\n".join(filled).strip()
    else:
        translated = "\n".join(rendered).strip()
    return attribute_speakers(translated)
