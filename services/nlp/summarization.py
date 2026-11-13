"""
services/nlp/summarization.py
--------------------------
BART-based meeting summarization service with token-aware map-reduce.

Features:
  - Token-aware chunking with BART tokenizer (sentence boundaries, ~700 tokens, 1-2 sentence overlap)
  - Map-reduce hierarchical summarization (up to 3 levels) for long transcripts
  - Deduplication: near-duplicate sentences (difflib ratio > 0.9) removed after map
    and again on the final output
  - Dynamic min/max length scaling per chunk
  - Safe handling of empty, short, and single-chunk inputs
  - Progress callback: progress(done_chunks, total_chunks)
  - Smart merge: automatically appends action items not already in the summary
  - Deprecated SUMMARY_MAX_INPUT_CHARS: preserved in signature, no longer a hard truncation cap
"""

import difflib
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import (
    BART_MODEL,
    SUMMARY_MAX_LENGTH,
    SUMMARY_MIN_LENGTH,
    SUMMARY_CHUNK_TOKENS,
    SUMMARY_MAX_INPUT_CHARS,
)

# Module-level model cache
_summarizer_model = None
_summarizer_tokenizer = None
_loaded_model_name = None

# Set to True when the BART model failed to load (OOM, disk full, etc.)
_summarizer_unavailable = False


def load_summarizer(model_name: str = BART_MODEL, device: int = -1):
    """
    Load (or return cached) HuggingFace summarizer model and tokenizer.
    Returns (None, None) if the model cannot be loaded (OOM / disk full).
    """
    global _summarizer_model, _summarizer_tokenizer, _loaded_model_name, _summarizer_unavailable
    if _summarizer_unavailable:
        return None, None
    if (
        _summarizer_model is None
        or _summarizer_tokenizer is None
        or _loaded_model_name != model_name
    ):
        try:
            from config import FREE_TIER
            if FREE_TIER:
                print("BART download skipped on the free plan. Using the extractive summary.")
                _summarizer_unavailable = True
                return None, None
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch

            print(f"Loading BART tokenizer ('{model_name}') ...")
            try:
                _summarizer_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, local_files_only=True,
                )
                _summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, local_files_only=True,
                )
            except (Exception, OSError):
                from utils.storage import assert_model_download_allowed
                assert_model_download_allowed("the meeting summarizer")
                print(f"Downloading BART tokenizer ('{model_name}') ...")
                _summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name)
                print(f"Downloading BART model ('{model_name}') ...")
                _summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            if device == 0 and torch.cuda.is_available():
                _summarizer_model = _summarizer_model.cuda()
            _loaded_model_name = model_name
            print(f"BART summarizer loaded ({model_name}).")
        except (Exception, MemoryError, OSError) as exc:
            print(f"⚠️  BART model unavailable: {exc}. Will use extractive fallback.")
            _summarizer_unavailable = True
            _summarizer_model = None
            _summarizer_tokenizer = None
            return None, None
    return _summarizer_model, _summarizer_tokenizer


def chunk_text_by_tokens(
    text: str,
    tokenizer,
    chunk_tokens: int = 700,
    overlap_sentences: int = 2,
) -> List[str]:
    """
    Split text into sentence-boundary chunks of at most `chunk_tokens` tokens,
    with `overlap_sentences` sentence overlap between adjacent chunks.

    Parameters
    ----------
    text : Input text to chunk.
    tokenizer : HuggingFace tokenizer (or mock) with encode() method.
    chunk_tokens : Maximum token count per chunk.
    overlap_sentences : Number of sentences carried over from the end of chunk N
                        to the beginning of chunk N+1.

    Returns
    -------
    List of chunk strings, each guaranteed to be <= chunk_tokens.
    """
    if not text or not text.strip():
        return []

    # Split into sentences along sentence-ending punctuation
    raw_sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text.strip())
        if s.strip()
    ]
    if not raw_sentences:
        return []

    # Safeguard: if any single sentence exceeds chunk_tokens, split it by words
    sentences = []
    for s in raw_sentences:
        s_tokens = tokenizer.encode(s, add_special_tokens=False)
        if len(s_tokens) > chunk_tokens:
            words = s.split()
            sub_sent = []
            for w in words:
                sub_sent.append(w)
                candidate_tokens = len(
                    tokenizer.encode(" ".join(sub_sent), add_special_tokens=False)
                )
                if candidate_tokens >= chunk_tokens - 10:
                    sentences.append(" ".join(sub_sent) + ".")
                    sub_sent = []
            if sub_sent:
                sentences.append(" ".join(sub_sent))
        else:
            sentences.append(s)

    chunks = []
    current_sentences = []
    current_tokens = 0

    for s in sentences:
        s_tokens = len(tokenizer.encode(s, add_special_tokens=False))
        if current_sentences and (current_tokens + s_tokens > chunk_tokens):
            chunks.append(" ".join(current_sentences))
            # Build overlap from the tail of current chunk
            overlap = (
                current_sentences[-overlap_sentences:]
                if overlap_sentences > 0
                else []
            )
            overlap_tokens = sum(
                len(tokenizer.encode(osent, add_special_tokens=False))
                for osent in overlap
            )
            # Ensure overlap + new sentence does not overflow the chunk budget
            while overlap and (overlap_tokens + s_tokens > chunk_tokens):
                overlap.pop(0)
                overlap_tokens = sum(
                    len(tokenizer.encode(osent, add_special_tokens=False))
                    for osent in overlap
                )
            current_sentences = list(overlap) + [s]
            current_tokens = overlap_tokens + s_tokens
        else:
            current_sentences.append(s)
            current_tokens += s_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def deduplicate_sentences(text: str, ratio_threshold: float = 0.9) -> str:
    """
    Remove near-duplicate sentences from *text*, keeping the first occurrence.

    Two sentences are considered duplicates when their normalised forms have a
    ``difflib.SequenceMatcher`` ratio > *ratio_threshold* (default 0.9).

    Normalisation: lowercase → strip punctuation → collapse whitespace.

    Parameters
    ----------
    text : Text that may contain duplicate or near-duplicate sentences.
    ratio_threshold : Similarity threshold above which the later sentence is
                      dropped (exclusive, i.e. ratio > threshold).

    Returns
    -------
    Text with duplicate sentences removed, joined by single spaces.
    """
    if not text or not text.strip():
        return text

    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    if len(sentences) <= 1:
        return text.strip()

    def _norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    kept: List[str] = []
    kept_norms: List[str] = []

    for sent in sentences:
        norm = _norm(sent)
        if not norm:
            continue
        is_dup = any(
            difflib.SequenceMatcher(None, norm, kn).ratio() > ratio_threshold
            for kn in kept_norms
        )
        if not is_dup:
            kept.append(sent)
            kept_norms.append(norm)

    return " ".join(kept)


def smart_merge(summary: str, tasks: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Append actionable tasks to summary that are not already covered,
    ensuring critical tasks are not missed without introducing redundancy.
    """
    if not summary or not tasks:
        return summary

    summary_lower = summary.lower()
    to_add = []
    for t in tasks:
        d = t.get("description", "")
        if not d:
            continue
        p = t.get("assigned_to", "unknown").lower()
        if p != "unknown" and p != "" and p in summary_lower:
            continue
        if d.lower() in summary_lower:
            continue
        to_add.append(d)

    if to_add:
        combined = summary.strip()
        if not combined.endswith("."):
            combined += "."
        return combined + " Furthermore, " + " ".join(to_add)
    return summary


def extractive_summarize(
    text: str,
    nlp=None,
    max_sentences: int = 8,
    *,
    tasks: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Extractive summarization fallback using spaCy sentence segmentation + TF-IDF-style scoring.
    Used when the BART model is unavailable (OOM / disk full).

    Scoring criteria (additive):
      - Sentence contains task keywords (STRONG_CUES / STRONG_ACTION_VERBS)
      - Sentence contains named entities (PERSON, ORG, DATE)
      - Sentence position: first 10% and last 10% of transcript weighted higher
      - Sentence length: 10-50 words preferred (too short/long penalised)

    Parameters
    ----------
    text          : Full transcript text.
    nlp           : Optional pre-loaded spaCy model; if None, spaCy is imported and loaded.
    max_sentences : Maximum number of sentences in the output.
    tasks         : Optional task list for smart_merge after extraction.

    Returns
    -------
    Extractive summary string.
    """
    if not text or not text.strip():
        return ""

    # Load spaCy if not provided
    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            # Last-resort: return first 1000 chars
            return text[:1000]

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        return text[:1000]

    n = len(sentences)

    # Keyword sets for scoring (import from module-level lists)
    from services.nlp.task_detection import STRONG_CUES, STRONG_ACTION_VERBS
    all_keyword_patterns = STRONG_CUES + STRONG_ACTION_VERBS

    scores = []
    for i, sent in enumerate(sentences):
        sl = sent.lower()
        score = 0.0

        # 1. Task keyword bonus
        keyword_hits = sum(1 for p in all_keyword_patterns if re.search(p, sl))
        score += keyword_hits * 0.5

        # 2. Named entity bonus
        sent_doc = nlp(sent)
        ne_count = sum(1 for ent in sent_doc.ents if ent.label_ in ("PERSON", "ORG", "DATE", "TIME"))
        score += ne_count * 0.4

        # 3. Position bonus (first and last 10% of transcript)
        position_ratio = i / max(n - 1, 1)
        if position_ratio <= 0.10 or position_ratio >= 0.90:
            score += 0.5

        # 4. Length preference (10-50 words)
        word_count = len(sent.split())
        if 10 <= word_count <= 50:
            score += 0.3
        elif word_count < 5:
            score -= 0.5

        # 5. Penalty for speaker-label-only lines (e.g. "[Nadia]:")
        if re.match(r"^\[?\w[\w\s]*\]?\s*:", sent) and word_count <= 3:
            score -= 1.0

        scores.append((score, i, sent))

    # Sort by score descending, pick top max_sentences
    scores.sort(key=lambda x: -x[0])
    top = scores[:max_sentences]

    # Restore original order
    top.sort(key=lambda x: x[1])

    result = " ".join(s for _, _, s in top)
    result = deduplicate_sentences(result)
    return smart_merge(result, tasks)


def _summarize_chunk(
    chunk_text: str,
    model,
    tokenizer,
    max_length: int,
    min_length: int,
    device: int = -1,
    num_beams: int = 4,
) -> str:
    """Summarize a single text chunk with scaled dynamic min/max lengths."""
    approx_input_tokens = len(
        tokenizer.encode(chunk_text, add_special_tokens=False)
    )
    if approx_input_tokens <= 30:
        return chunk_text.strip()

    safe_max = min(max_length, max(30, approx_input_tokens // 2))
    safe_min = min(min_length, max(10, safe_max // 3))
    if safe_min > safe_max:
        safe_min = max(5, safe_max // 2)

    try:
        import torch

        inputs = tokenizer(
            chunk_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )
        if device == 0 and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=safe_max,
            min_length=safe_min,
            num_beams=num_beams,
            early_stopping=True,
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as exc:
        print(f"  Chunk summarization warning: {exc}")
        return chunk_text


def summarize_text(
    text: str,
    max_length: int = 150,
    min_length: int = 50,
    model_name: str = BART_MODEL,
    device: int = -1,
    max_input_chars: int = 1000,
    summariser=None,
    *,
    tasks: Optional[List[Dict[str, Any]]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    chunk_tokens: Optional[int] = None,
    num_beams: int = 4,
    _info: Optional[dict] = None,
) -> str:
    """
    Summarize *text* using BART via token-aware map-reduce.

    Parameters
    ----------
    text : Raw meeting transcript to summarize.
    max_length : Upper bound on generated summary token length.
    min_length : Lower bound on generated summary token length.
    model_name : HuggingFace model variant.
    device : Device index (-1 for CPU, 0 for CUDA).
    max_input_chars : Deprecated: retained for signature compatibility.
                      Token-aware chunking processes the entire transcript.
    summariser : Pre-loaded (model, tokenizer) tuple (optional).
    tasks : Optional list of extracted tasks for smart-merging into the summary.
    progress : Optional callback progress(done_chunks, total_chunks).
    chunk_tokens : Token budget per chunk (default from config.SUMMARY_CHUNK_TOKENS).
    num_beams : Beam search width passed to model.generate() (default 4).
    _info : Optional dict populated with debug info:
            ``chunk_counts`` (list of chunk counts per level),
            ``levels`` (number of map-reduce levels executed).

    Returns
    -------
    str : Formatted meeting summary with duplicates removed.
    """
    if not text or not text.strip():
        return ""

    effective_chunk_tokens = (
        chunk_tokens if chunk_tokens is not None else SUMMARY_CHUNK_TOKENS
    )

    if _info is not None:
        _info.setdefault("chunk_counts", [])
        _info["levels"] = 0

    if summariser is None:
        model, tokenizer = load_summarizer(model_name, device)
    else:
        model, tokenizer = summariser

    # If model unavailable (OOM/disk full), use extractive fallback
    if model is None or tokenizer is None:
        print("⚠️  BART model not available — using extractive summarization fallback.")
        return extractive_summarize(text, tasks=tasks)

    # Check for very short inputs (<= 30 tokens) -> return as-is
    initial_tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(initial_tokens) <= 30:
        print(f"Input too short ({len(initial_tokens)} tokens) — returning as-is.")
        return smart_merge(text.strip(), tasks)

    level = 1
    current_text = text
    done_chunks = 0

    while level <= 3:
        chunks = chunk_text_by_tokens(
            current_text,
            tokenizer,
            chunk_tokens=effective_chunk_tokens,
            overlap_sentences=2,
        )

        total_chunks = len(chunks)

        if _info is not None:
            _info["chunk_counts"].append(total_chunks)
            _info["levels"] = level

        if total_chunks <= 1:
            # Single-pass: input fits entirely within one chunk
            if progress:
                progress(done_chunks, done_chunks + 1)
            chunk_content = chunks[0] if chunks else current_text
            summary = _summarize_chunk(
                chunk_content, model, tokenizer, max_length, min_length,
                device, num_beams,
            )
            done_chunks += 1
            if progress:
                progress(done_chunks, done_chunks)
            return smart_merge(deduplicate_sentences(summary), tasks)

        # Multi-chunk Map phase
        print(
            f"Map-reduce level {level}: processing {total_chunks} chunks "
            f"(chunk_limit={effective_chunk_tokens}) ..."
        )
        partial_summaries = []
        for i, chunk in enumerate(chunks):
            if progress:
                progress(done_chunks, done_chunks + (total_chunks - i))

            part = _summarize_chunk(
                chunk, model, tokenizer, max_length, min_length, device, num_beams,
            )
            partial_summaries.append(part)
            done_chunks += 1

            if progress:
                progress(done_chunks, done_chunks + (total_chunks - i - 1))

        # Deduplicate across partial summaries (removes chunk-overlap artefacts)
        joined = deduplicate_sentences(" ".join(partial_summaries))
        joined_tokens = len(tokenizer.encode(joined, add_special_tokens=False))

        # Check if reduction is complete or max levels reached
        if joined_tokens <= effective_chunk_tokens or level >= 3:
            # Final synthesis pass if joined partials exceed target max_length
            if joined_tokens > max_length:
                final_summary = _summarize_chunk(
                    joined, model, tokenizer, max_length, min_length,
                    device, num_beams,
                )
            else:
                final_summary = joined
            return smart_merge(deduplicate_sentences(final_summary), tasks)

        # Otherwise reduce in next level
        current_text = joined
        level += 1

    return smart_merge(deduplicate_sentences(current_text), tasks)


def save_summary(summary: str, output_path: str) -> None:
    """Persist a summary string to a plain-text file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(f"Summary saved to: {output_path}")


def load_summary(summary_path: str) -> str:
    """Load a previously-saved summary from disk."""
    with open(summary_path, "r", encoding="utf-8") as fh:
        return fh.read()
