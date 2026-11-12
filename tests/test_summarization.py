"""
tests/test_summarization.py
---------------------------
Unit tests for token-aware map-reduce summarization in Clario.
Uses fast mock model and tokenizer objects (zero heavy ML dependencies loaded).
"""

import pytest
from services.nlp.summarization import (
    chunk_text_by_tokens,
    deduplicate_sentences,
    smart_merge,
    summarize_text,
)


# ---------------------------------------------------------------------------
# Mock BART Tokenizer and Model
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Fast deterministic stand-in for HuggingFace AutoTokenizer."""

    def encode(self, text: str, add_special_tokens: bool = False):
        if not text or not text.strip():
            return []
        # Return a list of integer IDs (1 ID per whitespace-separated word)
        return [hash(w) % 100000 + 1 for w in text.split()]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, str):
            return token_ids
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], str):
            return token_ids[0]
        tokens = token_ids[0] if (token_ids and isinstance(token_ids[0], list)) else token_ids
        return f"Summary covering {len(tokens)} points."

    def __call__(self, text, return_tensors=None, max_length=None, truncation=None):
        ids = self.encode(text, add_special_tokens=False)
        return {"input_ids": [ids]}


class FakeModel:
    """Fast stand-in for BART AutoModelForSeq2SeqLM."""

    def __init__(self, preset_outputs=None):
        self.generate_calls = []
        self.preset_outputs = list(preset_outputs) if preset_outputs else []
        self._preset_idx = 0

    def generate(
        self,
        input_ids,
        max_length: int = 150,
        min_length: int = 50,
        num_beams: int = 4,
        early_stopping: bool = True,
        **kwargs,
    ):
        self.generate_calls.append({
            "input_tokens": len(input_ids[0]),
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "early_stopping": early_stopping,
        })
        if self._preset_idx < len(self.preset_outputs):
            out = self.preset_outputs[self._preset_idx]
            self._preset_idx += 1
            return out
        # Return a realistic token ID sequence
        count = min(max_length, max(min_length, len(input_ids[0]) // 3))
        return [[101] * count]


@pytest.fixture
def fake_summariser():
    """Return a fresh (FakeModel, FakeTokenizer) pair."""
    return FakeModel(), FakeTokenizer()


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestSummarizeEdgeCases:
    """Empty and very short inputs."""

    def test_summarize_empty(self, fake_summariser):
        model, tok = fake_summariser
        assert summarize_text("", summariser=(model, tok)) == ""
        assert summarize_text("   \n\t  ", summariser=(model, tok)) == ""
        assert len(model.generate_calls) == 0

    def test_summarize_very_short(self, fake_summariser):
        model, tok = fake_summariser
        short_text = "Alice discussed the project plan with Bob today."
        res = summarize_text(short_text, summariser=(model, tok))
        assert res == short_text
        # Model should not be called for <= 30 tokens
        assert len(model.generate_calls) == 0


class TestChunking:
    """Token-aware sentence chunking."""

    def test_no_chunk_exceeds_token_limit(self, fake_summariser):
        _, tok = fake_summariser
        text = " ".join([f"Sentence {i} covers key point {i}." for i in range(100)])
        limit = 40
        chunks = chunk_text_by_tokens(text, tok, chunk_tokens=limit, overlap_sentences=2)
        assert len(chunks) > 1
        for idx, chunk in enumerate(chunks):
            chunk_toks = len(tok.encode(chunk, add_special_tokens=False))
            assert chunk_toks <= limit, f"Chunk {idx} has {chunk_toks} tokens, exceeded limit {limit}!"

    def test_adversarial_runon_sentence_chunking(self, fake_summariser):
        """A huge run-on sentence with no punctuation must still respect the token limit."""
        _, tok = fake_summariser
        runon = "word " * 250  # 250 words without punctuation
        limit = 50
        chunks = chunk_text_by_tokens(runon, tok, chunk_tokens=limit, overlap_sentences=0)
        assert len(chunks) >= 5
        for chunk in chunks:
            assert len(tok.encode(chunk, add_special_tokens=False)) <= limit

    def test_chunk_overlap(self, fake_summariser):
        _, tok = fake_summariser
        sentences = [f"Unique sentence {i} discussing agenda." for i in range(20)]
        text = " ".join(sentences)
        chunks = chunk_text_by_tokens(text, tok, chunk_tokens=25, overlap_sentences=2)
        assert len(chunks) >= 2
        # Verify that chunk 1 starts with sentences present in chunk 0
        chunk_0_sentences = [s.strip() for s in chunks[0].split(".") if s.strip()]
        chunk_1_sentences = [s.strip() for s in chunks[1].split(".") if s.strip()]
        overlap_found = any(s in chunk_0_sentences for s in chunk_1_sentences[:2])
        assert overlap_found, "Consecutive chunks should share overlap sentences"


class TestMapReduceSummarization:
    """Hierarchical map-reduce execution."""

    def test_summarize_single_chunk(self, fake_summariser):
        model, tok = fake_summariser
        # Create text that is > 30 tokens but fits in 700 tokens
        sentences = [f"Team member {i} reported positive progress on milestone {i}." for i in range(10)]
        text = " ".join(sentences)
        res = summarize_text(text, summariser=(model, tok), chunk_tokens=700)
        assert "Summary covering" in res
        assert len(model.generate_calls) == 1
        call = model.generate_calls[0]
        assert call["num_beams"] == 4
        assert call["early_stopping"] is True

    def test_summarize_multichunk_map_reduce(self, fake_summariser):
        model, tok = fake_summariser
        # 60 sentences, chunk_tokens=50 -> forces multiple chunks + reduce pass
        sentences = [f"Topic {i}: Alice and Bob reviewed architecture component {i} in detail." for i in range(60)]
        text = " ".join(sentences)
        res = summarize_text(text, summariser=(model, tok), chunk_tokens=50)
        assert isinstance(res, str)
        assert len(res) > 0
        # Model should have been called multiple times (Map chunks + Reduce consolidation)
        assert len(model.generate_calls) > 1

    def test_progress_callback(self, fake_summariser):
        model, tok = fake_summariser
        sentences = [f"Discussion item {i} regarding budget and delivery timeline." for i in range(40)]
        text = " ".join(sentences)

        progress_history = []

        def on_progress(done, total):
            progress_history.append((done, total))

        summarize_text(
            text,
            summariser=(model, tok),
            chunk_tokens=40,
            progress=on_progress,
        )

        assert len(progress_history) > 0
        # Done should end equal to total
        final_done, final_total = progress_history[-1]
        assert final_done == final_total
        # Done count should increase monotonically
        done_counts = [d for d, _ in progress_history]
        assert done_counts == sorted(done_counts)

    def test_max_input_chars_deprecated_not_truncating(self, fake_summariser):
        """max_input_chars is deprecated and must no longer truncate long transcripts."""
        model, tok = fake_summariser
        # Generate > 2,000 characters (> 350 tokens)
        sentences = [f"Crucial item {i} with vital details." for i in range(60)]
        text = " ".join(sentences)
        assert len(text) > 2000

        # Pass old default max_input_chars=1000
        summarize_text(
            text,
            max_input_chars=1000,
            summariser=(model, tok),
            chunk_tokens=100,
        )

        # If it were truncated at 1000 chars, it would only have ~160 tokens (~2 chunks).
        # With full map-reduce chunking, 5 chunks are processed.
        assert len(model.generate_calls) >= 4, (
            "Expected multiple map-reduce calls covering all chunks beyond the 1000-char mark"
        )


class TestSmartMerge:
    """Smart merge of missing action items."""

    def test_smart_merge_appends_missing_tasks(self):
        summary = "Alice and Bob finalized the quarterly deliverables."
        tasks = [
            {"description": "John must review the security compliance audit.", "assigned_to": "John"},
            {"description": "Bob will deploy the database.", "assigned_to": "Bob"},
        ]
        merged = smart_merge(summary, tasks)
        # John is not mentioned in summary -> must be appended
        assert "John must review the security compliance audit" in merged
        assert "Furthermore," in merged
        # Bob is already mentioned -> should not duplicate Bob's task
        assert "Bob will deploy" not in merged

    def test_smart_merge_already_covered(self):
        summary = "Alice will finish the report by Friday."
        tasks = [{"description": "Alice will finish the report by Friday.", "assigned_to": "Alice"}]
        merged = smart_merge(summary, tasks)
        assert merged == summary
        assert "Furthermore" not in merged


class TestDeduplication:
    """Deduplication of repeated or near-repeated sentences."""

    def test_deduplicate_sentences_unit(self):
        # 1. Exact duplicate
        text1 = "Alice finalized the budget. Alice finalized the budget."
        assert deduplicate_sentences(text1) == "Alice finalized the budget."

        # 2. Case and punctuation difference
        text2 = "Dave investigated the database locking issue. dave investigated the database locking issue!"
        assert deduplicate_sentences(text2) == "Dave investigated the database locking issue."

        # 3. Near-duplicate with difflib ratio > 0.9 (keep first)
        text3 = "Sarah reviewed all pending pull requests for user management. Sarah reviewed all pending pull requests for user management today."
        assert deduplicate_sentences(text3) == "Sarah reviewed all pending pull requests for user management."

        # 4. Distinct sentences (ratio <= 0.9) must be preserved
        text4 = "First key decision made. Second completely different agenda item discussed."
        assert deduplicate_sentences(text4) == text4

    def test_summarize_removes_model_generated_duplicates(self):
        """When the model repeats sentences within a single summary, duplicates are stripped."""
        repeated_summary = (
            "Alice completed the security compliance audit. "
            "Alice completed the security compliance audit. "
            "Bob deployed the backend services."
        )
        model = FakeModel(preset_outputs=[[[repeated_summary]]])
        tok = FakeTokenizer()

        input_text = " ".join([f"Discussion sentence {i} on architecture." for i in range(15)])
        res = summarize_text(input_text, summariser=(model, tok), chunk_tokens=500)

        # Alice's sentence must appear only once
        assert res.count("Alice completed the security compliance audit.") == 1
        assert "Bob deployed the backend services." in res

    def test_summarize_removes_chunk_overlap_duplicates(self):
        """
        When chunk N ends with sentence A and chunk N+1 begins with sentence A
        due to 2-sentence chunk overlap, deduplication after map step strips the duplicate.
        """
        chunk0_summary = (
            "John reviewed the financial operating budget. "
            "Alice must finalize the SOC 2 compliance document."
        )
        chunk1_summary = (
            "Alice must finalize the SOC 2 compliance document. "
            "Mark verified the Stripe billing webhooks."
        )
        # Presets: map chunk 0, map chunk 1, then reduce pass if called
        model = FakeModel(preset_outputs=[
            [[chunk0_summary]],
            [[chunk1_summary]],
            [[chunk0_summary + " " + "Mark verified the Stripe billing webhooks."]],
        ])
        tok = FakeTokenizer()

        sentences = [f"Quarterly review sentence {i} discussing cross-functional milestones." for i in range(25)]
        text = " ".join(sentences)

        res = summarize_text(text, summariser=(model, tok), chunk_tokens=40)

        # Alice's action item appeared in both chunk summaries; it must appear only once in final result
        assert res.count("Alice must finalize the SOC 2 compliance document.") == 1
        assert "John reviewed the financial operating budget." in res
        assert "Mark verified the Stripe billing webhooks." in res
