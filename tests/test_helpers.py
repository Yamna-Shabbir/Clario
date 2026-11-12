"""
tests/test_helpers.py
---------------------
Unit tests for all public functions in utils/helpers.py.
Covers normal and edge inputs for:
  - save_json / load_json
  - clean_text / chunk_text
  - current_timestamp / today_date
  - ensure_dir / list_files
"""

import json
import os
import re
from datetime import datetime
import pytest

from utils.helpers import (
    save_json,
    load_json,
    clean_text,
    chunk_text,
    current_timestamp,
    today_date,
    ensure_dir,
    list_files,
)


class TestJsonHelpers:

    def test_save_and_load_json_roundtrip(self, tmp_path):
        data = {"meeting": "Quarterly Review", "attendees": ["Alice", "Bob"], "count": 42}
        out_file = str(tmp_path / "subdir" / "data.json")
        save_json(data, out_file)

        assert os.path.exists(out_file)
        loaded = load_json(out_file)
        assert loaded == data

    def test_save_json_unicode_and_nested(self, tmp_path):
        data = {"greeting": "こんにちは", "emojis": "🎙️⚡", "nested": {"a": [1, 2, 3]}}
        out_file = str(tmp_path / "unicode.json")
        save_json(data, out_file)

        loaded = load_json(out_file)
        assert loaded == data

    def test_save_json_empty_dict(self, tmp_path):
        out_file = str(tmp_path / "empty.json")
        save_json({}, out_file)
        assert load_json(out_file) == {}

    def test_load_json_nonexistent_raises_file_not_found(self, tmp_path):
        missing = str(tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError) as exc_info:
            load_json(missing)
        assert "not found" in str(exc_info.value).lower()


class TestTextHelpers:

    def test_clean_text_normal(self):
        raw = "   This   is   a   meeting \n\n with multiple    spaces.  "
        assert clean_text(raw) == "This is a meeting with multiple spaces."

    def test_clean_text_empty_and_whitespace(self):
        assert clean_text("") == ""
        assert clean_text("   \n\t\r   ") == ""

    def test_clean_text_already_clean(self):
        text = "Meeting summary is already clean."
        assert clean_text(text) == text

    def test_clean_text_unicode(self):
        text = "  チームミーティング  進捗確認  "
        assert clean_text(text) == "チームミーティング 進捗確認"

    def test_chunk_text_normal(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, max_chars=35)
        assert len(chunks) >= 2
        # Verify reconstructed text contains all sentences
        combined = " ".join(chunks)
        assert "First sentence." in combined
        assert "Fourth sentence." in combined

    def test_chunk_text_empty(self):
        assert chunk_text("") == []

    def test_chunk_text_fits_in_single_chunk(self):
        text = "Short text."
        chunks = chunk_text(text, max_chars=100)
        assert chunks == ["Short text."]

    def test_chunk_text_no_sentence_breaks(self):
        # Long string with no periods
        text = "word " * 100
        text = text.strip()
        chunks = chunk_text(text, max_chars=50)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestDateHelpers:

    def test_current_timestamp_default_format(self):
        ts = current_timestamp()
        assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", ts)

    def test_current_timestamp_custom_format(self):
        year = current_timestamp("%Y")
        assert year == str(datetime.now().year)

    def test_today_date_default_format(self):
        today = today_date()
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", today)
        assert today == datetime.now().strftime("%Y-%m-%d")

    def test_today_date_custom_format(self):
        fmt = "%d/%m/%Y"
        assert today_date(fmt) == datetime.now().strftime(fmt)


class TestDirectoryHelpers:

    def test_ensure_dir_creates_nested_directory(self, tmp_path):
        target = str(tmp_path / "deep" / "nested" / "dir")
        ret = ensure_dir(target)
        assert os.path.isdir(target)
        assert os.path.isabs(ret)

    def test_ensure_dir_idempotent_on_existing(self, tmp_path):
        target = str(tmp_path / "existing")
        os.makedirs(target, exist_ok=True)
        ret = ensure_dir(target)
        assert os.path.isdir(ret)

    def test_list_files_all(self, tmp_path):
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "b.mp3").write_text("b", encoding="utf-8")
        (tmp_path / "sub").mkdir()

        files = list_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in files]
        assert basenames == ["a.txt", "b.mp3"]

    def test_list_files_extension_filter_with_and_without_dot(self, tmp_path):
        (tmp_path / "file1.mp3").write_text("1", encoding="utf-8")
        (tmp_path / "file2.WAV").write_text("2", encoding="utf-8")
        (tmp_path / "file3.mp3").write_text("3", encoding="utf-8")

        # with dot
        mp3s = list_files(str(tmp_path), extension=".mp3")
        assert len(mp3s) == 2
        # without dot and case insensitivity
        wavs = list_files(str(tmp_path), extension="wav")
        assert len(wavs) == 1
        assert os.path.basename(wavs[0]) == "file2.WAV"

    def test_list_files_nonexistent_dir_returns_empty(self, tmp_path):
        assert list_files(str(tmp_path / "nonexistent")) == []

