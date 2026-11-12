"""spaCy model loading must survive the pure-Python msgpack stand-in."""

import sys
import types

from services.native_compat import _fill_unpacker, install
from services.nlp import task_detection

install()
from srsly.msgpack.fallback import Packer  # noqa: E402


def test_msgpack_stand_in_allows_integer_map_keys():
    module = types.ModuleType("srsly.msgpack._unpacker")
    _fill_unpacker(module)
    packed = Packer().pack({1: "task", 2: "date"})
    assert module.unpackb(packed) == {1: "task", 2: "date"}


def test_pasted_transcript_completes_when_spacy_cannot_load(monkeypatch):
    monkeypatch.setattr("services.pipeline._get_nlp", lambda: None)
    monkeypatch.setattr("services.pipeline._get_summariser", lambda: (None, None))
    from services.pipeline import run_clario_pipeline

    result = run_clario_pipeline(text_input="Sara will send the notes by Friday.")
    assert result is not None
    assert "Sara will send the notes by Friday." in result["transcript"]
    assert result["summary"]
    assert result["tasks"]


def test_load_nlp_falls_back_when_the_model_cannot_be_read(monkeypatch):
    previous_nlp = task_detection._nlp
    previous_flag = task_detection._nlp_unavailable
    fake = types.ModuleType("spacy")

    def _load(_name):
        raise ValueError("<class 'int'> is not allowed for map key")

    fake.load = _load
    monkeypatch.setitem(sys.modules, "spacy", fake)
    monkeypatch.setattr(task_detection, "_nlp", None)
    monkeypatch.setattr(task_detection, "_nlp_unavailable", False)
    try:
        assert task_detection.load_nlp() is None
        assert task_detection._nlp_unavailable is True
    finally:
        task_detection._nlp = previous_nlp
        task_detection._nlp_unavailable = previous_flag
