"""Unit tests for multilingual transcription helpers (no real Whisper weights)."""

from unittest.mock import MagicMock, patch

from services.preprocessing.language import (
    arabic_script_ratio,
    ensure_english_transcript,
    needs_english_normalization,
)
from services.preprocessing.transcription import (
    _HuggingFaceAsr,
    is_english_only_whisper_model,
    is_sukhan_model,
    transcribe_audio,
    whisper_transcribe_options,
)


def test_is_english_only_whisper_model():
    assert is_english_only_whisper_model("tiny.en")
    assert not is_english_only_whisper_model("base")
    assert not is_english_only_whisper_model("small")


def test_arabic_script_ratio():
    assert arabic_script_ratio("Hello world") == 0.0
    assert arabic_script_ratio("سلام") == 1.0
    mixed = "Hello " + "سلام"
    assert 0.0 < arabic_script_ratio(mixed) < 1.0


def test_needs_english_normalization_threshold():
    assert not needs_english_normalization("Team standup at nine.")
    assert needs_english_normalization("آج میٹنگ ہے۔")


def test_whisper_transcribe_options_multilingual_translate(monkeypatch):
    monkeypatch.setattr("config.WHISPER_TASK", "translate")
    monkeypatch.setattr("config.WHISPER_LANGUAGE", "auto")
    monkeypatch.setattr("config.WHISPER_INITIAL_PROMPT", "English and Urdu.")
    opts = whisper_transcribe_options(model_name="base")
    assert opts["task"] == "translate"
    assert "language" not in opts
    assert opts["initial_prompt"] == "English and Urdu."


def test_sukhan_options_transcribe_not_translate(monkeypatch):
    monkeypatch.setattr("config.WHISPER_TASK", "translate")
    monkeypatch.setattr("config.WHISPER_INITIAL_PROMPT", "ignore me")
    assert is_sukhan_model("Loudness-ai/urdu-sukhan-asr-v1")
    opts = whisper_transcribe_options(model_name="Loudness-ai/urdu-sukhan-asr-v1")
    assert opts["task"] == "transcribe"
    assert "initial_prompt" not in opts


def test_gated_model_uses_configured_fallback(monkeypatch):
    import services.preprocessing.transcription as transcription

    monkeypatch.setattr(transcription, "_whisper_model", None)
    monkeypatch.setattr("config.WHISPER_FALLBACK", "base")
    loaded = []

    def load(model_id):
        loaded.append(model_id)
        if "sukhan" in model_id:
            raise RuntimeError("403 Client Error. not in the authorized list")
        model = MagicMock()
        model.name = model_id
        return model

    monkeypatch.setattr(transcription, "_load_huggingface_asr", load)
    monkeypatch.setattr(
        transcription,
        "_load_openai_whisper",
        lambda model_size: (_ for _ in ()).throw(AssertionError("numba whisper was loaded")),
    )
    model = transcription.load_whisper("Loudness-ai/urdu-sukhan-asr-v1")
    assert loaded == ["Loudness-ai/urdu-sukhan-asr-v1", "openai/whisper-base"]
    assert model.name == "openai/whisper-base"
    monkeypatch.setattr(transcription, "_whisper_model", None)


def test_huggingface_load_failure_does_not_fall_back_without_fallback(monkeypatch):
    import services.preprocessing.transcription as transcription

    monkeypatch.setattr(transcription, "_whisper_model", None)
    monkeypatch.setattr("config.WHISPER_FALLBACK", "")

    def boom(model_id):
        raise RuntimeError("disk full")

    monkeypatch.setattr(transcription, "_load_huggingface_asr", boom)
    try:
        transcription.load_whisper("Loudness-ai/urdu-sukhan-asr-v1")
    except RuntimeError as exc:
        assert "did not fall back" in str(exc)
    else:
        raise AssertionError("expected the Sukhan load error to propagate")
    monkeypatch.setattr(transcription, "_whisper_model", None)


def test_any_primary_load_failure_uses_fallback(monkeypatch):
    import services.preprocessing.transcription as transcription

    monkeypatch.setattr(transcription, "_whisper_model", None)
    monkeypatch.setattr("config.WHISPER_FALLBACK", "small")
    loaded = []

    def load(model_id):
        loaded.append(model_id)
        if "sukhan" in model_id:
            raise ImportError("cannot import name 'HF_MODELS_DIR' from 'config'")
        model = MagicMock()
        model.name = model_id
        return model

    monkeypatch.setattr(transcription, "_load_huggingface_asr", load)
    model = transcription.load_whisper("Loudness-ai/urdu-sukhan-asr-v1")
    assert loaded == ["Loudness-ai/urdu-sukhan-asr-v1", "openai/whisper-small"]
    assert model.name == "openai/whisper-small"
    monkeypatch.setattr(transcription, "_whisper_model", None)


def test_huggingface_wrapper_forces_sukhan_transcribe():
    seen = {}

    def pipe(path, generate_kwargs=None):
        seen["path"] = path
        seen["generate_kwargs"] = generate_kwargs
        return {"text": "آج رپورٹ بھیج دیں۔"}

    model = _HuggingFaceAsr(pipe, "Loudness-ai/urdu-sukhan-asr-v1")
    out = model.transcribe("clip.wav", task="translate", language="en")
    assert out["text"].startswith("آج")
    assert seen["generate_kwargs"]["task"] == "transcribe"
    assert seen["generate_kwargs"]["language"] == "en"


def test_whisper_transcribe_options_english_only_forces_transcribe(monkeypatch):
    monkeypatch.setattr("config.WHISPER_TASK", "translate")
    opts = whisper_transcribe_options(model_name="tiny.en")
    assert opts["task"] == "transcribe"


def test_ensure_english_skips_mostly_latin():
    text = "Alice will ship the release tomorrow."
    assert ensure_english_transcript(text, translator=MagicMock()) == text


def test_ensure_english_uses_translator():
    urdu = "آج ہمیں رپورٹ بھیجنی ہے۔"
    fake = MagicMock(return_value="We need to send the report today.")
    out = ensure_english_transcript(urdu, translator=fake)
    assert out == "We need to send the report today."
    fake.assert_called_once()


def test_transcribe_audio_invokes_whisper_with_options(monkeypatch):
    seen = {}

    class FakeModel:
        def transcribe(self, audio_path, **kwargs):
            seen["kwargs"] = kwargs
            return {"text": "Sarah will update the dashboard."}

    monkeypatch.setattr("config.WHISPER_TASK", "translate")
    monkeypatch.setattr("config.WHISPER_LANGUAGE", "auto")
    monkeypatch.setattr("config.WHISPER_INITIAL_PROMPT", "")

    path = __file__
    with patch("os.path.exists", return_value=True):
        out = transcribe_audio(path, model=FakeModel(), model_size="base")
    assert out == "Sarah will update the dashboard."
    assert seen["kwargs"].get("task") == "translate"


def test_urdu_speaker_line_stays_latin():
    from services.preprocessing.urdu_norm import render_urdu_line

    line = render_urdu_line("Speaker 1 – احمد (Project Manager):", {})
    assert line.startswith("Speaker 1 – Ahmad")
    assert "احمد" not in line


def test_urdu_assignment_keeps_name_weekday_and_verb():
    from services.preprocessing.urdu_norm import render_urdu_line

    line = render_urdu_line(
        "علی، آپ جمعرات تک ڈیٹا بیس کی کارکردگی بہتر کر دیں۔",
        {"علی": "Ali"},
    )
    assert line is not None
    assert "Ali" in line
    assert "Thursday" in line
    assert "please improve" in line
    assert "database" in line

