"""
Generate gTTS audio fixtures for Urdu, English, and code-switched meetings.

Outputs under tests/fixtures/multilingual_audio/ with matching English
reference transcripts (expected output after Whisper translate).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "tests", "fixtures", "multilingual_audio")


SAMPLES = {
    "english_standup": {
        "audio": "english_standup.mp3",
        "reference": "english_standup.txt",
        "segments": [{"lang": "en", "text": (
            "Good morning team. Sarah please review the pull requests today. "
            "John will deploy the schema by tomorrow afternoon."
        )}],
    },
    "urdu_standup": {
        "audio": "urdu_standup.mp3",
        "reference": "urdu_standup.txt",
        "segments": [{"lang": "ur", "text": (
            "صبح بخیر ٹیم۔ سارہ آج پل ریکویسٹس کا جائزہ لے۔ "
            "جون کل دوپہر تک ڈیٹا بیس اسکیما لگائے گا۔"
        )}],
        "reference_en": (
            "Good morning team. Sarah will review the pull requests today. "
            "John will deploy the database schema by tomorrow afternoon."
        ),
    },
    "mixed_codeswitch": {
        "audio": "mixed_codeswitch.mp3",
        "reference": "mixed_codeswitch.txt",
        "segments": [
            {"lang": "en", "text": "Welcome everyone to the weekly sync."},
            {"lang": "ur", "text": "علی براہ کرم کل تک رپورٹ بھیجیں۔"},
            {"lang": "en", "text": "Thanks Ali, and please upload the notes after the call."},
        ],
        "reference_en": (
            "Welcome everyone to the weekly sync. "
            "Ali please send the report by tomorrow. "
            "Thanks Ali, and please upload the notes after the call."
        ),
    },
}


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return ""


def synthesize_segment(text: str, lang: str, dest: str) -> None:
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(dest)


def concat_mp3(paths: list[str], dest: str, ffmpeg: str) -> None:
    listing = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        for path in paths:
            listing.write(f"file '{path.replace(chr(92), '/')}'\n")
        listing.close()
        subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", listing.name, "-c", "copy", dest],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        os.unlink(listing.name)


def main() -> int:
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        print("FFmpeg is required to build mixed-language samples.", file=sys.stderr)
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    with tempfile.TemporaryDirectory() as tmp:
        for key, spec in SAMPLES.items():
            audio_path = os.path.join(OUT_DIR, spec["audio"])
            ref_path = os.path.join(OUT_DIR, spec["reference"])
            parts = []
            for idx, segment in enumerate(spec["segments"]):
                part = os.path.join(tmp, f"{key}_{idx}.mp3")
                synthesize_segment(segment["text"], segment["lang"], part)
                parts.append(part)
            if len(parts) == 1:
                shutil.copy(parts[0], audio_path)
            else:
                concat_mp3(parts, audio_path, ffmpeg)
            reference = spec.get("reference_en") or spec["segments"][0]["text"]
            with open(ref_path, "w", encoding="utf-8") as fh:
                fh.write(reference.strip())
            manifest.append({
                "id": key,
                "audio": spec["audio"],
                "reference": spec["reference"],
                "description": key.replace("_", " "),
            })
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({"samples": manifest}, fh, indent=2)
    print(f"Wrote {len(manifest)} samples to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
