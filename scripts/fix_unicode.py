"""
scripts/fix_unicode.py
----------------------
Strip emoji / non-cp1252 characters from print() calls in source files.

This is a one-shot maintenance script needed only if you are running on
a Windows terminal whose encoding is cp1252 (the default before Python 3.15).
app.py already reconfigures stdout/stderr to UTF-8 at startup, which is the
preferred long-term fix.

Run from the PROJECT ROOT (not from scripts/):
    python scripts/fix_unicode.py

Files that still contain emoji in print() calls (as of the architecture audit):
    generate_samples.py:41
    seed.py                (seed.py has been rewritten; now uses UTF-8 stdout)
    services/evaluation.py (lines 78, 90, 179, 206)
"""

import os
import pathlib
import re
import sys

# Resolve project root relative to this script's location
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

FILES = [
    ROOT / "models" / "database.py",
    ROOT / "services" / "pipeline.py",
    ROOT / "services" / "preprocessing" / "transcription.py",
    ROOT / "services" / "nlp" / "summarization.py",
    ROOT / "services" / "nlp" / "task_detection.py",
    ROOT / "services" / "evaluation.py",
    ROOT / "utils" / "visualization.py",
    ROOT / "utils" / "helpers.py",
    ROOT / "generate_samples.py",
]

# Broad emoji + common symbols that cp1252 cannot encode
EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FFFF"   # supplemental symbols
    "\U00002600-\U000027BF"   # misc symbols / dingbats
    "\U00002300-\U000023FF"   # technical symbols
    "\u2705\u274C\u26A0\u2764\u2B50\u2714\u2716"
    "]+",
    flags=re.UNICODE,
)

for path in FILES:
    if not path.exists():
        print(f"SKIP (not found): {path}")
        continue

    original = path.read_text(encoding="utf-8")
    cleaned  = EMOJI.sub("", original)

    if cleaned != original:
        path.write_text(cleaned, encoding="utf-8")
        print(f"Fixed: {path}")
    else:
        print(f"OK (no changes): {path}")

print("Done.")
