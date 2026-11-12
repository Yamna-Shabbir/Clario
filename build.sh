#!/usr/bin/env bash
# Render build for Clario.
# Free tier installs core deps only (no Torch/Whisper) so the web service fits
# in 512 MB. Set CLARIO_FREE_TIER=False and install requirements.txt for a
# paid instance that can run the full ML pipeline.
set -o errexit
set -o nounset
set -o pipefail

python -m pip install --upgrade pip

if [[ "${CLARIO_FREE_TIER:-True}" =~ ^([Tt]rue|1|yes|YES)$ ]]; then
  echo "[build] Free tier: installing requirements-core.txt (no Torch/Whisper)."
  pip install -r requirements-core.txt
else
  echo "[build] Full install: requirements.txt (includes ML)."
  pip install -r requirements.txt
fi

python -m spacy download en_core_web_sm
