# Multilingual transcription evaluation

Clario stores **English** transcripts for English, Urdu, and mixed meetings.

## Models

| Role | Model | Why |
|------|--------|-----|
| Primary ASR | [Loudness-ai/urdu-sukhan-asr-v1](https://huggingface.co/Loudness-ai/urdu-sukhan-asr-v1) | Fine-tune of Whisper large-v3-turbo (809M). Reported 19.7% WER on Urdu→English code-switching and 4.9% WER on English. Writes Urdu in Urdu script and English words in Latin letters. Translation is out of scope for this model. |
| English wording | `Helsinki-NLP/opus-mt-ur-en` | Turns any remaining Urdu script into English before summaries and tasks. |
| Fallback ASR | OpenAI Whisper `large-v3` (1.55B) | Used when Sukhan cannot be downloaded. `task=translate` emits English directly. Weaker on Urdu–English switches than Sukhan. |

Sukhan weights are gated (CC BY-NC 4.0). On each machine: request access on the model page, then `hf auth login`. If that download fails, Clario stops and names the error. It does not switch to `large-v3` unless `CLARIO_WHISPER_MODEL` is set to that checkpoint.

## How to reproduce local fixtures

```bash
python scripts/generate_multilingual_samples.py
python scripts/eval_multilingual_transcription.py --models tiny base small
```

Those gTTS clips are clean single-voice audio. They are not a substitute for the Sukhan paper’s noisy, multi-speaker sets, and they do not by themselves prove the published 19.7% / 4.9% figures.

## Known limitations

- **Sukhan does not translate.** Urdu speech becomes Urdu text first. English output depends on the Marian Urdu→English step, which can miss names, code-switched fragments, and long meetings.
- **Gated download.** Without an approved Hugging Face token, Sukhan never runs and large-v3 is used.
- **CPU cost.** Sukhan is about 1.6 GB in Transformers fp16; large-v3 is larger and slower. First meeting after a cold start downloads weights.
- **Roman Urdu** (Latin-script Urdu) is rewritten by `services/preprocessing/roman_urdu.py`
  before task detection. It is not sent through Marian unless Arabic script remains.
  Cover is limited to the meeting lexicon; lines the lexicon cannot read stay as evidence
  and may be flagged in **Needs review**.
- **Real rooms** (overlap, noise, accents) will be worse than gTTS or the model card’s curated sets.
- **English-only `*.en` checkpoints** cannot hear Urdu. Do not set `CLARIO_WHISPER_MODEL=tiny.en` if meetings include Urdu.
