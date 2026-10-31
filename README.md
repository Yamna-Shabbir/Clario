# Clario – AI Meeting Assistant

Clario is an AI-powered web application that transforms meeting audio into useful, structured information. It transcribes meetings, generates concise summaries, extracts action items with assignees, deadlines and priorities, and presents meeting insights through an analytics dashboard.

## Features

- Audio transcription using Whisper
- Meeting summarization using BART
- Action-item and task extraction using spaCy
- Task assignment, deadlines and priority detection
- Meeting history and search
- Business analytics and visualizations
- Knowledge graphs and task completion tracking
- PDF export

## Prerequisites

- Python 3.9 or higher
- FFmpeg
- Git

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Yamna-Shabbir/Clario.git
cd clario_ai_meeting_assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Run the application

```bash
python app.py
```

Open your browser and visit:

`http://127.0.0.1:5000`

## Testing with Demo Data

To populate the database with demonstration meeting data, run:

```bash
python seed.py
```

**Warning:** This script may replace existing database records. Use it only in a test environment.

## Notes

- FFmpeg is required for audio processing.
- Whisper and BART models may be downloaded automatically when first used.
- Initial model downloads can take time and require an internet connection.
- Keep API keys, credentials and environment variables out of the Git repository.