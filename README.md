Speech-to-Text Web App (Streamlit + Bash)
===========================================

Overview
--------
A Streamlit web app for Windows that transcribes microphone audio in (near) real-time using Faster-Whisper. Runs in Microsoft Edge browser from Bash terminal.

Features
--------
- Modern dark-themed web UI
- Start/Stop recording
- Real-time transcription updates
- Elapsed recording time
- Clear / Copy / Save transcript
- Configurable language and model size
- CPU or GPU device selection

Requirements
------------
- Windows with Python 3.10+
- Bash terminal (Git Bash, WSL, or MSYS2)
- Microphone access

Install (One-time)
------------------
1. In Bash, install dependencies:

```bash
pip install -r requirements.txt
# Install torch for your platform. Example (CPU-only):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you want GPU acceleration:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Run
---
In Bash from the project directory:

```bash
streamlit run main.py
```

Streamlit will:
- Launch automatically in your default browser (Edge, Chrome, etc.)
- Display the app at `http://localhost:8501`
- Show a note if it can't auto-launch (you can copy-paste the URL into Edge manually)

First run note
--------------
- Faster-Whisper will download the model weights on first run (may take a few minutes and disk space)
- Model downloads are cached locally for subsequent runs

Files
-----
- `main.py` - Streamlit web app
- `audio_recorder.py` - Audio capture using sounddevice
- `transcriber.py` - Worker thread running Faster-Whisper
- `requirements.txt` - Python dependencies

Tips
----
- To stop the app, press `Ctrl+C` in the Bash terminal
- To change the model size, use the sidebar (base, small, medium)
- To select language, enter code in sidebar (e.g., 'en', 'es', 'fr', 'de')
- Select CPU or GPU in the sidebar based on your hardware
