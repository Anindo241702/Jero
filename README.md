# Jero: Autonomous AI Assistant

**Jero** (pronounced like Zero) is an advanced, autonomous AI assistant designed for high-performance interaction on Windows. Built for efficiency, Jero is not just a chatbot—it is an OS-level agent that grows, adapts, and executes complex tasks autonomously.

## 🚀 Project Goal
To create a fully autonomous, voice-operated J.A.R.V.I.S.-inspired system that integrates deep into your OS, manages your workflow, and continuously upgrades its own capabilities.

## 🛠 Core Capabilities
* **Autonomous Development:** Jero can write, debug, and implement its own code modules.
* **Self-Learning:** Learns from past mistakes and optimizes its own performance.
* **OS-Level Control:** Possesses admin/root access to manage files, open applications, and automate system tasks.
* **Human-like Interaction:** Speaks and understands natural language (with native Bangla support).
* **Adaptability:** Learns your specific habits and workspace preferences over time.
* **Human-like Browsing:** Capable of web navigation and research to gather information.
* **Content Generation:** Creates, edits, and formats documents (Word/PDF) and handles system-level typing.
* **Continuous Self-Upgrade:** In idle time, Jero proactively uses the **Groq API** to research new features and the **NVIDIA NIM API** to generate the code and implement those features autonomously.

## 🏗 Architecture
Jero uses a **Producer-Consumer asynchronous pipeline** for zero-latency performance:
* **Brain:** NVIDIA NIM API (for heavy reasoning/coding) & Groq API (for fast research/ideation).
* **STT Engine:** `faster-whisper` (optimized for CPU).
* **TTS Engine:** `Piper` (ultra-fast, local Bangla synthesis).
* **Interface:** Integrated as a plugin for [Flow Launcher](https://www.flowlauncher.com/).
* **Automation:** Python-based OS interaction (PyAutoGUI, OS-native APIs).

## ⚙️ Requirements
* **Python 3.10+**
* **Hardware:** 8GB RAM, 4-core CPU (Optimized for local/cloud hybrid inference).
* **Environment:** A `.env` file containing your `NVIDIA_API_KEY` and `GROQ_API_KEY`.

## 🛠 Setup for Devin
1. **Clone/Fork:** This project is forked from [Original Repo Name].
2. **Environment:** Set up a virtual environment: `python -m venv venv`.
3. **Dependencies:** `pip install -r requirements.txt`.
4. **Config:** Update `config.json` with system paths and API credentials.

## ⚠️ Important Note
Jero is optimized for **speed and autonomy**. All AI tasks, including self-coding and research, run in background threads to keep the system responsive. **Do not modify the main execution loop to be sequential.**

## 📦 Project Structure
```
jero/
├── core/            # config, message bus, shared executor, logging, orchestrator
├── brain/           # NVIDIA NIM (reasoning/coding) + Groq (research) providers
├── audio/           # faster-whisper STT + Piper TTS + producer-consumer pipeline
├── autonomy/        # idle-time self-improvement engine (scaffold)
├── os_interaction/  # sandboxed filesystem / app / document handlers (scaffold)
└── utils/           # async helpers (executor offload, retry/backoff)
main.py              # async entrypoint
```

The whole system is `asyncio`-based. Every blocking call (model inference, audio
I/O, OS calls) is offloaded to a single shared, bounded `ThreadPoolExecutor`, so
the event loop never blocks. Modules communicate over an `asyncio.Queue` message
bus (producer-consumer).

## 🧑‍💻 Development
```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   Unix: source venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env               # add NVIDIA_API_KEY, GROQ_API_KEY
cp config.example.json config.json # adjust models/paths as needed

ruff check .                       # lint
pytest -q                          # tests (mock LLM/audio; no keys/models needed)
python main.py                     # run Jero
```

### Models
faster-whisper (`base`/`small`) downloads automatically into `models/whisper/`
on first run. For Piper, set `audio.tts.voice_url` and `audio.tts.config_url` in
`config.json` to the Bangla voice's `.onnx` / `.onnx.json` URLs; Jero downloads
them into `models/piper/` and verifies their presence before starting the
pipeline. The `models/` directory is git-ignored.
