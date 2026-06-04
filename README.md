# Jero: AI Assistant (Optimized)

Jero (pronounced like Zero), A high-performance, low-latency AI assistant built for Windows systems with limited hardware (8GB RAM, 4-core CPU). This project focuses on **asynchronous processing** to ensure an "instant" response feel using local engines and cloud-accelerated inference.

## 🚀 Project Goal
To create a responsive, J.A.R.V.I.S.-inspired assistant named **Jero** that:
* **Understands Bangla (bn)** natively.
* **Minimizes Latency** using an asynchronous pipeline (STT → LLM → TTS).
* **Conserves Resources** by offloading LLM inference to NVIDIA NIM while keeping audio processing local.

## 🏗 Architecture
This project utilizes a **Producer-Consumer pipeline** to prevent UI blocking:
* **STT Engine:** `faster-whisper` (optimized for CPU).
* **Brain:** NVIDIA NIM API (Llama 3.1 / Nemotron) with streaming enabled.
* **TTS Engine:** `Piper` (local, ultra-fast Bangla voice synthesis).
* **Interface:** Integrated as a plugin for [Flow Launcher](https://www.flowlauncher.com/).

## ⚙️ Requirements
* **Python 3.10+**
* **Hardware:** 8GB RAM minimum, 4-core CPU.
* **Environment:** A `.env` file containing your `NVIDIA_API_KEY`.

## 🛠 Setup for Devin
1. **Clone/Fork:** This project is forked from [Original Repo Name].
2. **Environment:** Set up a virtual environment: `python -m venv venv`.
3. **Dependencies:** `pip install -r requirements.txt`.
4. **Config:** Update `config.json` with your preferred paths and settings.

## ⚠️ Important Note
This project is optimized for **latency**. All AI tasks run in background threads to keep the UI responsive. **Do not modify the main execution loop to be sequential.**
