"""Jero: an autonomous, voice-operated, OS-level AI assistant.

The package is organized into independent async modules:

- ``jero.core``           : config, message bus, executor, logging, orchestrator
- ``jero.brain``          : NVIDIA NIM + Groq LLM providers
- ``jero.audio``          : faster-whisper STT + Piper TTS pipeline
- ``jero.autonomy``       : idle-time self-improvement engine
- ``jero.os_interaction`` : sandboxed filesystem / app / document handlers
"""

__version__ = "0.1.0"
