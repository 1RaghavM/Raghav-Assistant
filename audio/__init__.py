"""Audio processing module."""

from .stt import VoiceInput, SambanovaSTT, WakeWordListener, get_voice_input
from .tts import speak, get_tts, EdgeTTS, VOICES

# Streaming TTS (optional - requires realtimetts)
try:
    from .streaming_tts import (
        StreamingTTS, ChunkedStreamingTTS, get_streaming_tts,
        REALTIME_TTS_AVAILABLE
    )
except ImportError:
    REALTIME_TTS_AVAILABLE = False

__all__ = [
    "VoiceInput", "SambanovaSTT", "WakeWordListener", "get_voice_input",
    "speak", "get_tts", "EdgeTTS", "VOICES",
    "StreamingTTS", "ChunkedStreamingTTS", "get_streaming_tts",
    "REALTIME_TTS_AVAILABLE"
]
