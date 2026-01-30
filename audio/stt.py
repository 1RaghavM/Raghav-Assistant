"""Speech-to-Text using SambaNova Whisper API with wake word detection."""

import os
import io
import wave
import tempfile
import numpy as np
from typing import Optional, Tuple
from datetime import datetime

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None

# Wake word settings
WAKE_WORD = "menzi"
WAKE_WORD_VARIANTS = ["menzi", "manzi", "mensi", "mansi", "mensie", "men z"]


class SambanovaSTT:
    """Cloud STT using SambaNova's Whisper API."""

    def __init__(self):
        self.api_key = os.getenv("SAMBANOVA_API_KEY")
        if not self.api_key:
            raise ValueError("Set SAMBANOVA_API_KEY in .env")

        # SambaNova uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.sambanova.ai/v1"
        )
        print("[STT] Using SambaNova Whisper API")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using SambaNova Whisper.

        Args:
            audio_data: Float32 audio array normalized to [-1, 1]

        Returns:
            Transcribed text
        """
        if audio_data is None or len(audio_data) == 0:
            return ""

        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())

        wav_buffer.seek(0)

        try:
            # Use OpenAI-compatible transcription endpoint
            response = self.client.audio.transcriptions.create(
                model="Whisper-Large-v3",
                file=("audio.wav", wav_buffer, "audio/wav"),
            )
            return response.text.strip()
        except Exception as e:
            print(f"[STT] Error: {e}")
            return ""


class WakeWordListener:
    """Listens for wake word using local speech recognition."""

    def __init__(self):
        if not PYAUDIO_AVAILABLE:
            raise ImportError("pyaudio required for wake word detection")

        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Use Google Speech Recognition for wake word (lightweight)
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.pause_threshold = 0.8
            self.recognizer.dynamic_energy_threshold = False
            self.use_local_recognition = True
        except ImportError:
            self.use_local_recognition = False

        print(f"[WakeWord] Listening for '{WAKE_WORD}'")

    def _contains_wake_word(self, text: str) -> bool:
        """Check if text contains wake word or variants."""
        text_lower = text.lower()
        for variant in WAKE_WORD_VARIANTS:
            if variant in text_lower:
                return True
        return False

    def wait_for_wake_word(self, timeout: float = None) -> bool:
        """Wait for wake word to be spoken.

        Args:
            timeout: Max seconds to wait (None = infinite)

        Returns:
            True if wake word detected, False on timeout
        """
        import speech_recognition as sr

        print(f"\n... Waiting for '{WAKE_WORD}' ...")

        start_time = datetime.now()

        while True:
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    return False

            try:
                with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
                    # Short listen for wake word
                    audio = self.recognizer.listen(
                        source,
                        timeout=3,
                        phrase_time_limit=3
                    )

                # Quick transcription check
                try:
                    text = self.recognizer.recognize_google(audio)
                    if self._contains_wake_word(text):
                        print(f"[WakeWord] Detected: '{text}'")
                        return True
                except sr.UnknownValueError:
                    pass  # No speech detected
                except sr.RequestError:
                    pass  # Recognition service error

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"[WakeWord] Error: {e}")
                continue

        return False


class VoiceInput:
    """Combined wake word detection + cloud STT."""

    def __init__(self):
        self.stt = SambanovaSTT()
        self.wake_listener = WakeWordListener()
        self.wake_word_enabled = True

    def set_wake_word_enabled(self, enabled: bool):
        """Enable/disable wake word requirement."""
        self.wake_word_enabled = enabled

    def listen(self, require_wake_word: bool = None) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Listen for speech, optionally requiring wake word first.

        Args:
            require_wake_word: Override instance setting

        Returns:
            (transcribed_text, audio_array) or (None, None) on failure
        """
        import speech_recognition as sr

        use_wake = require_wake_word if require_wake_word is not None else self.wake_word_enabled

        # Wait for wake word if enabled
        if use_wake:
            if not self.wake_listener.wait_for_wake_word(timeout=60):
                return None, None

        # Now listen for the actual command
        print("\n Listening... (speak now)")

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.pause_threshold = 1.5
        recognizer.dynamic_energy_threshold = False

        try:
            with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)

            print(" Processing with Whisper...")

            # Get raw audio for voice ID
            audio_data = np.frombuffer(
                audio.get_raw_data(), dtype=np.int16
            ).astype(np.float32) / 32768.0

            # Transcribe with SambaNova Whisper
            text = self.stt.transcribe(audio_data)

            if text:
                return text, audio_data
            else:
                print(" Could not transcribe")
                return None, audio_data

        except sr.WaitTimeoutError:
            print(" No speech detected")
            return None, None
        except Exception as e:
            print(f" Error: {e}")
            return None, None


def get_voice_input() -> VoiceInput:
    """Get configured voice input instance."""
    return VoiceInput()
