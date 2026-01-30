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
        # Use Google Speech Recognition for wake word (lightweight)
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.pause_threshold = 0.8
            self.recognizer.dynamic_energy_threshold = False
        except ImportError:
            raise ImportError("speech_recognition required for wake word detection")

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
    """Combined wake word detection + cloud STT with conversation mode."""

    # Conversation timeout - how long to stay active after last interaction
    CONVERSATION_TIMEOUT = 30  # seconds

    def __init__(self):
        self.stt = SambanovaSTT()
        self.wake_listener = WakeWordListener()
        self.wake_word_enabled = True

        # Conversation mode state
        self.in_conversation = False
        self.last_interaction = None

    def set_wake_word_enabled(self, enabled: bool):
        """Enable/disable wake word requirement."""
        self.wake_word_enabled = enabled

    def start_conversation(self):
        """Start conversation mode - no wake word needed for follow-ups."""
        self.in_conversation = True
        self.last_interaction = datetime.now()

    def end_conversation(self):
        """End conversation mode - require wake word again."""
        self.in_conversation = False
        self.last_interaction = None

    def _is_conversation_active(self) -> bool:
        """Check if we're still in an active conversation."""
        if not self.in_conversation:
            return False

        if self.last_interaction is None:
            return False

        elapsed = (datetime.now() - self.last_interaction).total_seconds()
        if elapsed > self.CONVERSATION_TIMEOUT:
            print(f"\n[Conversation timed out after {self.CONVERSATION_TIMEOUT}s]")
            self.end_conversation()
            return False

        return True

    def listen(self, require_wake_word: bool = None) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Listen for speech with conversation mode support.

        In conversation mode, no wake word is needed for follow-up questions.
        After CONVERSATION_TIMEOUT seconds of silence, reverts to requiring wake word.

        Args:
            require_wake_word: Override instance setting (None = auto based on conversation state)

        Returns:
            (transcribed_text, audio_array) or (None, None) on failure
        """
        import speech_recognition as sr

        # Determine if we need wake word
        if require_wake_word is not None:
            use_wake = require_wake_word
        elif self._is_conversation_active():
            use_wake = False  # In conversation, no wake word needed
        else:
            use_wake = self.wake_word_enabled

        # Wait for wake word if needed
        if use_wake:
            if not self.wake_listener.wait_for_wake_word(timeout=120):
                return None, None
            # Wake word detected, start conversation
            self.start_conversation()

        # Now listen for the actual command
        remaining = ""
        if self._is_conversation_active():
            elapsed = (datetime.now() - self.last_interaction).total_seconds()
            remaining = f" [{int(self.CONVERSATION_TIMEOUT - elapsed)}s until timeout]"

        print(f"\nListening...{remaining}")

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.pause_threshold = 1.5
        recognizer.dynamic_energy_threshold = False

        try:
            with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
                # Shorter timeout during conversation
                timeout = 8 if self._is_conversation_active() else 10
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=20)

            print("Processing with Whisper...")

            # Get raw audio for voice ID
            audio_data = np.frombuffer(
                audio.get_raw_data(), dtype=np.int16
            ).astype(np.float32) / 32768.0

            # Transcribe with SambaNova Whisper
            text = self.stt.transcribe(audio_data)

            if text:
                # Update conversation timestamp
                self.last_interaction = datetime.now()
                return text, audio_data
            else:
                print("Could not transcribe")
                return None, audio_data

        except sr.WaitTimeoutError:
            # During conversation, timeout means end of conversation
            if self._is_conversation_active():
                print("[No follow-up detected]")
                self.end_conversation()
            else:
                print("No speech detected")
            return None, None
        except Exception as e:
            print(f"Error: {e}")
            return None, None


def get_voice_input() -> VoiceInput:
    """Get configured voice input instance."""
    return VoiceInput()
