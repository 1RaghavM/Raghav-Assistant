"""Real-time streaming TTS using edge-tts with chunked playback."""

import asyncio
import io
import re
import threading
import queue
from typing import Generator, Optional

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.playback import play
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except:
    PYGAME_AVAILABLE = False

# Mark as available if we have edge_tts
REALTIME_TTS_AVAILABLE = EDGE_TTS_AVAILABLE

# Voice options
VOICES = {
    "male_us": "en-US-GuyNeural",
    "female_us": "en-US-JennyNeural",
    "male_uk": "en-GB-RyanNeural",
    "female_uk": "en-GB-SoniaNeural",
}

DEFAULT_VOICE = "en-US-GuyNeural"


class ChunkedStreamingTTS:
    """Sentence-by-sentence streaming TTS for real-time speech."""

    def __init__(self, voice: str = DEFAULT_VOICE):
        if not EDGE_TTS_AVAILABLE:
            raise ImportError("Install edge-tts: pip install edge-tts")

        self.voice = voice
        self._stop_flag = False
        self._playback_queue = queue.Queue()
        self._playback_thread = None
        print(f"[TTS] Streaming with voice: {voice}")

    def _is_sentence_end(self, text: str) -> bool:
        """Check if text ends with sentence terminator."""
        text = text.strip()
        return bool(re.search(r'[.!?:;]\s*$', text))

    def _clean_text(self, text: str) -> str:
        """Clean text for TTS."""
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[\*#`\[\]]', '', text)
        return text.strip()

    async def _generate_audio(self, text: str) -> bytes:
        """Generate audio for text chunk."""
        communicate = edge_tts.Communicate(text, self.voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data

    def _play_audio_pygame(self, audio_data: bytes):
        """Play audio using pygame."""
        if not PYGAME_AVAILABLE:
            return

        try:
            audio_io = io.BytesIO(audio_data)
            pygame.mixer.music.load(audio_io)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"[TTS] Playback error: {e}")

    def _play_audio_afplay(self, audio_data: bytes):
        """Play audio using macOS afplay."""
        import tempfile
        import subprocess
        import os

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            subprocess.run(["afplay", temp_path], capture_output=True)
        finally:
            os.unlink(temp_path)

    def _play_audio(self, audio_data: bytes):
        """Play audio with best available method."""
        import platform

        if platform.system() == "Darwin":
            self._play_audio_afplay(audio_data)
        elif PYGAME_AVAILABLE:
            self._play_audio_pygame(audio_data)
        else:
            # Fallback: save and play
            self._play_audio_afplay(audio_data)

    def _playback_worker(self):
        """Background worker that plays audio from queue."""
        while True:
            item = self._playback_queue.get()
            if item is None:  # Poison pill
                break
            self._play_audio(item)
            self._playback_queue.task_done()

    def feed_and_play(self, text_generator: Generator[str, None, None]):
        """Feed text chunks and play as sentences complete.

        Generates TTS for each sentence and plays immediately.
        """
        buffer = ""
        self._stop_flag = False

        # Start playback thread
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

        async def process():
            nonlocal buffer

            for chunk in text_generator:
                if self._stop_flag:
                    break

                if not chunk:
                    continue

                buffer += chunk

                # Check if we have a complete sentence
                if self._is_sentence_end(buffer):
                    sentence = self._clean_text(buffer)
                    if sentence:
                        # Generate and queue audio
                        audio = await self._generate_audio(sentence + " ")
                        self._playback_queue.put(audio)
                    buffer = ""

            # Process remaining text
            if buffer.strip() and not self._stop_flag:
                remaining = self._clean_text(buffer)
                if remaining:
                    audio = await self._generate_audio(remaining)
                    self._playback_queue.put(audio)

        # Run async processing
        asyncio.run(process())

        # Wait for playback to finish
        self._playback_queue.join()

        # Stop playback thread
        self._playback_queue.put(None)
        self._playback_thread.join()

    def speak(self, text: str):
        """Speak complete text (non-streaming)."""
        text = self._clean_text(text)
        if not text:
            return

        async def generate_and_play():
            audio = await self._generate_audio(text)
            self._play_audio(audio)

        asyncio.run(generate_and_play())

    def stop(self):
        """Stop playback."""
        self._stop_flag = True


class StreamingTTS(ChunkedStreamingTTS):
    """Alias for ChunkedStreamingTTS."""
    pass


def get_streaming_tts(voice: str = DEFAULT_VOICE) -> Optional[ChunkedStreamingTTS]:
    """Get streaming TTS instance."""
    if EDGE_TTS_AVAILABLE:
        try:
            return ChunkedStreamingTTS(voice)
        except Exception as e:
            print(f"[TTS] Streaming TTS failed: {e}")
    return None
