"""Text-to-Speech using Edge TTS (Microsoft's natural voices)."""

import asyncio
import tempfile
import subprocess
import platform
from typing import Optional

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


# Natural voice options (Microsoft Edge voices)
VOICES = {
    "male_us": "en-US-GuyNeural",
    "female_us": "en-US-JennyNeural",
    "male_uk": "en-GB-RyanNeural",
    "female_uk": "en-GB-SoniaNeural",
    "male_au": "en-AU-WilliamNeural",
    "female_au": "en-AU-NatashaNeural",
}

# Default voice - natural male US voice
DEFAULT_VOICE = "en-US-GuyNeural"


class EdgeTTS:
    """Natural TTS using Microsoft Edge voices."""

    def __init__(self, voice: str = DEFAULT_VOICE):
        if not EDGE_TTS_AVAILABLE:
            raise ImportError("Install edge-tts: pip install edge-tts")
        self.voice = voice
        print(f"[TTS] Using Edge voice: {self.voice}")

    async def _speak_async(self, text: str):
        """Async speak implementation."""
        if not text.strip():
            return

        # Create temp file for audio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name

        try:
            # Generate speech
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(temp_path)

            # Play audio
            if platform.system() == "Darwin":
                subprocess.run(["afplay", temp_path], capture_output=True)
            elif platform.system() == "Linux":
                # Try mpv, then ffplay, then aplay
                for player in ["mpv --no-video", "ffplay -nodisp -autoexit", "aplay"]:
                    try:
                        subprocess.run(player.split() + [temp_path], capture_output=True)
                        break
                    except FileNotFoundError:
                        continue
            else:
                # Windows
                import os
                os.startfile(temp_path)

        finally:
            # Cleanup
            import os
            try:
                os.unlink(temp_path)
            except:
                pass

    def speak(self, text: str):
        """Speak text using Edge TTS."""
        if not text.strip():
            return

        # Clean text
        import re
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[\*#`\[\]]', '', text)
        text = text.strip()

        if not text:
            return

        # Run async in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create task
                asyncio.create_task(self._speak_async(text))
            else:
                loop.run_until_complete(self._speak_async(text))
        except RuntimeError:
            # No event loop, create new one
            asyncio.run(self._speak_async(text))


class FallbackTTS:
    """Fallback to system TTS if Edge TTS unavailable."""

    def speak(self, text: str):
        """Speak using system TTS."""
        import re
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[\*#`\[\]]', '', text)
        text = text.strip()

        if not text:
            return

        if platform.system() == "Darwin":
            # Use a better macOS voice if available
            subprocess.run(["say", "-v", "Samantha", "-r", "180", text], capture_output=True)
        elif platform.system() == "Linux":
            subprocess.run(["espeak", "-s", "150", text], capture_output=True)


def get_tts(voice: str = DEFAULT_VOICE):
    """Get TTS instance, with fallback."""
    if EDGE_TTS_AVAILABLE:
        try:
            return EdgeTTS(voice)
        except Exception as e:
            print(f"[TTS] Edge TTS failed: {e}, using fallback")

    return FallbackTTS()


# Convenience function
_tts_instance: Optional[EdgeTTS] = None

def speak(text: str):
    """Speak text using the best available TTS."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = get_tts()
    _tts_instance.speak(text)
