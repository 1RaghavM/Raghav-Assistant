import base64
import cv2
import numpy as np
from typing import Protocol
from datetime import datetime

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class VLMBackend(Protocol):
    def analyze(self, frame: np.ndarray, query: str) -> str:
        ...

class OllamaVLM:
    """Mac development - uses Ollama moondream locally."""

    def __init__(self, model: str = None):
        if model is None:
            from config import VLM_MODEL
            model = VLM_MODEL
        self.model = model

    def analyze(self, frame: np.ndarray, query: str) -> str:
        """Analyze image and return description."""
        if not OLLAMA_AVAILABLE:
            return "VLM not available (ollama not installed)"

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Encode as JPEG then base64
        _, buffer = cv2.imencode('.jpg', rgb)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        response = ollama.chat(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': query,
                'images': [image_base64]
            }]
        )

        return response['message']['content']


class HailoVLM:
    """Pi deployment - uses Hailo-10H via hailo-ollama."""

    def __init__(self, model: str = "qwen2.5-vl-3b"):
        self.model = model
        # TODO: Initialize hailo-ollama connection when deploying to Pi

    def analyze(self, frame: np.ndarray, query: str) -> str:
        # Placeholder - implement when migrating to Pi
        raise NotImplementedError("HailoVLM not yet implemented for Pi deployment")


def get_vlm() -> VLMBackend:
    """Factory function to get appropriate VLM backend."""
    from config import PLATFORM
    if PLATFORM == "mac":
        return OllamaVLM()
    else:
        return HailoVLM()


class VLM:
    """Main VLM interface with context management."""

    def __init__(self):
        self.backend = get_vlm()
        self.last_result = None
        self.last_timestamp = None

    def analyze(self, frame: np.ndarray, query: str) -> str:
        result = self.backend.analyze(frame, query)
        self.last_result = result
        self.last_timestamp = datetime.now()
        return result

    def get_context(self) -> dict:
        """Get current vision context for LLM."""
        if self.last_result is None:
            return None

        from config import VISION_CONTEXT_EXPIRY_MINUTES

        if self.last_timestamp:
            age = (datetime.now() - self.last_timestamp).total_seconds() / 60
            if age > VISION_CONTEXT_EXPIRY_MINUTES:
                self.clear_context()
                return None

        return {
            "timestamp": self.last_timestamp.isoformat() if self.last_timestamp else None,
            "description": self.last_result
        }

    def clear_context(self):
        self.last_result = None
        self.last_timestamp = None
