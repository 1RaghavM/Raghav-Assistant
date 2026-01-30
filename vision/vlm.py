"""Vision Language Model for scene understanding."""

import base64
import cv2
import numpy as np
from typing import Protocol, Optional
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
    """Uses Ollama for VLM (moondream, llava, etc.)."""

    def __init__(self, model: str = None):
        if model is None:
            from config import VLM_MODEL
            model = VLM_MODEL
        self.model = model
        print(f"[VLM] Using model: {self.model}")

    def analyze(self, frame: np.ndarray, query: str) -> str:
        """Analyze image and return description."""
        if not OLLAMA_AVAILABLE:
            return "Error: ollama not installed"

        if frame is None or frame.size == 0:
            return "Error: No image provided"

        try:
            # Ensure frame is valid
            if len(frame.shape) != 3:
                return "Error: Invalid image format"

            h, w = frame.shape[:2]
            print(f"[VLM] Image size: {w}x{h}")

            # Resize if too large (for speed)
            max_dim = 800
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                print(f"[VLM] Resized to: {frame.shape[1]}x{frame.shape[0]}")

            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Encode as JPEG then base64
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            success, buffer = cv2.imencode('.jpg', rgb, encode_params)
            if not success:
                return "Error: Failed to encode image"

            image_base64 = base64.b64encode(buffer).decode('utf-8')
            print(f"[VLM] Sending to {self.model}...")

            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': query,
                    'images': [image_base64]
                }]
            )

            result = response.get('message', {}).get('content', '')
            if not result:
                return "Error: Empty response from VLM"

            print(f"[VLM] Response: {result[:100]}...")
            return result

        except Exception as e:
            print(f"[VLM] Error: {e}")
            return f"Error: {str(e)}"


class HailoVLM:
    """Pi deployment - uses Hailo-10H AI HAT+ 2.

    Supported models on Hailo-10H (40 TOPS, 8GB RAM):
    - Qwen 2.5-VL 1.5B/3B (best for vision-language)
    - Florence-2 (232M/771M - very efficient)
    - Moondream 1.8B
    """

    def __init__(self, model: str = "qwen2.5-vl-3b"):
        self.model = model
        print(f"[VLM] Hailo mode: {self.model}")
        # hailo-ollama provides Ollama-compatible API for Hailo models

    def analyze(self, frame: np.ndarray, query: str) -> str:
        """Analyze using Hailo NPU."""
        # Hailo-ollama uses same API as Ollama
        try:
            import ollama

            if frame is None or frame.size == 0:
                return "Error: No image provided"

            # Resize for Hailo (640x480 optimal)
            h, w = frame.shape[:2]
            if w > 640 or h > 480:
                frame = cv2.resize(frame, (640, 480))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': query,
                    'images': [image_base64]
                }]
            )

            return response.get('message', {}).get('content', 'No response')

        except Exception as e:
            return f"Hailo VLM error: {str(e)}"


def get_vlm() -> VLMBackend:
    """Get appropriate VLM backend for platform."""
    from config import PLATFORM
    if PLATFORM == "mac":
        return OllamaVLM()
    else:
        return HailoVLM()


class VLM:
    """Main VLM interface."""

    def __init__(self):
        self.backend = get_vlm()
        self.last_result = None
        self.last_timestamp = None

    def analyze(self, frame: np.ndarray, query: str) -> str:
        """Analyze image with VLM."""
        print(f"[VLM] Analyzing: '{query}'")
        result = self.backend.analyze(frame, query)
        self.last_result = result
        self.last_timestamp = datetime.now()
        return result

    def get_context(self) -> Optional[dict]:
        """Get recent vision context."""
        if self.last_result is None:
            return None

        from config import VISION_CONTEXT_EXPIRY_MINUTES

        if self.last_timestamp:
            age = (datetime.now() - self.last_timestamp).total_seconds() / 60
            if age > VISION_CONTEXT_EXPIRY_MINUTES:
                self.clear_context()
                return None

        return {
            "timestamp": self.last_timestamp.isoformat(),
            "description": self.last_result
        }

    def clear_context(self):
        self.last_result = None
        self.last_timestamp = None
