import cv2
import numpy as np
from typing import Optional

class Camera:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._cap = None

    def _ensure_open(self) -> bool:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.device_id)
            if self._cap.isOpened():
                # Optimize settings
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self._cap is not None and self._cap.isOpened()

    def capture(self) -> Optional[np.ndarray]:
        """Capture a single frame. Returns BGR image or None."""
        if not self._ensure_open():
            return None

        # Flush buffer to get fresh frame
        self._cap.grab()
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        self.release()
