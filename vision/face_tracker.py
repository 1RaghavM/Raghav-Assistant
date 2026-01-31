"""
FaceTracker with MediaPipe BlazeFace detection.

Background thread that:
1. Captures frames from camera every 200ms
2. Detects faces using MediaPipe BlazeFace (very fast, ~5ms)
3. Identifies faces using face_recognition (only for new/changed faces)
4. Caches results so LLM queries have zero latency
5. Tracks speaker info from voice pipeline
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np

from config import (
    FACE_TRACK_INTERVAL_MS,
    FACE_STALE_TIMEOUT_S,
    FACE_REIDENTIFY_INTERVAL_S,
)


@dataclass
class DetectedFace:
    """Represents a detected and optionally identified face."""
    name: Optional[str] = None  # None if unknown
    confidence: float = 0.0
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # (x, y, w, h) normalized
    last_seen: float = field(default_factory=time.time)
    needs_reidentify: bool = True


class FaceTracker:
    """
    Background face detection and identification tracker.

    Uses MediaPipe BlazeFace for fast detection (~5ms) and
    face_recognition for identification (slower, cached).
    """

    def __init__(
        self,
        camera,
        face_id,
        interval_ms: int = None
    ):
        """
        Initialize FaceTracker.

        Args:
            camera: Camera instance for capturing frames
            face_id: FaceIdentifier instance for recognition
            interval_ms: Detection interval in milliseconds (default from config)
        """
        self.camera = camera
        self.face_id = face_id
        self.interval_ms = interval_ms if interval_ms is not None else FACE_TRACK_INTERVAL_MS

        # MediaPipe face detection
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (< 2m), 1 for full-range
            min_detection_confidence=0.5
        )

        # Thread management
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Face tracking state
        self._faces: Dict[str, DetectedFace] = {}  # position_key -> DetectedFace

        # Speaker tracking from voice pipeline
        self.speaker_id: Optional[str] = None
        self.speaker_confidence: float = 0.0

    def start(self):
        """Start background face tracking thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._track_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background face tracking thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _track_loop(self):
        """Main background tracking loop."""
        interval_s = self.interval_ms / 1000.0

        while self._running:
            start_time = time.time()

            try:
                self._detect_and_identify()
            except Exception as e:
                # Log but don't crash the thread
                print(f"[FaceTracker] Error in detection loop: {e}")

            # Sleep for remaining interval
            elapsed = time.time() - start_time
            sleep_time = max(0, interval_s - elapsed)
            time.sleep(sleep_time)

    def _detect_and_identify(self):
        """Detect faces in current frame and identify them."""
        frame = self.camera.capture()
        if frame is None:
            return

        # Convert BGR to RGB for MediaPipe
        rgb_frame = frame[:, :, ::-1]

        # Detect faces with MediaPipe BlazeFace
        results = self._detector.process(rgb_frame)

        current_time = time.time()
        seen_keys = set()

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                bbox_tuple = (bbox.xmin, bbox.ymin, bbox.width, bbox.height)
                pos_key = self._position_key(bbox_tuple)
                seen_keys.add(pos_key)

                with self._lock:
                    if pos_key in self._faces:
                        # Update existing face
                        face = self._faces[pos_key]
                        face.bbox = bbox_tuple
                        face.last_seen = current_time

                        # Check if we need to re-identify
                        time_since_identify = current_time - (
                            face.last_seen - FACE_STALE_TIMEOUT_S
                            if not face.needs_reidentify else 0
                        )
                        if face.needs_reidentify or time_since_identify > FACE_REIDENTIFY_INTERVAL_S:
                            name, conf = self._identify_face(frame, bbox_tuple)
                            if name is not None:
                                face.name = name
                                face.confidence = conf
                                face.needs_reidentify = False
                    else:
                        # New face detected
                        name, conf = self._identify_face(frame, bbox_tuple)
                        self._faces[pos_key] = DetectedFace(
                            name=name,
                            confidence=conf,
                            bbox=bbox_tuple,
                            last_seen=current_time,
                            needs_reidentify=(name is None)
                        )

        # Prune stale faces
        with self._lock:
            stale_keys = [
                key for key, face in self._faces.items()
                if current_time - face.last_seen > FACE_STALE_TIMEOUT_S
            ]
            for key in stale_keys:
                del self._faces[key]

    def _position_key(self, bbox: Tuple[float, float, float, float]) -> str:
        """
        Generate a position key for tracking faces across frames.

        Rounds position to 0.1 to allow for small movements while
        maintaining identity.
        """
        x, y, w, h = bbox
        # Use center point rounded to 0.1
        cx = round((x + w / 2) * 10) / 10
        cy = round((y + h / 2) * 10) / 10
        return f"{cx:.1f},{cy:.1f}"

    def _identify_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[Optional[str], float]:
        """
        Crop face from frame and identify using face_id.

        Args:
            frame: BGR image
            bbox: Normalized (x, y, w, h) bounding box

        Returns:
            (name, confidence) or (None, 0.0) if not identified
        """
        h, w = frame.shape[:2]
        x, y, bw, bh = bbox

        # Convert normalized coords to pixels with padding
        pad = 0.1  # 10% padding
        x1 = max(0, int((x - pad) * w))
        y1 = max(0, int((y - pad) * h))
        x2 = min(w, int((x + bw + pad) * w))
        y2 = min(h, int((y + bh + pad) * h))

        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        face_crop = frame[y1:y2, x1:x2]

        try:
            return self.face_id.identify_from_image(face_crop)
        except Exception:
            return None, 0.0

    def set_speaker(self, name: str, confidence: float):
        """
        Update speaker info from voice pipeline.

        Called by voice_assistant when speaker is identified.
        """
        self.speaker_id = name
        self.speaker_confidence = confidence

    def get_current_faces(self) -> List[DetectedFace]:
        """
        Get list of currently detected faces.

        Returns:
            List of DetectedFace objects
        """
        with self._lock:
            return list(self._faces.values())

    def get_identity_summary(self) -> str:
        """
        Get formatted identity summary for LLM system prompt.

        Combines face detection results with speaker info.
        """
        faces = self.get_current_faces()

        if not faces:
            if self.speaker_id:
                return f"The speaker is {self.speaker_id} (voice confidence: {self.speaker_confidence:.0%}). No one is visible on camera."
            return "No one is currently visible on camera."

        # Build face descriptions
        identified = [f for f in faces if f.name is not None]
        unknown_count = len(faces) - len(identified)

        parts = []

        if identified:
            names = [f"{f.name} ({f.confidence:.0%})" for f in identified]
            if len(names) == 1:
                parts.append(f"{names[0]} is visible")
            else:
                parts.append(f"{', '.join(names[:-1])} and {names[-1]} are visible")

        if unknown_count > 0:
            if unknown_count == 1:
                parts.append("1 unknown person is visible")
            else:
                parts.append(f"{unknown_count} unknown people are visible")

        summary = ". ".join(parts) + "."

        # Add speaker info if different from visual
        if self.speaker_id:
            speaker_visible = any(f.name == self.speaker_id for f in faces)
            if speaker_visible:
                summary += f" The speaker is {self.speaker_id}."
            else:
                summary += f" The speaker is {self.speaker_id} (identified by voice, not visible on camera)."

        return summary

    def get_faces_for_vlm(self) -> str:
        """
        Get formatted face info for VLM prompt enhancement.

        Returns concise info about detected/identified faces to
        help VLM understand who is in the scene.
        """
        faces = self.get_current_faces()

        if not faces:
            return "No faces detected in view."

        identified = [f for f in faces if f.name is not None]
        unknown_count = len(faces) - len(identified)

        parts = []

        if identified:
            for f in identified:
                # Describe position roughly
                cx = f.bbox[0] + f.bbox[2] / 2
                if cx < 0.33:
                    pos = "left"
                elif cx > 0.67:
                    pos = "right"
                else:
                    pos = "center"
                parts.append(f"{f.name} ({pos} of frame)")

        if unknown_count > 0:
            parts.append(f"{unknown_count} unidentified person(s)")

        return "Faces detected: " + ", ".join(parts)
