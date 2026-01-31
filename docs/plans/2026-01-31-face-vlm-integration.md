# Face-VLM Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correlate face recognition with VLM so the LLM always knows who is in frame, who is speaking, and what each person is doing.

**Architecture:** Background thread runs face detection every 200ms using MediaPipe BlazeFace (fast), caches detected faces with identities. Voice pipeline updates speaker ID. System prompt includes identity summary. Enhanced `see_camera` tool injects face annotations into VLM prompt.

**Tech Stack:** MediaPipe BlazeFace (detection), face_recognition (encoding/matching), threading (background tracker)

---

## Task 1: Add MediaPipe Dependency

**Files:**
- Modify: `requirements.txt`

**Step 1: Add mediapipe to requirements**

Add this line to `requirements.txt` after the Identity section:

```
mediapipe  # Fast face detection (BlazeFace)
```

**Step 2: Install the dependency**

Run: `pip install mediapipe`

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add mediapipe for fast face detection"
```

---

## Task 2: Add Config Constants for Face Tracking

**Files:**
- Modify: `config.py`

**Step 1: Add tracking config constants**

Add after line 36 (after `VISION_CONTEXT_EXPIRY_MINUTES`):

```python
# Face tracking
FACE_TRACK_INTERVAL_MS = 200      # Background detection interval
FACE_STALE_TIMEOUT_S = 2.0        # Remove faces not seen for this long
FACE_REIDENTIFY_INTERVAL_S = 5.0  # Re-run recognition every N seconds
```

**Step 2: Commit**

```bash
git add config.py
git commit -m "feat: add face tracking config constants"
```

---

## Task 3: Create FaceTracker with BlazeFace Detection

**Files:**
- Create: `vision/face_tracker.py`
- Test: `tests/test_face_tracker.py`

**Step 1: Write the failing test**

Create `tests/test_face_tracker.py`:

```python
import pytest
import numpy as np
from unittest.mock import Mock, patch

def test_face_tracker_init():
    """FaceTracker initializes with camera and face_id."""
    with patch('vision.face_tracker.mp') as mock_mp:
        mock_mp.solutions.face_detection.FaceDetection.return_value = Mock()
        from vision.face_tracker import FaceTracker

        mock_camera = Mock()
        mock_face_id = Mock()
        tracker = FaceTracker(mock_camera, mock_face_id)

        assert tracker.camera == mock_camera
        assert tracker.face_id == mock_face_id
        assert tracker._running == False

def test_get_current_faces_empty():
    """get_current_faces returns empty list when no faces detected."""
    with patch('vision.face_tracker.mp') as mock_mp:
        mock_mp.solutions.face_detection.FaceDetection.return_value = Mock()
        from vision.face_tracker import FaceTracker

        tracker = FaceTracker(Mock(), Mock())
        faces = tracker.get_current_faces()

        assert faces == []

def test_set_speaker():
    """set_speaker updates speaker_id and confidence."""
    with patch('vision.face_tracker.mp') as mock_mp:
        mock_mp.solutions.face_detection.FaceDetection.return_value = Mock()
        from vision.face_tracker import FaceTracker

        tracker = FaceTracker(Mock(), Mock())
        tracker.set_speaker("raghav", 0.85)

        assert tracker.speaker_id == "raghav"
        assert tracker.speaker_confidence == 0.85

def test_get_identity_summary_no_faces():
    """get_identity_summary returns appropriate message when no faces."""
    with patch('vision.face_tracker.mp') as mock_mp:
        mock_mp.solutions.face_detection.FaceDetection.return_value = Mock()
        from vision.face_tracker import FaceTracker

        tracker = FaceTracker(Mock(), Mock())
        summary = tracker.get_identity_summary()

        assert "No one" in summary or "no one" in summary.lower()
```

**Step 2: Run test to verify it fails**

Run: `/Users/raghavmehta/Downloads/Raghav-Assistant/.venv/bin/python -m pytest tests/test_face_tracker.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'vision.face_tracker'"

**Step 3: Write minimal implementation**

Create `vision/face_tracker.py`:

```python
"""Background face tracking with MediaPipe BlazeFace detection."""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class DetectedFace:
    """A detected face with identity and position."""
    name: str  # "unknown" if not identified
    confidence: float
    bbox: tuple  # (x, y, width, height) normalized 0-1
    last_seen: datetime = field(default_factory=datetime.now)
    needs_reidentify: bool = False


class FaceTracker:
    """Background thread that continuously detects and identifies faces."""

    def __init__(self, camera, face_id, interval_ms: int = None):
        """
        Args:
            camera: Camera instance for frame capture
            face_id: FaceIdentifier instance for recognition
            interval_ms: Detection interval (default from config)
        """
        self.camera = camera
        self.face_id = face_id

        if interval_ms is None:
            try:
                from config import FACE_TRACK_INTERVAL_MS
                interval_ms = FACE_TRACK_INTERVAL_MS
            except ImportError:
                interval_ms = 200
        self.interval_s = interval_ms / 1000.0

        # Load config
        try:
            from config import FACE_STALE_TIMEOUT_S, FACE_REIDENTIFY_INTERVAL_S
            self.stale_timeout = FACE_STALE_TIMEOUT_S
            self.reidentify_interval = FACE_REIDENTIFY_INTERVAL_S
        except ImportError:
            self.stale_timeout = 2.0
            self.reidentify_interval = 5.0

        # State
        self._faces: Dict[str, DetectedFace] = {}  # keyed by position hash
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Speaker info (set by voice pipeline)
        self.speaker_id: Optional[str] = None
        self.speaker_confidence: float = 0.0
        self.speaker_timestamp: Optional[datetime] = None

        # MediaPipe face detection (BlazeFace - very fast)
        if MEDIAPIPE_AVAILABLE:
            self._mp_face = mp.solutions.face_detection
            self._detector = self._mp_face.FaceDetection(
                model_selection=0,  # 0 = short range (< 2m), faster
                min_detection_confidence=0.5
            )
        else:
            self._detector = None

    def start(self):
        """Start background face tracking thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._track_loop, daemon=True)
        self._thread.start()
        print("[FaceTracker] Started background tracking")

    def stop(self):
        """Stop background face tracking."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        print("[FaceTracker] Stopped")

    def _track_loop(self):
        """Main tracking loop - runs in background thread."""
        while self._running:
            try:
                self._detect_and_identify()
            except Exception as e:
                print(f"[FaceTracker] Error: {e}")
            time.sleep(self.interval_s)

    def _detect_and_identify(self):
        """Detect faces in current frame and identify them."""
        if not MEDIAPIPE_AVAILABLE or not CV2_AVAILABLE:
            return

        frame = self.camera.capture()
        if frame is None:
            return

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces with BlazeFace
        results = self._detector.process(rgb)

        now = datetime.now()
        detected_positions = set()

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                pos_key = self._position_key(bbox)
                detected_positions.add(pos_key)

                with self._lock:
                    existing = self._faces.get(pos_key)

                    if existing and not existing.needs_reidentify:
                        # Update last_seen for existing face
                        existing.last_seen = now
                        # Check if we need to re-identify
                        age = (now - existing.last_seen).total_seconds()
                        if age > self.reidentify_interval:
                            existing.needs_reidentify = True
                    else:
                        # New face or needs re-identification - run recognition
                        name, confidence = self._identify_face(frame, bbox)
                        self._faces[pos_key] = DetectedFace(
                            name=name or "unknown",
                            confidence=confidence,
                            bbox=(bbox.xmin, bbox.ymin, bbox.width, bbox.height),
                            last_seen=now,
                            needs_reidentify=False
                        )

        # Prune stale faces
        with self._lock:
            stale_keys = [
                k for k, v in self._faces.items()
                if (now - v.last_seen).total_seconds() > self.stale_timeout
            ]
            for k in stale_keys:
                del self._faces[k]

    def _position_key(self, bbox) -> str:
        """Generate a key for face position (for tracking across frames)."""
        # Round to nearest 0.1 to allow some movement
        x = round(bbox.xmin, 1)
        y = round(bbox.ymin, 1)
        return f"{x:.1f},{y:.1f}"

    def _identify_face(self, frame: np.ndarray, bbox) -> tuple:
        """Identify face using face_recognition library.

        Returns: (name, confidence) or (None, 0.0)
        """
        if not CV2_AVAILABLE:
            return None, 0.0

        h, w = frame.shape[:2]

        # Convert normalized bbox to pixel coords
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # Add padding
        pad = int(min(bw, bh) * 0.2)
        x = max(0, x - pad)
        y = max(0, y - pad)
        bw = min(w - x, bw + 2 * pad)
        bh = min(h - y, bh + 2 * pad)

        # Crop face region
        face_crop = frame[y:y+bh, x:x+bw]

        if face_crop.size == 0:
            return None, 0.0

        # Use FaceIdentifier for recognition
        return self.face_id.identify_from_image(face_crop)

    def set_speaker(self, name: str, confidence: float):
        """Update current speaker (called by voice pipeline)."""
        self.speaker_id = name
        self.speaker_confidence = confidence
        self.speaker_timestamp = datetime.now()

    def get_current_faces(self) -> List[Dict]:
        """Get list of currently detected faces with identities."""
        with self._lock:
            return [
                {
                    "name": f.name,
                    "confidence": f.confidence,
                    "bbox": f.bbox,
                    "is_speaker": f.name == self.speaker_id and f.name != "unknown"
                }
                for f in self._faces.values()
            ]

    def get_identity_summary(self) -> str:
        """Get formatted identity summary for LLM system prompt."""
        faces = self.get_current_faces()

        if not faces:
            return "Current presence: No one visible in camera."

        lines = ["Current presence:"]

        for face in faces:
            name = face["name"]
            conf = face["confidence"]
            is_speaker = face["is_speaker"]

            if name == "unknown":
                if is_speaker:
                    lines.append("- Unknown person (speaking)")
                else:
                    lines.append("- Unknown person")
            else:
                status = []
                if is_speaker:
                    # Combine voice + face verification status
                    if self.speaker_confidence >= 0.75:
                        status.append("speaker, HIGH confidence: voice+face match")
                    elif self.speaker_confidence >= 0.6:
                        status.append("speaker, MEDIUM confidence: voice+face")
                    else:
                        status.append("speaker")

                if status:
                    lines.append(f"- {name.title()} ({', '.join(status)})")
                else:
                    lines.append(f"- {name.title()} (visible)")

        # Add verification status
        if self.speaker_id and self.speaker_confidence >= 0.75:
            lines.append("Speaker verified: YES")
        elif self.speaker_id and self.speaker_confidence >= 0.6:
            lines.append("Speaker verified: PARTIAL")

        return "\n".join(lines)

    def get_faces_for_vlm(self) -> str:
        """Get face annotations for VLM prompt enhancement."""
        faces = self.get_current_faces()

        if not faces:
            return ""

        lines = ["Known faces in frame:"]

        for face in faces:
            name = face["name"]
            bbox = face["bbox"]
            is_speaker = face["is_speaker"]

            # Describe position based on bbox center
            x_center = bbox[0] + bbox[2] / 2
            if x_center < 0.33:
                position = "left side"
            elif x_center > 0.66:
                position = "right side"
            else:
                position = "center"

            if name == "unknown":
                if is_speaker:
                    lines.append(f"- Unknown person (speaking): {position}")
                else:
                    lines.append(f"- Unknown person: {position}")
            else:
                if is_speaker:
                    lines.append(f"- {name.title()} (speaker): {position}")
                else:
                    lines.append(f"- {name.title()}: {position}")

        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `/Users/raghavmehta/Downloads/Raghav-Assistant/.venv/bin/python -m pytest tests/test_face_tracker.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vision/face_tracker.py tests/test_face_tracker.py
git commit -m "feat: add FaceTracker with MediaPipe BlazeFace detection"
```

---

## Task 4: Integrate FaceTracker into Menzi

**Files:**
- Modify: `menzi.py`

**Step 1: Add FaceTracker import and initialization**

Add import after line 27 (`from vision.vlm import VLM`):

```python
from vision.face_tracker import FaceTracker
```

**Step 2: Initialize FaceTracker in __init__**

Add after line 80 (`self.vlm = VLM()`):

```python
self.face_tracker = FaceTracker(self.camera, self.identity.face_id)
self.face_tracker.start()
```

**Step 3: Update _system_prompt to include identity context**

Replace the `_system_prompt` method (lines 90-103) with:

```python
def _system_prompt(self) -> str:
    parts = ["You are Menzi, a helpful voice assistant. Be conversational and brief (1-2 sentences)."]

    # Add face tracking identity context
    identity_ctx = self.face_tracker.get_identity_summary()
    parts.append(identity_ctx)

    if self.user and self.user != "unknown":
        parts.append(f"You're speaking with {self.user}.")
        mems = self.memory.get_all(self.user)
        if mems:
            parts.append(f"You know: {'; '.join(mems[:5])}")
        if self.is_admin:
            parts.append("They have admin access.")
    else:
        parts.append("You don't know who this is yet. If context requires knowing their name, ask naturally.")

    return " ".join(parts)
```

**Step 4: Update FaceTracker when speaker identified**

In the `run()` method, after line 357 (`self._set_user(identified)`), add:

```python
self.face_tracker.set_speaker(identified, 0.75)  # Voice ID confidence
```

**Step 5: Stop FaceTracker on cleanup**

In the `finally` block (after line 372 `self.camera.release()`), add:

```python
self.face_tracker.stop()
```

**Step 6: Pass FaceTracker to ToolExecutor**

Modify the `_set_user` method to pass face_tracker. Replace line 124-130:

```python
self.executor = ToolExecutor(
    current_user=name,
    is_admin=self.is_admin,
    camera=self.camera,
    vlm=self.vlm,
    face_tracker=self.face_tracker,
    on_user_change=self._on_user_change
)
```

**Step 7: Commit**

```bash
git add menzi.py
git commit -m "feat: integrate FaceTracker into Menzi for identity awareness"
```

---

## Task 5: Enhance ToolExecutor with Face Annotations

**Files:**
- Modify: `core/executor.py`
- Test: `tests/test_executor.py` (add test)

**Step 1: Add test for enhanced see_camera**

Add to `tests/test_executor.py`:

```python
def test_see_camera_with_face_tracker():
    """see_camera includes face annotations when face_tracker provided."""
    from core.executor import ToolExecutor
    from unittest.mock import Mock

    mock_camera = Mock()
    mock_camera.capture.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_vlm = Mock()
    mock_vlm.analyze.return_value = "A person is typing on a laptop."

    mock_face_tracker = Mock()
    mock_face_tracker.get_faces_for_vlm.return_value = "Known faces in frame:\n- Raghav (speaker): center"

    executor = ToolExecutor(
        current_user="raghav",
        is_admin=False,
        camera=mock_camera,
        vlm=mock_vlm,
        face_tracker=mock_face_tracker
    )

    result = executor._see_camera("what do you see?")

    # Verify VLM was called with enhanced query
    call_args = mock_vlm.analyze.call_args
    query = call_args[0][1]
    assert "Raghav" in query or "Known faces" in query
```

**Step 2: Run test to verify it fails**

Run: `/Users/raghavmehta/Downloads/Raghav-Assistant/.venv/bin/python -m pytest tests/test_executor.py::test_see_camera_with_face_tracker -v`
Expected: FAIL (face_tracker parameter not supported yet)

**Step 3: Update ToolExecutor to accept face_tracker**

Modify `core/executor.py`:

Add `face_tracker=None` to `__init__` signature (line 51):

```python
def __init__(
    self,
    current_user: str,
    is_admin: bool,
    data_dir: str = None,
    camera=None,
    vlm=None,
    face_tracker=None,
    on_user_change=None
):
```

Add after line 80 (`self.vlm = vlm`):

```python
self.face_tracker = face_tracker
```

**Step 4: Enhance _see_camera method**

Replace the `_see_camera` method (lines 192-204) with:

```python
def _see_camera(self, query: str) -> str:
    if self.camera is None or self.vlm is None:
        return "Camera not available."

    frame = self.camera.capture()
    if frame is None:
        return "Could not capture image from camera."

    try:
        # Enhance query with face annotations if available
        if self.face_tracker:
            face_context = self.face_tracker.get_faces_for_vlm()
            if face_context:
                enhanced_query = f"{face_context}\n\nDescribe what each person is doing, with emphasis on the speaker. {query}"
            else:
                enhanced_query = query
        else:
            enhanced_query = query

        result = self.vlm.analyze(frame, enhanced_query)
        return result
    except Exception as e:
        return f"Vision error: {str(e)}"
```

**Step 5: Run test to verify it passes**

Run: `/Users/raghavmehta/Downloads/Raghav-Assistant/.venv/bin/python -m pytest tests/test_executor.py::test_see_camera_with_face_tracker -v`
Expected: PASS

**Step 6: Commit**

```bash
git add core/executor.py tests/test_executor.py
git commit -m "feat: enhance see_camera with face annotations from FaceTracker"
```

---

## Task 6: Update Voice Pipeline to Set Speaker

**Files:**
- Modify: `menzi.py`

**Step 1: Update identify method to set speaker on tracker**

Replace the `identify` method (lines 295-301) with:

```python
def identify(self, audio: np.ndarray) -> Optional[str]:
    """Identify user from voice and update face tracker."""
    user, conf, method = self.identity.resolve(voice_audio=audio)
    if user and conf >= 0.6:
        print(f"Identified: {user} ({conf:.0%} via {method})")
        # Update face tracker with speaker info
        self.face_tracker.set_speaker(user, conf)
        return user
    return None
```

**Step 2: Commit**

```bash
git add menzi.py
git commit -m "feat: update voice pipeline to set speaker on FaceTracker"
```

---

## Task 7: Add Integration Test

**Files:**
- Create: `tests/test_face_vlm_integration.py`

**Step 1: Write integration test**

Create `tests/test_face_vlm_integration.py`:

```python
"""Integration tests for face detection + VLM context flow."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

def test_identity_summary_flows_to_system_prompt():
    """Verify face tracker identity flows to LLM system prompt."""
    with patch('vision.face_tracker.mp') as mock_mp, \
         patch('menzi.VoiceInput'), \
         patch('menzi.Cerebras'), \
         patch('menzi.Camera'), \
         patch('menzi.VLM'), \
         patch('menzi.CEREBRAS_API_KEY', 'test-key'):

        # Setup mock MediaPipe
        mock_mp.solutions.face_detection.FaceDetection.return_value = Mock()

        from menzi import Menzi

        # Mock the face tracker
        menzi = Menzi()
        menzi.face_tracker.get_identity_summary = Mock(
            return_value="Current presence:\n- Raghav (speaker, HIGH confidence)"
        )

        prompt = menzi._system_prompt()

        assert "Raghav" in prompt
        assert "speaker" in prompt.lower()

def test_see_camera_includes_face_context():
    """Verify see_camera tool includes face annotations."""
    from core.executor import ToolExecutor

    mock_camera = Mock()
    mock_camera.capture.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_vlm = Mock()
    mock_vlm.analyze.return_value = "Raghav is typing on a laptop."

    mock_face_tracker = Mock()
    mock_face_tracker.get_faces_for_vlm.return_value = (
        "Known faces in frame:\n- Raghav (speaker): center\n- Unknown person: left side"
    )

    executor = ToolExecutor(
        current_user="raghav",
        is_admin=False,
        camera=mock_camera,
        vlm=mock_vlm,
        face_tracker=mock_face_tracker
    )

    result = executor.execute("see_camera", {"query": "what's happening?"})

    # Check VLM was called with enhanced query
    call_args = mock_vlm.analyze.call_args
    enhanced_query = call_args[0][1]

    assert "Raghav (speaker)" in enhanced_query
    assert "Unknown person" in enhanced_query
    assert "emphasis on the speaker" in enhanced_query

def test_speaker_set_on_voice_identification():
    """Verify voice identification updates face tracker speaker."""
    with patch('vision.face_tracker.mp') as mock_mp, \
         patch('menzi.VoiceInput'), \
         patch('menzi.Cerebras'), \
         patch('menzi.Camera'), \
         patch('menzi.VLM'), \
         patch('menzi.CEREBRAS_API_KEY', 'test-key'):

        mock_mp.solutions.face_detection.FaceDetection.return_value = Mock()

        from menzi import Menzi

        menzi = Menzi()

        # Mock identity resolver to return a user
        menzi.identity.resolve = Mock(return_value=("raghav", 0.85, "voice"))

        # Spy on face tracker set_speaker
        menzi.face_tracker.set_speaker = Mock()

        # Call identify
        audio = np.zeros(16000, dtype=np.float32)
        result = menzi.identify(audio)

        assert result == "raghav"
        menzi.face_tracker.set_speaker.assert_called_once_with("raghav", 0.85)
```

**Step 2: Run tests**

Run: `/Users/raghavmehta/Downloads/Raghav-Assistant/.venv/bin/python -m pytest tests/test_face_vlm_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_face_vlm_integration.py
git commit -m "test: add integration tests for face-VLM context flow"
```

---

## Task 8: Run Full Test Suite and Verify

**Step 1: Run all tests**

Run: `/Users/raghavmehta/Downloads/Raghav-Assistant/.venv/bin/python -m pytest tests/ -v`

Expected: All new tests pass. The one pre-existing failure (GEMINI_API_KEY import) is unrelated.

**Step 2: Manual verification checklist**

- [ ] `mediapipe` installed successfully
- [ ] FaceTracker starts when Menzi initializes
- [ ] Background thread runs face detection
- [ ] System prompt includes identity summary
- [ ] `see_camera` includes face annotations in VLM query
- [ ] Voice identification updates speaker in tracker

**Step 3: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: address test feedback"
```

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | Add MediaPipe dependency | requirements.txt |
| 2 | Add config constants | config.py |
| 3 | Create FaceTracker | vision/face_tracker.py |
| 4 | Integrate into Menzi | menzi.py |
| 5 | Enhance ToolExecutor | core/executor.py |
| 6 | Update voice pipeline | menzi.py |
| 7 | Integration tests | tests/test_face_vlm_integration.py |
| 8 | Full test run | - |

**Expected latency improvement:**
- Face detection: ~100ms → ~5-10ms (BlazeFace vs dlib)
- Recognition only runs on new/changed faces
- Zero latency at query time (cached results)
