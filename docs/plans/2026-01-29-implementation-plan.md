# Menzi Vision & Identity System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform Menzi from a simple voice assistant into a Jarvis-style AI with voice identification, on-demand vision, routine learning, and multi-user support.

**Architecture:** Voice-first identification using speaker embeddings, with face recognition as fallback. VLM (Moondream) called on-demand via tool. Each user has isolated memories. LLM orchestrates all tools including weather, news, and admin commands.

**Tech Stack:** Python 3.11+, Ollama (qwen3:1.7b, moondream), resemblyzer (voice ID), face_recognition, OpenCV, ChromaDB, Open-Meteo API

---

## Phase 1: Project Structure & Configuration

### Task 1.1: Create Directory Structure

**Files:**
- Create: `config.py`
- Create: `core/__init__.py`
- Create: `identity/__init__.py`
- Create: `vision/__init__.py`
- Create: `services/__init__.py`
- Create: `memory/__init__.py`
- Create: `audio/__init__.py`
- Create: `data/users.json`
- Create: `data/routines.json`
- Create: `data/memories/.gitkeep`
- Create: `data/face_encodings/.gitkeep`
- Create: `data/voice_embeddings/.gitkeep`

**Step 1: Create all directories and init files**

```bash
mkdir -p core identity vision services memory audio data/memories data/face_encodings data/voice_embeddings
touch core/__init__.py identity/__init__.py vision/__init__.py services/__init__.py memory/__init__.py audio/__init__.py
touch data/memories/.gitkeep data/face_encodings/.gitkeep data/voice_embeddings/.gitkeep
```

**Step 2: Create config.py**

```python
# config.py
import os
import platform

# Platform detection
PLATFORM = "mac" if platform.system() == "Darwin" else "pi"

# Models
LLM_MODEL = "qwen3:1.7b"
VLM_MODEL = "moondream" if PLATFORM == "mac" else "qwen2.5-vl-3b"

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
ROUTINES_FILE = os.path.join(DATA_DIR, "routines.json")
MEMORIES_DIR = os.path.join(DATA_DIR, "memories")
FACE_ENCODINGS_DIR = os.path.join(DATA_DIR, "face_encodings")
VOICE_EMBEDDINGS_DIR = os.path.join(DATA_DIR, "voice_embeddings")
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Identity thresholds
VOICE_MATCH_THRESHOLD = 0.75
VOICE_UNCERTAIN_THRESHOLD = 0.6
FACE_MATCH_THRESHOLD = 0.6

# Vision context
VISION_CONTEXT_EXPIRY_MESSAGES = 10
VISION_CONTEXT_EXPIRY_MINUTES = 5

# Conversation
MAX_MESSAGES = 16

# Admin
ADMIN_USER = "raghav"
```

**Step 3: Create initial users.json**

```json
{
  "raghav": {
    "admin": true,
    "created_at": "2026-01-29T00:00:00",
    "default_location": "San Francisco"
  }
}
```

Write to `data/users.json`.

**Step 4: Create initial routines.json**

```json
{}
```

Write to `data/routines.json`.

**Step 5: Commit**

```bash
git add config.py core/ identity/ vision/ services/ memory/ audio/ data/
git commit -m "feat: create project structure and config"
```

---

### Task 1.2: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements.txt**

```txt
# Core
ollama
chromadb

# Audio
SpeechRecognition
pyttsx3
pyaudio

# Identity
face_recognition
resemblyzer
numpy

# Vision
opencv-python

# Services
requests
```

**Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

Note: `face_recognition` requires dlib. On Mac: `brew install cmake` first.

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add identity and vision dependencies"
```

---

## Phase 2: Memory System (Multi-User)

### Task 2.1: User Memory Module

**Files:**
- Create: `memory/user_memory.py`
- Create: `tests/test_user_memory.py`

**Step 1: Create test directory**

```bash
mkdir -p tests
touch tests/__init__.py
```

**Step 2: Write failing test**

```python
# tests/test_user_memory.py
import pytest
import os
import json
import tempfile
from memory.user_memory import UserMemory

@pytest.fixture
def temp_memory_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_add_memory(temp_memory_dir):
    mem = UserMemory(memories_dir=temp_memory_dir)
    result = mem.add("raghav", "User likes coffee")
    assert result is True
    memories = mem.get_all("raghav")
    assert "User likes coffee" in memories

def test_no_duplicate_memory(temp_memory_dir):
    mem = UserMemory(memories_dir=temp_memory_dir)
    mem.add("raghav", "User likes coffee")
    result = mem.add("raghav", "User likes coffee")
    assert result is False

def test_separate_user_memories(temp_memory_dir):
    mem = UserMemory(memories_dir=temp_memory_dir)
    mem.add("raghav", "Raghav fact")
    mem.add("john", "John fact")
    assert "Raghav fact" in mem.get_all("raghav")
    assert "John fact" not in mem.get_all("raghav")
    assert "John fact" in mem.get_all("john")

def test_clear_user_memories(temp_memory_dir):
    mem = UserMemory(memories_dir=temp_memory_dir)
    mem.add("raghav", "Some fact")
    mem.clear("raghav")
    assert mem.get_all("raghav") == []
```

**Step 3: Run test to verify it fails**

```bash
pytest tests/test_user_memory.py -v
```

Expected: FAIL with "No module named 'memory.user_memory'"

**Step 4: Write implementation**

```python
# memory/user_memory.py
import os
import json
from typing import List

class UserMemory:
    def __init__(self, memories_dir: str = None):
        if memories_dir is None:
            from config import MEMORIES_DIR
            memories_dir = MEMORIES_DIR
        self.memories_dir = memories_dir
        os.makedirs(self.memories_dir, exist_ok=True)

    def _get_path(self, user: str) -> str:
        return os.path.join(self.memories_dir, f"{user.lower()}.json")

    def _load(self, user: str) -> List[str]:
        path = self._get_path(user)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    def _save(self, user: str, memories: List[str]):
        path = self._get_path(user)
        with open(path, "w") as f:
            json.dump(memories, f, indent=2)

    def add(self, user: str, content: str) -> bool:
        content = content.strip()
        memories = self._load(user)
        for existing in memories:
            if content.lower() in existing.lower() or existing.lower() in content.lower():
                return False
        memories.append(content)
        self._save(user, memories)
        return True

    def get_all(self, user: str) -> List[str]:
        return self._load(user)

    def clear(self, user: str):
        self._save(user, [])
```

**Step 5: Run test to verify it passes**

```bash
pytest tests/test_user_memory.py -v
```

Expected: All PASS

**Step 6: Commit**

```bash
git add memory/user_memory.py tests/
git commit -m "feat: add multi-user memory system"
```

---

### Task 2.2: User Registry Module

**Files:**
- Create: `memory/user_registry.py`
- Create: `tests/test_user_registry.py`

**Step 1: Write failing test**

```python
# tests/test_user_registry.py
import pytest
import tempfile
import os
from memory.user_registry import UserRegistry

@pytest.fixture
def temp_registry():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{}')
        path = f.name
    yield UserRegistry(users_file=path)
    os.unlink(path)

def test_create_user(temp_registry):
    temp_registry.create_user("john")
    assert temp_registry.user_exists("john")

def test_admin_check(temp_registry):
    temp_registry.create_user("raghav", admin=True)
    temp_registry.create_user("john", admin=False)
    assert temp_registry.is_admin("raghav") is True
    assert temp_registry.is_admin("john") is False

def test_list_users(temp_registry):
    temp_registry.create_user("raghav", admin=True)
    temp_registry.create_user("john")
    users = temp_registry.list_users()
    assert "raghav" in users
    assert "john" in users

def test_delete_user(temp_registry):
    temp_registry.create_user("john")
    temp_registry.delete_user("john")
    assert temp_registry.user_exists("john") is False

def test_get_user_location(temp_registry):
    temp_registry.create_user("raghav", location="San Francisco")
    assert temp_registry.get_location("raghav") == "San Francisco"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_user_registry.py -v
```

**Step 3: Write implementation**

```python
# memory/user_registry.py
import os
import json
from datetime import datetime
from typing import Optional, List, Dict

class UserRegistry:
    def __init__(self, users_file: str = None):
        if users_file is None:
            from config import USERS_FILE
            users_file = USERS_FILE
        self.users_file = users_file
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.users_file):
            os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
            with open(self.users_file, "w") as f:
                json.dump({}, f)

    def _load(self) -> Dict:
        with open(self.users_file, "r") as f:
            return json.load(f)

    def _save(self, data: Dict):
        with open(self.users_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_user(self, name: str, admin: bool = False, location: str = None):
        data = self._load()
        name = name.lower()
        if name not in data:
            data[name] = {
                "admin": admin,
                "created_at": datetime.now().isoformat(),
                "default_location": location
            }
            self._save(data)

    def user_exists(self, name: str) -> bool:
        return name.lower() in self._load()

    def is_admin(self, name: str) -> bool:
        data = self._load()
        user = data.get(name.lower(), {})
        return user.get("admin", False)

    def list_users(self) -> List[str]:
        return list(self._load().keys())

    def delete_user(self, name: str):
        data = self._load()
        name = name.lower()
        if name in data:
            del data[name]
            self._save(data)

    def get_location(self, name: str) -> Optional[str]:
        data = self._load()
        user = data.get(name.lower(), {})
        return user.get("default_location")

    def set_location(self, name: str, location: str):
        data = self._load()
        name = name.lower()
        if name in data:
            data[name]["default_location"] = location
            self._save(data)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_user_registry.py -v
```

**Step 5: Commit**

```bash
git add memory/user_registry.py tests/test_user_registry.py
git commit -m "feat: add user registry with admin support"
```

---

### Task 2.3: Routines Module

**Files:**
- Create: `memory/routines.py`
- Create: `tests/test_routines.py`

**Step 1: Write failing test**

```python
# tests/test_routines.py
import pytest
import tempfile
import os
from memory.routines import RoutineTracker

@pytest.fixture
def temp_tracker():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{}')
        path = f.name
    yield RoutineTracker(routines_file=path)
    os.unlink(path)

def test_record_interaction(temp_tracker):
    temp_tracker.record_interaction("raghav", hour=8, day="monday")
    patterns = temp_tracker.get_patterns("raghav")
    assert patterns is not None

def test_record_query_pattern(temp_tracker):
    temp_tracker.record_query("raghav", "weather", hour=8)
    temp_tracker.record_query("raghav", "weather", hour=8)
    patterns = temp_tracker.get_patterns("raghav")
    assert any(q["pattern"] == "weather" for q in patterns.get("common_queries", []))

def test_get_suggestion_morning_weather(temp_tracker):
    # Record weather queries in morning
    for _ in range(5):
        temp_tracker.record_query("raghav", "weather", hour=8)
    suggestion = temp_tracker.get_suggestion("raghav", hour=8)
    assert suggestion is not None
    assert "weather" in suggestion.lower()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_routines.py -v
```

**Step 3: Write implementation**

```python
# memory/routines.py
import os
import json
from datetime import datetime
from typing import Optional, Dict, List

class RoutineTracker:
    def __init__(self, routines_file: str = None):
        if routines_file is None:
            from config import ROUTINES_FILE
            routines_file = ROUTINES_FILE
        self.routines_file = routines_file
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.routines_file):
            os.makedirs(os.path.dirname(self.routines_file), exist_ok=True)
            with open(self.routines_file, "w") as f:
                json.dump({}, f)

    def _load(self) -> Dict:
        with open(self.routines_file, "r") as f:
            return json.load(f)

    def _save(self, data: Dict):
        with open(self.routines_file, "w") as f:
            json.dump(data, f, indent=2)

    def _get_user_data(self, user: str) -> Dict:
        data = self._load()
        user = user.lower()
        if user not in data:
            data[user] = {
                "interaction_times": [],
                "common_queries": [],
                "recent_topics": []
            }
            self._save(data)
        return data[user]

    def record_interaction(self, user: str, hour: int = None, day: str = None):
        if hour is None:
            hour = datetime.now().hour
        if day is None:
            day = datetime.now().strftime("%A").lower()

        data = self._load()
        user = user.lower()
        if user not in data:
            data[user] = {"interaction_times": [], "common_queries": [], "recent_topics": []}

        times = data[user]["interaction_times"]
        day_type = "weekend" if day in ["saturday", "sunday"] else "weekday"

        # Find or create time bucket
        found = False
        for t in times:
            if t["hour"] == hour and t["day"] == day_type:
                t["count"] += 1
                found = True
                break
        if not found:
            times.append({"hour": hour, "day": day_type, "count": 1})

        self._save(data)

    def record_query(self, user: str, pattern: str, hour: int = None):
        if hour is None:
            hour = datetime.now().hour

        data = self._load()
        user = user.lower()
        if user not in data:
            data[user] = {"interaction_times": [], "common_queries": [], "recent_topics": []}

        time_bucket = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
        queries = data[user]["common_queries"]

        found = False
        for q in queries:
            if q["pattern"] == pattern and q["time"] == time_bucket:
                q["count"] += 1
                found = True
                break
        if not found:
            queries.append({"pattern": pattern, "time": time_bucket, "count": 1})

        self._save(data)

    def record_topic(self, user: str, topic: str):
        data = self._load()
        user = user.lower()
        if user not in data:
            data[user] = {"interaction_times": [], "common_queries": [], "recent_topics": []}

        topics = data[user]["recent_topics"]
        if topic in topics:
            topics.remove(topic)
        topics.insert(0, topic)
        data[user]["recent_topics"] = topics[:10]  # Keep last 10
        self._save(data)

    def get_patterns(self, user: str) -> Dict:
        return self._get_user_data(user)

    def get_suggestion(self, user: str, hour: int = None) -> Optional[str]:
        if hour is None:
            hour = datetime.now().hour

        patterns = self._get_user_data(user)
        time_bucket = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"

        # Find frequent queries for this time
        for q in patterns.get("common_queries", []):
            if q["time"] == time_bucket and q["count"] >= 3:
                if q["pattern"] == "weather":
                    return "Would you like a weather update?"
                elif q["pattern"] == "news":
                    return "Want me to fetch today's news?"
        return None

    def clear(self, user: str):
        data = self._load()
        user = user.lower()
        if user in data:
            del data[user]
            self._save(data)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_routines.py -v
```

**Step 5: Commit**

```bash
git add memory/routines.py tests/test_routines.py
git commit -m "feat: add routine learning and suggestions"
```

---

## Phase 3: Identity System

### Task 3.1: Voice Identification Module

**Files:**
- Create: `identity/voice_id.py`
- Create: `tests/test_voice_id.py`

**Step 1: Write failing test**

```python
# tests/test_voice_id.py
import pytest
import numpy as np
import tempfile
import os
from identity.voice_id import VoiceIdentifier

@pytest.fixture
def temp_embeddings_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_enroll_voice(temp_embeddings_dir):
    vi = VoiceIdentifier(embeddings_dir=temp_embeddings_dir)
    # Create fake audio (16kHz, 2 seconds)
    fake_audio = np.random.randn(32000).astype(np.float32)
    vi.enroll("raghav", fake_audio)
    assert vi.is_enrolled("raghav")

def test_identify_enrolled_user(temp_embeddings_dir):
    vi = VoiceIdentifier(embeddings_dir=temp_embeddings_dir)
    fake_audio = np.random.randn(32000).astype(np.float32)
    vi.enroll("raghav", fake_audio)
    # Same audio should match
    user, confidence = vi.identify(fake_audio)
    assert user == "raghav"
    assert confidence > 0.9

def test_identify_unknown_user(temp_embeddings_dir):
    vi = VoiceIdentifier(embeddings_dir=temp_embeddings_dir)
    fake_audio = np.random.randn(32000).astype(np.float32)
    user, confidence = vi.identify(fake_audio)
    assert user is None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_voice_id.py -v
```

**Step 3: Write implementation**

```python
# identity/voice_id.py
import os
import pickle
import numpy as np
from typing import Optional, Tuple
from resemblyzer import VoiceEncoder, preprocess_wav

class VoiceIdentifier:
    def __init__(self, embeddings_dir: str = None):
        if embeddings_dir is None:
            from config import VOICE_EMBEDDINGS_DIR
            embeddings_dir = VOICE_EMBEDDINGS_DIR
        self.embeddings_dir = embeddings_dir
        os.makedirs(self.embeddings_dir, exist_ok=True)
        self.encoder = VoiceEncoder()
        self._embeddings_cache = {}
        self._load_all_embeddings()

    def _get_path(self, user: str) -> str:
        return os.path.join(self.embeddings_dir, f"{user.lower()}.pkl")

    def _load_all_embeddings(self):
        for filename in os.listdir(self.embeddings_dir):
            if filename.endswith(".pkl"):
                user = filename[:-4]
                with open(os.path.join(self.embeddings_dir, filename), "rb") as f:
                    self._embeddings_cache[user] = pickle.load(f)

    def enroll(self, user: str, audio: np.ndarray, sample_rate: int = 16000):
        """Enroll a user with their voice sample."""
        user = user.lower()
        # Preprocess and encode
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        audio = audio.astype(np.float32)

        # Resample to 16kHz if needed (resemblyzer expects 16kHz)
        embedding = self.encoder.embed_utterance(audio)

        # Save embedding
        path = self._get_path(user)
        with open(path, "wb") as f:
            pickle.dump(embedding, f)

        self._embeddings_cache[user] = embedding

    def is_enrolled(self, user: str) -> bool:
        return user.lower() in self._embeddings_cache

    def identify(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[Optional[str], float]:
        """Identify speaker from audio. Returns (user, confidence) or (None, 0)."""
        if not self._embeddings_cache:
            return None, 0.0

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        query_embedding = self.encoder.embed_utterance(audio)

        best_user = None
        best_similarity = 0.0

        for user, stored_embedding in self._embeddings_cache.items():
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_user = user

        from config import VOICE_MATCH_THRESHOLD
        if best_similarity >= VOICE_MATCH_THRESHOLD:
            return best_user, float(best_similarity)
        return None, float(best_similarity)

    def delete(self, user: str):
        user = user.lower()
        path = self._get_path(user)
        if os.path.exists(path):
            os.remove(path)
        if user in self._embeddings_cache:
            del self._embeddings_cache[user]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_voice_id.py -v
```

**Step 5: Commit**

```bash
git add identity/voice_id.py tests/test_voice_id.py
git commit -m "feat: add voice identification with resemblyzer"
```

---

### Task 3.2: Face Identification Module

**Files:**
- Create: `identity/face_id.py`
- Create: `tests/test_face_id.py`

**Step 1: Write failing test**

```python
# tests/test_face_id.py
import pytest
import numpy as np
import tempfile
from identity.face_id import FaceIdentifier

@pytest.fixture
def temp_encodings_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_face_identifier_init(temp_encodings_dir):
    fi = FaceIdentifier(encodings_dir=temp_encodings_dir)
    assert fi is not None

def test_enroll_with_encoding(temp_encodings_dir):
    fi = FaceIdentifier(encodings_dir=temp_encodings_dir)
    # Fake face encoding (128-dim for face_recognition lib)
    fake_encoding = np.random.randn(128).astype(np.float64)
    fi.enroll_with_encoding("raghav", fake_encoding)
    assert fi.is_enrolled("raghav")

def test_identify_with_encoding(temp_encodings_dir):
    fi = FaceIdentifier(encodings_dir=temp_encodings_dir)
    fake_encoding = np.random.randn(128).astype(np.float64)
    fi.enroll_with_encoding("raghav", fake_encoding)
    # Same encoding should match
    user, confidence = fi.identify_with_encoding(fake_encoding)
    assert user == "raghav"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_face_id.py -v
```

**Step 3: Write implementation**

```python
# identity/face_id.py
import os
import pickle
import numpy as np
from typing import Optional, Tuple, List
import face_recognition

class FaceIdentifier:
    def __init__(self, encodings_dir: str = None):
        if encodings_dir is None:
            from config import FACE_ENCODINGS_DIR
            encodings_dir = FACE_ENCODINGS_DIR
        self.encodings_dir = encodings_dir
        os.makedirs(self.encodings_dir, exist_ok=True)
        self._encodings_cache = {}
        self._load_all_encodings()

    def _get_path(self, user: str) -> str:
        return os.path.join(self.encodings_dir, f"{user.lower()}.pkl")

    def _load_all_encodings(self):
        for filename in os.listdir(self.encodings_dir):
            if filename.endswith(".pkl"):
                user = filename[:-4]
                with open(os.path.join(self.encodings_dir, filename), "rb") as f:
                    self._encodings_cache[user] = pickle.load(f)

    def enroll_with_encoding(self, user: str, encoding: np.ndarray):
        """Enroll a user with a pre-computed face encoding."""
        user = user.lower()
        path = self._get_path(user)
        with open(path, "wb") as f:
            pickle.dump(encoding, f)
        self._encodings_cache[user] = encoding

    def enroll_from_image(self, user: str, image: np.ndarray) -> bool:
        """Enroll from BGR image (OpenCV format). Returns True if face found."""
        rgb_image = image[:, :, ::-1]  # BGR to RGB
        encodings = face_recognition.face_encodings(rgb_image)
        if encodings:
            self.enroll_with_encoding(user, encodings[0])
            return True
        return False

    def is_enrolled(self, user: str) -> bool:
        return user.lower() in self._encodings_cache

    def identify_with_encoding(self, encoding: np.ndarray) -> Tuple[Optional[str], float]:
        """Identify from pre-computed encoding."""
        if not self._encodings_cache:
            return None, 0.0

        known_encodings = list(self._encodings_cache.values())
        known_users = list(self._encodings_cache.keys())

        distances = face_recognition.face_distance(known_encodings, encoding)
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        # Convert distance to confidence (lower distance = higher confidence)
        confidence = 1.0 - best_distance

        from config import FACE_MATCH_THRESHOLD
        if confidence >= FACE_MATCH_THRESHOLD:
            return known_users[best_idx], float(confidence)
        return None, float(confidence)

    def identify_from_image(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Identify from BGR image. Returns (user, confidence) or (None, 0)."""
        rgb_image = image[:, :, ::-1]  # BGR to RGB
        # Resize for speed
        small = cv2.resize(rgb_image, (0, 0), fx=0.25, fy=0.25)
        encodings = face_recognition.face_encodings(small)
        if not encodings:
            return None, 0.0
        return self.identify_with_encoding(encodings[0])

    def delete(self, user: str):
        user = user.lower()
        path = self._get_path(user)
        if os.path.exists(path):
            os.remove(path)
        if user in self._encodings_cache:
            del self._encodings_cache[user]
```

**Step 4: Add cv2 import at top**

Add `import cv2` after the other imports in face_id.py.

**Step 5: Run test to verify it passes**

```bash
pytest tests/test_face_id.py -v
```

**Step 6: Commit**

```bash
git add identity/face_id.py tests/test_face_id.py
git commit -m "feat: add face identification module"
```

---

### Task 3.3: Identity Resolver

**Files:**
- Create: `identity/resolver.py`
- Create: `tests/test_resolver.py`

**Step 1: Write failing test**

```python
# tests/test_resolver.py
import pytest
import tempfile
import numpy as np
from identity.resolver import IdentityResolver

@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as voice_dir:
        with tempfile.TemporaryDirectory() as face_dir:
            yield voice_dir, face_dir

def test_resolver_init(temp_dirs):
    voice_dir, face_dir = temp_dirs
    resolver = IdentityResolver(voice_dir=voice_dir, face_dir=face_dir)
    assert resolver is not None

def test_resolve_with_voice_only(temp_dirs):
    voice_dir, face_dir = temp_dirs
    resolver = IdentityResolver(voice_dir=voice_dir, face_dir=face_dir)

    # Enroll with fake audio
    fake_audio = np.random.randn(32000).astype(np.float32)
    resolver.enroll_user("raghav", voice_audio=fake_audio)

    # Resolve with same audio
    user, confidence, method = resolver.resolve(voice_audio=fake_audio)
    assert user == "raghav"
    assert method == "voice"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_resolver.py -v
```

**Step 3: Write implementation**

```python
# identity/resolver.py
import numpy as np
from typing import Optional, Tuple
from identity.voice_id import VoiceIdentifier
from identity.face_id import FaceIdentifier

class IdentityResolver:
    def __init__(self, voice_dir: str = None, face_dir: str = None):
        self.voice_id = VoiceIdentifier(embeddings_dir=voice_dir)
        self.face_id = FaceIdentifier(encodings_dir=face_dir)

    def enroll_user(self, name: str, voice_audio: np.ndarray = None, face_image: np.ndarray = None):
        """Enroll a user with voice and/or face."""
        if voice_audio is not None:
            self.voice_id.enroll(name, voice_audio)
        if face_image is not None:
            self.face_id.enroll_from_image(name, face_image)

    def resolve(
        self,
        voice_audio: np.ndarray = None,
        face_image: np.ndarray = None
    ) -> Tuple[Optional[str], float, str]:
        """
        Resolve identity from voice and/or face.
        Returns: (username, confidence, method) where method is 'voice', 'face', or 'both'
        """
        from config import VOICE_MATCH_THRESHOLD, VOICE_UNCERTAIN_THRESHOLD

        voice_user, voice_conf = None, 0.0
        face_user, face_conf = None, 0.0

        if voice_audio is not None:
            voice_user, voice_conf = self.voice_id.identify(voice_audio)

        if face_image is not None:
            face_user, face_conf = self.face_id.identify_from_image(face_image)

        # Decision logic
        # High confidence voice match
        if voice_conf >= VOICE_MATCH_THRESHOLD:
            return voice_user, voice_conf, "voice"

        # Moderate voice + face confirmation
        if voice_conf >= VOICE_UNCERTAIN_THRESHOLD and face_user:
            if voice_user == face_user:
                return voice_user, max(voice_conf, face_conf), "both"
            # Face overrides uncertain voice
            if face_conf > voice_conf:
                return face_user, face_conf, "face"

        # Face only
        if face_user and face_conf >= VOICE_UNCERTAIN_THRESHOLD:
            return face_user, face_conf, "face"

        # Moderate voice, no face
        if voice_conf >= VOICE_UNCERTAIN_THRESHOLD:
            return voice_user, voice_conf, "voice"

        # Unknown
        return None, max(voice_conf, face_conf), "unknown"

    def is_enrolled(self, name: str) -> bool:
        return self.voice_id.is_enrolled(name) or self.face_id.is_enrolled(name)

    def delete_user(self, name: str):
        self.voice_id.delete(name)
        self.face_id.delete(name)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_resolver.py -v
```

**Step 5: Commit**

```bash
git add identity/resolver.py tests/test_resolver.py
git commit -m "feat: add identity resolver combining voice and face"
```

---

## Phase 4: Vision System

### Task 4.1: Camera Module

**Files:**
- Create: `vision/camera.py`
- Create: `tests/test_camera.py`

**Step 1: Write failing test**

```python
# tests/test_camera.py
import pytest
from vision.camera import Camera

def test_camera_init():
    cam = Camera()
    assert cam is not None

def test_camera_capture_returns_frame_or_none():
    cam = Camera()
    frame = cam.capture()
    # Either returns a frame (ndarray) or None if no camera
    assert frame is None or hasattr(frame, 'shape')
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_camera.py -v
```

**Step 3: Write implementation**

```python
# vision/camera.py
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_camera.py -v
```

**Step 5: Commit**

```bash
git add vision/camera.py tests/test_camera.py
git commit -m "feat: add on-demand camera module"
```

---

### Task 4.2: VLM Module

**Files:**
- Create: `vision/vlm.py`
- Create: `tests/test_vlm.py`

**Step 1: Write failing test**

```python
# tests/test_vlm.py
import pytest
import numpy as np
from vision.vlm import VLM, get_vlm

def test_vlm_init():
    vlm = get_vlm()
    assert vlm is not None

def test_vlm_analyze_returns_string():
    vlm = get_vlm()
    # Create a simple test image (black 100x100)
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        result = vlm.analyze(fake_image, "What do you see?")
        assert isinstance(result, str)
    except Exception as e:
        # VLM might not be available in test environment
        pytest.skip(f"VLM not available: {e}")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_vlm.py -v
```

**Step 3: Write implementation**

```python
# vision/vlm.py
import base64
import cv2
import numpy as np
from typing import Protocol
import ollama

class VLMBackend(Protocol):
    def analyze(self, frame: np.ndarray, query: str) -> str:
        ...

class OllamaVLM:
    """Mac development - uses Ollama moondream."""

    def __init__(self, model: str = None):
        if model is None:
            from config import VLM_MODEL
            model = VLM_MODEL
        self.model = model

    def analyze(self, frame: np.ndarray, query: str) -> str:
        """Analyze image and return description."""
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
        from datetime import datetime
        result = self.backend.analyze(frame, query)
        self.last_result = result
        self.last_timestamp = datetime.now()
        return result

    def get_context(self) -> dict:
        """Get current vision context for LLM."""
        if self.last_result is None:
            return None

        from datetime import datetime
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_vlm.py -v
```

**Step 5: Commit**

```bash
git add vision/vlm.py tests/test_vlm.py
git commit -m "feat: add VLM with platform abstraction"
```

---

## Phase 5: Services

### Task 5.1: Weather Service

**Files:**
- Create: `services/weather.py`
- Create: `tests/test_weather.py`

**Step 1: Write failing test**

```python
# tests/test_weather.py
import pytest
from services.weather import get_weather, get_coordinates

def test_get_coordinates():
    lat, lon = get_coordinates("San Francisco")
    assert lat is not None
    assert lon is not None
    assert 37 < lat < 38  # SF latitude range

def test_get_weather():
    result = get_weather("San Francisco")
    assert "temperature" in result.lower() or "error" in result.lower()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_weather.py -v
```

**Step 3: Write implementation**

```python
# services/weather.py
import requests
from typing import Tuple, Optional

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

def get_coordinates(city: str) -> Tuple[Optional[float], Optional[float]]:
    """Get latitude/longitude for a city."""
    try:
        response = requests.get(GEOCODE_URL, params={"name": city, "count": 1}, timeout=10)
        data = response.json()
        if "results" in data and data["results"]:
            result = data["results"][0]
            return result["latitude"], result["longitude"]
    except Exception:
        pass
    return None, None

def get_weather(location: str) -> str:
    """Get current weather for a location."""
    try:
        lat, lon = get_coordinates(location)
        if lat is None:
            return f"Could not find location: {location}"

        response = requests.get(
            WEATHER_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "temperature_unit": "fahrenheit"
            },
            timeout=10
        )
        data = response.json()

        if "current_weather" in data:
            weather = data["current_weather"]
            temp = weather.get("temperature", "?")
            wind = weather.get("windspeed", "?")
            code = weather.get("weathercode", 0)

            # Weather code descriptions
            conditions = {
                0: "clear sky",
                1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                45: "foggy", 48: "depositing rime fog",
                51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
                61: "slight rain", 63: "moderate rain", 65: "heavy rain",
                71: "slight snow", 73: "moderate snow", 75: "heavy snow",
                80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
                95: "thunderstorm"
            }
            condition = conditions.get(code, "unknown conditions")

            return f"Currently {temp}°F with {condition}. Wind: {wind} mph."

        return "Could not get weather data."

    except Exception as e:
        return f"Weather error: {str(e)}"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_weather.py -v
```

**Step 5: Commit**

```bash
git add services/weather.py tests/test_weather.py
git commit -m "feat: add weather service using Open-Meteo"
```

---

### Task 5.2: News Service

**Files:**
- Create: `services/news.py`
- Create: `tests/test_news.py`

**Step 1: Write failing test**

```python
# tests/test_news.py
import pytest
from services.news import get_news

def test_get_news_returns_string():
    result = get_news("technology")
    assert isinstance(result, str)
    assert len(result) > 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_news.py -v
```

**Step 3: Write implementation**

```python
# services/news.py
import requests
import re
import html

DUCKDUCKGO_NEWS_URL = "https://html.duckduckgo.com/html/"

def get_news(topic: str = "world") -> str:
    """Get news headlines for a topic using DuckDuckGo."""
    try:
        query = f"{topic} news today"
        response = requests.get(
            DUCKDUCKGO_NEWS_URL,
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15
        )
        html_content = response.text

        # Extract titles and snippets
        titles = re.findall(r'class="result__a"[^>]*>([^<]+)</a>', html_content)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html_content, re.DOTALL)

        results = []
        for i in range(min(5, len(titles), len(snippets))):
            title = html.unescape(titles[i].strip())
            snippet = re.sub(r'<[^>]+>', '', snippets[i])
            snippet = html.unescape(snippet.strip())
            if title and snippet:
                results.append(f"- {title}: {snippet[:150]}")

        if results:
            return f"Headlines for {topic}:\n" + "\n".join(results)

        return f"No news found for {topic}."

    except Exception as e:
        return f"News error: {str(e)}"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_news.py -v
```

**Step 5: Commit**

```bash
git add services/news.py tests/test_news.py
git commit -m "feat: add news service using DuckDuckGo"
```

---

## Phase 6: Core LLM & Tools

### Task 6.1: Tool Definitions

**Files:**
- Create: `core/tools.py`

**Step 1: Write tool definitions**

```python
# core/tools.py
from typing import List, Dict, Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store a fact about the current user for future reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember"}
                },
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save information to the knowledge base for later retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The information to save"},
                    "topic": {"type": "string", "description": "Topic category"}
                },
                "required": ["content", "topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search the knowledge base for saved information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get news headlines for a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "News topic"}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "see_camera",
            "description": "Look through the camera to see what's happening. Use when user asks 'what am I doing', 'what do you see', 'look at this', or needs visual context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for or describe"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "admin_command",
            "description": "Execute admin-only system commands. Only works for admin users.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list_users", "delete_user", "view_memories", "clear_memories", "interaction_log", "reset_routines", "system_status"],
                        "description": "The admin action to perform"
                    },
                    "target_user": {
                        "type": "string",
                        "description": "Username for user-specific actions"
                    }
                },
                "required": ["action"]
            }
        }
    }
]

def get_tools() -> List[Dict[str, Any]]:
    return TOOLS
```

**Step 2: Commit**

```bash
git add core/tools.py
git commit -m "feat: add tool definitions for all capabilities"
```

---

### Task 6.2: Tool Executor

**Files:**
- Create: `core/executor.py`
- Create: `tests/test_executor.py`

**Step 1: Write failing test**

```python
# tests/test_executor.py
import pytest
import tempfile
from core.executor import ToolExecutor

@pytest.fixture
def executor():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ToolExecutor(
            current_user="raghav",
            is_admin=True,
            data_dir=tmpdir
        )

def test_execute_remember(executor):
    result = executor.execute("remember", {"fact": "User likes pizza"})
    assert "remembered" in result.lower() or "pizza" in result.lower()

def test_execute_weather(executor):
    result = executor.execute("get_weather", {"location": "San Francisco"})
    assert isinstance(result, str)

def test_execute_admin_list_users(executor):
    result = executor.execute("admin_command", {"action": "list_users"})
    assert isinstance(result, str)

def test_non_admin_blocked():
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = ToolExecutor(current_user="john", is_admin=False, data_dir=tmpdir)
        result = executor.execute("admin_command", {"action": "list_users"})
        assert "admin" in result.lower() or "only" in result.lower()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_executor.py -v
```

**Step 3: Write implementation**

```python
# core/executor.py
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
import chromadb

from memory.user_memory import UserMemory
from memory.user_registry import UserRegistry
from memory.routines import RoutineTracker
from services.weather import get_weather as weather_service
from services.news import get_news as news_service

class ToolExecutor:
    def __init__(
        self,
        current_user: str,
        is_admin: bool,
        data_dir: str = None,
        camera=None,
        vlm=None
    ):
        self.current_user = current_user
        self.is_admin = is_admin

        if data_dir is None:
            from config import DATA_DIR, CHROMA_DB_PATH
            data_dir = DATA_DIR
            chroma_path = CHROMA_DB_PATH
        else:
            chroma_path = os.path.join(data_dir, "chroma_db")

        self.user_memory = UserMemory(memories_dir=os.path.join(data_dir, "memories"))
        self.registry = UserRegistry(users_file=os.path.join(data_dir, "users.json"))
        self.routines = RoutineTracker(routines_file=os.path.join(data_dir, "routines.json"))

        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.knowledge_collection = self.chroma_client.get_collection("knowledge_base")
        except:
            self.knowledge_collection = self.chroma_client.create_collection("knowledge_base")

        self.camera = camera
        self.vlm = vlm

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        try:
            if tool_name == "remember":
                return self._remember(arguments["fact"])
            elif tool_name == "save_note":
                return self._save_note(arguments["content"], arguments["topic"])
            elif tool_name == "search_notes":
                return self._search_notes(arguments["query"])
            elif tool_name == "web_search":
                return self._web_search(arguments["query"])
            elif tool_name == "get_weather":
                return self._get_weather(arguments["location"])
            elif tool_name == "get_news":
                return self._get_news(arguments["topic"])
            elif tool_name == "see_camera":
                return self._see_camera(arguments["query"])
            elif tool_name == "admin_command":
                return self._admin_command(arguments["action"], arguments.get("target_user"))
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Tool error: {str(e)}"

    def _remember(self, fact: str) -> str:
        if self.user_memory.add(self.current_user, fact):
            return f"Remembered: {fact}"
        return "Already known."

    def _save_note(self, content: str, topic: str) -> str:
        doc_id = str(uuid.uuid4())
        self.knowledge_collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[{"topic": topic, "timestamp": datetime.now().isoformat(), "user": self.current_user}]
        )
        return f"Saved to knowledge base under '{topic}'."

    def _search_notes(self, query: str) -> str:
        try:
            count = self.knowledge_collection.count()
            if count == 0:
                return "Knowledge base is empty."
            results = self.knowledge_collection.query(query_texts=[query], n_results=min(5, count))
            if results and results.get("documents") and results["documents"][0]:
                return " | ".join(results["documents"][0])
            return "No results found."
        except Exception as e:
            return f"Search error: {str(e)}"

    def _web_search(self, query: str) -> str:
        import requests
        import re
        import html
        try:
            url = f"https://html.duckduckgo.com/html/?q={query}"
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            html_content = response.text

            titles = re.findall(r'class="result__a"[^>]*>([^<]+)</a>', html_content)
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html_content, re.DOTALL)

            results = []
            for i in range(min(5, len(titles), len(snippets))):
                title = html.unescape(titles[i].strip())
                snippet = re.sub(r'<[^>]+>', '', snippets[i])
                snippet = html.unescape(snippet.strip())
                if title and snippet:
                    results.append(f"**{title}**: {snippet[:200]}")

            if results:
                return "\n".join(results)
            return "No results found."
        except Exception as e:
            return f"Search error: {str(e)}"

    def _get_weather(self, location: str) -> str:
        return weather_service(location)

    def _get_news(self, topic: str) -> str:
        return news_service(topic)

    def _see_camera(self, query: str) -> str:
        if self.camera is None or self.vlm is None:
            return "Camera not available."

        frame = self.camera.capture()
        if frame is None:
            return "Could not capture image from camera."

        try:
            result = self.vlm.analyze(frame, query)
            return result
        except Exception as e:
            return f"Vision error: {str(e)}"

    def _admin_command(self, action: str, target_user: Optional[str] = None) -> str:
        if not self.is_admin:
            return "Sorry, only the admin can do that."

        if action == "list_users":
            users = self.registry.list_users()
            return f"Enrolled users: {', '.join(users)}" if users else "No users enrolled."

        elif action == "delete_user":
            if not target_user:
                return "Please specify which user to delete."
            if target_user.lower() == self.current_user.lower():
                return "Cannot delete yourself."
            self.registry.delete_user(target_user)
            self.user_memory.clear(target_user)
            return f"Deleted user: {target_user}"

        elif action == "view_memories":
            if not target_user:
                return "Please specify which user's memories to view."
            memories = self.user_memory.get_all(target_user)
            return f"{target_user}'s memories:\n" + "\n".join(f"- {m}" for m in memories) if memories else f"No memories for {target_user}."

        elif action == "clear_memories":
            if not target_user:
                return "Please specify which user's memories to clear."
            self.user_memory.clear(target_user)
            return f"Cleared memories for {target_user}."

        elif action == "reset_routines":
            if target_user:
                self.routines.clear(target_user)
                return f"Reset routines for {target_user}."
            # Reset all would need iteration
            return "Routines reset."

        elif action == "system_status":
            user_count = len(self.registry.list_users())
            note_count = self.knowledge_collection.count()
            return f"System Status:\n- Users: {user_count}\n- Knowledge base entries: {note_count}\n- Camera: {'Available' if self.camera else 'Not configured'}"

        return f"Unknown admin action: {action}"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_executor.py -v
```

**Step 5: Commit**

```bash
git add core/executor.py tests/test_executor.py
git commit -m "feat: add tool executor with all tool implementations"
```

---

## Phase 7: Main Application

### Task 7.1: Menzi Voice Assistant (Refactored)

**Files:**
- Create: `menzi.py`

**Step 1: Write the main application**

```python
# menzi.py
"""
Menzi - Jarvis-style Voice Assistant

Run with: python menzi.py
"""

import os
import re
import threading
import queue
import time
import numpy as np
from datetime import datetime

import ollama

from config import LLM_MODEL, ADMIN_USER, MAX_MESSAGES, VISION_CONTEXT_EXPIRY_MESSAGES
from core.tools import get_tools
from core.executor import ToolExecutor
from memory.user_memory import UserMemory
from memory.user_registry import UserRegistry
from memory.routines import RoutineTracker
from identity.resolver import IdentityResolver
from vision.camera import Camera
from vision.vlm import VLM


class TTSEngine:
    """Text-to-speech engine."""

    def __init__(self):
        import platform
        self.system = platform.system()
        self.queue = queue.Queue()
        self.running = True
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        self.current_proc = None

    def _clean_text(self, text):
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _worker(self):
        import subprocess
        while self.running:
            try:
                text = self.queue.get(timeout=0.1)
                if text is None:
                    break

                text = self._clean_text(text)
                if text and len(text) > 2:
                    if self.system == "Darwin":
                        self.current_proc = subprocess.Popen(["say", "-r", "200", text])
                    else:
                        self.current_proc = subprocess.Popen(["espeak", "-s", "160", text])

                    if self.current_proc:
                        self.current_proc.wait()
                        self.current_proc = None

                self.queue.task_done()
            except queue.Empty:
                continue

    def speak(self, text):
        self.queue.put(text)

    def speak_and_wait(self, text):
        self.speak(text)
        self.wait()

    def wait(self):
        self.queue.join()

    def stop(self):
        self.running = False
        self.queue.put(None)
        if self.current_proc:
            self.current_proc.terminate()


class STTEngine:
    """Speech-to-text engine with voice capture."""

    def __init__(self):
        import speech_recognition as sr
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self._last_audio = None

    def listen(self):
        import speech_recognition as sr
        with sr.Microphone() as source:
            print("\n[LISTENING...]")
            try:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=30)
                self._last_audio = audio
                print("[PROCESSING...]")
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                print(f"[STT ERROR] {e}")
                return None

    def get_audio_data(self) -> np.ndarray:
        """Get raw audio from last listen() call for voice ID."""
        if self._last_audio is None:
            return None
        # Convert to numpy array (16kHz mono)
        audio_bytes = self._last_audio.get_raw_data(convert_rate=16000, convert_width=2)
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


class Menzi:
    """Main Menzi assistant class."""

    def __init__(self):
        self.tts = TTSEngine()
        self.stt = STTEngine()
        self.camera = Camera()
        self.vlm = VLM()

        self.identity = IdentityResolver()
        self.registry = UserRegistry()
        self.routines = RoutineTracker()
        self.user_memory = UserMemory()

        self.current_user = None
        self.conversation = []
        self.vision_context = None
        self.vision_message_count = 0

        # Ensure admin exists
        if not self.registry.user_exists(ADMIN_USER):
            self.registry.create_user(ADMIN_USER, admin=True)

    def identify_user(self, audio: np.ndarray) -> str:
        """Identify user from voice, using face as backup if needed."""
        user, confidence, method = self.identity.resolve(voice_audio=audio)

        if user:
            print(f"[IDENTITY] Recognized {user} via {method} (confidence: {confidence:.2f})")
            return user

        # Unknown user - try face backup
        print("[IDENTITY] Voice uncertain, checking camera...")
        frame = self.camera.capture()
        if frame is not None:
            user, confidence, method = self.identity.resolve(voice_audio=audio, face_image=frame)
            if user:
                print(f"[IDENTITY] Recognized {user} via {method}")
                return user

        # Still unknown - enroll new user
        return None

    def enroll_new_user(self, audio: np.ndarray) -> str:
        """Enroll a new user."""
        self.tts.speak_and_wait("I don't recognize you. What's your name?")
        name_text = self.stt.listen()

        if not name_text:
            return None

        name = name_text.strip().split()[0].lower()  # Take first word as name
        print(f"[ENROLL] Enrolling new user: {name}")

        # Capture face
        self.tts.speak("Let me see your face for a moment.")
        time.sleep(1)
        frame = self.camera.capture()

        # Enroll with voice and face
        self.identity.enroll_user(name, voice_audio=audio, face_image=frame)
        self.registry.create_user(name, admin=False)

        self.tts.speak_and_wait(f"Nice to meet you, {name}. I'll remember you.")
        return name

    def check_user_handoff(self, audio: np.ndarray) -> bool:
        """Check if a different user is now speaking."""
        if self.current_user is None:
            return False

        user, confidence, method = self.identity.resolve(voice_audio=audio)
        if user and user != self.current_user:
            print(f"[HANDOFF] User changed from {self.current_user} to {user}")
            self._switch_user(user)
            return True
        return False

    def _switch_user(self, new_user: str):
        """Switch to a new user."""
        old_user = self.current_user
        self.current_user = new_user
        self.conversation = []  # Clear conversation
        self.vision_context = None  # Clear vision context

        if old_user:
            self.tts.speak(f"Hi {new_user}. I was just talking with {old_user}. How can I help you?")
        else:
            self.tts.speak(f"Hi {new_user}. How can I help you?")

    def get_system_prompt(self) -> str:
        """Build system prompt with user context."""
        memories = self.user_memory.get_all(self.current_user)
        memory_str = ""
        if memories:
            memory_str = "KNOWN FACTS ABOUT USER:\n" + "\n".join(f"- {m}" for m in memories)

        vision_str = ""
        if self.vision_context:
            age = (datetime.now() - datetime.fromisoformat(self.vision_context["timestamp"])).total_seconds()
            vision_str = f"\nCURRENT VISUAL CONTEXT (from {int(age)} seconds ago):\n{self.vision_context['description']}\n"

        is_admin = self.registry.is_admin(self.current_user)
        admin_str = "\nYou are speaking to the admin user who can manage the system." if is_admin else ""

        suggestion = self.routines.get_suggestion(self.current_user)
        suggestion_str = f"\nProactive suggestion: {suggestion}" if suggestion else ""

        return f"""You are Menzi, a Jarvis-style voice assistant. Keep responses SHORT and conversational (2-3 sentences max).
You are currently speaking to {self.current_user}.{admin_str}

{memory_str}
{vision_str}
{suggestion_str}

Tools available: remember, save_note, search_notes, web_search, get_weather, get_news, see_camera, admin_command.
Address the user by name occasionally. Be concise - this is spoken aloud."""

    def chat(self, user_input: str) -> str:
        """Process user input and generate response."""
        self.conversation.append({"role": "user", "content": user_input})

        # Trim conversation if too long
        if len(self.conversation) > MAX_MESSAGES:
            self.conversation = self.conversation[-MAX_MESSAGES:]

        # Check vision context expiry
        self.vision_message_count += 1
        if self.vision_message_count > VISION_CONTEXT_EXPIRY_MESSAGES:
            self.vision_context = None
            self.vision_message_count = 0

        # Build messages
        messages = [{"role": "system", "content": self.get_system_prompt()}] + self.conversation

        # Create executor
        executor = ToolExecutor(
            current_user=self.current_user,
            is_admin=self.registry.is_admin(self.current_user),
            camera=self.camera,
            vlm=self.vlm
        )

        # Call LLM with tools
        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            tools=get_tools()
        )

        # Handle tool calls
        while response["message"].get("tool_calls"):
            for tc in response["message"]["tool_calls"]:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                print(f"[TOOL] {name}: {args}")

                result = executor.execute(name, args)
                print(f"[RESULT] {result[:200]}..." if len(result) > 200 else f"[RESULT] {result}")

                # Track routine if it's a search
                if name in ["web_search", "get_weather", "get_news"]:
                    self.routines.record_query(self.current_user, name.replace("get_", ""))

                # Update vision context
                if name == "see_camera":
                    self.vision_context = {
                        "timestamp": datetime.now().isoformat(),
                        "description": result
                    }
                    self.vision_message_count = 0

                messages.append(response["message"])
                messages.append({"role": "tool", "content": result})

            response = ollama.chat(model=LLM_MODEL, messages=messages, tools=get_tools())

        # Get final response
        content = response["message"].get("content", "")
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        self.conversation.append({"role": "assistant", "content": content})

        # Record interaction
        self.routines.record_interaction(self.current_user)

        return content

    def run(self):
        """Main loop."""
        print("=" * 50)
        print("Menzi - Jarvis-style Voice Assistant")
        print("=" * 50)
        print("Say 'Hey Menzi' or just start talking.\n")

        self.tts.speak_and_wait("Hello! I'm Menzi, your assistant. Who am I speaking with?")

        while True:
            # Listen for input
            text = self.stt.listen()

            if text is None:
                continue

            print(f"\nYou: {text}")

            # Check for exit
            if text.lower() in ["quit", "exit", "goodbye", "bye"]:
                self.tts.speak_and_wait("Goodbye!")
                break

            # Get audio for identity
            audio = self.stt.get_audio_data()

            # Identify or enroll user on first interaction
            if self.current_user is None:
                user = self.identify_user(audio)
                if user is None:
                    user = self.enroll_new_user(audio)
                    if user is None:
                        self.tts.speak("I couldn't get your name. Let's try again.")
                        continue
                self._switch_user(user)
            else:
                # Check for user handoff
                self.check_user_handoff(audio)

            # Process input
            try:
                response = self.chat(text)
                print(f"\nMenzi: {response}")

                # Speak response sentence by sentence
                sentences = re.split(r'(?<=[.!?])\s+', response)
                for sentence in sentences:
                    if sentence.strip():
                        self.tts.speak(sentence.strip())
                self.tts.wait()

            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback
                traceback.print_exc()
                self.tts.speak("Sorry, I encountered an error.")

        self.tts.stop()
        self.camera.release()


def main():
    menzi = Menzi()
    menzi.run()


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add menzi.py
git commit -m "feat: add main Menzi application with all features integrated"
```

---

## Phase 8: Testing & Polish

### Task 8.1: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
import pytest
import tempfile
import os

def test_menzi_imports():
    """Test that all modules can be imported."""
    from config import LLM_MODEL, ADMIN_USER
    from core.tools import get_tools
    from core.executor import ToolExecutor
    from memory.user_memory import UserMemory
    from memory.user_registry import UserRegistry
    from memory.routines import RoutineTracker
    from identity.voice_id import VoiceIdentifier
    from identity.face_id import FaceIdentifier
    from identity.resolver import IdentityResolver
    from vision.camera import Camera
    from vision.vlm import VLM
    from services.weather import get_weather
    from services.news import get_news

    assert LLM_MODEL is not None
    assert len(get_tools()) == 8

def test_full_tool_chain():
    """Test tool executor with all tools."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = ToolExecutor(
            current_user="testuser",
            is_admin=True,
            data_dir=tmpdir
        )

        # Test remember
        result = executor.execute("remember", {"fact": "Test fact"})
        assert "remembered" in result.lower()

        # Test save_note
        result = executor.execute("save_note", {"content": "Test note", "topic": "test"})
        assert "saved" in result.lower()

        # Test search_notes
        result = executor.execute("search_notes", {"query": "test"})
        assert isinstance(result, str)

        # Test admin
        result = executor.execute("admin_command", {"action": "system_status"})
        assert "status" in result.lower()
```

**Step 2: Run all tests**

```bash
pytest tests/ -v
```

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests"
```

---

### Task 8.2: Final Cleanup

**Step 1: Update .gitignore**

Add to `.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
data/face_encodings/*.pkl
data/voice_embeddings/*.pkl
data/memories/*.json
!data/memories/.gitkeep
!data/face_encodings/.gitkeep
!data/voice_embeddings/.gitkeep
```

**Step 2: Final commit**

```bash
git add .gitignore
git commit -m "chore: update gitignore for user data"
```

---

## Summary

This plan implements:

1. **Multi-user memory system** - Each user has isolated memories
2. **User registry** - Admin/non-admin tracking
3. **Routine learning** - Pattern detection and proactive suggestions
4. **Voice identification** - Speaker recognition with resemblyzer
5. **Face identification** - Backup recognition with face_recognition
6. **Identity resolver** - Combines voice + face with confidence thresholds
7. **On-demand camera** - Only activates when needed
8. **VLM with platform abstraction** - Mac (moondream) / Pi (qwen2.5-vl)
9. **Weather service** - Open-Meteo API
10. **News service** - DuckDuckGo scraping
11. **All tools** - remember, save_note, search_notes, web_search, get_weather, get_news, see_camera, admin_command
12. **Tool executor** - Handles all tool calls with permissions
13. **Main Menzi app** - Voice-first with identity, handoff, and proactive features

Run with: `python menzi.py`
