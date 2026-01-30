# Menzi Vision & Identity System Design

**Date:** 2026-01-29
**Status:** Approved
**Target:** Mac M4 (development) → Pi 5 + AI HAT+ 2 (deployment)

---

## Overview

Menzi is a Jarvis-style voice assistant with:
- **Voice-primary identification** - recognizes users by voice, camera as backup
- **Scene understanding** - VLM answers "What am I doing?" queries on-demand
- **Dynamic user enrollment** - new users automatically prompted for name
- **Routine learning** - learns patterns, offers proactive assistance
- **Multi-user support** - each user has separate memories, Raghav is admin

---

## Architecture

### Core Principle: Voice-First, Camera On-Demand

```
[Always Listening for Wake Word]
              ↓
[Wake Word: "Hey Menzi"]
              ↓
[Voice Embedding Captured from speech]
              ↓
[Identity Resolver]
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
[Voice Match]     [Uncertain]
    ↓                   ↓
[Load User]       [Camera ON → Face Check → Camera OFF]
                        ↓
                  [Still Unknown?]
                        ↓
                  ["What's your name?" → Enroll]
              ↓
[Conversation Loop with User Context]
              ↓
[LLM may call tools: web_search, remember, see_camera, etc.]
              ↓
[Conversation Ends]
```

### Camera Activates Only When:
1. Voice ID is uncertain (backup identification)
2. LLM calls `see_camera` tool ("What am I doing?")
3. New user enrollment (capture face after they say name)
4. User handoff confirmation (voice changed mid-conversation)

---

## Features

### 1. Voice-Primary Identification

**Voice Embedding:**
- Library: `resemblyzer`
- ~256-dim embedding from ~2 seconds of speech
- Captured from wake word + first sentence
- ~100ms processing time

**Face Recognition (Backup):**
- Library: `face_recognition` (dlib-based)
- Only activated when voice is uncertain
- Camera ON → capture → encode → compare → Camera OFF

**Identity Resolution Logic:**
```python
def resolve_identity(voice_embedding, face_embedding=None):
    voice_match, voice_confidence = match_voice(voice_embedding)

    if voice_confidence > 0.8:
        return voice_match  # High confidence voice match

    if voice_confidence > 0.6 and face_embedding:
        face_match, face_confidence = match_face(face_embedding)
        if voice_match == face_match:
            return voice_match  # Both agree
        if face_confidence > 0.8:
            return face_match  # Trust face when voice uncertain

    if voice_confidence > 0.6:
        return voice_match  # Moderate voice confidence, no face

    return None  # Unknown - trigger enrollment
```

### 2. Scene Continuity (VLM Context)

**Problem:** User asks "What am I doing?" then "What about the thing on my left?"

**Solution:** VLM results persist in conversation context.

```python
vision_context = {
    "timestamp": "2026-01-29T10:30:00",
    "description": "User at desk with laptop, coffee mug right, notebook left...",
    "expires_after_messages": 10,
    "expires_after_minutes": 5
}
```

**Injected into system prompt when present:**
```
CURRENT VISUAL CONTEXT (from 2 minutes ago):
You see: User at desk with laptop, coffee mug on right, notebook on left...
```

**Expiry:** Vision context expires after 5 minutes OR 10 conversation turns (whichever first).

### 3. Routine Learning

**Tracked per user:**
```python
routines = {
    "raghav": {
        "interaction_times": [
            {"hour": 8, "day": "weekday", "count": 45},
            {"hour": 22, "day": "weekday", "count": 38},
        ],
        "common_queries": [
            {"pattern": "weather", "time": "morning", "count": 30},
            {"pattern": "news", "time": "morning", "count": 22},
        ],
        "recent_topics": ["DormDish", "Raspberry Pi", "Hailo"],
        "preferences": {
            "greeting_style": "brief",
            "wants_proactive_info": True
        }
    }
}
```

**Proactive Jarvis-style behaviors:**

| Pattern Detected | Menzi Behavior |
|------------------|----------------|
| Morning + user usually asks weather | "Good morning Raghav. It's 72°F and clear today." |
| User researching topic X frequently | "I found some new articles about DormDish - want a summary?" |
| Time matches usual work session start | "Ready when you are." |
| New user at unusual hour | "Hi John, Raghav usually handles things at this hour - need me to note something for him?" |

**Key principle:** Proactive but not annoying - suggestions offered once, not repeated if ignored.

### 4. User Handoff

**Detection:** Voice embedding compared on every utterance.

```
[Ongoing Conversation with Raghav]
              ↓
[New speech input received]
              ↓
[Voice Embedding extracted]
              ↓
[Compare to current_user embedding]
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
[Match: Same User]  [Mismatch: Different Voice]
    ↓                   ↓
[Continue]          [Identify New Speaker]
                          ↓
                    [Camera ON → Face backup → Camera OFF]
                          ↓
              ┌───────────┴───────────┐
              ↓                       ↓
        [Known User]           [Unknown User]
              ↓                       ↓
        [Switch Context]       [Enrollment Flow]
```

**Handoff behavior:**
```python
# Menzi responds:
"Hi John. I was just talking with Raghav. How can I help you?"

# Context switch:
# - Save Raghav's conversation to his history
# - Load John's memories
# - Start fresh conversation for John
# - Vision context cleared (stale for new user)
```

**Thresholds:**
- Voice similarity threshold: 0.75 (below = different person)
- Brief pauses (< 3 seconds) assume same speaker

### 5. Admin Commands (Raghav Only)

| Command | Action |
|---------|--------|
| "List all users" | Shows enrolled users with join dates |
| "Delete user [name]" | Removes user, their memories, face/voice data |
| "Show [user]'s memories" | View another user's stored facts |
| "Clear [user]'s memories" | Wipe another user's data |
| "Who's been talking to you?" | Interaction log - who, when, how many conversations |
| "Reset routines" | Clears learned patterns for all users |
| "System status" | Camera working, storage used, users count |

**Permission check:**
```python
def execute_admin_command(action, target_user, current_user):
    if not is_admin(current_user):
        return "Sorry, only Raghav can do that."
    # Execute action...
```

---

## Tools

| Tool | Description | When LLM Calls It |
|------|-------------|-------------------|
| `remember` | Store fact about current user | User shares personal info |
| `save_note` | Save to knowledge base (RAG) | User wants to store information |
| `search_notes` | Query ChromaDB knowledge base | User asks about saved info |
| `web_search` | DuckDuckGo search | User needs current info |
| `get_weather` | Fetch weather for location | User asks weather, or proactive morning |
| `get_news` | Fetch headlines by topic | User asks news, or proactive briefing |
| `see_camera` | Activate VLM for scene analysis | User asks what they're doing, visual context needed |
| `admin_command` | System administration | Admin user gives system commands |

**Tool Schema Example (see_camera):**
```python
{
    "type": "function",
    "function": {
        "name": "see_camera",
        "description": "Look through the camera to see what's happening. Use when user asks 'what am I doing', 'what do you see', 'look at this', or needs visual context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific question about the scene (e.g., 'What is the user doing?', 'Describe what you see')"
                }
            },
            "required": ["query"]
        }
    }
}
```

---

## Project Structure

```
raghav-assistant/
├── main.py                 # Entry point (text mode)
├── voice_assistant.py      # Voice mode entry point
├── config.py               # Settings, thresholds, API endpoints
│
├── core/
│   ├── __init__.py
│   ├── llm.py              # Ollama wrapper, tool execution
│   ├── conversation.py     # Context management, history
│   └── tools.py            # Tool definitions & handlers
│
├── identity/
│   ├── __init__.py
│   ├── voice_id.py         # Speaker embedding (resemblyzer)
│   ├── face_id.py          # Face recognition (face_recognition lib)
│   ├── resolver.py         # Combines voice + face, resolves identity
│   └── enrollment.py       # New user registration flow
│
├── vision/
│   ├── __init__.py
│   ├── camera.py           # Camera capture (cv2), on-demand
│   └── vlm.py              # VLM wrapper with backend abstraction
│
├── services/
│   ├── __init__.py
│   ├── weather.py          # Open-Meteo API
│   ├── news.py             # News fetching
│   └── web_search.py       # DuckDuckGo search
│
├── memory/
│   ├── __init__.py
│   ├── user_memory.py      # Per-user facts (JSON)
│   ├── knowledge_base.py   # ChromaDB RAG
│   └── routines.py         # Pattern learning & storage
│
├── audio/
│   ├── __init__.py
│   ├── stt.py              # Speech-to-text
│   ├── tts.py              # Text-to-speech
│   └── wake_word.py        # Wake word detection
│
└── data/
    ├── users.json          # User registry
    ├── routines.json       # Learned patterns
    ├── memories/
    │   └── {user}.json     # Per-user memories
    ├── face_encodings/
    │   └── {user}.pkl      # Face embeddings
    ├── voice_embeddings/
    │   └── {user}.pkl      # Voice embeddings
    └── chroma_db/          # ChromaDB storage
```

---

## Dependencies

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

---

## Platform Differences: Mac vs Pi

| Component | Mac M4 (Dev) | Pi 5 + AI HAT+ 2 (Deploy) |
|-----------|--------------|---------------------------|
| LLM | Ollama (qwen3:1.7b) | Ollama (qwen3:1.7b) |
| VLM | Ollama (moondream) | Hailo (qwen2.5-vl-3b) via hailo-ollama |
| TTS | `say` command (macOS) | `espeak` (Linux) |
| Face lib | face_recognition | face_recognition |
| Camera | cv2 VideoCapture (webcam) | cv2 with picamera2 backend |
| Voice ID | resemblyzer (CPU) | resemblyzer (CPU, 1 thread) |

### VLM Backend Abstraction

```python
# vision/vlm.py

class VLMBackend:
    def analyze(self, frame, query: str) -> str:
        raise NotImplementedError

class OllamaVLM(VLMBackend):
    """Mac development - uses Ollama moondream"""
    def __init__(self):
        self.model = "moondream"

    def analyze(self, frame, query: str) -> str:
        # Convert frame to base64, send to ollama
        ...

class HailoVLM(VLMBackend):
    """Pi deployment - uses Hailo-10H"""
    def __init__(self):
        # Initialize hailo-ollama connection
        self.model = "qwen2.5-vl-3b"

    def analyze(self, frame, query: str) -> str:
        # Send to Hailo NPU
        ...

# Factory function
def get_vlm_backend() -> VLMBackend:
    if config.PLATFORM == "mac":
        return OllamaVLM()
    else:
        return HailoVLM()
```

### Pi-Specific Optimizations

**CPU Thread Limiting:**
```python
import torch
torch.set_num_threads(1)  # Limit face_recognition/resemblyzer
```

**Ollama Thread Limiting:**
```bash
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
```

**Camera Resolution (Pi):**
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

---

## Hardware Budget (Pi 5 Deployment)

| Component | Resource | Notes |
|-----------|----------|-------|
| Wake word detection | CPU (minimal) | Always on |
| Voice ID (resemblyzer) | CPU, 1 thread | ~100ms per embedding |
| Face recognition | CPU, 1 thread | Only when needed, 0.5s intervals |
| LLM (Qwen 1.7B) | CPU, 2-3 cores | Burst during response |
| VLM (Qwen2.5-VL-3B) | NPU (Hailo-10H) | On-demand, 8GB dedicated RAM |

**Thermal:** Active cooler required on Pi 5.

---

## Data Storage Schema

### users.json
```json
{
  "raghav": {
    "admin": true,
    "created_at": "2026-01-29T10:00:00",
    "default_location": "San Francisco"
  },
  "john": {
    "admin": false,
    "created_at": "2026-01-30T14:30:00",
    "default_location": null
  }
}
```

### memories/{user}.json
```json
[
  "User's name is Raghav",
  "User is a CS student",
  "User is building DormDish.ai",
  "User lives in San Francisco"
]
```

### routines.json
```json
{
  "raghav": {
    "interaction_times": [
      {"hour": 8, "day": "weekday", "count": 45}
    ],
    "common_queries": [
      {"pattern": "weather", "time": "morning", "count": 30}
    ],
    "recent_topics": ["DormDish", "Raspberry Pi"]
  }
}
```

---

## Migration Checklist: Mac → Pi

When ready to deploy to Pi 5 + AI HAT+ 2:

- [ ] Install Hailo SDK and hailo-ollama
- [ ] Download Qwen2.5-VL-3B .hef model from Hailo Model Zoo
- [ ] Update `config.py`: set `PLATFORM = "pi"`
- [ ] Install picamera2: `sudo apt install python3-picamera2`
- [ ] Set thread limits in startup script
- [ ] Install espeak: `sudo apt install espeak`
- [ ] Test camera with `libcamera-hello`
- [ ] Verify thermal management (active cooler)
- [ ] Run performance benchmarks
- [ ] Test all tools end-to-end

---

## API References

**Open-Meteo (Weather):**
```
https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true
```

**DuckDuckGo (Search):**
```
https://html.duckduckgo.com/html/?q={query}
```

---

## Future Enhancements (Not in Initial Build)

- Gesture commands (wave to wake)
- Home automation integration
- Local Whisper STT (offline)
- Emotion detection from face
- Visual memory ("Where did I leave my keys?")
