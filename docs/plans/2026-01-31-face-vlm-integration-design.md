# Face Detection + VLM Integration Design

## Problem Statement

When the VLM sees multiple people in frame, it describes what it sees but has no way to identify WHO each person is. If Raghav is speaking and another person is present, the LLM cannot correlate "which person is doing what" with identity.

**Goal:** Seamless multi-modal identity awareness - the LLM always knows who is in frame, who is speaking, and what each person is doing.

## Design Decisions

| Decision | Choice |
|----------|--------|
| Annotation style | Full annotation with speaker emphasis |
| Speaker detection | Voice ID (primary) + VAD timing (fallback) |
| Face + VLM integration | Pre-process face detection, inject into VLM prompt |
| Face detection timing | Continuous background thread (200ms interval) |
| LLM context | System prompt (identity summary) + enriched `see_camera` tool |
| Confidence tiers | HIGH/MEDIUM/LOW with transparent behavior |
| Enrollment trigger | Speaker-triggered (unknown person speaks) |

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     FaceTracker (Background Thread)             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Camera    │───▶│  BlazeFace   │───▶│  FaceIdentifier  │   │
│  │  (capture)  │    │  (detect)    │    │  (recognize)     │   │
│  └─────────────┘    └──────────────┘    └──────────────────┘   │
│                              │                    │              │
│                              ▼                    ▼              │
│                     ┌─────────────────────────────────┐         │
│                     │   detected_faces (cached state) │         │
│                     │   - name, confidence, bbox      │         │
│                     │   - speaker_id, last_seen       │         │
│                     └─────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │  System Prompt  │    │   see_camera    │    │  Voice Pipeline │
   │  (identity ctx) │    │  (VLM + faces)  │    │  (set_speaker)  │
   └─────────────────┘    └─────────────────┘    └─────────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    ▼
                            ┌─────────────┐
                            │     LLM     │
                            │  (Cerebras) │
                            └─────────────┘
```

### Data Flow

1. **Background thread** captures frames every 200ms
2. **BlazeFace** detects faces with bounding boxes (~5ms)
3. **FaceIdentifier** recognizes only new/changed faces (~25ms per face)
4. **Voice pipeline** updates `speaker_id` when speech detected
5. **System prompt** always includes identity summary
6. **see_camera tool** injects face annotations into VLM prompt

## Optimized Face Detection Pipeline

### Two-Stage Approach

**Stage 1: Detection (every 200ms) - FAST**
- Use MediaPipe BlazeFace (200-1000 FPS capable)
- Returns bounding boxes + landmarks
- Speed: ~5-10ms per frame

**Stage 2: Recognition (on-demand)**
- Only runs when new face appears or after N seconds
- Crops face from bbox, runs encoding
- Compares against pre-loaded encodings
- Speed: ~20-30ms per face

### Optimizations

1. **MediaPipe BlazeFace** for detection instead of dlib
2. **Skip recognition if face unchanged** - track across frames
3. **Pre-loaded encodings** - already implemented
4. **Face tracking** - don't re-identify same face every frame

### Expected Latency

| Operation | Current | Optimized |
|-----------|---------|-----------|
| Face detection | ~100ms | ~5-10ms |
| Face recognition | ~50ms/face | ~25ms/face (only when needed) |
| Total per cycle | ~150ms+ | ~10-30ms |

## Speaker Detection Logic

```python
def get_current_speaker(self) -> dict:
    voice = self._last_voice_id
    faces = self.detected_faces

    # HIGH: Voice confident + face matches
    if voice.confidence >= 0.75:
        matching_face = find_face_by_name(faces, voice.name)
        if matching_face:
            return {"name": voice.name, "verified": "voice+face", "confidence": "HIGH"}
        return {"name": voice.name, "verified": "voice_only", "confidence": "MEDIUM"}

    # MEDIUM: Weak voice + single known face
    if voice.confidence >= 0.6 and has_single_known_face(faces):
        return {"name": faces[0].name, "verified": "face+weak_voice", "confidence": "MEDIUM"}

    # LOW: No confident match
    return {"name": "Unknown", "verified": "none", "confidence": "LOW"}
```

## Confidence Tiers & Behavior

| Confidence | Verification | LLM Behavior |
|------------|--------------|--------------|
| **HIGH** | voice + face match | Full access, personal responses |
| **MEDIUM** | voice OR face only | Normal responses, confirm sensitive actions |
| **LOW** | mismatch or unknown | Generic responses, transparent about uncertainty |

**Mismatch handling:** LLM is transparent:
> "I hear Raghav's voice but I'm seeing someone I don't recognize. Is that you, Raghav?"

## LLM Context Integration

### System Prompt Injection

Every LLM call includes identity context:

```
Current presence:
- Raghav (speaker, HIGH confidence: voice+face match)
- Unknown person (background)
Speaker verified: YES
```

### Enhanced see_camera Tool

VLM prompt includes face annotations:

```
Known faces in frame:
- Raghav (speaker): bbox [100,50,200,180]
- Unknown: bbox [400,60,480,170]

Describe what each person is doing, with emphasis on the speaker.
```

**VLM output example:**
> "Raghav (speaker) is sitting at a desk typing on a MacBook, looking focused. An unknown person is standing behind him near the doorway, holding a coffee cup."

## Enrollment Flow

**Trigger:** Unknown person speaks (VAD detects audio, voice ID < 0.6 confidence)

**Flow:**
1. Unknown voice detected
2. Check if unrecognized face visible
3. LLM naturally asks: "I don't think we've met. What's your name?"
4. User responds: "I'm John"
5. System enrolls voice + face for "John"

**Safeguards:**
- Only enroll if single clear face visible
- Require face looking at camera (use landmarks)
- No auto-update of existing users

## File Structure

### New Files

```
vision/
└── face_tracker.py      # Background thread + cached face state
```

### Modified Files

| File | Changes |
|------|---------|
| `vision/vlm.py` | Accept face annotations in prompt |
| `identity/face_id.py` | Swap detector to MediaPipe BlazeFace |
| `identity/resolver.py` | Integrate with FaceTracker |
| `menzi.py` | Start/stop FaceTracker, inject identity into system prompt |
| `core/executor.py` | Enhance `_see_camera` with face annotations |
| `config.py` | Add tracking config constants |

### Configuration Additions

```python
# config.py
FACE_TRACK_INTERVAL_MS = 200      # Background detection interval
FACE_STALE_TIMEOUT_S = 2          # Remove faces not seen for this long
FACE_REIDENTIFY_INTERVAL_S = 5    # Re-run recognition every N seconds
```

### Dependency Changes

```
# Add
mediapipe    # For BlazeFace detection (fast, lightweight)

# face_recognition library kept for encoding/matching
```

## Implementation Order

1. **Create `vision/face_tracker.py`** - Background thread with cached state
2. **Modify `identity/face_id.py`** - Add MediaPipe BlazeFace detector
3. **Modify `menzi.py`** - Initialize FaceTracker, inject into system prompt
4. **Modify `core/executor.py`** - Enhance `see_camera` with annotations
5. **Modify `identity/resolver.py`** - Wire speaker detection
6. **Update `config.py`** - Add new constants
7. **Test end-to-end** - Verify latency targets met
