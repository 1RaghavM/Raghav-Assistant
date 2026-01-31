import os
import platform
from dotenv import load_dotenv

load_dotenv()

# Platform detection
PLATFORM = "mac" if platform.system() == "Darwin" else "pi"

# LLM - Cerebras API (ultra-fast inference)
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
LLM_MODEL = "gpt-oss-120b"

# STT - SambaNova Whisper API (cloud)
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

# VLM - Local (Ollama on Mac, Hailo on Pi)
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

# Face tracking
FACE_TRACK_INTERVAL_MS = 200      # Background detection interval
FACE_STALE_TIMEOUT_S = 2.0        # Remove faces not seen for this long
FACE_REIDENTIFY_INTERVAL_S = 5.0  # Re-run recognition every N seconds

# Conversation
MAX_MESSAGES = 16

# Admin
ADMIN_USER = "raghav"
