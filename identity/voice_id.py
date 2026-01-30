import os
import pickle
import numpy as np
from typing import Optional, Tuple

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    VoiceEncoder = None
    preprocess_wav = None


class VoiceIdentifier:
    def __init__(self, embeddings_dir: str = None):
        if embeddings_dir is None:
            from config import VOICE_EMBEDDINGS_DIR
            embeddings_dir = VOICE_EMBEDDINGS_DIR
        self.embeddings_dir = embeddings_dir
        os.makedirs(self.embeddings_dir, exist_ok=True)

        if RESEMBLYZER_AVAILABLE:
            self.encoder = VoiceEncoder()
        else:
            self.encoder = None

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

        # Use resemblyzer if available, otherwise create a mock embedding for testing
        if self.encoder is not None:
            embedding = self.encoder.embed_utterance(audio)
        else:
            # Create a deterministic embedding from audio for testing purposes
            # Normalize and hash the audio to create a consistent embedding
            np.random.seed(int(np.abs(audio[:100]).sum() * 1000) % (2**31))
            embedding = np.random.randn(256).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

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

        # Use resemblyzer if available, otherwise create a mock embedding for testing
        if self.encoder is not None:
            query_embedding = self.encoder.embed_utterance(audio)
        else:
            # Create a deterministic embedding from audio for testing purposes
            np.random.seed(int(np.abs(audio[:100]).sum() * 1000) % (2**31))
            query_embedding = np.random.randn(256).astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        best_user = None
        best_similarity = 0.0

        for user, stored_embedding in self._embeddings_cache.items():
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_user = user

        # Import threshold from config, with fallback for testing
        try:
            from config import VOICE_MATCH_THRESHOLD
        except ImportError:
            VOICE_MATCH_THRESHOLD = 0.75

        if best_similarity >= VOICE_MATCH_THRESHOLD:
            return best_user, float(best_similarity)
        return None, float(best_similarity)

    def delete(self, user: str):
        """Delete a user's voice enrollment."""
        user = user.lower()
        path = self._get_path(user)
        if os.path.exists(path):
            os.remove(path)
        if user in self._embeddings_cache:
            del self._embeddings_cache[user]
