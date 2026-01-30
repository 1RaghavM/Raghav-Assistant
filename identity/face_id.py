import os
import pickle
import numpy as np
from typing import Optional, Tuple, List

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

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
        if not FACE_RECOGNITION_AVAILABLE:
            return False
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

        if FACE_RECOGNITION_AVAILABLE:
            distances = face_recognition.face_distance(known_encodings, encoding)
        else:
            # Fallback: use numpy for distance calculation
            distances = np.array([np.linalg.norm(encoding - known) for known in known_encodings])

        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        # Convert distance to confidence (lower distance = higher confidence)
        confidence = 1.0 - best_distance

        # Import threshold from config, with fallback for testing
        try:
            from config import FACE_MATCH_THRESHOLD
        except ImportError:
            FACE_MATCH_THRESHOLD = 0.6

        if confidence >= FACE_MATCH_THRESHOLD:
            return known_users[best_idx], float(confidence)
        return None, float(confidence)

    def identify_from_image(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Identify from BGR image. Returns (user, confidence) or (None, 0)."""
        if not FACE_RECOGNITION_AVAILABLE:
            return None, 0.0
        rgb_image = image[:, :, ::-1]  # BGR to RGB
        # Resize for speed if cv2 is available
        if CV2_AVAILABLE:
            small = cv2.resize(rgb_image, (0, 0), fx=0.25, fy=0.25)
        else:
            small = rgb_image
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
