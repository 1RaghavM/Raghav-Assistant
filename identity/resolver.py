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
