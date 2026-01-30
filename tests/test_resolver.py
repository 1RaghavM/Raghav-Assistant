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
