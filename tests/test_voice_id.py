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
