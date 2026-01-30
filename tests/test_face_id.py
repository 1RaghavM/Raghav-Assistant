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
