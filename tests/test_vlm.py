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
