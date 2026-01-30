import pytest
from vision.camera import Camera

def test_camera_init():
    cam = Camera()
    assert cam is not None

def test_camera_capture_returns_frame_or_none():
    cam = Camera()
    frame = cam.capture()
    # Either returns a frame (ndarray) or None if no camera
    assert frame is None or hasattr(frame, 'shape')
