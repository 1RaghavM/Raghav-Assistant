import pytest
import numpy as np
from unittest.mock import Mock, patch

def test_face_tracker_init():
    """FaceTracker initializes with camera and face_id."""
    with patch('vision.face_tracker._MP_FACE_DETECTION', Mock()):
        from vision.face_tracker import FaceTracker

        mock_camera = Mock()
        mock_face_id = Mock()
        tracker = FaceTracker(mock_camera, mock_face_id)

        assert tracker.camera == mock_camera
        assert tracker.face_id == mock_face_id
        assert tracker._running == False

def test_get_current_faces_empty():
    """get_current_faces returns empty list when no faces detected."""
    with patch('vision.face_tracker._MP_FACE_DETECTION', Mock()):
        from vision.face_tracker import FaceTracker

        tracker = FaceTracker(Mock(), Mock())
        faces = tracker.get_current_faces()

        assert faces == []

def test_set_speaker():
    """set_speaker updates speaker_id and confidence."""
    with patch('vision.face_tracker._MP_FACE_DETECTION', Mock()):
        from vision.face_tracker import FaceTracker

        tracker = FaceTracker(Mock(), Mock())
        tracker.set_speaker("raghav", 0.85)

        assert tracker.speaker_id == "raghav"
        assert tracker.speaker_confidence == 0.85

def test_get_identity_summary_no_faces():
    """get_identity_summary returns appropriate message when no faces."""
    with patch('vision.face_tracker._MP_FACE_DETECTION', Mock()):
        from vision.face_tracker import FaceTracker

        tracker = FaceTracker(Mock(), Mock())
        summary = tracker.get_identity_summary()

        assert "No one" in summary or "no one" in summary.lower()
