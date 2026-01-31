import pytest
import tempfile
import numpy as np
from core.executor import ToolExecutor

@pytest.fixture
def executor():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ToolExecutor(
            current_user="raghav",
            is_admin=True,
            data_dir=tmpdir
        )

def test_execute_remember(executor):
    result = executor.execute("remember", {"fact": "User likes pizza"})
    assert "remembered" in result.lower() or "pizza" in result.lower()

def test_execute_weather(executor):
    result = executor.execute("get_weather", {"location": "San Francisco"})
    assert isinstance(result, str)

def test_execute_admin_list_users(executor):
    result = executor.execute("admin_command", {"action": "list_users"})
    assert isinstance(result, str)

def test_non_admin_blocked():
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = ToolExecutor(current_user="john", is_admin=False, data_dir=tmpdir)
        result = executor.execute("admin_command", {"action": "list_users"})
        assert "admin" in result.lower() or "only" in result.lower()

def test_see_camera_with_face_tracker():
    """see_camera includes face annotations when face_tracker provided."""
    from core.executor import ToolExecutor
    from unittest.mock import Mock

    mock_camera = Mock()
    mock_camera.capture.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_vlm = Mock()
    mock_vlm.analyze.return_value = "A person is typing on a laptop."

    mock_face_tracker = Mock()
    mock_face_tracker.get_faces_for_vlm.return_value = "Known faces in frame:\n- Raghav (speaker): center"

    with tempfile.TemporaryDirectory() as tmpdir:
        executor = ToolExecutor(
            current_user="raghav",
            is_admin=False,
            data_dir=tmpdir,
            camera=mock_camera,
            vlm=mock_vlm,
            face_tracker=mock_face_tracker
        )

        result = executor._see_camera("what do you see?")

        # Verify VLM was called with enhanced query
        call_args = mock_vlm.analyze.call_args
        query = call_args[0][1]
        assert "Raghav" in query or "Known faces" in query
