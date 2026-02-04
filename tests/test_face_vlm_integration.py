"""Integration tests for face detection + VLM context flow."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

def test_identity_summary_flows_to_system_prompt():
    """Verify face tracker identity flows to LLM system prompt."""
    with patch('vision.face_tracker._MP_FACE_DETECTION', Mock()), \
         patch('menzi.VoiceInput'), \
         patch('menzi.Cerebras'), \
         patch('menzi.Camera'), \
         patch('menzi.VLM'), \
         patch('menzi.CEREBRAS_API_KEY', 'test-key'):

        from menzi import Menzi

        # Mock the face tracker
        menzi = Menzi()
        menzi.face_tracker.get_identity_summary = Mock(
            return_value="Current presence:\n- Raghav (speaker, HIGH confidence)"
        )

        prompt = menzi._system_prompt()

        assert "Raghav" in prompt
        assert "speaker" in prompt.lower()

def test_see_camera_includes_face_context():
    """Verify see_camera tool includes face annotations in VLM query."""
    from core.executor import ToolExecutor

    mock_camera = Mock()
    mock_camera.capture.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_vlm = Mock()
    mock_vlm.analyze.return_value = "Raghav is typing on a laptop."

    mock_face_tracker = Mock()
    mock_face_tracker.get_faces_for_vlm.return_value = (
        "Known faces in frame:\n- Raghav (speaker): center\n- Unknown person: left side"
    )

    executor = ToolExecutor(
        current_user="raghav",
        is_admin=False,
        camera=mock_camera,
        vlm=mock_vlm,
        face_tracker=mock_face_tracker
    )

    result = executor.execute("see_camera", {"query": "what's happening?"})

    # Check VLM was called with enhanced query
    call_args = mock_vlm.analyze.call_args
    enhanced_query = call_args[0][1]

    assert "Raghav (speaker)" in enhanced_query
    assert "Unknown person" in enhanced_query
    assert "emphasis on the speaker" in enhanced_query

def test_speaker_set_on_voice_identification():
    """Verify voice identification updates face tracker speaker."""
    with patch('vision.face_tracker._MP_FACE_DETECTION', Mock()), \
         patch('menzi.VoiceInput'), \
         patch('menzi.Cerebras'), \
         patch('menzi.Camera'), \
         patch('menzi.VLM'), \
         patch('menzi.CEREBRAS_API_KEY', 'test-key'):

        from menzi import Menzi

        menzi = Menzi()

        # Mock identity resolver to return a user
        menzi.identity.resolve = Mock(return_value=("raghav", 0.85, "voice"))

        # Spy on face tracker set_speaker
        menzi.face_tracker.set_speaker = Mock()

        # Call identify
        audio = np.zeros(16000, dtype=np.float32)
        result = menzi.identify(audio)

        assert result == "raghav"
        menzi.face_tracker.set_speaker.assert_called_once_with("raghav", 0.85)
