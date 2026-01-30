# tests/test_integration.py
import pytest
import tempfile
import os

def test_menzi_imports():
    """Test that all modules can be imported."""
    from config import LLM_MODEL, ADMIN_USER, GEMINI_API_KEY
    from core.tools import get_gemini_tools
    from core.executor import ToolExecutor
    from memory.user_memory import UserMemory
    from memory.user_registry import UserRegistry
    from memory.routines import RoutineTracker
    from identity.voice_id import VoiceIdentifier
    from identity.face_id import FaceIdentifier
    from identity.resolver import IdentityResolver
    from vision.camera import Camera
    from vision.vlm import VLM
    from services.weather import get_weather
    from services.news import get_news

    assert LLM_MODEL is not None
    # get_gemini_tools returns a Tool object, not a list
    assert get_gemini_tools() is not None

def test_full_tool_chain():
    """Test tool executor with all tools."""
    from core.executor import ToolExecutor

    with tempfile.TemporaryDirectory() as tmpdir:
        executor = ToolExecutor(
            current_user="testuser",
            is_admin=True,
            data_dir=tmpdir
        )

        # Test remember
        result = executor.execute("remember", {"fact": "Test fact"})
        assert "remembered" in result.lower()

        # Test save_note
        result = executor.execute("save_note", {"content": "Test note", "topic": "test"})
        assert "saved" in result.lower()

        # Test search_notes
        result = executor.execute("search_notes", {"query": "test"})
        assert isinstance(result, str)

        # Test admin
        result = executor.execute("admin_command", {"action": "system_status"})
        assert "status" in result.lower()
