import pytest
import tempfile
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
