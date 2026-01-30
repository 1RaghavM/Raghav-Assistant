import pytest
import tempfile
import os
from memory.user_registry import UserRegistry

@pytest.fixture
def temp_registry():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{}')
        path = f.name
    yield UserRegistry(users_file=path)
    os.unlink(path)

def test_create_user(temp_registry):
    temp_registry.create_user("john")
    assert temp_registry.user_exists("john")

def test_admin_check(temp_registry):
    temp_registry.create_user("raghav", admin=True)
    temp_registry.create_user("john", admin=False)
    assert temp_registry.is_admin("raghav") is True
    assert temp_registry.is_admin("john") is False

def test_list_users(temp_registry):
    temp_registry.create_user("raghav", admin=True)
    temp_registry.create_user("john")
    users = temp_registry.list_users()
    assert "raghav" in users
    assert "john" in users

def test_delete_user(temp_registry):
    temp_registry.create_user("john")
    temp_registry.delete_user("john")
    assert temp_registry.user_exists("john") is False

def test_get_user_location(temp_registry):
    temp_registry.create_user("raghav", location="San Francisco")
    assert temp_registry.get_location("raghav") == "San Francisco"
