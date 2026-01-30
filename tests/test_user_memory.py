import pytest
import os
import json
import tempfile
from memory.user_memory import UserMemory

@pytest.fixture
def temp_memory_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_add_memory(temp_memory_dir):
    mem = UserMemory(memories_dir=temp_memory_dir)
    result = mem.add("raghav", "User likes coffee")
    assert result is True
    memories = mem.get_all("raghav")
    assert "User likes coffee" in memories

def test_no_duplicate_memory(temp_memory_dir):
    mem = UserMemory(memories_dir=temp_memory_dir)
    mem.add("raghav", "User likes coffee")
    result = mem.add("raghav", "User likes coffee")
    assert result is False

def test_separate_user_memories(temp_memory_dir):
    mem = UserMemory(memories_dir=temp_memory_dir)
    mem.add("raghav", "Raghav fact")
    mem.add("john", "John fact")
    assert "Raghav fact" in mem.get_all("raghav")
    assert "John fact" not in mem.get_all("raghav")
    assert "John fact" in mem.get_all("john")

def test_clear_user_memories(temp_memory_dir):
    mem = UserMemory(memories_dir=temp_memory_dir)
    mem.add("raghav", "Some fact")
    mem.clear("raghav")
    assert mem.get_all("raghav") == []
