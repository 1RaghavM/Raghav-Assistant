import pytest
import tempfile
import os
from memory.routines import RoutineTracker

@pytest.fixture
def temp_tracker():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{}')
        path = f.name
    yield RoutineTracker(routines_file=path)
    os.unlink(path)

def test_record_interaction(temp_tracker):
    temp_tracker.record_interaction("raghav", hour=8, day="monday")
    patterns = temp_tracker.get_patterns("raghav")
    assert patterns is not None

def test_record_query_pattern(temp_tracker):
    temp_tracker.record_query("raghav", "weather", hour=8)
    temp_tracker.record_query("raghav", "weather", hour=8)
    patterns = temp_tracker.get_patterns("raghav")
    assert any(q["pattern"] == "weather" for q in patterns.get("common_queries", []))

def test_get_suggestion_morning_weather(temp_tracker):
    # Record weather queries in morning
    for _ in range(5):
        temp_tracker.record_query("raghav", "weather", hour=8)
    suggestion = temp_tracker.get_suggestion("raghav", hour=8)
    assert suggestion is not None
    assert "weather" in suggestion.lower()
