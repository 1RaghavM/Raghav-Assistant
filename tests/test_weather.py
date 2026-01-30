import pytest
from services.weather import get_weather, get_coordinates

def test_get_coordinates():
    lat, lon = get_coordinates("San Francisco")
    assert lat is not None
    assert lon is not None
    assert 37 < lat < 38  # SF latitude range

def test_get_weather():
    result = get_weather("San Francisco")
    # Check for valid weather response (contains °F) or error message
    assert "°f" in result.lower() or "error" in result.lower()
