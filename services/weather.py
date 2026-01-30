import requests
from typing import Tuple, Optional

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

def get_coordinates(city: str) -> Tuple[Optional[float], Optional[float]]:
    """Get latitude/longitude for a city."""
    try:
        response = requests.get(GEOCODE_URL, params={"name": city, "count": 1}, timeout=10)
        data = response.json()
        if "results" in data and data["results"]:
            result = data["results"][0]
            return result["latitude"], result["longitude"]
    except Exception:
        pass
    return None, None

def get_weather(location: str) -> str:
    """Get current weather for a location."""
    try:
        lat, lon = get_coordinates(location)
        if lat is None:
            return f"Could not find location: {location}"

        response = requests.get(
            WEATHER_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "temperature_unit": "fahrenheit"
            },
            timeout=10
        )
        data = response.json()

        if "current_weather" in data:
            weather = data["current_weather"]
            temp = weather.get("temperature", "?")
            wind = weather.get("windspeed", "?")
            code = weather.get("weathercode", 0)

            # Weather code descriptions
            conditions = {
                0: "clear sky",
                1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                45: "foggy", 48: "depositing rime fog",
                51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
                61: "slight rain", 63: "moderate rain", 65: "heavy rain",
                71: "slight snow", 73: "moderate snow", 75: "heavy snow",
                80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
                95: "thunderstorm"
            }
            condition = conditions.get(code, "unknown conditions")

            return f"Currently {temp}°F with {condition}. Wind: {wind} mph."

        return "Could not get weather data."

    except Exception as e:
        return f"Weather error: {str(e)}"
