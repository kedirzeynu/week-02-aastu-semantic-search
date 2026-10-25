# api_demo.py - calling an API safely (slides 9 and 10).
import json
import os

import pandas as pd
import requests

STATUS_MESSAGES = {
    401: "Bad or missing API key.",
    404: "Not found - check the URL or the parameters.",
    429: "Too many requests - slow down and try again later.",
}


class APIError(Exception):
    """Raised when an API call fails for any reason."""


def get_api_key(name="API_KEY"):
    """Read a secret from an environment variable (never hard-code keys)."""
    key = os.environ.get(name)
    if not key:
        raise APIError(f"Environment variable {name} is not set.")
    return key


def build_headers(key):
    """Headers for APIs that use a Bearer token (slide 10)."""
    return {"Authorization": f"Bearer {key}"}


def get_json(url, params=None, headers=None, timeout=10):
    """GET a URL and return the JSON as a Python dict/list, or raise APIError."""
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
    except requests.exceptions.RequestException as error:
        raise APIError(f"Network problem: {error}") from error

    if response.status_code == 200:
        return response.json()
    message = STATUS_MESSAGES.get(response.status_code, "Unexpected status code.")
    raise APIError(f"HTTP {response.status_code}: {message}")


def fetch_addis_temperature():
    """Current temperature in Addis Ababa (free API, no key needed)."""
    data = get_json(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": 9.03, "longitude": 38.74, "current": "temperature_2m"},
    )
    return data["current"]["temperature_2m"]


def offline_json_demo():
    """JSON text -> dict -> DataFrame, with no internet needed."""
    text = '''
    {"city": "Addis Ababa",
     "main": {"temp": 18.5, "humidity": 60},
     "forecast": [{"day": "Mon", "temp": 19}, {"day": "Tue", "temp": 21}]}
    '''
    data = json.loads(text)
    print("type:", type(data).__name__)
    print("temp:", data["main"]["temp"])
    print("missing key, safe way:", data.get("wind", "no wind data"))
    print(pd.DataFrame(data["forecast"]))
    print(json.dumps(data["main"]))


if __name__ == "__main__":
    offline_json_demo()
    try:
        print("Addis temperature now:", fetch_addis_temperature(), "C")
    except APIError as error:
        print("API call failed:", error)