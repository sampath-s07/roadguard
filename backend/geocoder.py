"""
geocoder.py  –  Reverse geocoding via Nominatim (OpenStreetMap)
No API key needed. Free for reasonable use.
"""

import urllib.request
import urllib.parse
import json
import time

_cache = {}
_last_call = 0
MIN_INTERVAL = 1.0   # Nominatim rate limit: 1 req/sec


def reverse_geocode(lat, lng):
    """
    Returns road/address string for given coordinates.
    Uses Nominatim (OSM). Cached to avoid re-fetching.
    Returns dict: {road, suburb, city, country, display}
    """
    key = f"{round(lat,4)},{round(lng,4)}"
    if key in _cache:
        return _cache[key]

    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    url = (
        "https://nominatim.openstreetmap.org/reverse"
        f"?lat={lat}&lon={lng}&format=json&addressdetails=1"
    )
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "RoadGuard-TrafficSignApp/3.0"
        })
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        addr = data.get("address", {})
        road    = addr.get("road") or addr.get("pedestrian") or ""
        suburb  = addr.get("suburb") or addr.get("neighbourhood") or ""
        city    = addr.get("city") or addr.get("town") or addr.get("village") or ""
        country = addr.get("country") or ""

        parts = [p for p in [road, suburb, city] if p]
        display = ", ".join(parts) if parts else data.get("display_name", "Unknown location")[:60]

        result = {"road": road, "suburb": suburb, "city": city,
                  "country": country, "display": display}
        _cache[key] = result
        _last_call = time.time()
        return result

    except Exception as e:
        return {"road": "", "suburb": "", "city": "", "country": "",
                "display": f"{round(lat,4)}, {round(lng,4)}"}
