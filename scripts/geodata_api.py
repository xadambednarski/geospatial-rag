import requests
from geopy.distance import geodesic


NOMINATIM_API = (
    "https://nominatim.openstreetmap.org/search?q={query}&format=json&addressdetails=1&limit=1"
)
NOMINATIM_HEADERS = {"User-Agent": "GeoSpatialLLM/1.0"}


def reverse_geocode(address):
    url = NOMINATIM_API.format(query=address)
    response = requests.get(url, headers=NOMINATIM_HEADERS)
    response.raise_for_status()
    return response.json()


def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers