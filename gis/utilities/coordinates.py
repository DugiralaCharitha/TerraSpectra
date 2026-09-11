"""
Geodetic and coordinate utilities for TerraSpectra GIS Module.
Calculates geodesic distances, spherical polygon acreage, bounding boxes, and CRS transformations.
"""

import math
from typing import List, Tuple, Dict, Any

EARTH_RADIUS_METERS = 6371000.0  # WGS84 mean radius
METERS_SQ_PER_ACRE = 4046.8564224
METERS_SQ_PER_HECTARE = 10000.0


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Compute Haversine distance between two (lat, lon) pairs in meters.
    """
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def calculate_polygon_area_sq_meters(coordinates: List[Tuple[float, float]]) -> float:
    """
    Calculate spherical excess polygon area in square meters for WGS84 coordinates.
    Input coordinates are expected as [(lat, lon), ...] in counter-clockwise or clockwise order.
    The polygon may be closed or open (first and last point identical).
    """
    if len(coordinates) < 3:
        return 0.0

    # Ensure list of points without duplicate closing point
    points = list(coordinates)
    if points[0] == points[-1] and len(points) > 3:
        points = points[:-1]

    if len(points) < 3:
        return 0.0

    # Spherical excess using Girard's theorem / trapezoidal spherical method
    total_rad_area = 0.0
    num_points = len(points)

    for i in range(num_points):
        lat1, lon1 = math.radians(points[i][0]), math.radians(points[i][1])
        lat2, lon2 = math.radians(points[(i + 1) % num_points][0]), math.radians(points[(i + 1) % num_points][1])
        total_rad_area += (lon2 - lon1) * (2.0 + math.sin(lat1) + math.sin(lat2))

    total_rad_area = abs(total_rad_area * 0.5)
    return total_rad_area * (EARTH_RADIUS_METERS ** 2)


def calculate_polygon_acres(coordinates: List[Tuple[float, float]]) -> float:
    """
    Calculate polygon area directly in acres.
    """
    sq_meters = calculate_polygon_area_sq_meters(coordinates)
    return sq_meters / METERS_SQ_PER_ACRE


def calculate_bounding_box(coordinates: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Compute [min_lat, min_lon, max_lat, max_lon] for a polygon.
    """
    lats = [c[0] for c in coordinates]
    lons = [c[1] for c in coordinates]
    return {
        "min_lat": min(lats),
        "min_lon": min(lons),
        "max_lat": max(lats),
        "max_lon": max(lons),
        "center_lat": (min(lats) + max(lats)) / 2.0,
        "center_lon": (min(lons) + max(lons)) / 2.0,
    }


def wgs84_to_web_mercator(lat: float, lon: float) -> Tuple[float, float]:
    """
    Project EPSG:4326 (WGS84 Lat/Lon) to EPSG:3857 (Web Mercator meters).
    """
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return (x, y)


def web_mercator_to_wgs84(x: float, y: float) -> Tuple[float, float]:
    """
    Inverse project EPSG:3857 (Web Mercator meters) to EPSG:4326 (WGS84 Lat/Lon).
    """
    lon = (x / 20037508.34) * 180.0
    lat = (y / 20037508.34) * 180.0
    lat = 180.0 / math.pi * (2.0 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return (lat, lon)
