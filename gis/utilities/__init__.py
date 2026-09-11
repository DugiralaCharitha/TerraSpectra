"""GIS utilities package."""
from .coordinates import (
    haversine_distance,
    calculate_polygon_area_sq_meters,
    calculate_polygon_acres,
    calculate_bounding_box,
    wgs84_to_web_mercator,
    web_mercator_to_wgs84,
)
from .colormaps import interpolate_color, get_hex_color, get_severity_label
from .validation import validate_geojson_file, run_full_gis_validation

__all__ = [
    "haversine_distance",
    "calculate_polygon_area_sq_meters",
    "calculate_polygon_acres",
    "calculate_bounding_box",
    "wgs84_to_web_mercator",
    "web_mercator_to_wgs84",
    "interpolate_color",
    "get_hex_color",
    "get_severity_label",
    "validate_geojson_file",
    "run_full_gis_validation",
]
