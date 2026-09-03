"""
Geospatial Layer definitions and renderers for TerraSpectra GIS.
Produces standardized layer payloads for WebGL, Deck.gl, and Leaflet map clients.
"""

from typing import Dict, Any, List
import json
from pathlib import Path

GEOJSON_DIR = Path(__file__).resolve().parent.parent / "geojson"


def get_farm_boundary_layer() -> Dict[str, Any]:
    """
    Returns the farm perimeter and management parcel layer payload.
    """
    path = GEOJSON_DIR / "farm_boundary.geojson"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            geojson = json.load(f)
    else:
        geojson = {"type": "FeatureCollection", "features": []}

    return {
        "layer_id": "farm-boundary-layer",
        "name": "Wadgaon 1,000-Acre Farm Boundary",
        "type": "vector",
        "crs": "EPSG:4326",
        "visible": True,
        "style": {
            "stroke": "#163a24",
            "stroke_width": 3,
            "fill": "#2d6a4f",
            "fill_opacity": 0.05,
        },
        "data": geojson,
    }


def get_disease_hotspot_layer() -> Dict[str, Any]:
    """
    Returns the 5.2-acre fungal blight early outbreak hotspot layer payload.
    """
    path = GEOJSON_DIR / "disease_hotspots.geojson"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            geojson = json.load(f)
    else:
        geojson = {"type": "FeatureCollection", "features": []}

    return {
        "layer_id": "disease-hotspot-layer",
        "name": "5.2-Acre Fungal Blight Early Hotspot (Parcel C)",
        "type": "vector",
        "crs": "EPSG:4326",
        "visible": True,
        "alert_level": "CRITICAL",
        "lead_time_days": 21,
        "style": {
            "stroke": "#dc2626",
            "stroke_width": 3.5,
            "fill": "#ef4444",
            "fill_opacity": 0.45,
            "dash_array": [6, 4],
        },
        "data": geojson,
    }
