"""
Leaflet and React-Leaflet GIS layer adapter.
Produces Leaflet GeoJSON layer styling functions, popups, and layer configurations.
"""

from typing import Dict, Any


def get_leaflet_layer_manifest() -> Dict[str, Any]:
    """
    Returns layer manifest for React-Leaflet integration.
    """
    return {
        "base_layers": [
            {
                "id": "esri-satellite",
                "name": "High-Res Satellite (Esri)",
                "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "attribution": "&copy; Esri, DigitalGlobe, Earthstar Geographics",
                "is_default": True,
            },
            {
                "id": "osm-streets",
                "name": "Street Map (OpenStreetMap)",
                "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                "attribution": "&copy; OpenStreetMap contributors",
                "is_default": False,
            },
        ],
        "overlay_layers": [
            {
                "id": "spectral-grid",
                "name": "3D Hyperspectral Grid Cells",
                "is_default": True,
            },
            {
                "id": "farm-parcels",
                "name": "Agricultural Parcels (A through F)",
                "is_default": True,
            },
            {
                "id": "disease-hotspot",
                "name": "5.2-Acre Fungal Blight Hotspot",
                "is_default": True,
            },
        ],
    }
