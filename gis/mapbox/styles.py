"""
Mapbox and satellite basemap style presets for TerraSpectra GIS.
Supports high-resolution satellite topography, terrain DEM meshes, and hybrid vector overlays.
"""

from typing import Dict, Any


MAP_STYLES = {
    "satellite_esri": {
        "name": "Esri World Imagery (Satellite Topography)",
        "type": "raster",
        "tiles": [
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ],
        "attribution": "&copy; Esri, DigitalGlobe, GeoEye, Earthstar Geographics",
        "max_zoom": 19,
        "is_satellite": True,
    },
    "satellite_mapbox": {
        "name": "Mapbox Satellite Streets",
        "style_url": "mapbox://styles/mapbox/satellite-streets-v12",
        "dem_source": "mapbox://mapbox.mapbox-terrain-dem-v1",
        "max_zoom": 22,
        "is_satellite": True,
    },
    "osm_streets": {
        "name": "OpenStreetMap Standard",
        "type": "raster",
        "tiles": [
            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        ],
        "attribution": "&copy; OpenStreetMap contributors",
        "max_zoom": 19,
        "is_satellite": False,
    },
    "carto_dark": {
        "name": "CartoDB Dark Matter (High Contrast GIS)",
        "type": "raster",
        "tiles": [
            "https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png"
        ],
        "attribution": "&copy; CartoDB",
        "max_zoom": 19,
        "is_satellite": False,
    },
}


def get_default_satellite_style() -> Dict[str, Any]:
    """
    Returns default high-resolution satellite imagery configuration without requiring an API key.
    """
    return MAP_STYLES["satellite_esri"]
