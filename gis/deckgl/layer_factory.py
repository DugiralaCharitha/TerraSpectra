"""
Deck.gl 3D Layer Factory and Configuration Generator for TerraSpectra.
Constructs JSON-serializable layer configurations for Deck.gl / pydeck / WebGL consumers.
"""

from typing import Dict, Any, List


def create_deckgl_farm_viewstate(
    center: List[float] = None,
    zoom: float = 14.8,
    pitch: float = 45.0,
    bearing: float = -15.0
) -> Dict[str, Any]:
    """
    Returns initial view state for 3D Deck.gl rendering with realistic tilt and bearing.
    """
    center = center or [76.6061, 20.7505]
    return {
        "longitude": center[0],
        "latitude": center[1],
        "zoom": zoom,
        "pitch": pitch,
        "bearing": bearing,
        "maxZoom": 20,
        "minZoom": 11,
    }


def create_deckgl_gridcell_layer_config(
    data: List[Dict[str, Any]],
    layer_id: str = "farm-disease-grid",
    extruded: bool = True
) -> Dict[str, Any]:
    """
    Returns declarative configuration for Deck.gl GridCellLayer / PolygonLayer.
    """
    return {
        "id": layer_id,
        "type": "PolygonLayer",
        "data": data,
        "pickable": True,
        "stroked": True,
        "filled": True,
        "extruded": extruded,
        "wireframe": False,
        "lineWidthMinPixels": 1,
        "getPolygon": "@@=polygon",
        "getElevation": "@@=elevation",
        "getFillColor": "@@=rgba",
        "getLineColor": [255, 255, 255, 60],
        "elevationScale": 1.5,
        "material": {
            "ambient": 0.35,
            "diffuse": 0.6,
            "shininess": 32,
            "specularColor": [50, 50, 50],
        },
    }


def create_deckgl_hotspot_boundary_config(
    hotspot_geojson: Dict[str, Any],
    layer_id: str = "hotspot-boundary-layer"
) -> Dict[str, Any]:
    """
    Returns configuration for rendering the 5.2-acre fungal outbreak boundary.
    """
    return {
        "id": layer_id,
        "type": "GeoJsonLayer",
        "data": hotspot_geojson,
        "pickable": True,
        "stroked": True,
        "filled": True,
        "extruded": False,
        "lineWidthMinPixels": 3,
        "getFillColor": [239, 68, 68, 80],
        "getLineColor": [220, 38, 38, 255],
    }
