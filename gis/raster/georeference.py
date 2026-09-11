"""
Raster Georeferencing and Grid Generation for TerraSpectra GIS.
Transforms hyperspectral raster predictions into georeferenced geographic cells
with WGS84 coordinates, NDVI, chlorophyll stress, and disease severity.
"""

from typing import List, Dict, Any, Tuple
import math
from ..utilities.colormaps import interpolate_color, get_hex_color, get_severity_label
from ..utilities.coordinates import calculate_polygon_acres


def generate_farm_grid_cells(
    bbox: Dict[str, float],
    rows: int = 24,
    cols: int = 24,
    hotspot_bbox: Dict[str, float] = None,
    hotspot_severity_multiplier: float = 1.0,
    base_noise_seed: float = 42.0
) -> List[Dict[str, Any]]:
    """
    Generate an array of georeferenced grid cells covering the farm bounding box.
    Each cell contains GPS polygon bounds, center (lat, lon), NDVI, disease severity,
    chemical anomaly indicators, and color attributes.
    """
    min_lat, max_lat = bbox["min_lat"], bbox["max_lat"]
    min_lon, max_lon = bbox["min_lon"], bbox["max_lon"]

    lat_step = (max_lat - min_lat) / rows
    lon_step = (max_lon - min_lon) / cols

    cells = []
    cell_idx = 0

    h_min_lat = hotspot_bbox["min_lat"] if hotspot_bbox else 20.75355
    h_max_lat = hotspot_bbox["max_lat"] if hotspot_bbox else 20.75485
    h_min_lon = hotspot_bbox["min_lon"] if hotspot_bbox else 76.61110
    h_max_lon = hotspot_bbox["max_lon"] if hotspot_bbox else 76.61250

    h_center_lat = (h_min_lat + h_max_lat) / 2.0
    h_center_lon = (h_min_lon + h_max_lon) / 2.0

    for r in range(rows):
        c_lat_min = min_lat + r * lat_step
        c_lat_max = c_lat_min + lat_step
        cell_lat = (c_lat_min + c_lat_max) / 2.0

        for c in range(cols):
            c_lon_min = min_lon + c * lon_step
            c_lon_max = c_lon_min + lon_step
            cell_lon = (c_lon_min + c_lon_max) / 2.0

            # Distance to the 5.2-acre fungal outbreak epicenter
            d_lat = (cell_lat - h_center_lat) / (lat_step * 2.5)
            d_lon = (cell_lon - h_center_lon) / (lon_step * 2.5)
            dist_sq = d_lat * d_lat + d_lon * d_lon

            # Base healthy farm natural variation
            pseudo_rand = math.sin(r * 12.9898 + c * 78.233 + base_noise_seed) * 43758.5453
            pseudo_rand = pseudo_rand - math.floor(pseudo_rand)
            baseline_stress = 0.08 + pseudo_rand * 0.12

            # Hotspot Gaussian dispersion
            hotspot_factor = math.exp(-dist_sq * 1.8) * hotspot_severity_multiplier
            raw_severity = baseline_stress + (0.85 * hotspot_factor)
            severity = max(0.04, min(0.98, raw_severity))

            # NDVI inversely correlates with stress
            ndvi = max(0.22, min(0.88, 0.82 - (severity * 0.55)))

            # 3D elevation extrusion for Deck.gl / visualization (meters or visual z-height)
            elevation = round(severity * 150.0, 1)

            rgba = interpolate_color(severity)
            hex_color = get_hex_color(severity)
            label = get_severity_label(severity)

            is_in_outbreak_zone = dist_sq < 1.4 and hotspot_severity_multiplier > 0.4

            polygon = [
                [c_lon_min, c_lat_min],
                [c_lon_min, c_lat_max],
                [c_lon_max, c_lat_max],
                [c_lon_max, c_lat_min],
                [c_lon_min, c_lat_min],
            ]

            cells.append({
                "cell_id": f"cell_{r}_{c}",
                "row": r,
                "col": c,
                "center": [cell_lat, cell_lon],
                "bounds": {
                    "min_lat": c_lat_min,
                    "max_lat": c_lat_max,
                    "min_lon": c_lon_min,
                    "max_lon": c_lon_max,
                },
                "polygon": polygon,
                "ndvi": round(ndvi, 3),
                "severity": round(severity, 3),
                "severity_label": label,
                "elevation": elevation,
                "is_hotspot": is_in_outbreak_zone,
                "rgba": list(rgba),
                "hex_color": hex_color,
                "fill_opacity": 0.55 if is_in_outbreak_zone else 0.35,
            })
            cell_idx += 1

    return cells
