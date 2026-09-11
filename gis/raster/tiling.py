"""
Geospatial tiling and coordinate alignment engine for hyperspectral imagery rasters.
Handles spatial partitioning, tiling matrix blocks, and spatial index lookups.
"""

from typing import Dict, Any, List, Tuple
from .georeference import generate_farm_grid_cells


class FarmRasterTilingEngine:
    """
    Slices a 1,000-acre farm area into multi-resolution geospatial tiles.
    """

    def __init__(self, farm_bbox: Dict[str, float] = None):
        self.bbox = farm_bbox or {
            "min_lat": 20.74145,
            "max_lat": 20.75955,
            "min_lon": 76.59643,
            "max_lon": 76.61577,
        }

    def get_tile_grid(self, zoom_level: int = 15, week_progression: float = 1.0) -> Dict[str, Any]:
        """
        Generate tiled raster representation suitable for Mapbox/Deck.gl/Leaflet rendering.
        """
        grid_resolution = 24 if zoom_level <= 15 else 36
        cells = generate_farm_grid_cells(
            bbox=self.bbox,
            rows=grid_resolution,
            cols=grid_resolution,
            hotspot_severity_multiplier=week_progression,
        )

        # Convert to GeoJSON FeatureCollection for standard GIS interoperability
        features = []
        for cell in cells:
            features.append({
                "type": "Feature",
                "id": cell["cell_id"],
                "properties": {
                    "ndvi": cell["ndvi"],
                    "severity": cell["severity"],
                    "severity_label": cell["severity_label"],
                    "elevation": cell["elevation"],
                    "is_hotspot": cell["is_hotspot"],
                    "hex_color": cell["hex_color"],
                    "fill_opacity": cell["fill_opacity"],
                    "center": cell["center"],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [cell["polygon"]],
                },
            })

        return {
            "type": "FeatureCollection",
            "name": f"TerraSpectra_Raster_Grid_z{zoom_level}",
            "grid_dimensions": {"rows": grid_resolution, "cols": grid_resolution},
            "cell_count": len(cells),
            "features": features,
            "raw_cells": cells,
        }
