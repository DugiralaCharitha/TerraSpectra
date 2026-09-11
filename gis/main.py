"""
TerraSpectra GIS & Visualization Microservice.
Runs on port 8080 (as specified in docker-compose.yml).
Serves georeferenced farm boundaries, disease hotspot zones,
multi-temporal timeline slices, chemical anomaly analytics, and GIS validation reports.
"""

import os
import json
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .utilities.validation import run_full_gis_validation
from .overlays.timeline import get_timeline_metadata, get_timeline_slice
from .overlays.spectral_indices import get_benchmark_chemical_anomalies
from .deckgl.layer_factory import create_deckgl_farm_viewstate, create_deckgl_gridcell_layer_config
from .mapbox.styles import MAP_STYLES

BASE_DIR = Path(__file__).resolve().parent
GEOJSON_DIR = BASE_DIR / "geojson"


def load_geojson(filename: str):
    path = GEOJSON_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"type": "FeatureCollection", "features": []}


if HAS_FASTAPI:
    app = FastAPI(
        title="TerraSpectra GIS & Visualization API",
        version="1.0.0",
        description="Geospatial engine for 1,000-acre farm monitoring and 3D hyperspectral crop disease forecasting."
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {
            "service": "TerraSpectra GIS & Visualization Engine",
            "status": "online",
            "port": 8080,
            "crs": "EPSG:4326 (WGS84)",
            "farm_scale": "1,000 Acres (Wadgaon, Maharashtra)",
            "outbreak_lead_time": "3 Weeks Early (Hyperspectral Red-Edge Shift)",
        }

    @app.get("/gis/farm-boundary")
    def get_farm_boundary():
        return load_geojson("farm_boundary.geojson")

    @app.get("/gis/disease-hotspots")
    def get_disease_hotspots():
        return load_geojson("disease_hotspots.geojson")

    @app.get("/gis/timeline")
    def get_timeline():
        return {
            "steps": get_timeline_metadata(),
            "total_steps": len(get_timeline_metadata()),
            "current_step_index": 0,
        }

    @app.get("/gis/grid-cells")
    def get_grid_cells(week: int = Query(0, ge=-2, le=3)):
        return get_timeline_slice(week)

    @app.get("/gis/analytics")
    def get_analytics():
        chemical = get_benchmark_chemical_anomalies()
        current_slice = get_timeline_slice(0)
        return {
            "total_farm_acres": 1000.14,
            "target_outbreak_zone_acres": 5.2,
            "lead_time_days": 21,
            "affected_parcel": "Parcel C (Northeast Field)",
            "acreage_breakdown": current_slice["acreage_breakdown"],
            "chemical_anomalies": chemical,
        }

    @app.get("/gis/validation")
    def get_validation():
        return run_full_gis_validation()

    @app.get("/gis/deckgl-config")
    def get_deckgl_config():
        view_state = create_deckgl_farm_viewstate()
        slice_data = get_timeline_slice(0)
        layer_cfg = create_deckgl_gridcell_layer_config(slice_data["grid_cells"])
        return {
            "view_state": view_state,
            "layer_config": layer_cfg,
            "basemap_styles": MAP_STYLES,
        }

else:
    app = None
