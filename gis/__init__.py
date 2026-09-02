"""
TerraSpectra GIS Subsystem.
Core geospatial intelligence engine for 1,000-acre precision agriculture
and hyperspectral early crop disease forecasting.
"""

from .utilities.coordinates import calculate_polygon_acres, haversine_distance
from .utilities.validation import run_full_gis_validation
from .overlays.spectral_indices import get_benchmark_chemical_anomalies
from .overlays.timeline import get_timeline_metadata, get_timeline_slice

__version__ = "1.0.0"
__all__ = [
    "calculate_polygon_acres",
    "haversine_distance",
    "run_full_gis_validation",
    "get_benchmark_chemical_anomalies",
    "get_timeline_metadata",
    "get_timeline_slice",
]
