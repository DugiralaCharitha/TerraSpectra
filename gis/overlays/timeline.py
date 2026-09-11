"""
Timeline Multi-Temporal Progression Engine for TerraSpectra GIS.
Generates multi-temporal spectral states for historical analysis and disease forecasting:
- Week -2: Historical Baseline (Healthy)
- Week -1: Sub-visual Inception
- Week 0: Current Day (Sub-visual Chemical Anomaly detected by 3D-CNN + ViT, 3 weeks early)
- Week +1: Forecasted Early Cellular Degradation
- Week +2: Forecasted Visible Chlorosis
- Week +3: Forecasted Severe Foliar Blight Outbreak (if untreated)
"""

from typing import List, Dict, Any
from .spectral_indices import get_chemical_anomaly_profile
from ..raster.georeference import generate_farm_grid_cells


TIMELINE_STEPS = [
    {
        "week_index": -2,
        "step_id": "week_minus_2",
        "label": "Week -2 (Baseline)",
        "relative_days": -14,
        "date_offset": "14 Days Ago",
        "progression_factor": 0.05,
        "affected_acres": 0.0,
        "chlorophyll_dip_pct": "0.0%",
        "pri_value": 0.041,
        "stage": "Healthy Vegetative Baseline",
        "symptom_visibility": "None (Canopy Completely Healthy)",
        "status_color": "#22c55e",
        "description": "Normal chlorophyll absorption across all 200+ bands. No fungal presence detected.",
    },
    {
        "week_index": -1,
        "step_id": "week_minus_1",
        "label": "Week -1 (Incubation)",
        "relative_days": -7,
        "date_offset": "7 Days Ago",
        "progression_factor": 0.22,
        "affected_acres": 1.2,
        "chlorophyll_dip_pct": "-6.8%",
        "pri_value": 0.015,
        "stage": "Microscopic Spore Germination",
        "symptom_visibility": "Sub-Visual (Undetectable by RGB or Human Eye)",
        "status_color": "#84cc16",
        "description": "Initial spore adhesion and hyphal infiltration. Photochemical reflectance begins minor drift.",
    },
    {
        "week_index": 0,
        "step_id": "week_0_today",
        "label": "Week 0 (Today - 3 Wks Early)",
        "relative_days": 0,
        "date_offset": "Today (Detection Date)",
        "progression_factor": 0.75,
        "affected_acres": 5.2,
        "chlorophyll_dip_pct": "-28.4%",
        "pri_value": -0.142,
        "stage": "Sub-Visual Chlorophyll Dip Detected (Targeted Window)",
        "symptom_visibility": "Sub-Visual (Leaves still appear green to naked eye)",
        "status_color": "#ef4444",
        "description": "3D-CNN & ViT flag 5.2-acre zone in Parcel C. Preventative fungicide application window is OPEN.",
    },
    {
        "week_index": 1,
        "step_id": "week_plus_1",
        "label": "Week +1 (Forecast)",
        "relative_days": 7,
        "date_offset": "+7 Days Forecast",
        "progression_factor": 0.90,
        "affected_acres": 9.4,
        "chlorophyll_dip_pct": "-42.1%",
        "pri_value": -0.210,
        "stage": "Cellular Membrane Breakdown",
        "symptom_visibility": "Faint Underleaf Micro-Lesions",
        "status_color": "#f97316",
        "description": "Pathogen consumes cellular starch. Water absorption band starts severe attenuation.",
    },
    {
        "week_index": 2,
        "step_id": "week_plus_2",
        "label": "Week +2 (Forecast)",
        "relative_days": 14,
        "date_offset": "+14 Days Forecast",
        "progression_factor": 1.15,
        "affected_acres": 16.8,
        "chlorophyll_dip_pct": "-61.5%",
        "pri_value": -0.325,
        "stage": "Foliar Chlorosis & Spore Dispersion",
        "symptom_visibility": "Visible Yellowing on Upper Canopy",
        "status_color": "#dc2626",
        "description": "Macroscopic symptoms visible. Traditional RGB satellite sensors would first detect stress here (2 weeks late).",
    },
    {
        "week_index": 3,
        "step_id": "week_plus_3",
        "label": "Week +3 (Outbreak)",
        "relative_days": 21,
        "date_offset": "+21 Days Forecast",
        "progression_factor": 1.45,
        "affected_acres": 28.5,
        "chlorophyll_dip_pct": "-78.9%",
        "pri_value": -0.440,
        "stage": "Full Necrotic Blight Outbreak",
        "symptom_visibility": "Severe Brown Lesions & Crop Loss",
        "status_color": "#b91c1c",
        "description": "Catastrophic crop damage if untreated. 28+ acres infested across Parcel C and surrounding fields.",
    },
]


def get_timeline_metadata() -> List[Dict[str, Any]]:
    """
    Returns the array of timeline steps and their metadata for UI sliders.
    """
    return TIMELINE_STEPS


def get_timeline_slice(week_index: int, farm_bbox: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Get the specific geospatial grid and chemical metrics for a requested timeline step.
    """
    # Find matching step, defaulting to week 0 (today)
    step = next((s for s in TIMELINE_STEPS if s["week_index"] == week_index), TIMELINE_STEPS[2])

    bbox = farm_bbox or {
        "min_lat": 20.74145,
        "max_lat": 20.75955,
        "min_lon": 76.59643,
        "max_lon": 76.61577,
    }

    cells = generate_farm_grid_cells(
        bbox=bbox,
        rows=24,
        cols=24,
        hotspot_severity_multiplier=step["progression_factor"],
        base_noise_seed=42.0 + float(week_index) * 1.5,
    )

    # Acreage risk breakdown for this timeline slice
    high_risk_cells = [c for c in cells if c["severity"] >= 0.70]
    moderate_risk_cells = [c for c in cells if 0.40 <= c["severity"] < 0.70]
    healthy_cells = [c for c in cells if c["severity"] < 0.40]

    total_cells = len(cells)
    total_farm_acres = 1000.14
    acres_per_cell = total_farm_acres / float(total_cells)

    high_risk_acres = round(len(high_risk_cells) * acres_per_cell, 1)
    moderate_risk_acres = round(len(moderate_risk_cells) * acres_per_cell, 1)
    healthy_acres = round(total_farm_acres - high_risk_acres - moderate_risk_acres, 1)

    return {
        "timeline_step": step,
        "total_farm_acres": total_farm_acres,
        "acreage_breakdown": {
            "high_risk_outbreak_acres": high_risk_acres,
            "moderate_stress_acres": moderate_risk_acres,
            "healthy_canopy_acres": healthy_acres,
        },
        "chemical_anomalies": {
            "chlorophyll_absorption_dip": step["chlorophyll_dip_pct"],
            "pri_index": step["pri_value"],
            "target_lead_time": f"{21 - max(0, step['relative_days'])} days prior to visual symptoms",
        },
        "grid_cells": cells,
    }
