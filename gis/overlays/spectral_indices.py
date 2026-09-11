"""
Hyperspectral Chemical & Spectral Indices for TerraSpectra.
Computes and models subtle chemical changes in crop chlorophyll reflection,
photochemical reflectance, and water indices for early fungal disease detection.
"""

from typing import Dict, Any, List


def calculate_ndre(nir: float, red_edge: float) -> float:
    """
    Normalized Difference Red Edge Index (NDRE).
    Sensitive to chlorophyll content in dense canopies.
    Formula: (NIR - RedEdge) / (NIR + RedEdge)
    Bands: ~790nm (NIR) and ~705nm (Red Edge)
    """
    denom = nir + red_edge
    if denom == 0.0:
        return 0.0
    return (nir - red_edge) / denom


def calculate_pri(b531: float, b570: float) -> float:
    """
    Photochemical Reflectance Index (PRI).
    Indicates photosynthetic light use efficiency and xanthophyll cycle activity.
    Formula: (R531 - R570) / (R531 + R570)
    Fungal infection causes an early dip in PRI before visual symptoms.
    """
    denom = b531 + b570
    if denom == 0.0:
        return 0.0
    return (b531 - b570) / denom


def calculate_cwi(b900: float, b970: float) -> float:
    """
    Canopy Water Index (CWI).
    Detects subtle cellular water loss induced by fungal pathogen colonization.
    Formula: R900 / R970
    """
    if b970 == 0.0:
        return 1.0
    return b900 / b970


def get_chemical_anomaly_profile(severity: float) -> Dict[str, Any]:
    """
    Returns the estimated biochemical anomaly metrics based on predicted disease severity [0.0 - 1.0].
    """
    # Healthy baseline:
    # Chlorophyll reflection at 705nm: ~0.42
    # PRI: +0.02 to +0.04
    # CWI: 1.15
    # Carotenoid ratio: normal (1.0x)

    chlorophyll_dip_pct = -1.0 * round(severity * 32.0, 1)  # Up to -32% dip
    pri_val = round(0.035 - (severity * 0.18), 3)           # Shifts negative as stress rises
    cwi_drop_pct = -1.0 * round(severity * 21.5, 1)         # Cellular moisture drop
    carotenoid_increase_pct = round(severity * 38.0, 1)     # Carotenoids increase under fungal attack

    return {
        "chlorophyll_dip_percent": chlorophyll_dip_pct,
        "photochemical_reflectance_index": pri_val,
        "canopy_water_index_drop_percent": cwi_drop_pct,
        "carotenoid_accumulation_percent": carotenoid_increase_pct,
        "pathogen_risk_status": "Severe Outbreak" if severity > 0.75 else ("Sub-Visual Stress" if severity > 0.4 else "Healthy"),
    }


def get_benchmark_chemical_anomalies() -> Dict[str, Any]:
    """
    Returns the global farm chemical anomaly summary for the analytics panel.
    """
    return {
        "outbreak_lead_time_days": 21,
        "chlorophyll_reflection_dip": {
            "name": "Chlorophyll-a/b Reflection Dip (705nm)",
            "affected_zone_delta": "-28.4%",
            "healthy_control_delta": "+0.8%",
            "status": "Critical Early Anomaly",
            "detection_band": "Band 88 (705.4 nm, Red Edge inflection)",
        },
        "photochemical_reflectance_index": {
            "name": "Photochemical Reflectance Index (PRI)",
            "affected_zone_value": -0.142,
            "healthy_control_value": 0.038,
            "status": "Photosynthetic Inhibition",
            "detection_band": "Band 32 (531.2 nm) vs Band 44 (570.1 nm)",
        },
        "canopy_water_index": {
            "name": "Canopy Water Index (CWI / Water Band)",
            "affected_zone_delta": "-18.8%",
            "healthy_control_delta": "-1.2%",
            "status": "Cellular Desiccation",
            "detection_band": "Band 148 (970.0 nm water absorption band)",
        },
        "carotenoid_accumulation": {
            "name": "Carotenoid-to-Chlorophyll Ratio",
            "affected_zone_delta": "+34.1%",
            "healthy_control_delta": "-0.5%",
            "status": "Oxidative Defense Activation",
            "detection_band": "Band 22 (480.0 nm) / Band 82 (680.0 nm)",
        },
    }
