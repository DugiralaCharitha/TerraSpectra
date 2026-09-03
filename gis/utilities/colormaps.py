"""
Colormap definitions and interpolation utilities for TerraSpectra GIS visualization.
Supports RGBA, Hex, and Deck.gl [R, G, B, A] output.
"""

from typing import List, Tuple, Dict, Any


# Standard Agricultural Crop Health Diverging Scale
# Green (Optimal) -> Lime (Good) -> Yellow (Moderate Stress) -> Orange (Elevated Stress) -> Red (Severe Outbreak)
CROP_HEALTH_RAMP = [
    (0.00, [34, 197, 94, 220], "#22c55e", "Optimal Canopy"),
    (0.25, [132, 204, 22, 220], "#84cc16", "Normal Vegetative"),
    (0.50, [234, 179, 8, 220], "#eab308", "Sub-Visual Chemical Stress"),
    (0.70, [249, 115, 22, 230], "#f97316", "Elevated Pathogen Risk"),
    (1.00, [239, 68, 68, 240], "#ef4444", "Critical Fungal Outbreak"),
]


def interpolate_color(val: float, ramp=CROP_HEALTH_RAMP) -> Tuple[int, int, int, int]:
    """
    Interpolate RGBA color from normalized value in range [0.0, 1.0].
    """
    val = max(0.0, min(1.0, float(val)))

    for i in range(len(ramp) - 1):
        v1, c1, _, _ = ramp[i]
        v2, c2, _, _ = ramp[i + 1]

        if v1 <= val <= v2:
            t = (val - v1) / (v2 - v1) if v2 > v1 else 0.0
            r = int(c1[0] + t * (c2[0] - c1[0]))
            g = int(c1[1] + t * (c2[1] - c1[1]))
            b = int(c1[2] + t * (c2[2] - c1[2]))
            a = int(c1[3] + t * (c2[3] - c1[3]))
            return (r, g, b, a)

    return tuple(ramp[-1][1])


def get_hex_color(val: float, ramp=CROP_HEALTH_RAMP) -> str:
    """
    Get hex string "#rrggbb" for a value [0.0, 1.0].
    """
    r, g, b, _ = interpolate_color(val, ramp)
    return f"#{r:02x}{g:02x}{b:02x}"


def get_severity_label(val: float) -> str:
    """
    Return descriptive category label for disease severity.
    """
    if val < 0.25:
        return "Optimal (Healthy)"
    elif val < 0.45:
        return "Normal Canopy"
    elif val < 0.65:
        return "Early Chemical Anomaly"
    elif val < 0.80:
        return "Elevated Pathogen Risk"
    else:
        return "Critical Outbreak Hotspot"
