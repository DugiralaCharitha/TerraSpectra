"""
Unit tests for TerraSpectra GIS & Visualization Module.
Verifies geodetic area calculations, GeoJSON integrity, GIS validation, and timeline slices.
"""

import unittest
import json
from pathlib import Path

from gis.utilities.coordinates import (
    haversine_distance,
    calculate_polygon_acres,
    calculate_bounding_box,
    wgs84_to_web_mercator,
    web_mercator_to_wgs84,
)
from gis.utilities.validation import run_full_gis_validation, validate_geojson_file
from gis.utilities.colormaps import interpolate_color, get_hex_color, get_severity_label
from gis.overlays.timeline import get_timeline_metadata, get_timeline_slice
from gis.overlays.spectral_indices import get_benchmark_chemical_anomalies


class TestTerraSpectraGIS(unittest.TestCase):

    def setUp(self):
        self.gis_dir = Path(__file__).resolve().parent
        self.geojson_dir = self.gis_dir / "geojson"

    def test_haversine_known_distance(self):
        # 1 deg latitude near equator ~ 111,195m
        dist = haversine_distance((0.0, 0.0), (1.0, 0.0))
        self.assertAlmostEqual(dist, 111195.0, delta=1000.0)

    def test_farm_boundary_acreage(self):
        farm_path = self.geojson_dir / "farm_boundary.geojson"
        self.assertTrue(farm_path.exists())

        with open(farm_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        outer_feat = next(f for f in data["features"] if f["id"] == "farm_outer_boundary")
        ring = outer_feat["geometry"]["coordinates"][0]
        pts = [(p[1], p[0]) for p in ring]
        acres = calculate_polygon_acres(pts)

        # Must be approximately 1,000 acres (tolerance: within 1%)
        self.assertAlmostEqual(acres, 1000.0, delta=10.0)

    def test_disease_hotspot_acreage(self):
        hotspot_path = self.geojson_dir / "disease_hotspots.geojson"
        self.assertTrue(hotspot_path.exists())

        with open(hotspot_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        hotspot_feat = next(f for f in data["features"] if f["id"] == "hotspot_primary_5acre")
        ring = hotspot_feat["geometry"]["coordinates"][0]
        pts = [(p[1], p[0]) for p in ring]
        acres = calculate_polygon_acres(pts)

        # Must be approximately 5.2 acres (tolerance: within 0.2 acres)
        self.assertAlmostEqual(acres, 5.2, delta=0.2)

    def test_full_gis_validation_pass(self):
        report = run_full_gis_validation()
        self.assertEqual(report["overall_status"], "PASS")
        self.assertTrue(report["spatial_containment_verified"])
        self.assertLessEqual(report["max_coordinate_offset_meters"], 0.1)

    def test_web_mercator_roundtrip(self):
        lat, lon = 20.7505, 76.6061
        x, y = wgs84_to_web_mercator(lat, lon)
        r_lat, r_lon = web_mercator_to_wgs84(x, y)
        self.assertAlmostEqual(lat, r_lat, places=4)
        self.assertAlmostEqual(lon, r_lon, places=4)

    def test_colormaps(self):
        color_green = interpolate_color(0.0)
        color_red = interpolate_color(1.0)
        # Green has high G, Red has high R
        self.assertGreater(color_green[1], color_green[0])
        self.assertGreater(color_red[0], color_red[1])

        hex_str = get_hex_color(0.5)
        self.assertTrue(hex_str.startswith("#"))
        self.assertEqual(len(hex_str), 7)

    def test_timeline_slices(self):
        steps = get_timeline_metadata()
        self.assertEqual(len(steps), 6)

        slice_0 = get_timeline_slice(0)
        self.assertIn("grid_cells", slice_0)
        self.assertGreaterEqual(len(slice_0["grid_cells"]), 500)
        self.assertIn("acreage_breakdown", slice_0)
        self.assertAlmostEqual(slice_0["total_farm_acres"], 1000.14, places=1)

    def test_chemical_anomalies(self):
        anomalies = get_benchmark_chemical_anomalies()
        self.assertIn("chlorophyll_reflection_dip", anomalies)
        self.assertIn("photochemical_reflectance_index", anomalies)
        self.assertIn("canopy_water_index", anomalies)
        self.assertEqual(anomalies["outbreak_lead_time_days"], 21)


if __name__ == "__main__":
    unittest.main()
