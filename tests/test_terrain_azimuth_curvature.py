from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from backend.terrain_horizon import (
    MOON_REFERENCE_RADIUS_M,
    build_horizon_profile_ray_march,
    geodesic_destination_sphere,
    inverse_south_polar_stereographic,
    projected_axis_bearing_deg,
)


class TerrainAzimuthCurvatureTests(unittest.TestCase):
    def test_projected_positive_y_axis_bearing_is_not_site_longitude(self):
        # At the provisional tile-centre point, east longitude is 222.510447°.
        # The positive projected Y axis points at local bearing 137.489553°.
        bearing = projected_axis_bearing_deg(-11000.0, -12000.0, axis="y")
        self.assertAlmostEqual(bearing, 137.4895529, places=5)
        self.assertNotAlmostEqual(bearing, 222.5104471, places=2)

    def test_synthetic_pillar_recovers_local_bearing_and_spherical_curvature(self):
        # A 5 m synthetic DEM with one 1 km-high pillar on the positive grid-Y
        # bearing. This exercises the full raster -> geodesic -> ENU ray marcher.
        width = height = 801
        pixel = 5.0
        observer_x, observer_y = -11000.0, -12000.0
        transform = Affine(
            pixel, 0.0, observer_x - 400.5 * pixel,
            0.0, -pixel, observer_y + 400.5 * pixel,
        )
        crs = CRS.from_proj4(
            "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 "
            "+R=1737400 +units=m +no_defs +type=crs"
        )
        dem = np.zeros((height, width), dtype=np.float32)

        expected_bearing = projected_axis_bearing_deg(observer_x, observer_y, axis="y")
        lat0, lon0 = inverse_south_polar_stereographic(observer_x, observer_y)
        lat1, lon1 = geodesic_destination_sphere(
            lat0, lon0, np.radians(expected_bearing), 500.0, MOON_REFERENCE_RADIUS_M
        )
        colat = lat1 + 0.5 * np.pi
        rho = 2.0 * MOON_REFERENCE_RADIUS_M * np.tan(0.5 * colat)
        target_x = rho * np.sin(lon1)
        target_y = rho * np.cos(lon1)
        col = int(round((target_x - transform.c) / pixel - 0.5))
        row = int(round((transform.f - target_y) / pixel - 0.5))
        dem[row - 5:row + 6, col - 5:col + 6] = 1000.0

        with tempfile.TemporaryDirectory() as folder:
            dem_path = Path(folder) / "synthetic_pillar.tif"
            with rasterio.open(
                dem_path, "w", driver="GTiff", width=width, height=height,
                count=1, dtype="float32", crs=crs, transform=transform, nodata=-9999.0,
            ) as dst:
                dst.write(dem, 1)

            result = build_horizon_profile_ray_march(
                dem_path,
                observer_x_m=observer_x,
                observer_y_m=observer_y,
                observer_height_m=1.0,
                azimuth_step_deg=0.5,
                ray_step_m=5.0,
                minimum_ray_distance_m=5.0,
                maximum_ray_distance_m=1000.0,
                rays_per_chunk=12,
            )

        azimuth, horizon, *_rest, metadata = result
        peak = int(np.nanargmax(horizon))
        peak_azimuth = float(azimuth[peak])
        angular_error = abs((peak_azimuth - expected_bearing + 180.0) % 360.0 - 180.0)
        self.assertLessEqual(angular_error, 0.51)

        # Verify the reported angle against exact spherical 3-D geometry at
        # the sampled peak range, not a flat-Euclidean approximation.
        peak_distance = float(result[5][peak])
        observer_radius = MOON_REFERENCE_RADIUS_M + 1.0
        target_radius = MOON_REFERENCE_RADIUS_M + 1000.0
        delta = peak_distance / MOON_REFERENCE_RADIUS_M
        expected_elevation = np.degrees(np.arctan2(
            target_radius * np.cos(delta) - observer_radius,
            target_radius * np.sin(delta),
        ))
        self.assertAlmostEqual(float(horizon[peak]), float(expected_elevation), places=5)
        flat_elevation = np.degrees(np.arctan2(999.0, peak_distance))
        self.assertGreater(float(flat_elevation - horizon[peak]), 0.005)
        self.assertIn("Spherical Moon curvature included", metadata["curvature_model"])
        self.assertAlmostEqual(
            metadata["projected_grid_y_bearing_from_local_north_deg"], expected_bearing, places=5
        )


if __name__ == "__main__":
    unittest.main()
