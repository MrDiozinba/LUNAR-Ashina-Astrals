from pathlib import Path
import sys
import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.terrain_horizon import (
    inverse_south_polar_stereographic,
    local_enu_basis,
    far_field_required_range_m,
    validate_lunar_south_polar_grid,
)

DEM = ROOT / "data" / "dem" / "Site01_final_adj_5mpp_surf.tif"


def test_inverse_projection_tile_centre():
    lat, lon = inverse_south_polar_stereographic(-11000.0, -12000.0)
    assert abs(np.degrees(lat) - (-89.4631628340)) < 1e-8
    assert abs(np.degrees(lon) - 222.5104470780) < 1e-8


def test_enu_is_orthonormal_and_right_handed():
    lat, lon = inverse_south_polar_stereographic(-11000.0, -12000.0)
    east, north, up = local_enu_basis(lat, lon)
    basis = np.vstack([east, north, up])
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    assert np.allclose(np.cross(east, north), up, atol=1e-12)


def test_far_field_required_range_is_positive_or_unclosed():
    result = far_field_required_range_m(2.0, 1944.8, 1.0, 5000.0)
    assert result is None or result > 0


@pytest.mark.skipif(not DEM.exists(), reason="Local Site01 DEM is not bundled.")
def test_site01_crs_against_proj_control_points():
    with rasterio.open(DEM) as ds:
        result = validate_lunar_south_polar_grid(ds)
    assert result["control_points"] == 25
    assert result["max_latitude_difference_deg"] < 1e-8
    assert result["max_longitude_difference_deg"] < 1e-8
