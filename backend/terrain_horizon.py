"""Ray-marched LOLA DEM terrain horizon profile.

This replaces point-cloud binning of DEM cell centres with per-azimuth ray
sampling over a bilinearly interpolated DEM surface.

SCIENTIFIC STATUS:
- Uses the Site01 south-polar stereographic grid and spherical Moon radius.
- Uses a provisional observer at the raster centre unless coordinates are given.
- Treats DEM Z as radial surface height above the 1,737,400 m reference sphere.
  NASA PGDA describes Z as surface height in metres; confirm the exact vertical
  datum/reference against the complete product README before final science.
- Horizon is limited to the supplied DEM footprint; distant terrain is omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import rasterio

MOON_REFERENCE_RADIUS_M = 1_737_400.0



def validate_lunar_south_polar_grid(dataset, control_points_per_axis=5):
    """Validate affine geometry and compare inverse projection with PROJ numerically.

    This is deliberately not tied to one PROJ export dictionary: equivalent CRS
    WKT/PROJ representations can normalize differently between PROJ versions.
    """
    if dataset.crs is None:
        raise ValueError("DEM has no CRS; refusing to assume a lunar projection.")

    tr = dataset.transform
    tol = 1e-10
    if abs(tr.b) > tol or abs(tr.d) > tol:
        raise ValueError("Rotated/sheared rasters are unsupported by this sampler.")
    if not np.isclose(abs(tr.a), abs(tr.e), rtol=0, atol=1e-9):
        raise ValueError("Sampler requires square pixels (|pixel width| = |pixel height|).")
    if tr.a <= 0 or tr.e >= 0:
        raise ValueError("Expected a north-up raster with positive X and negative Y pixel size.")

    # Test pixel-centre coordinates, not corner coordinates.
    rows = np.linspace(0, dataset.height - 1, control_points_per_axis)
    cols = np.linspace(0, dataset.width - 1, control_points_per_axis)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr = rr.ravel()
    cc = cc.ravel()
    xs, ys = rasterio.transform.xy(dataset.transform, rr, cc, offset="center")
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    # Independent PROJ transformation using the same lunar sphere radius.
    from rasterio.warp import transform as warp_transform
    target_geographic = (
        f"+proj=longlat +R={MOON_REFERENCE_RADIUS_M:.1f} "
        "+no_defs +type=crs"
    )
    proj_lon, proj_lat = warp_transform(
        dataset.crs, target_geographic, xs.tolist(), ys.tolist()
    )
    proj_lon = np.asarray(proj_lon, dtype=np.float64)
    proj_lat = np.asarray(proj_lat, dtype=np.float64)

    manual_lat, manual_lon = inverse_south_polar_stereographic(xs, ys)
    manual_lat = np.degrees(manual_lat)
    manual_lon = np.degrees(manual_lon)
    lon_delta = (proj_lon - manual_lon + 180.0) % 360.0 - 180.0
    lat_error = np.abs(proj_lat - manual_lat)
    lon_error = np.abs(lon_delta)

    max_lat_error = float(np.max(lat_error))
    max_lon_error = float(np.max(lon_error))
    if max(max_lat_error, max_lon_error) > 1e-8:
        raise ValueError(
            "Raster CRS does not numerically agree with the assumed spherical "
            f"south-polar stereographic transform (max lat error={max_lat_error:.3g}°, "
            f"max lon error={max_lon_error:.3g}°)."
        )

    return {
        "validation_method": "25 pixel-centre checks against rasterio/PROJ transform",
        "control_points": int(xs.size),
        "max_latitude_difference_deg": max_lat_error,
        "max_longitude_difference_deg": max_lon_error,
        "pixel_width_m": float(tr.a),
        "pixel_height_m": float(tr.e),
        "area_or_point_tag": dataset.tags().get("AREA_OR_POINT", "unspecified"),
        "crs_wkt": dataset.crs.to_wkt(),
    }


def inverse_south_polar_stereographic(x_m, y_m, radius_m=MOON_REFERENCE_RADIUS_M):
    """Inverse spherical south-polar stereographic: planetocentric lat, east lon."""
    rho = np.hypot(x_m, y_m)
    colat = 2.0 * np.arctan2(rho, 2.0 * radius_m)
    lat = -0.5 * np.pi + colat
    lon = np.arctan2(x_m, y_m) % (2.0 * np.pi)
    return lat, lon


def local_enu_basis(lat_rad, lon_rad):
    """Return East, North, Up unit vectors in Moon-fixed Cartesian coordinates."""
    up = np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ], dtype=np.float64)
    east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0], dtype=np.float64)
    north = np.array([
        -np.sin(lat_rad) * np.cos(lon_rad),
        -np.sin(lat_rad) * np.sin(lon_rad),
        np.cos(lat_rad),
    ], dtype=np.float64)
    return east, north, up


def projected_axis_bearing_deg(observer_x_m, observer_y_m, axis="y", delta_m=1.0):
    """Bearing of a positive projected-grid axis measured clockwise from local north.

    This is a local differential calculation. It prevents confusing the site's
    east longitude with the azimuth of a projected raster axis.
    """
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    if not np.isfinite(delta_m) or delta_m <= 0:
        raise ValueError("delta_m must be positive and finite")
    lat0, lon0 = inverse_south_polar_stereographic(observer_x_m, observer_y_m)
    x1 = observer_x_m + (delta_m if axis == "x" else 0.0)
    y1 = observer_y_m + (delta_m if axis == "y" else 0.0)
    lat1, lon1 = inverse_south_polar_stereographic(x1, y1)

    def radial_unit(lat, lon):
        return np.array([np.cos(lat) * np.cos(lon),
                         np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=np.float64)

    east, north, _up = local_enu_basis(lat0, lon0)
    tangent = MOON_REFERENCE_RADIUS_M * (radial_unit(lat1, lon1) - radial_unit(lat0, lon0))
    e_component = float(np.dot(tangent, east))
    n_component = float(np.dot(tangent, north))
    return float(np.degrees(np.arctan2(e_component, n_component)) % 360.0)


def geodesic_destination_sphere(lat1, lon1, azimuth_rad, distance_m, radius_m):
    """Direct great-circle problem on a sphere; bearing is clockwise from north."""
    delta = distance_m / radius_m
    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_delta, cos_delta = np.sin(delta), np.cos(delta)

    sin_lat2 = (
        sin_lat1 * cos_delta
        + cos_lat1 * sin_delta * np.cos(azimuth_rad)
    )
    lat2 = np.arcsin(np.clip(sin_lat2, -1.0, 1.0))
    dlon = np.arctan2(
        np.sin(azimuth_rad) * sin_delta * cos_lat1,
        cos_delta - sin_lat1 * sin_lat2,
    )
    lon2 = lon1 + dlon
    return lat2, lon2


def bilinear_sample_many(array, transform, x_m, y_m, nodata=None):
    """Vectorized bilinear sample at projected coordinates.

    Returns (values, valid_mask). Pixel-centre indexing accounts for the
    AREA_OR_POINT=Area registration and the GDAL affine transform.
    """
    col_f = (x_m - transform.c) / transform.a - 0.5
    row_f = (y_m - transform.f) / transform.e - 0.5
    c0 = np.floor(col_f).astype(np.int64)
    r0 = np.floor(row_f).astype(np.int64)
    dc = col_f - c0
    dr = row_f - r0

    inside = (
        (c0 >= 0) & (r0 >= 0)
        & (c0 + 1 < array.shape[1])
        & (r0 + 1 < array.shape[0])
    )
    values = np.full(np.shape(x_m), np.nan, dtype=np.float64)
    if not np.any(inside):
        valid = np.zeros(np.shape(x_m), dtype=bool)
        return values, valid, inside

    rr = r0[inside]
    cc = c0[inside]
    z00 = array[rr, cc].astype(np.float64, copy=False)
    z01 = array[rr, cc + 1].astype(np.float64, copy=False)
    z10 = array[rr + 1, cc].astype(np.float64, copy=False)
    z11 = array[rr + 1, cc + 1].astype(np.float64, copy=False)

    valid_local = (
        np.isfinite(z00) & np.isfinite(z01)
        & np.isfinite(z10) & np.isfinite(z11)
    )
    if nodata is not None and np.isfinite(nodata):
        valid_local &= (
            (z00 != nodata) & (z01 != nodata)
            & (z10 != nodata) & (z11 != nodata)
        )

    sampled = (
        (1.0 - dr[inside]) * (
            (1.0 - dc[inside]) * z00 + dc[inside] * z01
        )
        + dr[inside] * (
            (1.0 - dc[inside]) * z10 + dc[inside] * z11
        )
    )
    inside_indices = np.flatnonzero(inside)
    good_indices = inside_indices[valid_local]
    values.flat[good_indices] = sampled[valid_local]
    valid = np.zeros(np.shape(x_m), dtype=bool)
    valid.flat[good_indices] = True
    return values, valid, inside


def build_horizon_profile_ray_march(
    dem_path,
    observer_x_m=None,
    observer_y_m=None,
    observer_height_m=1.0,
    azimuth_step_deg=0.1,
    ray_step_m=0.625,
    minimum_ray_distance_m=0.625,
    maximum_ray_distance_m=15000.0,
    rays_per_chunk=12,
):
    """Build DEM-footprint horizon by sampling one geodesic ray per azimuth.

    For each azimuth, the terrain DEM is bilinearly sampled every `ray_step_m`
    along the corresponding spherical geodesic. The maximum 3-D line-of-sight
    elevation angle is stored for that azimuth. The profile is not extrapolated
    beyond the DEM footprint.
    """
    calculation_started = time.perf_counter()
    if azimuth_step_deg <= 0 or azimuth_step_deg > 2:
        raise ValueError("azimuth_step_deg must be > 0 and <= 2 degrees.")
    n_bins_float = 360.0 / azimuth_step_deg
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins * azimuth_step_deg, 360.0, atol=1e-8):
        raise ValueError("azimuth_step_deg must divide 360 exactly.")
    if ray_step_m <= 0 or minimum_ray_distance_m < 0:
        raise ValueError("Ray step must be positive and minimum distance non-negative.")

    with rasterio.open(dem_path) as ds:
        crs_validation = validate_lunar_south_polar_grid(ds)
        dem_ma = ds.read(1, masked=True, out_dtype="float32")
        dem = np.asarray(dem_ma.filled(np.nan), dtype=np.float32)
        transform = ds.transform
        nodata = ds.nodata
        area_or_point = ds.tags().get("AREA_OR_POINT")

        if observer_x_m is None:
            observer_x_m = (ds.bounds.left + ds.bounds.right) / 2.0
        if observer_y_m is None:
            observer_y_m = (ds.bounds.bottom + ds.bounds.top) / 2.0

        lat0, lon0 = inverse_south_polar_stereographic(observer_x_m, observer_y_m)
        ground_values, ground_valid, _ground_inside = bilinear_sample_many(
            dem, transform,
            np.asarray([observer_x_m]), np.asarray([observer_y_m]),
            nodata=nodata,
        )
        if not ground_valid[0]:
            raise ValueError("Observer point cannot be bilinearly sampled from valid DEM cells.")
        ground_z_m = float(ground_values[0])

        east, north, up = local_enu_basis(lat0, lon0)
        observer_radius = MOON_REFERENCE_RADIUS_M + ground_z_m + observer_height_m
        observer_position = observer_radius * up

        # Azimuth bin centres; north=0°, east=90°, clockwise positive.
        azimuth_centres_deg = (
            np.arange(n_bins, dtype=np.float64) + 0.5
        ) * azimuth_step_deg
        azimuth_centres_rad = np.radians(azimuth_centres_deg)

        distances = np.arange(
            minimum_ray_distance_m,
            maximum_ray_distance_m + 0.5 * ray_step_m,
            ray_step_m,
            dtype=np.float64,
        )

        horizon = np.full(n_bins, np.nan, dtype=np.float64)
        sample_count = np.zeros(n_bins, dtype=np.int32)
        nearest_sample_distance = np.full(n_bins, np.nan, dtype=np.float64)
        farthest_sample_distance = np.full(n_bins, np.nan, dtype=np.float64)
        distance_at_horizon = np.full(n_bins, np.nan, dtype=np.float64)
        tile_exit_distance = np.full(n_bins, np.nan, dtype=np.float64)
        coverage_end_distance = np.full(n_bins, np.nan, dtype=np.float64)
        tile_boundary_stop = np.zeros(n_bins, dtype=bool)
        range_limited = np.zeros(n_bins, dtype=bool)
        nodata_stop = np.zeros(n_bins, dtype=bool)
        nodata_sample_count = np.zeros(n_bins, dtype=np.int32)

        sin_lat0, cos_lat0 = np.sin(lat0), np.cos(lat0)

        for start in range(0, n_bins, rays_per_chunk):
            stop = min(start + rays_per_chunk, n_bins)
            a = azimuth_centres_rad[start:stop, None]
            d = distances[None, :]
            delta = d / MOON_REFERENCE_RADIUS_M
            sin_delta = np.sin(delta)
            cos_delta = np.cos(delta)

            sin_lat2 = (
                sin_lat0 * cos_delta
                + cos_lat0 * sin_delta * np.cos(a)
            )
            lat2 = np.arcsin(np.clip(sin_lat2, -1.0, 1.0))
            dlon = np.arctan2(
                np.sin(a) * sin_delta * cos_lat0,
                cos_delta - sin_lat0 * sin_lat2,
            )
            lon2 = lon0 + dlon

            # Inverse spherical south-polar stereographic:
            # rho=2R*tan(colat/2), X=rho*sin(lon), Y=rho*cos(lon).
            colat2 = lat2 + 0.5 * np.pi
            rho = 2.0 * MOON_REFERENCE_RADIUS_M * np.tan(0.5 * colat2)
            x2 = rho * np.sin(lon2)
            y2 = rho * np.cos(lon2)

            z2, valid, inside = bilinear_sample_many(
                dem, transform, x2, y2, nodata=nodata
            )

            terrain_radius = MOON_REFERENCE_RADIUS_M + z2
            cos_lat2 = np.cos(lat2)
            px = terrain_radius * cos_lat2 * np.cos(lon2)
            py = terrain_radius * cos_lat2 * np.sin(lon2)
            pz = terrain_radius * np.sin(lat2)

            dx = px - observer_position[0]
            dy = py - observer_position[1]
            dz = pz - observer_position[2]

            east_component = dx * east[0] + dy * east[1] + dz * east[2]
            north_component = dx * north[0] + dy * north[1] + dz * north[2]
            up_component = dx * up[0] + dy * up[1] + dz * up[2]
            horizontal = np.hypot(east_component, north_component)
            elevation = np.degrees(np.arctan2(up_component, horizontal))

            valid &= np.isfinite(elevation)
            for local_i in range(stop - start):
                idx = start + local_i
                inside_ray = inside[local_i]
                valid_ray = valid[local_i]
                if np.any(inside_ray):
                    inside_indices = np.flatnonzero(inside_ray)
                    support_end_idx = int(inside_indices[-1])
                    coverage_end_distance[idx] = float(distances[support_end_idx])
                    # Distinguish the raster interpolation boundary, requested
                    # distance cap, and a NoData stop.
                    range_limited[idx] = bool(inside_ray[-1] and valid_ray[-1])
                    tile_boundary_stop[idx] = bool(not inside_ray[-1])
                    nodata_stop[idx] = bool(inside_ray[-1] and not valid_ray[-1])
                    if tile_boundary_stop[idx]:
                        tile_exit_distance[idx] = float(distances[support_end_idx])
                nodata_sample_count[idx] = int(
                    np.count_nonzero(inside_ray & ~valid_ray)
                )

                good = valid_ray
                if not np.any(good):
                    continue
                vals = elevation[local_i, good]
                ds_good = distances[good]
                imax = int(np.argmax(vals))
                horizon[idx] = float(vals[imax])
                sample_count[idx] = int(vals.size)
                nearest_sample_distance[idx] = float(ds_good[0])
                farthest_sample_distance[idx] = float(ds_good[-1])
                distance_at_horizon[idx] = float(ds_good[imax])

        missing = ~np.isfinite(horizon)
        if np.any(missing):
            missing_degrees = int(np.count_nonzero(missing))
            raise ValueError(
                f"{missing_degrees} azimuth rays had no valid terrain samples. "
                "Do not interpolate silently; check observer position, DEM coverage and ray range."
            )

        # 2D grid coverage is different by azimuth; report a warning threshold.
        metadata = {
            "dem_file": Path(dem_path).name,
            "dem_width_pixels": int(ds.width),
            "dem_height_pixels": int(ds.height),
            "pixel_size_x_m": float(transform.a),
            "pixel_size_y_m": float(abs(transform.e)),
            "raster_area_or_point": ds.tags().get("AREA_OR_POINT", "unspecified"),
            "dem_nodata_value": (
                float(nodata) if nodata is not None and np.isfinite(nodata) else None
            ),
            "crs_validation": crs_validation,
            "area_or_point": area_or_point,
            "dem_frame_from_pgda": "MOON_ME, DE421",
            "projection": "South Polar Stereographic, true at pole, lon_0=0, k=1",
            "observer_projected_x_m": float(observer_x_m),
            "observer_projected_y_m": float(observer_y_m),
            "observer_latitude_planetocentric_deg": float(np.degrees(lat0)),
            "observer_longitude_east_deg": float(np.degrees(lon0)),
            "observer_ground_z_m": ground_z_m,
            "observer_height_above_dem_m": float(observer_height_m),
            "vertical_reference_assumption": (
                "DEM Z treated as radial surface height above the 1,737,400 m "
                "reference sphere. PGDA describes pixels as surface height Z in metres; "
                "confirm the precise vertical reference from the complete product README."
            ),
            "azimuth_convention": "local north = 0 degrees, clockwise positive",
            "projected_grid_x_bearing_from_local_north_deg": projected_axis_bearing_deg(observer_x_m, observer_y_m, axis="x"),
            "projected_grid_y_bearing_from_local_north_deg": projected_axis_bearing_deg(observer_x_m, observer_y_m, axis="y"),
            "azimuth_ray_definition": "Spherical geodesic direct solution using local true-north bearing; not projected-grid azimuth.",
            "curvature_model": "Spherical Moon curvature included by converting sampled radial DEM elevations to Moon-fixed 3D Cartesian positions before computing observer-local ENU elevation.",
            "azimuth_bin_width_deg": float(azimuth_step_deg),
            "ray_step_m": float(ray_step_m),
            "minimum_ray_distance_m": float(minimum_ray_distance_m),
            "maximum_ray_distance_m_requested": float(maximum_ray_distance_m),
            "sample_count_min": int(sample_count.min()),
            "sample_count_median": float(np.median(sample_count)),
            "sample_count_max": int(sample_count.max()),
            "nearest_sample_distance_min_m": float(np.nanmin(nearest_sample_distance)),
            "farthest_sample_distance_min_m": float(np.nanmin(farthest_sample_distance)),
            "farthest_sample_distance_max_m": float(np.nanmax(farthest_sample_distance)),
            "tile_exit_distance_min_m": (
                float(np.nanmin(tile_exit_distance))
                if np.any(np.isfinite(tile_exit_distance)) else None
            ),
            "tile_exit_distance_median_m": (
                float(np.nanmedian(tile_exit_distance))
                if np.any(np.isfinite(tile_exit_distance)) else None
            ),
            "tile_exit_distance_max_m": (
                float(np.nanmax(tile_exit_distance))
                if np.any(np.isfinite(tile_exit_distance)) else None
            ),
            "coverage_end_distance_min_m": float(np.nanmin(coverage_end_distance)),
            "coverage_end_distance_median_m": float(np.nanmedian(coverage_end_distance)),
            "coverage_end_distance_max_m": float(np.nanmax(coverage_end_distance)),
            "tile_boundary_stop_azimuth_bins": int(np.count_nonzero(tile_boundary_stop)),
            "range_limited_azimuth_bins": int(np.count_nonzero(range_limited)),
            "nodata_stop_azimuth_bins": int(np.count_nonzero(nodata_stop)),
            "nodata_sample_count_total": int(np.sum(nodata_sample_count)),
            "distance_at_horizon_min_m": float(np.nanmin(distance_at_horizon)),
            "distance_at_horizon_max_m": float(np.nanmax(distance_at_horizon)),
            "horizon_min_deg": float(np.min(horizon)),
            "horizon_max_deg": float(np.max(horizon)),
            "horizon_mean_deg": float(np.mean(horizon)),
            "horizon_median_deg": float(np.median(horizon)),
            "coverage_limitation": (
                "Profile uses only the supplied 16 km x 16 km DEM tile. The terrain "
                "horizon beyond each ray's tile exit is unknown and may be higher."
            ),
            "horizon_interpretation": "tile-limited lower bound with respect to omitted terrain only; DEM error is not bounded",
            "observer_location_status": (
                "Provisional geometric centre of raster; not a verified landing coordinate."
            ),
            "algorithm": (
                "Per-azimuth great-circle ray march on a sphere; DEM elevations "
                "bilinearly sampled in projected X/Y; 3-D line-of-sight elevation "
                "computed in observer-local ENU."
            ),
            "runtime_seconds": float(time.perf_counter() - calculation_started),
        }

    return (
        azimuth_centres_deg, horizon, sample_count, nearest_sample_distance,
        farthest_sample_distance, distance_at_horizon, tile_exit_distance,
        coverage_end_distance, tile_boundary_stop,
        range_limited, nodata_stop, nodata_sample_count, metadata
    )



def far_field_required_range_m(
    horizon_elevation_deg,
    observer_ground_z_m,
    observer_height_m,
    maximum_terrain_z_m,
    radius_m=MOON_REFERENCE_RADIUS_M,
):
    """Farthest range at which a target at maximum terrain Z can remain above
    the current horizon angle, assuming spherical radial heights.

    Returns None if even the maximum-height target remains above that angle
    through its geometric line-of-sight limit. This is a conservative global
    bound only if maximum_terrain_z_m is a valid upper bound for all omitted terrain.
    """
    r_observer = radius_m + observer_ground_z_m + observer_height_m
    r_target = radius_m + maximum_terrain_z_m
    if r_target <= radius_m or r_observer <= radius_m:
        raise ValueError("Observer and maximum terrain radii must exceed reference radius.")

    # Maximum surface distance to a target at r_target, tangent to the reference sphere.
    sight_limit = radius_m * (
        np.arccos(np.clip(radius_m / r_observer, -1.0, 1.0))
        + np.arccos(np.clip(radius_m / r_target, -1.0, 1.0))
    )

    h = np.radians(float(horizon_elevation_deg))

    def apparent_elevation(distance_m):
        theta = distance_m / radius_m
        up = r_target * np.cos(theta) - r_observer
        horizontal = r_target * np.sin(theta)
        return np.arctan2(up, horizontal)

    terminal_angle = apparent_elevation(sight_limit)
    # If the max-height target is still at/above the horizon at sight limit,
    # the bound does not close within the geometric line of sight.
    if h <= terminal_angle:
        return None

    lo, hi = 0.0, sight_limit
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if apparent_elevation(mid) > h:
            lo = mid
        else:
            hi = mid
    return float(hi)



def sha256_file(path, chunk_bytes=8 * 1024 * 1024):
    """Return SHA-256 for a local file; returns None for remote/non-file paths."""
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_for(path):
    """Return repository HEAD when the project is inside a Git repository."""
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(path).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=3,
        )
        return result.stdout.strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", required=True)
    parser.add_argument("--output-dir", default="data/horizon_masks")
    parser.add_argument("--observer-x-m", type=float, default=None)
    parser.add_argument("--observer-y-m", type=float, default=None)
    parser.add_argument("--heights-m", type=float, nargs="+", default=[1.0, 5.0])
    parser.add_argument("--azimuth-step-deg", type=float, default=0.1)
    parser.add_argument("--ray-step-m", type=float, default=0.625)
    parser.add_argument("--minimum-ray-distance-m", type=float, default=0.625)
    parser.add_argument("--maximum-ray-distance-m", type=float, default=15000.0)
    parser.add_argument("--rays-per-chunk", type=int, default=12)
    parser.add_argument(
        "--max-terrain-z-m", type=float, default=None,
        help="Optional defensible upper bound on all omitted terrain Z values (m). "
             "Only then can a far-field closure distance be estimated."
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_meta = {"profiles": []}

    for height in args.heights_m:
        (
            az, hor, count, near_d, far_d, at_hor,
            tile_exit_d, coverage_end_d, tile_stop, range_limited,
            nodata_stop, nodata_count, meta
        ) = build_horizon_profile_ray_march(
            args.dem,
            observer_x_m=args.observer_x_m,
            observer_y_m=args.observer_y_m,
            observer_height_m=height,
            azimuth_step_deg=args.azimuth_step_deg,
            ray_step_m=args.ray_step_m,
            minimum_ray_distance_m=args.minimum_ray_distance_m,
            maximum_ray_distance_m=args.maximum_ray_distance_m,
            rays_per_chunk=args.rays_per_chunk,
        )

        meta["dem_sha256"] = sha256_file(args.dem)
        meta["processor_sha256"] = sha256_file(__file__)
        meta["git_commit"] = git_commit_for(__file__)
        meta["python_version"] = platform.python_version()
        meta["rasterio_version"] = rasterio.__version__
        meta["gdal_version"] = getattr(rasterio, "__gdal_version__", None)
        meta["proj_version"] = getattr(rasterio, "__proj_version__", None)

        required_range = np.full(len(az), np.nan, dtype=np.float64)
        # -1 means not evaluated, 0 means evaluated but not closed, 1 means closed.
        closed_by_bound = np.full(len(az), -1, dtype=np.int8)
        if args.max_terrain_z_m is not None:
            for i, horizon_angle in enumerate(hor):
                req = far_field_required_range_m(
                    horizon_angle,
                    meta["observer_ground_z_m"],
                    height,
                    args.max_terrain_z_m,
                )
                if req is not None:
                    required_range[i] = req
                    if np.isfinite(coverage_end_d[i]):
                        closed_by_bound[i] = int(req <= coverage_end_d[i])
                    else:
                        closed_by_bound[i] = 0

        meta["maximum_omitted_terrain_z_bound_m"] = args.max_terrain_z_m
        meta["closed_by_bound_azimuth_bins"] = (
            int(np.count_nonzero(closed_by_bound == 1))
            if args.max_terrain_z_m is not None else None
        )
        meta["unclosed_azimuth_bins"] = (
            int(np.count_nonzero(closed_by_bound == 0))
            if args.max_terrain_z_m is not None else None
        )
        meta["closure_not_evaluated_bins"] = (
            int(np.count_nonzero(closed_by_bound == -1))
        )
        meta["closure_method"] = (
            "spherical upper bound using maximum_omitted_terrain_z_bound_m"
            if args.max_terrain_z_m is not None else
            "not evaluated: supply --max-terrain-z-m from a defensible regional upper bound"
        )

        stem = f"site01_tile_center_h{height:g}m_RAYMARCH_PROVISIONAL"
        csv_path = out / f"{stem}.csv"
        data = np.column_stack([
            az, hor, count, near_d, far_d, at_hor,
            tile_exit_d, coverage_end_d, tile_stop.astype(np.int8),
            range_limited.astype(np.int8), nodata_stop.astype(np.int8),
            required_range, closed_by_bound.astype(np.int8), nodata_count
        ])
        np.savetxt(
            csv_path, data, delimiter=",",
            header=(
                "azimuth_center_deg,horizon_elevation_deg,sample_count,"
                "nearest_sample_distance_m,farthest_sample_distance_m,"
                "distance_at_horizon_m,tile_exit_distance_m,coverage_end_distance_m,"
                "tile_boundary_stop,range_limited,nodata_stop,required_range_m,"
                "closed_by_bound,nodata_sample_count"
            ),
            comments="", fmt="%.6f",
        )
        meta["output_csv"] = csv_path.name
        all_meta["profiles"].append(meta)
        print(
            f"{csv_path}: bins={len(az)}, "
            f"horizon=[{hor.min():.3f}, {hor.max():.3f}] deg, "
            f"samples/bin median={np.median(count):.0f}, "
            f"range max={far_d.max()/1000:.2f} km"
        )

    (out / "site01_tile_center_RAYMARCH_PROVISIONAL_metadata.json").write_text(
        json.dumps(all_meta, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
