"""Moon-fixed Sun/Earth geometry and terrain-horizon visibility helpers."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

MOON_RADIUS_M = 1_737_400.0
MOON_FRAME = "MOON_ME_DE440_ME421"


def local_enu_basis(latitude_deg: float, longitude_east_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return east, north, up unit vectors in a planetocentric body-fixed frame."""
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_east_deg)
    east = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=float)
    north = np.array(
        [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
        dtype=float,
    )
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)], dtype=float)
    return east, north, up


def observer_position_fixed_km(
    latitude_deg: float,
    longitude_east_deg: float,
    ground_elevation_m: float,
    observer_height_m: float,
    moon_radius_m: float = MOON_RADIUS_M,
) -> np.ndarray:
    """Site position in Moon-fixed XYZ, assuming spherical radial height datum."""
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_east_deg)
    radius_km = (moon_radius_m + ground_elevation_m + observer_height_m) / 1000.0
    return radius_km * np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        dtype=float,
    )


def vector_to_azimuth_elevation(
    vector_fixed_km: Iterable[float],
    latitude_deg: float,
    longitude_east_deg: float,
) -> tuple[float, float, float]:
    """Convert a local line-of-sight vector to azimuth, elevation and range."""
    vec = np.asarray(vector_fixed_km, dtype=float)
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        raise ValueError("vector_fixed_km must be a finite 3-vector")
    east, north, up = local_enu_basis(latitude_deg, longitude_east_deg)
    e = float(np.dot(vec, east))
    n = float(np.dot(vec, north))
    u = float(np.dot(vec, up))
    horizontal = math.hypot(e, n)
    azimuth = math.degrees(math.atan2(e, n)) % 360.0
    elevation = math.degrees(math.atan2(u, horizontal))
    distance = float(np.linalg.norm(vec))
    if distance <= 0:
        raise ValueError("line-of-sight vector has zero length")
    return azimuth, elevation, distance


def interpolate_circular_horizon(
    azimuth_deg: float,
    bin_centers_deg: Iterable[float],
    horizon_elevation_deg: Iterable[float],
) -> float:
    """Linearly interpolate a 0..360 degree horizon profile with wraparound."""
    az = np.asarray(list(bin_centers_deg), dtype=float)
    hz = np.asarray(list(horizon_elevation_deg), dtype=float)
    if az.ndim != 1 or hz.ndim != 1 or az.size != hz.size or az.size < 2:
        raise ValueError("azimuth and horizon arrays must be same-length 1-D arrays")
    if not np.all(np.isfinite(az)) or not np.all(np.isfinite(hz)):
        raise ValueError("horizon profile contains non-finite values")
    order = np.argsort(az)
    az, hz = az[order], hz[order]
    if np.any(np.diff(az) <= 0):
        raise ValueError("azimuth bin centers must be unique")
    if az[0] < 0 or az[-1] >= 360:
        raise ValueError("azimuth bin centers must lie in [0, 360)")
    x = float(azimuth_deg) % 360.0
    xx = np.concatenate((az - 360.0, az, az + 360.0))
    yy = np.concatenate((hz, hz, hz))
    return float(np.interp(x, xx, yy))


def angular_semidiameter_deg(radius_km: float, distance_km: float) -> float:
    """Return angular semidiameter using asin(R/d); inputs must share units."""
    if not (math.isfinite(radius_km) and math.isfinite(distance_km)):
        raise ValueError("radius and distance must be finite")
    if radius_km <= 0 or distance_km <= radius_km:
        raise ValueError("distance must exceed positive body radius")
    return math.degrees(math.asin(radius_km / distance_km))


def circular_disk_visible_fraction(center_elevation_deg: float, horizon_elevation_deg: float, semidiameter_deg: float) -> float:
    """Approximate fraction of a circular body disk above a locally straight horizon.

    The terrain horizon is treated as constant across the disk at its centre azimuth.
    This is a geometric clipping approximation, not a communications/power model.
    """
    if not all(map(math.isfinite, (center_elevation_deg, horizon_elevation_deg, semidiameter_deg))):
        raise ValueError("disk visibility inputs must be finite")
    if semidiameter_deg < 0:
        raise ValueError("semidiameter_deg cannot be negative")
    gap = center_elevation_deg - horizon_elevation_deg
    if semidiameter_deg == 0:
        return 1.0 if gap > 0 else 0.0
    z = gap / semidiameter_deg
    if z <= -1.0:
        return 0.0
    if z >= 1.0:
        return 1.0
    radicand = max(0.0, 1.0 - z * z)
    fraction = (math.acos(-z) + z * math.sqrt(radicand)) / math.pi
    return min(1.0, max(0.0, fraction))


def visibility_state(fraction: float, full_tolerance: float = 1e-9) -> str:
    """Classify a body disk as hidden, partial or full."""
    f = float(fraction)
    if not math.isfinite(f):
        raise ValueError("visibility fraction must be finite")
    if f <= 0.0:
        return "hidden"
    if f >= 1.0 - full_tolerance:
        return "full"
    return "partial"
