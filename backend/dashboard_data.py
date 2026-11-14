"""Load and validate the currently generated Site01 terrain/SPICE outputs.

This module intentionally refuses to invent missing sites or silently accept
mismatched terrain-profile inputs. The data describe one provisional tile-centre
point and a 48-hour SPICE window only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .scientific_constants import DISK_VISIBILITY_MATERIALITY_FRACTION
from .spice_site_geometry import circular_disk_visible_fraction, visibility_state

TIMELINE_FILES = {
    1: "site01_spice_timeline_h1m.csv",
    5: "site01_spice_timeline_h5m.csv",
}
HORIZON_FILES = {
    1: "site01_multires_h1m.csv",
    5: "site01_multires_h5m.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required scientific data file is missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Scientific data file is empty: {path}")
    return path


def _load_timeline(path: Path, expected_height: int) -> pd.DataFrame:
    frame = pd.read_csv(_require_file(path))
    required = {
        "utc", "observer_height_m", "sun_azimuth_deg", "sun_center_elevation_deg",
        "sun_terrain_horizon_deg", "sun_angular_semidiameter_deg",
        "sun_visible_fraction_approx", "sun_visibility_state", "earth_azimuth_deg",
        "earth_center_elevation_deg", "earth_terrain_horizon_deg",
        "earth_angular_semidiameter_deg", "earth_visible_fraction_approx",
        "earth_visibility_state", "earth_visibility_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path.name} contains no observations")

    frame["timestamp_utc"] = pd.to_datetime(frame["utc"], utc=True, errors="coerce")
    if frame["timestamp_utc"].isna().any():
        raise ValueError(f"{path.name} contains invalid UTC timestamps")
    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    if frame["timestamp_utc"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate timestamps")

    heights = pd.to_numeric(frame["observer_height_m"], errors="coerce")
    if heights.isna().any() or not np.allclose(heights.to_numpy(), expected_height, atol=1e-9):
        raise ValueError(f"{path.name} does not consistently represent a {expected_height} m observer")

    numeric = [
        "sun_azimuth_deg", "sun_center_elevation_deg", "sun_terrain_horizon_deg",
        "sun_angular_semidiameter_deg", "sun_visible_fraction_approx",
        "earth_azimuth_deg", "earth_center_elevation_deg", "earth_terrain_horizon_deg",
        "earth_angular_semidiameter_deg", "earth_visible_fraction_approx",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name} has non-finite values in {column}")

    for column in ("sun_visible_fraction_approx", "earth_visible_fraction_approx"):
        if ((frame[column] < -1e-9) | (frame[column] > 1.0 + 1e-9)).any():
            raise ValueError(f"{path.name} has disk fractions outside [0, 1] in {column}")

    for body in ("sun", "earth"):
        azimuth = frame[f"{body}_azimuth_deg"].to_numpy(float)
        if ((azimuth < 0.0) | (azimuth >= 360.0)).any():
            raise ValueError(f"{path.name} has {body} azimuth outside [0, 360)")
        expected_fraction = np.fromiter((
            circular_disk_visible_fraction(el, hz, semi)
            for el, hz, semi in zip(
                frame[f"{body}_center_elevation_deg"].to_numpy(float),
                frame[f"{body}_terrain_horizon_deg"].to_numpy(float),
                frame[f"{body}_angular_semidiameter_deg"].to_numpy(float),
            )
        ), dtype=float, count=len(frame))
        stored_fraction = frame[f"{body}_visible_fraction_approx"].to_numpy(float)
        if not np.allclose(stored_fraction, expected_fraction, atol=5e-7, rtol=0):
            max_error = float(np.max(np.abs(stored_fraction - expected_fraction)))
            raise ValueError(
                f"{path.name} {body} disk fractions do not reproduce the circular-disk "
                f"clipping equation (maximum difference {max_error:.6g})"
            )
        state = frame[f"{body}_visibility_state"].astype(str)
        invalid_state = ~state.isin({"hidden", "partial", "full"})
        if invalid_state.any():
            raise ValueError(f"{path.name} has invalid {body} visibility states")
        expected_states = np.array([visibility_state(value) for value in expected_fraction], dtype=object)
        if not np.array_equal(state.to_numpy(dtype=object), expected_states):
            raise ValueError(f"{path.name} {body} visibility states disagree with disk fractions")

    frame["sun_visible"] = frame["sun_visible_fraction_approx"] >= DISK_VISIBILITY_MATERIALITY_FRACTION
    frame["earth_visible"] = frame["earth_visible_fraction_approx"] >= DISK_VISIBILITY_MATERIALITY_FRACTION
    frame["both_visible"] = frame["sun_visible"] & frame["earth_visible"]
    frame["timestamp_ist"] = frame["timestamp_utc"].dt.tz_convert("Asia/Kolkata")
    frame["elapsed_hours"] = (
        frame["timestamp_utc"] - frame["timestamp_utc"].iloc[0]
    ).dt.total_seconds() / 3600.0
    return frame


def _load_horizon(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(_require_file(path))
    required = {
        "azimuth_center_deg", "combined_horizon_elevation_deg",
        "combined_distance_at_horizon_m", "regional_80m_coverage_end_m",
        "regional_80m_distance_at_horizon_m", "regional_80m_dominates",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if len(frame) != 3600:
        raise ValueError(f"{path.name} should contain 3600 azimuth bins; found {len(frame)}")
    frame = frame.sort_values("azimuth_center_deg").reset_index(drop=True)
    az = pd.to_numeric(frame["azimuth_center_deg"], errors="coerce").to_numpy(dtype=float)
    hz = pd.to_numeric(frame["combined_horizon_elevation_deg"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(az).all() or not np.isfinite(hz).all():
        raise ValueError(f"{path.name} contains non-finite horizon samples")
    for column in ("combined_distance_at_horizon_m", "regional_80m_coverage_end_m",
                   "regional_80m_distance_at_horizon_m", "regional_80m_dominates"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name} contains non-finite values in {column}")
    if not np.allclose(az, np.arange(0.05, 360.0, 0.1), atol=1e-6):
        raise ValueError(f"{path.name} does not contain the expected 0.1-degree azimuth bins")
    if (frame[["combined_distance_at_horizon_m", "regional_80m_coverage_end_m",
               "regional_80m_distance_at_horizon_m"]] < 0).any().any():
        raise ValueError(f"{path.name} contains negative terrain distances")
    if ((frame["regional_80m_dominates"] < 0) | (frame["regional_80m_dominates"] > 1)).any():
        raise ValueError(f"{path.name} has invalid regional_80m_dominates flags")
    return frame


def load_scientific_bundle(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    spice_dir = root / "data" / "spice"
    horizon_dir = root / "data" / "horizon_masks"

    spice_metadata_path = _require_file(spice_dir / "site01_spice_timeline_metadata.json")
    terrain_metadata_path = _require_file(horizon_dir / "site01_multires_horizon_metadata.json")
    spice_meta = json.loads(spice_metadata_path.read_text(encoding="utf-8"))
    terrain_meta = json.loads(terrain_metadata_path.read_text(encoding="utf-8"))

    timelines: dict[int, pd.DataFrame] = {}
    horizons: dict[int, pd.DataFrame] = {}
    hash_checks: dict[str, bool] = {}
    for height in (1, 5):
        timeline_path = spice_dir / TIMELINE_FILES[height]
        horizon_path = horizon_dir / HORIZON_FILES[height]
        timelines[height] = _load_timeline(timeline_path, height)
        horizons[height] = _load_horizon(horizon_path)
        expected_hash = spice_meta.get("horizon_inputs", {}).get(HORIZON_FILES[height])
        actual_hash = sha256_file(horizon_path)
        if not expected_hash:
            raise ValueError(f"SPICE metadata does not record the hash of {HORIZON_FILES[height]}")
        hash_checks[HORIZON_FILES[height]] = (actual_hash == expected_hash)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Horizon provenance mismatch for {HORIZON_FILES[height]}: "
                "the SPICE timeline metadata does not match the current horizon CSV."
            )

    t1 = timelines[1]["timestamp_utc"].reset_index(drop=True)
    t5 = timelines[5]["timestamp_utc"].reset_index(drop=True)
    if not t1.equals(t5):
        raise ValueError("The 1 m and 5 m SPICE timelines do not have identical timestamps")
    if len(t1) > 1:
        steps = t1.diff().dropna().dt.total_seconds().to_numpy() / 60.0
        nominal_step = float(spice_meta.get("nominal_step_minutes", 5.0))
        if not np.allclose(steps, nominal_step, atol=1e-6):
            raise ValueError(f"The SPICE timeline is not uniformly sampled at the recorded {nominal_step:g}-minute interval")

    # Verify the terrain metadata itself, not just the two horizon CSVs.
    actual_terrain_meta_hash = sha256_file(terrain_metadata_path)
    expected_terrain_meta_hash = spice_meta.get("terrain_metadata_sha256")
    if not expected_terrain_meta_hash or actual_terrain_meta_hash != expected_terrain_meta_hash:
        raise ValueError("Terrain metadata SHA-256 does not match the SPICE metadata record")
    hash_checks[terrain_metadata_path.name] = True

    far_limit_km = spice_meta.get("terrain_horizon", {}).get("far_terrain_limit_km")
    if far_limit_km is None or not np.isfinite(float(far_limit_km)) or float(far_limit_km) <= 0:
        raise ValueError("SPICE metadata must record a positive far-terrain range limit")
    far_limit_m = float(far_limit_km) * 1000.0

    # Every per-epoch stored terrain angle must match circular interpolation
    # from the exact horizon CSV that was loaded and hash-verified above.
    for height in (1, 5):
        horizon = horizons[height]
        profile_meta = next((item for item in terrain_meta.get("heights", [])
                             if int(item.get("observer_height_m", -1)) == height), None)
        if profile_meta is None:
            raise ValueError(f"Terrain metadata is missing the {height} m observer profile")
        regional = profile_meta.get("regional_80m", {})
        sampled_range = regional.get("sampled_range_m", [])
        if len(sampled_range) != 2 or not np.isclose(float(sampled_range[1]), far_limit_m, atol=1e-6):
            raise ValueError(f"{height} m terrain metadata range does not match the SPICE far-terrain limit")
        coverage = horizon["regional_80m_coverage_end_m"].to_numpy(float)
        if np.any(coverage > far_limit_m + 1e-6):
            raise ValueError(f"{height} m horizon claims coverage beyond the recorded far-terrain limit")
        measured_range_limited = int(np.count_nonzero(coverage >= far_limit_m - 0.5))
        recorded_range_limited = int(regional.get("range_limited_azimuth_bins", -1))
        if measured_range_limited != recorded_range_limited:
            raise ValueError(
                f"{height} m horizon CSV coverage count ({measured_range_limited}) "
                f"does not match metadata ({recorded_range_limited})"
            )
        for body, az_col, stored_col in (
            ("Sun", "sun_azimuth_deg", "sun_terrain_horizon_deg"),
            ("Earth", "earth_azimuth_deg", "earth_terrain_horizon_deg"),
        ):
            body_azimuth = timelines[height][az_col].to_numpy(float)
            expected_horizon = np.interp(
                body_azimuth,
                horizon["azimuth_center_deg"].to_numpy(float),
                horizon["combined_horizon_elevation_deg"].to_numpy(float),
                period=360.0,
            )
            actual_horizon = timelines[height][stored_col].to_numpy(float)
            if not np.allclose(actual_horizon, expected_horizon, atol=1e-5, rtol=0):
                error = float(np.max(np.abs(actual_horizon - expected_horizon)))
                raise ValueError(
                    f"{height} m timeline {body} horizon does not match circular interpolation "
                    f"of the loaded terrain CSV (maximum difference {error:.6g} deg)"
                )

    # This checked-in manifest pins the timeline CSVs and metadata artifacts
    # used for this release; it must be regenerated when SPICE/terrain is recomputed.
    manifest_path = spice_dir / "scientific_file_manifest.json"
    manifest_checks: dict[str, bool] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative_path, expected in manifest.get("sha256", {}).items():
            candidate = root / relative_path
            if not candidate.is_file():
                raise FileNotFoundError(f"Manifest input file is missing: {relative_path}")
            actual = sha256_file(candidate)
            manifest_checks[relative_path] = actual == expected
            if actual != expected:
                raise ValueError(f"Scientific input SHA-256 mismatch: {relative_path}")
    hash_checks.update(manifest_checks)

    # DEM and kernel inputs are checked when present locally. Missing source
    # files are reported as unchecked because the dashboard consumes cached outputs.
    optional_source_checks: dict[str, bool | None] = {}
    for profile in terrain_meta.get("heights", []):
        for source_key in ("local_5m", "regional_80m"):
            source = profile.get(source_key, {})
            raw_path = source.get("file")
            expected = source.get("sha256")
            if raw_path and expected:
                candidate = root / raw_path
                if candidate.is_file():
                    ok = sha256_file(candidate) == expected
                    optional_source_checks[str(raw_path)] = ok
                    if not ok:
                        raise ValueError(f"DEM source hash mismatch: {raw_path}")
                else:
                    optional_source_checks[str(raw_path)] = None
    for kernel_name, expected in spice_meta.get("kernel_files", {}).items():
        candidate = spice_dir / "kernels" / kernel_name
        if candidate.is_file():
            ok = sha256_file(candidate) == expected
            optional_source_checks[f"data/spice/kernels/{kernel_name}"] = ok
            if not ok:
                raise ValueError(f"SPICE kernel hash mismatch: {kernel_name}")
        else:
            optional_source_checks[f"data/spice/kernels/{kernel_name}"] = None

    # Check that metadata describes the exact temporal extent and row counts
    # represented by the files, rather than trusting stale summary fields.
    recorded_window = spice_meta.get("time_window_utc", {})
    recorded_start = pd.to_datetime(recorded_window.get("start"), utc=True, errors="coerce")
    recorded_end = pd.to_datetime(recorded_window.get("end"), utc=True, errors="coerce")
    if pd.isna(recorded_start) or pd.isna(recorded_end):
        raise ValueError("SPICE metadata is missing valid UTC start/end coverage")
    if t1.iloc[0] != recorded_start or t1.iloc[-1] != recorded_end:
        raise ValueError("SPICE timeline timestamps do not match the metadata UTC coverage window")
    for height in (1, 5):
        output_meta = spice_meta.get("outputs", {}).get(str(height), {})
        if int(output_meta.get("sample_count", -1)) != len(timelines[height]):
            raise ValueError(f"SPICE metadata sample count does not match the {height} m timeline")

    observer = spice_meta.get("observer", {})
    return {
        "timelines": timelines,
        "horizons": horizons,
        "spice_metadata": spice_meta,
        "terrain_metadata": terrain_meta,
        "hash_checks": hash_checks,
        "optional_source_hash_checks": optional_source_checks,
        "observer": observer,
        "spice_metadata_path": spice_metadata_path,
        "terrain_metadata_path": terrain_metadata_path,
    }


def summarize_timeline(
    frame: pd.DataFrame, threshold: float = DISK_VISIBILITY_MATERIALITY_FRACTION
) -> dict[str, float | int]:
    """Summarize apparent-disk visibility using the shared materiality threshold."""
    if frame.empty:
        raise ValueError("Cannot summarize an empty timeline")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Visibility threshold must be in [0,1]")
    sun = pd.to_numeric(frame["sun_visible_fraction_approx"], errors="coerce").to_numpy(dtype=float)
    earth = pd.to_numeric(frame["earth_visible_fraction_approx"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(sun).all() or not np.isfinite(earth).all():
        raise ValueError("Visibility fractions must be finite")
    sun_state = frame["sun_visibility_state"].astype(str).to_numpy()
    earth_state = frame["earth_visibility_state"].astype(str).to_numpy()
    sun_visible = sun >= threshold
    earth_visible = earth >= threshold
    return {
        "samples": int(len(frame)),
        "visibility_materiality_fraction": float(threshold),
        "sun_any_visible_pct": float(100.0 * np.mean(sun_visible)),
        "sun_full_visible_pct": float(100.0 * np.mean(sun_state == "full")),
        "sun_mean_disk_fraction_pct": float(100.0 * np.mean(sun)),
        "earth_any_visible_pct": float(100.0 * np.mean(earth_visible)),
        "earth_full_visible_pct": float(100.0 * np.mean(earth_state == "full")),
        "earth_mean_disk_fraction_pct": float(100.0 * np.mean(earth)),
        "both_any_visible_pct": float(100.0 * np.mean(sun_visible & earth_visible)),
    }


def snapshot_slider_bounds(sample_count: int) -> tuple[int, int] | None:
    """Return integer slider bounds only when at least two samples exist."""
    if not isinstance(sample_count, (int, np.integer)) or isinstance(sample_count, bool) or sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer")
    return (0, int(sample_count) - 1) if sample_count > 1 else None
