"""Auditable decision-support functions for the current LUNAR scientific bundle.

The module summarizes sampled inputs only. It does not extrapolate omitted
terrain or recompute SPICE ephemerides.
"""
from __future__ import annotations
from typing import Final
import numpy as np
import pandas as pd

COMPASS_SECTORS: Final[tuple[str, ...]] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
DISK_VISIBILITY_MATERIALITY_FRACTION: Final[float] = 0.01


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} contains non-finite values")
    return values


def summarize_horizon_sectors(
    horizon: pd.DataFrame,
    far_range_limit_m: float = 45_000.0,
    *,
    coverage_tolerance_m: float = 0.5,
    cap_pin_tolerance_m: float = 0.5,
) -> pd.DataFrame:
    """Summarize sector medians and explicit coverage/cap diagnostics.

    `Range-limited azimuth bins` is derived per row from the regional ray's
    actual coverage end. `Combined cap-pinned bins` counts rows where the
    composite argmax is at the far-field endpoint (within tolerance); this is
    a warning signal, not a horizon-closure test.
    """
    required = {
        "azimuth_center_deg", "combined_horizon_elevation_deg",
        "combined_distance_at_horizon_m", "regional_80m_dominates",
        "regional_80m_coverage_end_m", "regional_80m_distance_at_horizon_m",
    }
    missing = sorted(required - set(horizon.columns))
    if missing:
        raise ValueError(f"Horizon profile is missing columns: {missing}")
    if len(horizon) != 3600:
        raise ValueError(f"Expected 3600 azimuth bins; found {len(horizon)}")
    for name, value in (("far_range_limit_m", far_range_limit_m),
                        ("coverage_tolerance_m", coverage_tolerance_m),
                        ("cap_pin_tolerance_m", cap_pin_tolerance_m)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if far_range_limit_m <= 0:
        raise ValueError("far_range_limit_m must be positive")

    frame = horizon.copy().sort_values("azimuth_center_deg").reset_index(drop=True)
    az = _numeric(frame, "azimuth_center_deg")
    elev = _numeric(frame, "combined_horizon_elevation_deg")
    combined_dist = _numeric(frame, "combined_distance_at_horizon_m")
    regional_end = _numeric(frame, "regional_80m_coverage_end_m")
    regional_dist = _numeric(frame, "regional_80m_distance_at_horizon_m")
    reg = _numeric(frame, "regional_80m_dominates")
    if not np.allclose(az, np.arange(.05, 360, .1), atol=1e-6):
        raise ValueError("Expected 0.1-degree azimuth bins")
    if (regional_end < 0).any() or (regional_dist < 0).any() or (combined_dist < 0).any():
        raise ValueError("Terrain distances must be non-negative")

    range_limited = regional_end >= (far_range_limit_m - coverage_tolerance_m)
    coverage_short = regional_end < (far_range_limit_m - coverage_tolerance_m)
    combined_cap_pinned = combined_dist >= (far_range_limit_m - cap_pin_tolerance_m)
    regional_cap_pinned = regional_dist >= (far_range_limit_m - cap_pin_tolerance_m)
    sector_idx = np.floor(((az + 22.5) % 360) / 45).astype(int)

    rows = []
    for i, sector in enumerate(COMPASS_SECTORS):
        mask = sector_idx == i
        if not mask.any():
            continue
        rows.append({
            "Compass sector": sector,
            "Median skyline (°)": float(np.median(elev[mask])),
            "80 m controlling bins": int(np.count_nonzero((reg > .5) & mask)),
            "Range-limited azimuth bins": int(np.count_nonzero(range_limited & mask)),
            "Coverage-short azimuth bins": int(np.count_nonzero(coverage_short & mask)),
            "Combined cap-pinned bins": int(np.count_nonzero(combined_cap_pinned & mask)),
            "Regional profile cap-pinned bins": int(np.count_nonzero(regional_cap_pinned & mask)),
            "Azimuth bins": int(np.count_nonzero(mask)),
        })
    return pd.DataFrame(rows)


def compare_horizon_profiles(horizon_1m: pd.DataFrame, horizon_5m: pd.DataFrame):
    req = {"azimuth_center_deg", "combined_horizon_elevation_deg"}
    for label, frame in (("1 m", horizon_1m), ("5 m", horizon_5m)):
        missing = req - set(frame.columns)
        if missing:
            raise ValueError(f"{label} horizon missing columns: {sorted(missing)}")
    a = horizon_1m.sort_values("azimuth_center_deg").reset_index(drop=True)
    b = horizon_5m.sort_values("azimuth_center_deg").reset_index(drop=True)
    az = _numeric(a, "azimuth_center_deg")
    az_b = _numeric(b, "azimuth_center_deg")
    if len(az) != len(az_b) or not np.allclose(az, az_b, atol=1e-9):
        raise ValueError("Mismatched azimuth bins")
    h1 = _numeric(a, "combined_horizon_elevation_deg")
    h5 = _numeric(b, "combined_horizon_elevation_deg")
    difference = h1 - h5
    return pd.DataFrame({"azimuth_center_deg": az, "horizon_1m_minus_5m_deg": difference}), {
        "mean_deg": float(difference.mean()),
        "median_deg": float(np.median(difference)),
        "minimum_deg": float(difference.min()),
        "maximum_deg": float(difference.max()),
        "p95_deg": float(np.percentile(difference, 95)),
        "positive_bins": int((difference > 0).sum()),
        "negative_bins": int((difference < 0).sum()),
    }


def sample_horizon_difference_at_azimuths(
    horizon_1m: pd.DataFrame,
    horizon_5m: pd.DataFrame,
    azimuths_deg,
) -> dict[str, float | int]:
    """Sample (1 m horizon - 5 m horizon) along a body's actual azimuth track."""
    required = {"azimuth_center_deg", "combined_horizon_elevation_deg"}
    if required - set(horizon_1m.columns) or required - set(horizon_5m.columns):
        raise ValueError("Horizon files lack required azimuth/elevation columns")
    a = horizon_1m.sort_values("azimuth_center_deg")
    b = horizon_5m.sort_values("azimuth_center_deg")
    x = _numeric(a, "azimuth_center_deg")
    xb = _numeric(b, "azimuth_center_deg")
    if len(x) != len(xb) or not np.allclose(x, xb, atol=1e-9):
        raise ValueError("Mismatched horizon azimuth grids")
    y1 = _numeric(a, "combined_horizon_elevation_deg")
    y5 = _numeric(b, "combined_horizon_elevation_deg")
    target = np.asarray(azimuths_deg, dtype=float)
    if target.size == 0 or not np.isfinite(target).all():
        raise ValueError("Body azimuth track must be non-empty and finite")
    h1 = np.interp(target, x, y1, period=360.0)
    h5 = np.interp(target, x, y5, period=360.0)
    delta = h1 - h5
    return {
        "samples": int(delta.size),
        "mean_deg": float(delta.mean()),
        "median_deg": float(np.median(delta)),
        "minimum_deg": float(delta.min()),
        "maximum_deg": float(delta.max()),
    }


def compare_timeline_visibility(timeline_1m: pd.DataFrame, timeline_5m: pd.DataFrame) -> dict:
    columns = ["timestamp_utc", "sun_visibility_state", "earth_visibility_state",
               "sun_visible_fraction_approx", "earth_visible_fraction_approx"]
    for label, frame in (("1 m", timeline_1m), ("5 m", timeline_5m)):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{label} timeline missing: {sorted(missing)}")
    a = timeline_1m[columns].sort_values("timestamp_utc").reset_index(drop=True)
    b = timeline_5m[columns].sort_values("timestamp_utc").reset_index(drop=True)
    if len(a) != len(b) or not a.timestamp_utc.astype(str).equals(b.timestamp_utc.astype(str)):
        raise ValueError("Observer timelines are not timestamp-aligned")
    return {
        "samples": len(a),
        "sun_state_different_rows": int((a.sun_visibility_state.astype(str) != b.sun_visibility_state.astype(str)).sum()),
        "earth_state_different_rows": int((a.earth_visibility_state.astype(str) != b.earth_visibility_state.astype(str)).sum()),
        "sun_fraction_different_rows": int((~np.isclose(a.sun_visible_fraction_approx.astype(float), b.sun_visible_fraction_approx.astype(float), atol=1e-9)).sum()),
        "earth_fraction_different_rows": int((~np.isclose(a.earth_visible_fraction_approx.astype(float), b.earth_visible_fraction_approx.astype(float), atol=1e-9)).sum()),
    }


def select_half_open_window(frame: pd.DataFrame, start_utc, duration_hours: int) -> pd.DataFrame:
    """Return samples in [start, start + duration), the app's canonical convention."""
    if not isinstance(duration_hours, (int, np.integer)) or isinstance(duration_hours, bool) or duration_hours <= 0:
        raise ValueError("duration_hours must be a positive integer")
    if "timestamp_utc" not in frame:
        raise ValueError("Timeline requires timestamp_utc")
    start = pd.Timestamp(start_utc)
    if start.tzinfo is None:
        raise ValueError("start_utc must be timezone-aware")
    start = start.tz_convert("UTC")
    end = start + pd.Timedelta(hours=int(duration_hours))
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    if timestamps.isna().any() or timestamps.empty:
        raise ValueError("Timeline contains invalid or missing timestamps")
    if end > timestamps.max():
        raise ValueError("Requested half-open window extends beyond computed timeline coverage")
    return frame.loc[(timestamps >= start) & (timestamps < end)].copy()


def build_delay_sweep(
    frame: pd.DataFrame,
    window_hours: int,
    step_hours: int | None = None,
    *,
    step_minutes: int | None = None,
    visibility_threshold: float = DISK_VISIBILITY_MATERIALITY_FRACTION,
) -> pd.DataFrame:
    """Sweep complete half-open windows on the source cadence.

    Set exactly one of step_hours or step_minutes. If neither is supplied,
    the default step is one hour. The last complete start is included when it
    aligns to the requested step.
    """
    required = {"timestamp_utc", "timestamp_ist", "sun_visible_fraction_approx",
                "earth_visible_fraction_approx", "sun_visibility_state", "earth_visibility_state",
                "sun_center_elevation_deg", "sun_terrain_horizon_deg",
                "sun_angular_semidiameter_deg", "earth_center_elevation_deg",
                "earth_terrain_horizon_deg", "earth_angular_semidiameter_deg"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"SPICE timeline missing columns: {sorted(missing)}")
    if not isinstance(window_hours, (int, np.integer)) or isinstance(window_hours, bool) or window_hours <= 0:
        raise ValueError("window_hours must be a positive integer")
    if step_hours is not None and step_minutes is not None:
        raise ValueError("Specify step_hours or step_minutes, not both")
    if step_minutes is None:
        if step_hours is None:
            step_minutes = 60
        else:
            if not isinstance(step_hours, (int, np.integer)) or isinstance(step_hours, bool) or step_hours <= 0:
                raise ValueError("step_hours must be a positive integer; fractional hours are not supported")
            step_minutes = int(step_hours) * 60
    if not isinstance(step_minutes, (int, np.integer)) or isinstance(step_minutes, bool) or step_minutes <= 0:
        raise ValueError("step_minutes must be a positive integer")
    if not np.isfinite(visibility_threshold) or not 0 <= visibility_threshold <= 1:
        raise ValueError("visibility_threshold must be in [0,1]")

    data = frame.sort_values("timestamp_utc").reset_index(drop=True)
    times = pd.to_datetime(data.timestamp_utc, utc=True, errors="coerce")
    if times.isna().any() or len(times) < 2:
        raise ValueError("Need at least two valid UTC timestamps")
    differences = times.diff().dropna().dt.total_seconds().to_numpy()
    if (differences <= 0).any() or not np.allclose(differences, differences[0], atol=1e-6):
        raise ValueError("Timestamps must be strictly increasing and regularly sampled")
    cadence_s = float(differences[0])
    window_s = int(window_hours) * 3600
    if not np.isclose(window_s / cadence_s, round(window_s / cadence_s), atol=1e-9):
        raise ValueError("Window duration must align to the source sample cadence")
    step_s = int(step_minutes) * 60
    if not np.isclose(step_s / cadence_s, round(step_s / cadence_s), atol=1e-9):
        raise ValueError("Sweep step must align to the source sample cadence")

    window_samples = int(round(window_s / cadence_s))
    step_samples = int(round(step_s / cadence_s))
    # The window end boundary must be <= the last computed timestamp.
    # With N+1 timestamps spanning N cadence intervals, the final complete
    # start is N - window_samples, not len(data) - window_samples.
    last_start = len(data) - 1 - window_samples
    if last_start < 0:
        raise ValueError("Timeline shorter than requested window")

    sun = _numeric(data, "sun_visible_fraction_approx")
    earth = _numeric(data, "earth_visible_fraction_approx")
    if ((sun < 0) | (sun > 1) | (earth < 0) | (earth > 1)).any():
        raise ValueError("Visibility fractions must be finite and in [0,1]")
    sun_el = _numeric(data, "sun_center_elevation_deg")
    sun_horizon = _numeric(data, "sun_terrain_horizon_deg")
    sun_semi = _numeric(data, "sun_angular_semidiameter_deg")
    earth_el = _numeric(data, "earth_center_elevation_deg")
    earth_horizon = _numeric(data, "earth_terrain_horizon_deg")
    earth_semi = _numeric(data, "earth_angular_semidiameter_deg")
    # Positive Sun lower-limb clearance means the complete solar disk clears
    # the sampled skyline. Positive Earth upper-limb clearance means some of
    # the apparent Earth disk reaches above the sampled skyline.
    sun_lower_limb_clearance = sun_el - sun_semi - sun_horizon
    earth_upper_limb_clearance = earth_el + earth_semi - earth_horizon
    sun_state = data.sun_visibility_state.astype(str).to_numpy()
    ist = pd.to_datetime(data.timestamp_ist, errors="coerce")
    if ist.isna().any():
        raise ValueError("Timeline contains invalid IST timestamps")
    expected_ist = times.dt.tz_convert("Asia/Kolkata")
    if not ist.equals(expected_ist):
        raise ValueError("timestamp_ist must equal timestamp_utc converted to Asia/Kolkata")

    starts = np.arange(0, last_start + 1, step_samples, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("No complete windows fit inside cached SPICE coverage")
    ends = starts + window_samples

    def window_means(values: np.ndarray) -> np.ndarray:
        prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        return (prefix[ends] - prefix[starts]) / window_samples

    def window_percent(flags: np.ndarray) -> np.ndarray:
        prefix = np.concatenate(([0], np.cumsum(flags.astype(np.int64))))
        return 100.0 * (prefix[ends] - prefix[starts]) / window_samples

    # Window-edge diagnostics expose changing geometry even when visibility
    # percentages are saturated (e.g. Sun always full, Earth always hidden).
    sun_min_clearance_all = pd.Series(sun_lower_limb_clearance).rolling(
        window_samples, min_periods=window_samples
    ).min().to_numpy(dtype=float)
    earth_max_clearance_all = pd.Series(earth_upper_limb_clearance).rolling(
        window_samples, min_periods=window_samples
    ).max().to_numpy(dtype=float)
    sun_mean_elevation = window_means(sun_el)
    earth_mean_elevation = window_means(earth_el)

    sun_visible = sun >= visibility_threshold
    earth_visible = earth >= visibility_threshold
    sun_mean = 100.0 * window_means(sun)
    earth_mean = 100.0 * window_means(earth)
    sun_any = window_percent(sun_visible)
    earth_any = window_percent(earth_visible)
    both_any = window_percent(sun_visible & earth_visible)
    sun_full = window_percent(sun_state == "full")
    delays_min = np.rint((times.iloc[starts] - times.iloc[0]).dt.total_seconds().to_numpy() / 60.0).astype(int)
    starts_ist = ist.iloc[starts].reset_index(drop=True)
    return pd.DataFrame({
        "delay_minutes": delays_min,
        "delay_hours": delays_min / 60.0,
        "window_start_ist": starts_ist,
        "window_end_ist": starts_ist + pd.to_timedelta(int(window_hours), unit="h"),
        "sample_count": np.full(starts.size, window_samples, dtype=int),
        "sun_mean_disk_fraction_pct": sun_mean,
        "sun_any_visible_pct": sun_any,
        "sun_full_visible_pct": sun_full,
        "earth_mean_disk_fraction_pct": earth_mean,
        "earth_any_visible_pct": earth_any,
        "both_any_visible_pct": both_any,
        "sun_mean_center_elevation_deg": sun_mean_elevation,
        "earth_mean_center_elevation_deg": earth_mean_elevation,
        "sun_min_lower_limb_clearance_deg": sun_min_clearance_all[ends - 1],
        "earth_max_upper_limb_clearance_deg": earth_max_clearance_all[ends - 1],
    })


def classify_objective(
    sweep: pd.DataFrame,
    column: str,
    *,
    materiality_pct: float = 100.0 * DISK_VISIBILITY_MATERIALITY_FRACTION,
    tolerance_pp: float = 0.01,
) -> dict:
    """Classify an objective stored on a 0..100 percentage scale.

    Below `materiality_pct` the objective is unachieved, not a tie. `tolerance_pp`
    is the equivalence tolerance in percentage points and is deliberately a
    separate parameter.
    """
    values = pd.to_numeric(sweep[column], errors="coerce").to_numpy(float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Objective values must be finite and non-empty")
    if not np.isfinite(materiality_pct) or materiality_pct < 0:
        raise ValueError("materiality_pct must be finite and non-negative")
    if not np.isfinite(tolerance_pp) or tolerance_pp < 0:
        raise ValueError("tolerance_pp must be finite and non-negative")
    best = float(values.max())
    if best < materiality_pct:
        return {"status": "unachieved", "best_value": best, "best_rows": sweep.iloc[0:0].copy()}
    mask = np.abs(values - best) <= tolerance_pp
    return {"status": "unique" if int(mask.sum()) == 1 else "tied",
            "best_value": best, "best_rows": sweep.loc[mask].copy()}


def longest_runs(
    frame: pd.DataFrame,
    column: str,
    *,
    threshold: float,
    mode: str,
    sample_minutes: int = 5,
) -> dict:
    """Find the longest sampled run; report interval span and sample-slot span.

    `elapsed_span_minutes=(n-1)*cadence` is the timestamp-to-timestamp lower
    bound. `sample_slot_minutes=n*cadence` describes n cadence-sized sample
    slots. Both are returned to avoid silently conflating the conventions.
    """
    values = _numeric(frame, column)
    if values.size == 0:
        raise ValueError("Run input must be non-empty")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if mode not in {"below", "at_or_above"}:
        raise ValueError("mode must be below or at_or_above")
    if not isinstance(sample_minutes, (int, np.integer)) or isinstance(sample_minutes, bool) or sample_minutes <= 0:
        raise ValueError("sample_minutes must be a positive integer")
    flags = values < threshold if mode == "below" else values >= threshold
    best = current = 0
    best_start = None
    current_start = None
    for idx, flag in enumerate(flags):
        if flag and current == 0:
            current_start = idx
        current = current + 1 if flag else 0
        if current > best:
            best = current
            best_start = current_start
    elapsed = max(0, best - 1) * int(sample_minutes)
    slots = best * int(sample_minutes)
    return {
        "samples": int(best),
        "minutes": int(elapsed),
        "elapsed_span_minutes": int(elapsed),
        "sample_slot_minutes": int(slots),
        "start_index": best_start,
        "left_censored": bool(best_start == 0) if best_start is not None else False,
        "right_censored": bool(best_start is not None and best_start + best == len(flags)),
    }


def _disk_fraction_above_horizon(center_elevation_deg, horizon_elevation_deg, semidiameter_deg):
    center = np.asarray(center_elevation_deg, dtype=float)
    horizon = np.asarray(horizon_elevation_deg, dtype=float)
    semi = np.asarray(semidiameter_deg, dtype=float)
    if not (np.isfinite(center).all() and np.isfinite(horizon).all() and np.isfinite(semi).all()):
        raise ValueError("Disk geometry contains non-finite values")
    if (semi < 0).any():
        raise ValueError("Angular semidiameter cannot be negative")
    gap = center - horizon
    result = np.zeros(np.broadcast_shapes(gap.shape, semi.shape), dtype=float)
    gap, semi = np.broadcast_arrays(gap, semi)
    zero = semi == 0
    result[zero] = (gap[zero] > 0).astype(float)
    nonzero = ~zero
    z = np.zeros_like(gap)
    z[nonzero] = gap[nonzero] / semi[nonzero]
    result[nonzero & (z >= 1)] = 1.0
    middle = nonzero & (z > -1) & (z < 1)
    zm = z[middle]
    result[middle] = (np.arccos(-zm) + zm * np.sqrt(np.maximum(0.0, 1.0 - zm * zm))) / np.pi
    return result


def compare_timeline_to_flat_horizon(
    frame: pd.DataFrame,
    *,
    visibility_threshold: float = DISK_VISIBILITY_MATERIALITY_FRACTION,
) -> dict[str, dict[str, float | int | str]]:
    """Diagnose whether sampled terrain changes disk visibility vs a 0° horizon.

    This is a sensitivity diagnostic for the cached interval, not proof that
    terrain is irrelevant outside that interval.
    """
    if not 0 <= visibility_threshold <= 1:
        raise ValueError("visibility_threshold must be in [0,1]")
    result = {}
    for body in ("sun", "earth"):
        el = _numeric(frame, f"{body}_center_elevation_deg")
        horizon = _numeric(frame, f"{body}_terrain_horizon_deg")
        semi = _numeric(frame, f"{body}_angular_semidiameter_deg")
        actual = _numeric(frame, f"{body}_visible_fraction_approx")
        flat = _disk_fraction_above_horizon(el, np.zeros_like(el), semi)
        actual_visible = actual >= visibility_threshold
        flat_visible = flat >= visibility_threshold
        lower_limb_clearance = el - semi - horizon
        upper_limb_clearance = el + semi - horizon
        if np.all(lower_limb_clearance >= 0):
            state = "entire disk above sampled terrain horizon throughout interval"
        elif np.all(upper_limb_clearance <= 0):
            state = "entire disk below sampled terrain horizon throughout interval"
        else:
            state = "disk intersects sampled terrain horizon during interval"
        result[body] = {
            "samples": int(len(frame)),
            "terrain_vs_flat_classification_changed_rows": int(np.count_nonzero(actual_visible != flat_visible)),
            "terrain_vs_flat_fraction_changed_rows": int(np.count_nonzero(~np.isclose(actual, flat, atol=1e-6, rtol=0))),
            "mean_terrain_minus_flat_disk_fraction_pp": float(100.0 * np.mean(actual - flat)),
            "minimum_lower_limb_clearance_deg": float(np.min(lower_limb_clearance)),
            "maximum_upper_limb_clearance_deg": float(np.max(upper_limb_clearance)),
            "terrain_geometry_state": state,
        }
    return result
