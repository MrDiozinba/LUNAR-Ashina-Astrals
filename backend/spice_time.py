"""UTC datetime formatting helpers for CSPICE."""
from __future__ import annotations

from datetime import datetime, timezone


def format_spice_utc(dt: datetime) -> str:
    """Format an aware datetime in the calendar form accepted by CSPICE STR2ET.

    CSPICE rejects the mixed form ``YYYY-MM-DDTHH:MM:SS UTC``. This helper
    emits ``YYYY MON DD HH:MM:SS[.fraction] UTC`` after converting to UTC.
    """
    if dt.tzinfo is None:
        raise ValueError("SPICE UTC conversion requires a timezone-aware datetime")
    utc_dt = dt.astimezone(timezone.utc)
    value = utc_dt.strftime("%Y %b %d %H:%M:%S")
    if utc_dt.microsecond:
        value += "." + f"{utc_dt.microsecond:06d}".rstrip("0")
    return f"{value} UTC".upper()
