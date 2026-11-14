#!/usr/bin/env python3
"""Regression check for CSV-rounded cached azimuth bin centres."""
import numpy as np

step = 0.1
n = int(round(360.0 / step))
expected = (np.arange(n, dtype=np.float64) + 0.5) * step
# The local profile CSV is written with fmt=%.6f.
cached = np.round(expected, 6)
farfield = expected.copy()
assert np.array_equal(cached, farfield) is False
assert np.allclose(cached, expected, rtol=0.0, atol=1.0e-6)
assert np.allclose(farfield, expected, rtol=0.0, atol=1.0e-6)
# A real 0.01-degree bin error must still be rejected.
wrong = farfield.copy()
wrong[10] += 0.01
assert not np.allclose(wrong, expected, rtol=0.0, atol=1.0e-6)
print(f"PASS: {n} bins; rounded-cache max error="
      f"{np.max(np.abs(cached - expected)):.3g} deg; material mismatch rejected")
