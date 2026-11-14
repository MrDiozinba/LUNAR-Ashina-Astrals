# LUNAR — Lunar Environment Intelligence and Mission Planning System

**Team:** Ashina Astrals  
**Event:** 2026 NASA Space Apps Challenge  
**Status:** Independent student-built scientific prototype; not endorsed by NASA.

LUNAR is a terrain-aware decision-support prototype for examining apparent solar-disk visibility and geometric Earth-disk visibility at a provisional lunar South Pole surface point. It combines NASA LOLA multi-resolution terrain products, a 3,600-direction horizon profile, DE440-based NAIF SPICE ephemerides, and independent JPL Horizons comparisons in a Streamlit dashboard.

## Current demonstration scope

- One provisional Site01 tile-centre test point (not a verified landing coordinate).
- Observer heights of 1 m and 5 m above the local DEM surface.
- 5-minute SPICE sampling over a 48-hour interval (577 observations per observer height).
- Multi-resolution terrain horizon sampled to 45 km. The horizon is **not globally closed**; omitted terrain beyond 45 km could raise the true skyline.
- Earth visibility is geometric disk visibility, not DSN/network link availability.
- Sun visibility is geometric disk visibility, not solar-panel power prediction.

## Run the dashboard

Python 3.10+ is recommended. The dashboard uses precomputed scientific CSV/JSON outputs included in this repository; the large source DEM rasters and SPICE kernel binaries are not included.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Run the release audit suite

```bash
source .venv/bin/activate
python scripts/run_v6_audit_tests.py
python -m py_compile app.py backend/*.py scripts/*.py
```

The test log included in the repository records the team's 44-test run. Re-run the commands above after cloning; the current environment and dependency versions may differ.

## Reproducing the scientific calculations

The repository includes cached outputs and provenance, not the full input DEMs or kernel files. To regenerate results, obtain the precise NASA PGDA LOLA products and NAIF SPICE kernels listed in the metadata and cited resources, follow their own data-use terms, and place them in the expected project data locations. Install `requirements-spice.txt` for the SPICE calculation layer. Terrain regeneration additionally requires compatible Rasterio/GDAL/PROJ installations and the source DEM tiles.

## Key repository structure

```text
app.py
backend/                 dashboard data, geometry and decision-support modules
scripts/                 explicit release audit runner
tests/                   geometry, data-integrity and UI regression tests
data/horizon_masks/      1 m and 5 m 3,600-bin horizon profiles + metadata
data/spice/              1 m and 5 m SPICE timelines + metadata + SHA-256 manifest
data/horizons_validation/ independent JPL Horizons comparison report
docs/                    project summary, description, references, AI disclosure,
                         release notes, checklist and 7-slide demo deck
```

## Independent ephemeris validation

The saved JPL Horizons comparison covers Sun and Earth at 1 m and 5 m observer heights, each with 577 aligned observations. The maximum reported elevation residual was approximately 0.000024° (Earth); azimuth residuals were at or below approximately 0.000001°. This validates apparent ephemeris directions for the tested observer and interval; it does **not** validate terrain obstruction or globally close the horizon.

## Data and attribution

Primary resources used in the computation:

- NASA PGDA Site01 LOLA product: https://pgda.gsfc.nasa.gov/products/78
- NASA PGDA south-polar regional LOLA products: https://pgda.gsfc.nasa.gov/products/90
- NAIF SPICE kernels and documentation: https://naif.jpl.nasa.gov/naif/data.html
- JPL Horizons and API documentation: https://ssd.jpl.nasa.gov/horizons/ and https://ssd-api.jpl.nasa.gov/doc/horizons.html

Other software: Python, NumPy, Pandas, Rasterio/GDAL, PROJ, SpiceyPy, Streamlit and Plotly. See `docs/03_DATA_AND_RESOURCE_REFERENCES.txt` for fuller references and usage cautions.

LROC QuickMap and NASA Moon Trek were explored as reference tools; they are not claimed as computational data inputs in the current bundle. CSA resources are contextual references, not computational inputs.

## AI/SI disclosure

ChatGPT assisted with software architecture, code drafting, debugging, tests, scientific-method review, data interpretation, documentation and presentation wording. Claude provided independent technical reviews. The team ran the software, acquired/processed scientific inputs, inspected outputs, ran tests and performed the JPL Horizons comparison. See `docs/04_AI_AND_SI_DISCLOSURE.txt`.

## Limitations

This is a reproducible prototype, not an operational lunar mission tool or landing-site certification. It currently has one provisional test point, a 45 km range-limited regional horizon, a 48-hour ephemeris interval, incomplete DEM uncertainty propagation, no comprehensive local slope/hazard model, no DSN link budget and no solar power-system model. Do not interpret the current 100% Sun / 0% Earth result as universal for the lunar South Pole.

NASA/JPL/CSA resources are cited as data and reference sources. This project is independently developed and is not endorsed by NASA, JPL or CSA. Follow product-specific usage terms before redistributing source data or imagery.
