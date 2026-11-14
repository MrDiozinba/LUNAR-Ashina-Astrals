from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.dashboard_data import load_scientific_bundle, summarize_timeline, snapshot_slider_bounds
from backend.decision_support import (
    build_delay_sweep, compare_horizon_profiles, summarize_horizon_sectors,
    compare_timeline_visibility, classify_objective, sample_horizon_difference_at_azimuths,
    select_half_open_window, longest_runs, compare_timeline_to_flat_horizon,
)
from backend.scientific_constants import DISK_VISIBILITY_MATERIALITY_FRACTION

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="LUNAR | Mission Environment Explorer",
    page_icon="🌘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#080D18; --panel:#101A2B; --line:rgba(148,166,196,.16); --muted:#91A2B9; --text:#EDF3FC; --cyan:#65D7E8; --gold:#F2B66D; }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp { background: radial-gradient(ellipse at 78% -10%, rgba(47,91,143,.22), transparent 36%), var(--bg); color:var(--text); }
.block-container { padding-top:1.6rem; padding-bottom:3rem; max-width:1500px; }
[data-testid="stSidebar"] { background:#0B1322; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#DCE5F2; }
.hero { padding:1.45rem 1.6rem; border:1px solid var(--line); border-radius:18px; background:linear-gradient(115deg,rgba(24,43,69,.96),rgba(13,23,39,.9)); margin-bottom:1.15rem; position:relative; overflow:hidden; }
.hero:after { content:''; position:absolute; right:4.2rem; top:50%; transform:translateY(-50%); width:142px; height:142px; border:1px solid rgba(101,215,232,.16); border-radius:50%; background:radial-gradient(circle at 38% 36%, rgba(101,215,232,.17), rgba(101,215,232,.035) 42%, transparent 69%); box-shadow:inset 0 0 34px rgba(101,215,232,.06), 0 0 40px rgba(101,215,232,.035); pointer-events:none; }
.hero:before { content:''; position:absolute; right:calc(4.2rem + 50px); top:calc(50% - 50px); width:42px; height:42px; border-radius:50%; background:linear-gradient(145deg, rgba(214,237,246,.26), rgba(80,127,158,.05)); box-shadow:0 0 28px rgba(101,215,232,.08); pointer-events:none; }
.eyebrow { color:var(--cyan); font:500 11px 'DM Mono',monospace; letter-spacing:2px; text-transform:uppercase; }
.hero h1 { color:#F5F8FD; font-size:clamp(27px,3vw,42px); line-height:1.12; margin:.45rem 0 .55rem; font-weight:800; }
.hero p { color:#AEBED2; max-width:850px; margin:0; font-size:14px; line-height:1.65; position:relative; z-index:1; }
.hero .eyebrow, .hero h1 { position:relative; z-index:1; }
.hero h1, .hero p { max-width:calc(100% - 165px); }
[data-testid="stSidebar"] div[data-baseweb="select"] > div { min-height:40px; background:#111D2E; border-color:rgba(148,166,196,.25); }
[data-testid="stSidebar"] div[data-baseweb="select"] span { font-size:12px; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#B8C7D9; font-size:12px; font-weight:600; }
[data-testid="stSidebar"] .stCaption { color:#879AB2; font-size:11px; }
@media (max-width: 900px) { .hero:before, .hero:after { display:none; } .hero h1, .hero p { max-width:100%; } .block-container { padding-left:1rem; padding-right:1rem; } }
.section-kicker { color:var(--cyan); font:500 10px 'DM Mono',monospace; letter-spacing:1.7px; text-transform:uppercase; margin:1.2rem 0 .4rem; }
.section-title { color:#EDF3FC; font-size:21px; font-weight:800; margin:0 0 .8rem; }
.metric-card { background:linear-gradient(145deg,#142238,#0F1929); border:1px solid var(--line); border-radius:14px; padding:15px 17px; min-height:112px; }
.metric-card.blue { border-top:2px solid #65D7E8; }
.metric-card.gold { border-top:2px solid #F2B66D; }
.metric-card.violet { border-top:2px solid #A99BFF; }
.metric-card.green { border-top:2px solid #70D6A1; }
.metric-label { color:#9FB0C7; font-size:11px; text-transform:uppercase; letter-spacing:1.1px; font-weight:700; }
.metric-value { color:#F3F7FD; font-size:29px; font-weight:800; margin:5px 0 1px; }
.metric-note { color:#8294AD; font-size:10px; }
.info-box { border:1px solid rgba(101,215,232,.2); background:rgba(34,89,111,.12); padding:12px 14px; border-radius:10px; color:#B9CADB; font-size:12px; line-height:1.6; }
.warning-box { border:1px solid rgba(242,182,109,.32); background:rgba(126,79,25,.13); padding:12px 14px; border-radius:10px; color:#E9C99F; font-size:12px; line-height:1.6; }
.site-chip { display:inline-block; background:rgba(101,215,232,.1); border:1px solid rgba(101,215,232,.2); border-radius:20px; color:#8CE6F0; padding:4px 9px; font:11px 'DM Mono',monospace; }
.small-muted { color:#91A2B9; font-size:11px; }
.chart-heading { color:#EDF3FC; font-size:14px; font-weight:800; margin:0 0 .45rem; }
.chart-legend { display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin:0 0 .35rem; color:#AEBED2; font-size:10px; }
.chart-legend span { display:inline-flex; align-items:center; gap:6px; }
.legend-line { display:inline-block; width:18px; border-top:2px solid currentColor; }
.legend-dash { border-top-style:dashed; }
.legend-dot { border-top-style:dotted; }
.sky-key { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:0 0 .35rem; color:#AEBED2; font-size:10px; }
.sky-key .dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px; }
.sky-key .diamond { width:8px; height:8px; display:inline-block; transform:rotate(45deg); margin:0 7px 0 2px; }
div[data-testid="stTabs"] button { color:#AEBED2; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#65D7E8; }
.stButton button, .stDownloadButton button { border-radius:9px; }
hr { border-color:var(--line); }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_bundle(root_string: str):
    return load_scientific_bundle(root_string)


@st.cache_data(show_spinner=False)
def get_delay_sweep_cached(frame: pd.DataFrame, window_hours: int, step_minutes: int):
    return build_delay_sweep(frame, window_hours=window_hours, step_minutes=step_minutes)


def card(label: str, value: str, note: str, accent: str = "blue") -> None:
    st.markdown(
        f"""<div class="metric-card {accent}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def style_chart(
    fig: go.Figure,
    height: int = 340,
    *,
    show_legend: bool = False,
    top_margin: int | None = None,
) -> go.Figure:
    """Apply the shared dark theme while reserving separate rows for title and legend."""
    title_text = fig.layout.title.text if fig.layout.title else None
    if top_margin is None:
        top_margin = 90 if show_legend else 66
    if show_legend:
        top_margin = max(top_margin, 90)
    fig.update_layout(
        height=height,
        margin=dict(l=16, r=16, t=top_margin, b=54),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color="#DCE5F2", size=11),
        showlegend=show_legend,
        hovermode="x unified",
        title=(dict(text=title_text, x=0, xanchor="left", y=0.985, yanchor="top",
                    font=dict(size=14, color="#EDF3FC")) if title_text else None),
    )
    if show_legend:
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=9, color="#AEBED2"),
            bgcolor="rgba(0,0,0,0)",
        ))
    fig.update_xaxes(showgrid=False, linecolor="rgba(160,180,210,.18)", tickfont=dict(color="#91A2B9"))
    fig.update_yaxes(gridcolor="rgba(160,180,210,.12)", zerolinecolor="rgba(160,180,210,.25)", tickfont=dict(color="#91A2B9"))
    return fig


def elevation_chart(frame: pd.DataFrame, body: str, body_color: str) -> go.Figure:
    prefix = body.lower()
    elevation = f"{prefix}_center_elevation_deg"
    horizon = f"{prefix}_terrain_horizon_deg"
    semidiameter = f"{prefix}_angular_semidiameter_deg"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame["timestamp_ist"], y=frame[elevation], mode="lines", name="Centre elevation",
        line=dict(color=body_color, width=2.5),
        customdata=np.column_stack([frame["utc"], frame[f"{prefix}_visibility_state"]]),
        hovertemplate="%{x|%d %b %H:%M IST}<br>Centre: %{y:.3f}°<br>UTC: %{customdata[0]}<br>Disk state: %{customdata[1]}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=frame["timestamp_ist"], y=frame[horizon], mode="lines", name="Terrain horizon · regional lower bound",
        line=dict(color="#F2B66D", width=1.6, dash="dot"),
        hovertemplate="Terrain horizon: %{y:.3f}°<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=frame["timestamp_ist"], y=frame[elevation] + frame[semidiameter], mode="lines", name="Upper limb",
        line=dict(color=body_color, width=1, dash="dash"), opacity=0.55, showlegend=False,
        hovertemplate="Upper limb: %{y:.3f}°<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=frame["timestamp_ist"], y=frame[elevation] - frame[semidiameter], mode="lines", name="Lower limb",
        line=dict(color=body_color, width=1, dash="dash"), opacity=0.55, showlegend=False,
        hovertemplate="Lower limb: %{y:.3f}°<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,.2)")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Time (India Standard Time)")
    fig.update_yaxes(title="Elevation (degrees)")
    return style_chart(fig, height=330, show_legend=False, top_margin=26)


def render_elevation_chart(frame: pd.DataFrame, body: str, body_color: str) -> None:
    """Render chart title and legend outside the plotting canvas to prevent collisions."""
    st.markdown(
        f"""<div class='chart-heading'>{body} elevation · terrain clearance</div>
        <div class='chart-legend'>
          <span style='color:{body_color}'><i class='legend-line'></i>Centre elevation</span>
          <span style='color:#F2B66D'><i class='legend-line legend-dot'></i>Terrain horizon (regional lower bound)</span>
          <span style='color:{body_color}'><i class='legend-line legend-dash'></i>Apparent disk limbs</span>
        </div>""", unsafe_allow_html=True
    )
    st.plotly_chart(elevation_chart(frame, body, body_color), width="stretch")


def render_figure_with_legend(fig: go.Figure, height: int = 340) -> None:
    """Move a multi-series chart title and legend into dedicated Streamlit rows."""
    title_text = fig.layout.title.text if fig.layout.title else None
    fig.update_layout(title=None)
    if title_text:
        st.markdown(f"<div class='chart-heading'>{title_text}</div>", unsafe_allow_html=True)
    styled = style_chart(fig, height=height, show_legend=True, top_margin=90)
    st.plotly_chart(styled, width="stretch")


def fraction_chart(frame: pd.DataFrame, body: str, color: str) -> go.Figure:
    col = f"{body.lower()}_visible_fraction_approx"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame["timestamp_ist"], y=100.0 * frame[col], mode="lines", name=f"{body} disk visible fraction",
        line=dict(color=color, width=2.2), fill="tozeroy", fillcolor="rgba(101,215,232,.08)",
        hovertemplate="%{x|%d %b %H:%M IST}<br>Visible disk fraction: %{y:.3f}%<extra></extra>",
    ))
    fig.update_yaxes(title="Apparent disk visible (%)", range=[0, 100])
    fig.update_xaxes(title="Time (India Standard Time)")
    fig.update_layout(title=dict(text=f"{body} apparent disk visibility"), showlegend=False)
    return style_chart(fig, 280, show_legend=False, top_margin=62)


def sky_dome_chart(row: pd.Series, horizon: pd.DataFrame) -> go.Figure:
    az = horizon["azimuth_center_deg"].to_numpy(dtype=float)
    horizon_el = horizon["combined_horizon_elevation_deg"].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=90.0 - horizon_el, theta=az, mode="lines", name="Terrain skyline",
        line=dict(color="#F2B66D", width=2),
        customdata=horizon_el,
        hovertemplate="Azimuth: %{theta:.1f}°<br>Terrain horizon: %{customdata:.3f}°<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[90.0 - float(row["sun_center_elevation_deg"])],
        theta=[float(row["sun_azimuth_deg"])], mode="markers+text", text=["SUN"], textposition="top center",
        name="Sun", marker=dict(size=13, color="#65D7E8", symbol="circle"),
        hovertemplate=f"Sun<br>Azimuth: {float(row['sun_azimuth_deg']):.2f}°<br>Elevation: {float(row['sun_center_elevation_deg']):.3f}°<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[90.0 - float(row["earth_center_elevation_deg"])],
        theta=[float(row["earth_azimuth_deg"])], mode="markers+text", text=["EARTH"], textposition="top center",
        name="Earth", marker=dict(size=13, color="#A99BFF", symbol="diamond"),
        hovertemplate=f"Earth<br>Azimuth: {float(row['earth_azimuth_deg']):.2f}°<br>Elevation: {float(row['earth_center_elevation_deg']):.3f}°<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Local sky dome · azimuth-dependent terrain skyline"),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], tickvals=[0, 30, 60, 90, 100],
                            ticktext=["Zenith", "60°", "30°", "0°", "−10°"],
                            gridcolor="rgba(160,180,210,.16)", color="#91A2B9", angle=24,
                            tickfont=dict(size=9)),
            angularaxis=dict(direction="clockwise", rotation=90,
                             tickmode="array", tickvals=[0,45,90,135,180,225,270,315],
                             ticktext=["N","NE","E","SE","S","SW","W","NW"],
                             gridcolor="rgba(160,180,210,.16)", color="#91A2B9"),
        ),
        showlegend=False,
    )
    return style_chart(fig, 420, show_legend=False, top_margin=44)


def availability_band_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    series = [
        ("Sun disk visible", "sun_visible", "#65D7E8", 3),
        ("Geometric Earth disk visible", "earth_visible", "#F2B66D", 2),
        ("Both simultaneously", "both_visible", "#70D6A1", 1),
    ]
    step = frame["timestamp_ist"].iloc[1] - frame["timestamp_ist"].iloc[0] if len(frame) > 1 else pd.Timedelta(minutes=5)
    for label, column, color, y in series:
        values = frame[column].to_numpy(dtype=bool)
        start = None
        for index, visible in enumerate(values):
            if visible and start is None:
                start = index
            is_last = index == len(values) - 1
            if start is not None and (not visible or is_last):
                end = index if (visible and is_last) else index - 1
                x0 = frame.iloc[start]["timestamp_ist"]
                x1 = frame.iloc[end]["timestamp_ist"] + step
                fig.add_shape(type="rect", x0=x0, x1=x1, y0=y - 0.27, y1=y + 0.27,
                              line=dict(width=0), fillcolor=color, opacity=0.88)
                start = None
    fig.add_trace(go.Scatter(
        x=[frame["timestamp_ist"].iloc[0], frame["timestamp_ist"].iloc[-1]], y=[0.5, 3.5],
        mode="markers", marker=dict(size=1, opacity=0), showlegend=False, hoverinfo="skip",
    ))
    fig.update_yaxes(tickmode="array", tickvals=[1,2,3],
                     ticktext=["Sun + Earth", "Earth visible", "Sun visible"], range=[0.5,3.5])
    fig.update_xaxes(type="date", range=[frame["timestamp_ist"].iloc[0], frame["timestamp_ist"].iloc[-1]],
                     title="Time (India Standard Time)", showgrid=False, tickformat="%d %b\n%H:%M")
    fig.update_layout(title=dict(text="Computed geometric visibility windows", x=0, xanchor="left", y=0.985, yanchor="top", font=dict(size=14, color="#EDF3FC")),
                      height=235, showlegend=False, margin=dict(l=16,r=16,t=58,b=48),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Manrope, sans-serif", color="#DCE5F2", size=10))
    return fig


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def summarize_timeline_material(
    frame: pd.DataFrame, threshold: float = DISK_VISIBILITY_MATERIALITY_FRACTION
) -> dict[str, float | int]:
    """Summarize visibility using a 1% apparent-disk materiality threshold."""
    sun = pd.to_numeric(frame["sun_visible_fraction_approx"], errors="coerce").to_numpy(float)
    earth = pd.to_numeric(frame["earth_visible_fraction_approx"], errors="coerce").to_numpy(float)
    if len(frame) == 0 or not np.isfinite(sun).all() or not np.isfinite(earth).all():
        raise ValueError("Timeline visibility fractions must be finite and non-empty")
    return {
        "samples": int(len(frame)),
        "sun_any_visible_pct": float(100 * np.mean(sun >= threshold)),
        "sun_full_visible_pct": float(100 * np.mean(frame["sun_visibility_state"].astype(str).to_numpy() == "full")),
        "earth_any_visible_pct": float(100 * np.mean(earth >= threshold)),
        "both_any_visible_pct": float(100 * np.mean((sun >= threshold) & (earth >= threshold))),
        "sun_mean_disk_fraction_pct": float(100 * sun.mean()),
        "earth_mean_disk_fraction_pct": float(100 * earth.mean()),
    }


def daily_window(frame: pd.DataFrame, start: pd.Timestamp, hours: int = 24) -> pd.DataFrame:
    return select_half_open_window(frame, start, hours)


try:
    bundle = get_bundle(str(ROOT))
except Exception as exc:
    st.error("Scientific dashboard data could not be loaded or failed validation.")
    st.exception(exc)
    st.stop()

spice_meta = bundle["spice_metadata"]
observer = bundle["observer"]
timelines = bundle["timelines"]
horizons = bundle["horizons"]
full_reference = timelines[1]
local_start = full_reference["timestamp_ist"].iloc[0]
local_end = full_reference["timestamp_ist"].iloc[-1]
local_dates = sorted(full_reference["timestamp_ist"].dt.date.unique().tolist())

with st.sidebar:
    st.markdown('<div class="eyebrow">MISSION CONTROL / 01</div>', unsafe_allow_html=True)
    st.markdown("## LUNAR")
    st.caption("South-pole environment explorer")
    page = st.radio(
        "NAVIGATION",
        ["Mission overview", "Site explorer", "Environment timeline", "Site comparison", "Mission delay simulator"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**SCIENTIFIC SCENARIO**")
    st.selectbox("Available terrain site", ["Site01 · Connecting Ridge"], index=0, disabled=True)
    st.caption("Single provisional tile-centre test point")
    observer_height = st.selectbox("Observer height above DEM", [1, 5], index=0, format_func=lambda value: f"{value} m")
    scope = st.radio("Timeline scope", ["Full computed window", "Selected IST date"], index=0)
    selected_date = st.date_input(
        "IST date", value=local_dates[0], min_value=local_dates[0], max_value=local_dates[-1],
        disabled=(scope == "Full computed window"),
    )
    st.divider()
    st.markdown("**DATA STATUS**")
    st.markdown('<span class="site-chip">● SCIENTIFIC OUTPUTS LOADED</span>', unsafe_allow_html=True)
    st.caption("SPICE DE440 + NASA LOLA terrain · provisional regional model")

full_df = timelines[observer_height]
if scope == "Selected IST date":
    df = full_df[full_df["timestamp_ist"].dt.date == selected_date].copy()
else:
    df = full_df.copy()
if df.empty:
    st.error("No samples exist for the selected IST date in the computed SPICE window.")
    st.stop()
df = df.reset_index(drop=True)
summary = summarize_timeline_material(df)
horizon_df = horizons[observer_height]

st.markdown(f"""
<div class="hero">
  <div class="eyebrow">LUNAR SOUTH POLE · MISSION ENVIRONMENT INTELLIGENCE</div>
  <h1>See the window.<br>Understand the trade-off.</h1>
  <p>Explore terrain-obstructed Sun and geometric Earth visibility using the current NASA LOLA multi-resolution horizon profiles and DE440 SPICE ephemerides. This build contains one provisional Site01 test point and a computed 48-hour interval.</p>
</div>
""", unsafe_allow_html=True)

if page == "Mission overview":
    st.markdown('<div class="section-kicker">01 / Mission snapshot</div><div class="section-title">Computed environment at a glance</div>', unsafe_allow_html=True)
    st.markdown('<span class="site-chip">SITE_01 · PROVISIONAL TILE-CENTRE TEST POINT</span>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Sun disk visible", format_pct(summary["sun_any_visible_pct"]),
             f"Full disk in {summary['sun_full_visible_pct']:.1f}% of samples", "blue")
    with c2:
        card("Earth disk visible", format_pct(summary["earth_any_visible_pct"]),
             "Geometric visibility, not radio-link availability", "gold")
    with c3:
        card("Sun + Earth overlap", format_pct(summary["both_any_visible_pct"]),
             "Both apparent disks have some visible area", "green")
    with c4:
        span_hours = (df["timestamp_utc"].iloc[-1] - df["timestamp_utc"].iloc[0]).total_seconds() / 3600.0
        card("Computed interval", f"{span_hours:.1f} h", f"{summary['samples']} samples · 5-minute cadence", "violet")

    terrain_sensitivity = compare_timeline_to_flat_horizon(df)
    changed_sun = terrain_sensitivity["sun"]["terrain_vs_flat_classification_changed_rows"]
    changed_earth = terrain_sensitivity["earth"]["terrain_vs_flat_classification_changed_rows"]
    st.info(
        f"Terrain-sensitivity check for this cached interval: replacing the sampled terrain skyline "
        f"with a flat 0° horizon changes the 1% disk-visibility classification in "
        f"{changed_sun} Sun rows and {changed_earth} Earth rows. "
        "This interval therefore does not demonstrate terrain-driven visibility transitions."
    )

    st.markdown('<div class="section-kicker">02 / Environmental trends</div><div class="section-title">Celestial geometry and apparent-disk visibility</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        render_elevation_chart(df, "Sun", "#65D7E8")
    with right:
        render_elevation_chart(df, "Earth", "#F2B66D")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(fraction_chart(df, "Sun", "#65D7E8"), width="stretch")
    with right:
        st.plotly_chart(fraction_chart(df, "Earth", "#F2B66D"), width="stretch")

    st.markdown('<div class="section-kicker">03 / Local sky geometry</div><div class="section-title">Where are the Sun and Earth relative to the terrain skyline?</div>', unsafe_allow_html=True)
    snapshot_bounds = snapshot_slider_bounds(len(df))
    if snapshot_bounds is not None:
        sample_index = st.slider(
            "Move through the selected timeline", snapshot_bounds[0], snapshot_bounds[1],
            min(snapshot_bounds[1], len(df) // 2), step=1, key="overview_sky_time"
        )
    else:
        sample_index = 0
        st.caption("Only one sample is available on this final IST date; the snapshot control is disabled.")
    sky_col, sky_note = st.columns([1.05, 1])
    with sky_col:
        st.markdown(
            "<div class='sky-key'><span><i class='legend-line legend-dot' style='color:#F2B66D'></i>Terrain skyline</span>"
            "<span><i class='dot' style='background:#65D7E8'></i>Sun centre</span>"
            "<span><i class='diamond' style='background:#A99BFF'></i>Earth centre</span></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(sky_dome_chart(df.iloc[sample_index], horizon_df), width="stretch")
    with sky_note:
        selected_row = df.iloc[sample_index]
        st.markdown(f"**Snapshot:** {selected_row['timestamp_ist']:%d %b %Y, %H:%M IST}")
        st.metric("Sun centre", f"{selected_row['sun_center_elevation_deg']:.3f}° elevation",
                  f"Azimuth {selected_row['sun_azimuth_deg']:.2f}° · {selected_row['sun_visibility_state']}")
        st.metric("Earth centre", f"{selected_row['earth_center_elevation_deg']:.3f}° elevation",
                  f"Azimuth {selected_row['earth_azimuth_deg']:.2f}° · {selected_row['earth_visibility_state']}")
        st.markdown('<div class="info-box">The skyline is the 3,600-bin, azimuth-dependent multi-resolution terrain horizon. Body markers show apparent disk centres; visibility fractions account approximately for finite disk size using a locally straight horizon at the disk-centre azimuth.</div>', unsafe_allow_html=True)

elif page == "Site explorer":
    st.markdown('<div class="section-kicker">02 / Site intelligence</div><div class="section-title">Site01 terrain and coordinate record</div>', unsafe_allow_html=True)
    lat = float(observer.get("latitude_planetocentric_deg", -89.463162834037))
    lon = float(observer.get("longitude_east_deg", 222.51044707800085))
    ground_z = float(observer.get("ground_elevation_from_local_5m_dem_m", 1944.8283386230469))
    a, b = st.columns([1.1, 1])
    with a:
        st.markdown("### Connecting Ridge · Site01")
        st.markdown('<span class="site-chip">PGDA SITE01 · TEST POINT ONLY</span>', unsafe_allow_html=True)
        st.write("The current observer is the geometric centre of the supplied Site01 local DEM tile. It is not a surveyed lander coordinate or an endorsed landing target.")
        st.write("**Terrain sources:** 5 m local LOLA DEM + regional 80 m LOLA DEM. The regional 20 m crop was used for cross-resolution checks.")
    with b:
        st.markdown("### Coordinate record")
        st.metric("Planetocentric latitude", f"{lat:.6f}°")
        st.metric("East longitude", f"{lon:.6f}°")
        st.metric("Local DEM surface Z", f"{ground_z:,.3f} m")
        st.caption(f"Observer height: {observer_height} m above the local DEM surface")

    st.markdown('<div class="section-kicker">Terrain and visibility</div><div class="section-title">Multi-resolution horizon profile</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=horizon_df["azimuth_center_deg"], y=horizon_df["combined_horizon_elevation_deg"],
        mode="lines", name="Combined 5 m + 80 m horizon", line=dict(color="#65D7E8", width=1.8),
        hovertemplate="Azimuth: %{x:.2f}°<br>Horizon: %{y:.3f}°<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,.25)")
    fig.update_layout(title="Terrain horizon by local azimuth", xaxis_title="Azimuth (degrees clockwise from north)",
                      yaxis_title="Horizon elevation (degrees)")
    st.plotly_chart(style_chart(fig, 390, show_legend=False, top_margin=66), width="stretch")
    h_stats = horizon_df["combined_horizon_elevation_deg"]
    h1, h2, h3 = st.columns(3)
    with h1:
        card("Minimum horizon", f"{h_stats.min():.2f}°", "Lowest sampled azimuth bin", "blue")
    with h2:
        card("Median horizon", f"{h_stats.median():.2f}°", "Across 3,600 azimuth bins", "violet")
    with h3:
        card("Maximum horizon", f"{h_stats.max():.2f}°", "Highest sampled azimuth bin", "gold")

    far_limit_km = float(spice_meta.get("terrain_horizon", {}).get("far_terrain_limit_km", 45.0))
    sector_summary = summarize_horizon_sectors(horizon_df, far_limit_km * 1000.0)
    range_limited_bins = int(sector_summary["Range-limited azimuth bins"].sum())
    coverage_short_bins = int(sector_summary["Coverage-short azimuth bins"].sum())
    combined_cap_pinned_bins = int(sector_summary["Combined cap-pinned bins"].sum())
    far_field_dominates = int(sector_summary["80 m controlling bins"].sum())
    controlling_distance_km = float(horizon_df["combined_distance_at_horizon_m"].median()) / 1000.0
    st.markdown('<div class="section-kicker">Directional assessment</div><div class="section-title">Skyline by compass sector</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        card("Range-limited rays", f"{range_limited_bins:,} / 3,600", "Regional rays reach the 45 km data cap", "gold")
    with m2:
        card("Composite cap-pinned", f"{combined_cap_pinned_bins:,} / 3,600", "Sampled combined maximum occurs at the far-range endpoint", "gold")
    with m3:
        card("80 m controlling bins", f"{far_field_dominates:,} / 3,600", "Regional DEM sets the composite maximum", "violet")
    with m4:
        card("Median range to skyline", f"{controlling_distance_km:.2f} km", "Distance to the controlling sampled terrain point", "blue")

    sector_chart = go.Figure()
    sector_chart.add_trace(go.Bar(
        x=sector_summary["Compass sector"], y=sector_summary["Median skyline (°)"],
        name="Median sampled skyline", marker_color="#65D7E8",
        hovertemplate="%{x}<br>Median skyline: %{y:.3f}°<extra></extra>",
    ))
    sector_chart.update_layout(title="Median sampled skyline by compass sector",
                               xaxis_title="Compass sector", yaxis_title="Horizon elevation (degrees)")
    render_figure_with_legend(sector_chart, height=330)
    st.dataframe(
        sector_summary[["Compass sector", "Median skyline (°)", "80 m controlling bins", "Range-limited azimuth bins", "Combined cap-pinned bins", "Azimuth bins"]].round({"Median skyline (°)": 3}),
        width="stretch", hide_index=True,
    )
    st.markdown(
        f'<div class="warning-box">{range_limited_bins:,} of 3,600 regional azimuth rays reach the {far_limit_km:.0f} km range cap; {coverage_short_bins:,} stop short of it. The composite sampled maximum is pinned at the cap in {combined_cap_pinned_bins:,} azimuths. An interior maximum does not prove closure: omitted terrain may raise the true skyline in any direction.</div>',
        unsafe_allow_html=True,
    )
    st.caption("Directional summaries describe the sampled regional skyline only. They are not slope, rock, boulder, or landing-hazard assessments.")

    with st.expander("Terrain provenance and modelling limitations"):
        st.write(f"Horizon source: {spice_meta.get('terrain_horizon', {}).get('source', 'not recorded')}.")
        st.write(f"Far-field range limit: {far_limit_km:.0f} km.")
        st.write("The profile is a regional lower bound with respect to omitted terrain. Terrain beyond the far-field limit may raise the true horizon. The 80 m far-field raster does not resolve smaller features; DEM uncertainties are not fully propagated.")
        st.write("The horizon merge uses the maximum sampled elevation angle from the local and regional profiles. No automatic vertical shift or tilt was applied to the DEMs.")

elif page == "Environment timeline":
    st.markdown('<div class="section-kicker">03 / Time-domain analysis</div><div class="section-title">Sun and Earth geometry through time</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Positions are computed with CSPICE using DE440 and the MOON_ME_DE440_ME421 frame. Terrain obstruction uses the selected observer-height horizon profile. Times below are displayed in India Standard Time; UTC is retained in the source CSV.</div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["Solar environment", "Earth visibility"])
    with t1:
        render_elevation_chart(df, "Sun", "#65D7E8")
        st.plotly_chart(fraction_chart(df, "Sun", "#65D7E8"), width="stretch")
        st.write(f"Sun disk has some visible area in **{summary['sun_any_visible_pct']:.1f}%** of the selected samples; mean sampled disk-visible fraction is **{summary['sun_mean_disk_fraction_pct']:.1f}%**.")
    with t2:
        render_elevation_chart(df, "Earth", "#F2B66D")
        st.plotly_chart(fraction_chart(df, "Earth", "#A99BFF"), width="stretch")
        st.write(f"Geometric Earth disk has some visible area in **{summary['earth_any_visible_pct']:.1f}%** of the selected samples.")
    st.markdown('<div class="section-kicker">04 / Operational windows</div><div class="section-title">Sampled geometric visibility windows</div>', unsafe_allow_html=True)
    st.plotly_chart(availability_band_chart(df), width="stretch")
    st.markdown("**Continuity within the selected interval**")
    cadence_minutes = int(round(df["timestamp_utc"].diff().dropna().dt.total_seconds().median() / 60.0)) if len(df) > 1 else int(spice_meta.get("nominal_step_minutes", 5))
    longest_sun_dark = longest_runs(
        df, "sun_visible_fraction_approx", threshold=DISK_VISIBILITY_MATERIALITY_FRACTION,
        mode="below", sample_minutes=cadence_minutes,
    )
    longest_earth_visible = longest_runs(
        df, "earth_visible_fraction_approx", threshold=DISK_VISIBILITY_MATERIALITY_FRACTION,
        mode="at_or_above", sample_minutes=cadence_minutes,
    )
    r1, r2 = st.columns(2)
    with r1:
        card("Longest sampled Sun-dark run", f"{longest_sun_dark['elapsed_span_minutes'] / 60:.2f} h",
             f"{longest_sun_dark['samples']} consecutive samples · timestamp-span lower bound", "gold")
    with r2:
        card("Longest sampled Earth-visible run", f"{longest_earth_visible['elapsed_span_minutes'] / 60:.2f} h",
             f"{longest_earth_visible['samples']} consecutive samples · 1% materiality threshold", "violet")
    st.caption("Run durations use (sample count − 1) × cadence as the timestamp-to-timestamp lower bound; runs touching either interval edge are censored by the available coverage.")
    st.markdown('<div class="warning-box">Earth visibility here is geometric only; it is not DSN availability or a communications-link prediction. Solar disk visibility is not electrical power or a solar-array energy model. The terrain horizon is provisional and truncated at 45 km.</div>', unsafe_allow_html=True)

elif page == "Site comparison":
    st.markdown('<div class="section-kicker">04 / Comparative assessment</div><div class="section-title">Compare observer heights at the same test point</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Only one terrain location currently has a generated and validated data bundle. This page compares 1 m and 5 m observer heights at the same provisional Site01 point; it does not rank separate landing sites.</div>', unsafe_allow_html=True)
    rows = []
    for height in (1, 5):
        stats = summarize_timeline_material(timelines[height])
        rows.append({
            "Observer height": f"{height} m",
            "Sun any-visible samples (%)": round(stats["sun_any_visible_pct"], 2),
            "Sun full-visible samples (%)": round(stats["sun_full_visible_pct"], 2),
            "Mean Sun disk-visible fraction (%)": round(stats["sun_mean_disk_fraction_pct"], 2),
            "Earth any-visible samples (%)": round(stats["earth_any_visible_pct"], 2),
            "Sun + Earth overlap (%)": round(stats["both_any_visible_pct"], 2),
        })
    st.markdown("**Computed visibility over the same 48-hour interval**")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    difference, difference_stats = compare_horizon_profiles(horizons[1], horizons[5])
    st.markdown('<div class="section-kicker">Observer-height effect</div><div class="section-title">How much does raising the observer change the skyline?</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        card("Median horizon depression", f"{difference_stats['median_deg']:.2f}°", "5 m observer relative to 1 m", "blue")
    with d2:
        card("Mean horizon depression", f"{difference_stats['mean_deg']:.2f}°", "Across 3,600 matched azimuth bins", "violet")
    with d3:
        card("Maximum horizon depression", f"{difference_stats['maximum_deg']:.2f}°", "Largest 1 m minus 5 m difference", "gold")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=difference["azimuth_center_deg"], y=difference["horizon_1m_minus_5m_deg"],
        mode="lines", name="1 m horizon minus 5 m horizon", line=dict(color="#A99BFF", width=1.8),
        hovertemplate="Azimuth: %{x:.2f}°<br>Horizon depression at 5 m: %{y:.3f}°<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,.25)")
    fig.update_layout(title="Azimuth-wise skyline change from 1 m to 5 m observer height",
                      xaxis_title="Azimuth (degrees clockwise from north)",
                      yaxis_title="1 m horizon − 5 m horizon (degrees)")
    st.plotly_chart(style_chart(fig, height=330, show_legend=False, top_margin=64), width="stretch")
    sun_track_delta = sample_horizon_difference_at_azimuths(
        horizons[1], horizons[5], timelines[1]["sun_azimuth_deg"].to_numpy(float)
    )
    earth_track_delta = sample_horizon_difference_at_azimuths(
        horizons[1], horizons[5], timelines[1]["earth_azimuth_deg"].to_numpy(float)
    )
    st.markdown("**Observer-height sensitivity along the actual celestial tracks**")
    st.dataframe(pd.DataFrame([
        {"Body track": "Sun", "Samples": sun_track_delta["samples"],
         "Mean 1 m − 5 m horizon (°)": sun_track_delta["mean_deg"],
         "Range (°)": f"{sun_track_delta['minimum_deg']:.4f} to {sun_track_delta['maximum_deg']:.4f}"},
        {"Body track": "Earth", "Samples": earth_track_delta["samples"],
         "Mean 1 m − 5 m horizon (°)": earth_track_delta["mean_deg"],
         "Range (°)": f"{earth_track_delta['minimum_deg']:.4f} to {earth_track_delta['maximum_deg']:.4f}"},
    ]).round({"Mean 1 m − 5 m horizon (°)": 4}), width="stretch", hide_index=True)
    st.caption("Track-weighted values are more relevant to the current SPICE geometry than an unweighted mean over all azimuths. Near-field maxima can still be sensitive to interpolated DEM detail.")
    visibility_delta = compare_timeline_visibility(timelines[1], timelines[5])
    st.info(
        f"Across {visibility_delta['samples']} aligned SPICE samples, observer height changes "
        f"Sun state in {visibility_delta['sun_state_different_rows']} rows and Earth state in "
        f"{visibility_delta['earth_state_different_rows']} rows. Disk fractions differ in "
        f"{visibility_delta['sun_fraction_different_rows']} Sun rows and "
        f"{visibility_delta['earth_fraction_different_rows']} Earth rows."
    )
    st.caption(f"The 1 m minus 5 m horizon difference ranges from {difference_stats['minimum_deg']:.3f}° to {difference_stats['maximum_deg']:.3f}°. Positive values mean the sampled skyline is lower at 5 m. This is a geometric invariant for the same terrain and observer point, not an independent performance result.")

    horizon_rows = []
    for height in (1, 5):
        hz = horizons[height]["combined_horizon_elevation_deg"]
        horizon_rows.append({
            "Observer height": f"{height} m",
            "Minimum sampled horizon (°)": round(float(hz.min()), 3),
            "Median sampled horizon (°)": round(float(hz.median()), 3),
            "Maximum sampled horizon (°)": round(float(hz.max()), 3),
        })
    st.markdown("**Terrain skyline summary**")
    st.dataframe(pd.DataFrame(horizon_rows), width="stretch", hide_index=True)
    st.caption("A separate-site comparison requires additional candidate coordinates, DEM coverage, horizon profiles and matching SPICE timelines. No synthetic candidate ranking is shown.")

elif page == "Mission delay simulator":
    st.markdown('<div class="section-kicker">05 / Schedule resilience</div><div class="section-title">Screen surface-visibility windows within cached SPICE coverage</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-box">This is not a launch-window optimizer. It screens fixed-duration surface-visibility windows already computed by SPICE, within the cached 48-hour interval. The baseline is the first cached sample. Any-visible metrics require at least 1% of the apparent disk to be exposed. No ephemerides are recomputed.</div>', unsafe_allow_html=True)
    full = timelines[observer_height]
    duration_label = st.selectbox("Surface-visibility window duration", ["6 hours", "12 hours", "24 hours"], index=2)
    window_hours = int(duration_label.split()[0])
    sweep = get_delay_sweep_cached(full, window_hours=window_hours, step_minutes=5)
    visibility_sweep_columns = [
        "sun_mean_disk_fraction_pct", "earth_mean_disk_fraction_pct", "both_any_visible_pct"
    ]
    if all(sweep[column].nunique(dropna=False) == 1 for column in visibility_sweep_columns):
        st.info(
            "The visibility curves are flat because every candidate window in this cached 48-hour interval "
            "has the same outcome: the Sun's disk remains fully visible, while the Earth disk remains hidden. "
            "This is a property of the current data, not a plotting error. The geometry-margin chart below "
            "shows whether either body is moving closer to crossing the sampled terrain skyline."
        )
    max_delay_hours = float(sweep["delay_hours"].max())
    delay_hours = st.slider("Start delay from baseline (hours)", 0.0, max_delay_hours,
                            min(12.0, max_delay_hours), 5/60)
    delay_minutes = int(round(delay_hours * 60))
    baseline = daily_window(full, full["timestamp_utc"].iloc[0], window_hours)
    revised_start = full["timestamp_utc"].iloc[0] + pd.Timedelta(minutes=delay_minutes)
    revised = daily_window(full, revised_start, window_hours)
    if baseline.empty or revised.empty:
        st.warning("The selected delay does not have a complete sampled window.")
    else:
        base_stats = summarize_timeline_material(baseline)
        revised_stats = summarize_timeline_material(revised)
        c1, c2, c3 = st.columns(3)
        with c1:
            card("Change in Sun disk visibility", f"{revised_stats['sun_mean_disk_fraction_pct'] - base_stats['sun_mean_disk_fraction_pct']:+.2f} pp",
                 "Shifted window minus baseline", "blue")
        with c2:
            card("Change in Earth disk visibility", f"{revised_stats['earth_mean_disk_fraction_pct'] - base_stats['earth_mean_disk_fraction_pct']:+.2f} pp",
                 "Geometric disk fraction", "gold")
        with c3:
            card("Change in simultaneous visibility", f"{revised_stats['both_any_visible_pct'] - base_stats['both_any_visible_pct']:+.2f} pp",
                 "Time-sample percentage points", "green")

        comparison = pd.DataFrame([
            {"Window": "Baseline", "Start (IST)": baseline["timestamp_ist"].iloc[0].strftime("%d %b %H:%M"),
             "Sun mean disk fraction (%)": round(base_stats["sun_mean_disk_fraction_pct"], 2),
             "Earth mean disk fraction (%)": round(base_stats["earth_mean_disk_fraction_pct"], 2),
             "Samples": int(len(baseline))},
            {"Window": f"Shifted +{delay_minutes/60:g} h", "Start (IST)": revised["timestamp_ist"].iloc[0].strftime("%d %b %H:%M"),
             "Sun mean disk fraction (%)": round(revised_stats["sun_mean_disk_fraction_pct"], 2),
             "Earth mean disk fraction (%)": round(revised_stats["earth_mean_disk_fraction_pct"], 2),
             "Samples": int(len(revised))},
        ])
        st.dataframe(comparison, width="stretch", hide_index=True)

    st.markdown("**Sampled visibility across every feasible 5-minute start**")
    objective_options = {
        "Maximize mean visible Sun disk fraction": "sun_mean_disk_fraction_pct",
        "Maximize mean geometric Earth disk fraction": "earth_mean_disk_fraction_pct",
        "Maximize simultaneous Sun + Earth any-visibility": "both_any_visible_pct",
    }
    objective_label = st.selectbox("Screening objective (no weighted composite score)", list(objective_options))
    objective_column = objective_options[objective_label]
    objective = classify_objective(
        sweep, objective_column,
        materiality_pct=100.0 * DISK_VISIBILITY_MATERIALITY_FRACTION,
        tolerance_pp=0.01,
    )
    best_value = objective["best_value"]
    best_rows = objective["best_rows"]
    if objective["status"] == "unachieved":
        st.warning(f"This objective is not achieved at the 1% disk-visibility materiality level in the cached data: its maximum is only {best_value:.3f}%. No feasible start reaches 1.0%.")
    elif objective["status"] == "tied":
        st.info(f"{len(best_rows)} sampled start times are equivalent within 0.01 percentage points (maximum {best_value:.2f}%). No unique start is identified in this cached interval.")
    else:
        best_start = best_rows.iloc[0]["window_start_ist"]
        st.success(f"Best sampled surface-visibility window start: {best_start:%d %b %Y, %H:%M IST} ({best_value:.2f}%).")

    sweep_fig = go.Figure()
    sweep_fig.add_trace(go.Scatter(x=sweep["delay_hours"], y=sweep["sun_mean_disk_fraction_pct"],
                                   mode="lines+markers", name="Mean Sun disk fraction", line=dict(color="#65D7E8", width=2)))
    sweep_fig.add_trace(go.Scatter(x=sweep["delay_hours"], y=sweep["earth_mean_disk_fraction_pct"],
                                   mode="lines+markers", name="Mean Earth disk fraction", line=dict(color="#F2B66D", width=2)))
    sweep_fig.add_trace(go.Scatter(x=sweep["delay_hours"], y=sweep["both_any_visible_pct"],
                                   mode="lines+markers", name="Simultaneous any-visibility", line=dict(color="#70D6A1", width=2)))
    sweep_fig.update_layout(title="Visibility metrics by sampled start delay", xaxis_title="Start delay from baseline (hours)",
                            yaxis_title="Window sample percentage (%)")
    sweep_fig.update_yaxes(range=[0, 100])
    render_figure_with_legend(sweep_fig, height=330)

    geometry_fig = go.Figure()
    geometry_fig.add_trace(go.Scatter(
        x=sweep["delay_hours"], y=sweep["sun_min_lower_limb_clearance_deg"],
        mode="lines", name="Sun minimum lower-limb clearance",
        line=dict(color="#65D7E8", width=2),
        hovertemplate="Delay: %{x:.2f} h<br>Sun lower-limb clearance: %{y:.3f}°<extra></extra>",
    ))
    geometry_fig.add_trace(go.Scatter(
        x=sweep["delay_hours"], y=sweep["earth_max_upper_limb_clearance_deg"],
        mode="lines", name="Earth maximum upper-limb clearance",
        line=dict(color="#F2B66D", width=2),
        hovertemplate="Delay: %{x:.2f} h<br>Earth upper-limb clearance: %{y:.3f}°<extra></extra>",
    ))
    geometry_fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,.5)",
                           annotation_text="Disk-limb / skyline contact")
    geometry_fig.update_layout(
        title="Body-to-skyline clearance across candidate windows",
        xaxis_title="Start delay from baseline (hours)",
        yaxis_title="Limb clearance above sampled skyline (degrees)",
    )
    render_figure_with_legend(geometry_fig, height=330)
    st.caption(
        "Positive Sun lower-limb clearance means the entire solar disk clears the sampled skyline. "
        "Positive Earth upper-limb clearance means at least part of the Earth disk clears it. "
        "These margins use the sampled 45 km-limited terrain horizon and are not globally closed."
    )

    display_sweep = sweep[[
        "delay_hours", "window_start_ist", "sample_count", "sun_mean_disk_fraction_pct",
        "earth_mean_disk_fraction_pct", "both_any_visible_pct",
    ]].copy()
    display_sweep["window_start_ist"] = display_sweep["window_start_ist"].dt.strftime("%d %b %H:%M IST")
    display_sweep = display_sweep.rename(columns={
        "delay_hours": "Delay (h)", "window_start_ist": "Window start",
        "sample_count": "Samples", "sun_mean_disk_fraction_pct": "Sun mean disk (%)",
        "earth_mean_disk_fraction_pct": "Earth mean disk (%)", "both_any_visible_pct": "Sun + Earth overlap (%)",
    })
    with st.expander("Show all feasible 5-minute windows"):
        st.dataframe(display_sweep.round(2), width="stretch", hide_index=True)
    st.caption("The baseline is the first cached timestamp, not a user-selected landing time. The sweep uses 5-minute start increments and half-open windows [start, end). It cannot answer delays outside the existing 48-hour ephemeris coverage.")

st.divider()
with st.expander("Scientific provenance, computation settings and limitations"):
    st.write(f"**Ephemeris:** {spice_meta.get('ephemeris', 'not recorded')}")
    st.write(f"**Reference frame:** {spice_meta.get('target_frame', 'not recorded')} (frame ID {spice_meta.get('frame_id_loaded_from_FK', 'not recorded')})")
    st.write(f"**Aberration correction:** {spice_meta.get('aberration_correction', 'not recorded')}")
    st.write(f"**UTC coverage:** {spice_meta.get('time_window_utc', {}).get('start')} to {spice_meta.get('time_window_utc', {}).get('end')}")
    st.write(f"**Terrain horizon range:** {spice_meta.get('terrain_horizon', {}).get('far_terrain_limit_km')} km")
    st.write("**Known limitations:** provisional tile-centre observer; 45 km regional terrain cutoff; no fully propagated DEM uncertainty; constant local horizon approximation across each apparent disk; radial spherical surface normal (no local DEM slope); geometric Earth visibility is not a link budget; solar visibility is not array power.")
    st.write("**Required scientific artifact hash checks:**")
    st.json(bundle["hash_checks"])
    st.write("**Source DEM/kernel hashes (null means source file was not present locally):**")
    st.json(bundle["optional_source_hash_checks"])

left, right = st.columns([3, 1])
with left:
    st.markdown('<div class="small-muted">LUNAR · Mission Environment Explorer · Scientific integration v7</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="small-muted" style="text-align:right">SITE01 · PROVISIONAL</div>', unsafe_allow_html=True)
