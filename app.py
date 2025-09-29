import io
import gzip
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="HLMA Lightning Mapper (.dat)",
    layout="wide",
)

st.title("⚡ HLMA Lightning — .dat Uploader & Map")
st.caption(
    "Upload an HLMA .dat or .dat.gz file, map the strikes, and explore with filters."
)

# ---------------------------
# Helpers
# ---------------------------

def _is_gz(bytes_buf: bytes) -> bool:
    """Return True if bytes look like a gzip file."""
    return len(bytes_buf) >= 2 and bytes_buf[0] == 0x1F and bytes_buf[1] == 0x8B


def _open_text_from_upload(uploaded) -> io.StringIO:
    """Return a text stream from an uploaded file (supports .gz)."""
    raw = uploaded.read()
    if _is_gz(raw) or str(uploaded.name).lower().endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            text = gz.read().decode("utf-8", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    return io.StringIO(text)


def _detect_header_and_sep(sample_text: str):
    """Heuristically detect if a header is present and choose a separator.

    Returns: (has_header: bool, sep: str | None, delim_whitespace: bool)
    """
    # Find first non-comment, non-empty line
    first_data_line = None
    for ln in sample_text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        first_data_line = s
        break

    # Default assumptions
    has_header = False
    sep = None
    delim_ws = False

    if first_data_line is None:
        return has_header, sep, True

    # If commas appear, use comma sep; otherwise treat as whitespace
    if "," in first_data_line:
        sep = ","
        tokens = [t.strip() for t in first_data_line.split(",")]
    else:
        delim_ws = True
        tokens = first_data_line.split()

    # If any token is non-numeric, likely a header
    def _is_num(tok: str) -> bool:
        try:
            float(tok)
            return True
        except Exception:
            return False

    non_num = sum(0 if _is_num(t) else 1 for t in tokens)
    has_header = non_num >= max(1, len(tokens) // 3)
    return has_header, sep, delim_ws


def _read_dat_to_df(uploaded) -> pd.DataFrame:
    """Parse .dat/.dat.gz into a DataFrame, skipping comment lines.

    The function tries (1) detected header; (2) user-provided mapping if needed.
    """
    sio = _open_text_from_upload(uploaded)
    # Keep a small sample for detection
    sample = sio.getvalue()[:10_000]
    has_header, sep, delim_ws = _detect_header_and_sep(sample)

    # Reset pointer and read
    sio.seek(0)
    try:
        if delim_ws:
            df = pd.read_csv(
                sio,
                comment="#",
                header=0 if has_header else None,
                delim_whitespace=True,
                engine="python",
            )
        else:
            df = pd.read_csv(
                sio,
                comment="#",
                header=0 if has_header else None,
                sep=sep,
                engine="python",
            )
    except Exception as e:
        # Fallback: split on any whitespace or comma
        sio.seek(0)
        df = pd.read_csv(
            sio,
            comment="#",
            header=0 if has_header else None,
            sep=r"\s+|,",
            engine="python",
        )

    # Assign generic column names if there's no header
    if df.columns.dtype == "int64" or any(isinstance(c, (int, np.integer)) for c in df.columns):
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    return df


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _lon_wrap(lon: pd.Series) -> pd.Series:
    """Wrap longitudes to [-180, 180] if they look like [0, 360]."""
    s = lon.copy()
    if s.max(skipna=True) > 180 and s.min(skipna=True) >= 0:
        s = ((s + 180) % 360) - 180
    return s


def _initial_view(df: pd.DataFrame, lat_col: str, lon_col: str):
    lat = df[lat_col].astype(float)
    lon = df[lon_col].astype(float)
    lat_m = float(np.nanmean(lat)) if len(lat) else 0.0
    lon_m = float(np.nanmean(lon)) if len(lon) else 0.0
    return pdk.ViewState(latitude=lat_m, longitude=lon_m, zoom=6, pitch=0)


# ---------------------------
# Sidebar — Upload & Mapping
# ---------------------------
with st.sidebar:
    st.header("1) Upload .dat / .dat.gz")
    up = st.file_uploader(
        "Choose a .dat or .dat.gz file",
        type=["dat", "gz"],
        accept_multiple_files=False,
    )

    st.header("2) Column Mapping")
    st.caption(
        "If your file has headers like 'lat'/'lon', I'll detect them. Otherwise, pick the columns below."
    )

# Work area
if up is None:
    st.info(
        "Upload an HLMA .dat (or .dat.gz) file in the sidebar to begin.\n\n"
        "Tip: comment lines starting with '#' are ignored."
    )
    st.stop()

# Parse
raw_df = _read_dat_to_df(up)

# Auto-detect likely latitude/longitude column names
lower_names = {c.lower(): c for c in raw_df.columns.astype(str)}

cand_lat = [
    n for n in ["lat", "latitude", "lat_deg", "y", "phi", "col_1", "col_2"] if n in lower_names
]
cand_lon = [
    n for n in ["lon", "longitude", "long", "lon_deg", "x", "lambda", "col_0", "col_3"] if n in lower_names
]

lat_default = lower_names.get(cand_lat[0], None) if cand_lat else None
lon_default = lower_names.get(cand_lon[0], None) if cand_lon else None

with st.sidebar:
    col_options = list(raw_df.columns.astype(str))
    lat_col = st.selectbox("Latitude column", options=col_options, index=(col_options.index(lat_default) if lat_default in col_options else 0))
    lon_col = st.selectbox("Longitude column", options=col_options, index=(col_options.index(lon_default) if lon_default in col_options else min(1, len(col_options)-1)))
    alt_col = st.selectbox("Altitude column (optional)", options=["(none)"] + col_options, index=0)
    time_col = st.selectbox("Time column (optional)", options=["(none)"] + col_options, index=0)

# Clean and coerce
work_df = raw_df.copy()
work_df[lat_col] = _coerce_numeric(work_df[lat_col])
work_df[lon_col] = _coerce_numeric(work_df[lon_col])
if alt_col != "(none)":
    work_df[alt_col] = _coerce_numeric(work_df[alt_col])
else:
    work_df["altitude_m"] = np.nan
    alt_col = "altitude_m"

# Wrap longitudes if in 0..360
after_wrap = _lon_wrap(work_df[lon_col])
work_df[lon_col] = after_wrap

# Drop invalid rows
work_df = work_df[np.isfinite(work_df[lat_col]) & np.isfinite(work_df[lon_col])].copy()

if work_df.empty:
    st.error(
        "No valid latitude/longitude rows found after parsing. Check the column mapping and try again."
    )
    st.stop()

# ---------------------------
# Sidebar — Filters
# ---------------------------
with st.sidebar:
    st.header("3) Filters")
    lat_min = float(np.nanmin(work_df[lat_col]))
    lat_max = float(np.nanmax(work_df[lat_col]))
    lon_min = float(np.nanmin(work_df[lon_col]))
    lon_max = float(np.nanmax(work_df[lon_col]))

    lat_rng = st.slider("Latitude range", min_value=lat_min, max_value=lat_max, value=(lat_min, lat_max))
    lon_rng = st.slider("Longitude range", min_value=lon_min, max_value=lon_max, value=(lon_min, lon_max))

    alt_rng = None
    if alt_col:
        a_min = float(np.nanmin(work_df[alt_col])) if work_df[alt_col].notna().any() else 0.0
        a_max = float(np.nanmax(work_df[alt_col])) if work_df[alt_col].notna().any() else 0.0
        if a_min != a_max:
            alt_rng = st.slider(
                "Altitude range (units as in file)",
                min_value=a_min,
                max_value=a_max,
                value=(a_min, a_max),
            )

# Apply filters
flt = (
    (work_df[lat_col].between(lat_rng[0], lat_rng[1]))
    & (work_df[lon_col].between(lon_rng[0], lon_rng[1]))
)
if alt_rng is not None:
    flt &= work_df[alt_col].between(alt_rng[0], alt_rng[1])

plot_df = work_df.loc[flt, [lat_col, lon_col, alt_col] + ([time_col] if time_col != "(none)" else [])].copy()
plot_df.rename(columns={lat_col: "lat", lon_col: "lon", alt_col: "alt"}, inplace=True)

st.success(f"Parsed rows: {len(work_df):,} • Plotted rows: {len(plot_df):,}")

# ---------------------------
# Map
# ---------------------------
if plot_df.empty:
    st.warning("No points within the selected filters.")
else:
    view = _initial_view(plot_df, "lat", "lon")

    # Build tooltip text
    def build_tooltip(row):
        parts = [f"lat: {row['lat']:.4f}", f"lon: {row['lon']:.4f}"]
        if not (np.isnan(row.get("alt", np.nan))):
            parts.append(f"alt: {row['alt']}")
        if time_col != "(none)":
            tval = row.get(time_col)
            parts.append(f"time: {tval}")
        return " | ".join(parts)

    plot_df["tooltip"] = plot_df.apply(build_tooltip, axis=1)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[lon, lat]",
        get_radius=1000,  # meters; adjust as needed
        auto_highlight=True,
        pickable=True,
        opacity=0.6,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style="mapbox://styles/mapbox/light-v11",
        tooltip={"text": "{tooltip}"},
    )

    st.pydeck_chart(deck, use_container_width=True)

# ---------------------------
# Data Preview & Export
# ---------------------------
with st.expander("Preview parsed table"):
    st.dataframe(plot_df.head(500), use_container_width=True)

csv = plot_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered CSV",
    data=csv,
    file_name=(Path(up.name).stem + "_filtered.csv"),
    mime="text/csv",
)

st.caption(
    "Notes: If your longitudes are 0..360 they are wrapped to -180..180 for mapping. "
    "If auto-detection picks wrong columns, fix them in the sidebar."
)
