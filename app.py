import io
import gzip
from pathlib import Path
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="HLMA Lightning Mapper (.dat)", layout="wide")
st.title("⚡ HLMA Lightning — .dat Uploader & Map")
st.caption("Upload an HLMA .dat or .dat.gz file, parse robustly, and plot on an interactive map.")

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

def _read_hlma_block(s: str) -> pd.DataFrame | None:
    """
    HLMA-export aware parser:
      - find '*** data ***'
      - keep only rows AFTER it that start with a number
      - drop literal '...' lines
      - parse as whitespace-delimited (8 columns typical)
      - assign HLMA column names
    """
    lines_all = s.splitlines()
    # locate sentinel
    start_idx = None
    for i, ln in enumerate(lines_all):
        if '*** data ***' in ln.lower():
            start_idx = i + 1
            break
    if start_idx is None:
        return None  # not an HLMA-export structure

    num_pat = re.compile(r'^\s*[+-]?\d')
    data_lines = []
    for ln in lines_all[start_idx:]:
        if not ln.strip():
            continue
        if ln.strip() == '...':
            continue
        if not num_pat.match(ln):
            continue
        data_lines.append(ln)

    if not data_lines:
        return pd.DataFrame()

    # Parse all data lines as whitespace-delimited
    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=r"\s+",
        engine="python",
        header=None,
        on_bad_lines="skip",
        skip_blank_lines=True,
    )

    # Typical HLMA export has 8 columns (time, lat, lon, alt, chi2, nstations, p_dbw, mask)
    # If column count differs, we still proceed but name what we can.
    names_8 = ["time_uts", "lat", "lon", "alt_m", "chi2", "nstations", "p_dbw", "mask"]
    if df.shape[1] == 8:
        df.columns = names_8
    else:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        # If we have at least 4 numeric columns, map the first four to HLMA names
        if df.shape[1] >= 4:
            rename_map = {df.columns[0]: "time_uts",
                          df.columns[1]: "lat",
                          df.columns[2]: "lon",
                          df.columns[3]: "alt_m"}
            df = df.rename(columns=rename_map)

    return df

def _generic_parse(s: str) -> pd.DataFrame:
    """Generic robust parser for non-HLMA files."""
    # Remove comment/blank lines for detection
    data_lines = [ln for ln in s.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    if not data_lines:
        return pd.DataFrame()

    # Try sep=None (csv.Sniffer)
    df = None
    try:
        df = pd.read_csv(io.StringIO("\n".join(data_lines)),
                         sep=None, engine="python",
                         header=None,
                         on_bad_lines="skip", skip_blank_lines=True)
    except Exception:
        df = None

    # Fallback: regex whitespace or comma
    if df is None or df.empty:
        try:
            df = pd.read_csv(io.StringIO("\n".join(data_lines)),
                             sep=r"\s+|,", engine="python",
                             header=None,
                             on_bad_lines="skip", skip_blank_lines=True)
        except Exception:
            df = None

    # Last resort: manual split normalized to most-common width
    if df is None or df.empty:
        splitter = re.compile(r"[, \t]+")
        rows, widths = [], []
        for ln in data_lines:
            parts = [p for p in splitter.split(ln.strip()) if p != ""]
            if parts:
                rows.append(parts)
                widths.append(len(parts))
        if not rows:
            return pd.DataFrame()
        most_w = Counter(widths).most_common(1)[0][0]
        fixed = []
        for parts in rows:
            if len(parts) < most_w:
                parts = parts + [""] * (most_w - len(parts))
            elif len(parts) > most_w:
                parts = parts[:most_w]
            fixed.append(parts)
        df = pd.DataFrame(fixed)

    # Name columns if unnamed
    if df.columns.dtype == "int64" or any(isinstance(c, (int, np.integer)) for c in df.columns):
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
    return df

def _read_dat_to_df(uploaded) -> pd.DataFrame:
    """
    Unified entry:
      - Try HLMA-aware parsing (*** data *** sentinel)
      - Else, use generic robust parsing
    """
    sio = _open_text_from_upload(uploaded)
    text = sio.getvalue()

    df = _read_hlma_block(text)
    if df is None:
        df = _generic_parse(text)

    # Final cleanups
    if df.empty:
        return df

    # Normalize any Unnamed columns
    new_cols = []
    for i, c in enumerate(df.columns):
        cn = str(c)
        if cn.lower().startswith("unnamed") or cn.strip() == "":
            cn = f"col_{i}"
        new_cols.append(cn)
    df.columns = new_cols

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
    return pdk.ViewState(latitude=lat_m, longitude=lon_m, zoom=7, pitch=0)

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
    st.caption("If your file has headers like 'lat'/'lon', I'll detect them. Otherwise, pick them below.")

if up is None:
    st.info("Upload an HLMA .dat (or .dat.gz) file in the sidebar to begin.\n\n"
            "Tip: header block and '*** data ***' are handled automatically.")
    st.stop()

# Parse (robust; HLMA-aware)
raw_df = _read_dat_to_df(up)

# Auto defaults: prefer HLMA names if present, else guess
if set(["lat", "lon"]).issubset(raw_df.columns):
    lat_default = "lat"
    lon_default = "lon"
else:
    # Guess by value ranges
    lat_default = None
    lon_default = None
    for c in raw_df.columns:
        s = pd.to_numeric(raw_df[c], errors="coerce")
        if s.notna().mean() < 0.8:
            continue
        mn, mx = float(s.min()), float(s.max())
        if lat_default is None and -90.5 <= mn <= 90.5 and -90.5 <= mx <= 90.5:
            lat_default = c
        if lon_default is None and -180.5 <= mn <= 360.5 and -180.5 <= mx <= 360.5:
            lon_default = c
    # Fallbacks
    if lat_default is None and len(raw_df.columns) >= 2:
        lat_default = raw_df.columns[1]
    if lon_default is None and len(raw_df.columns) >= 3:
        lon_default = raw_df.columns[2]

with st.sidebar:
    col_options = list(raw_df.columns.astype(str))
    lat_col = st.selectbox("Latitude column", options=col_options,
                           index=(col_options.index(lat_default) if lat_default in col_options else 0))
    lon_col = st.selectbox("Longitude column", options=col_options,
                           index=(col_options.index(lon_default) if lon_default in col_options else min(1, len(col_options)-1)))
    alt_candidates = [c for c in raw_df.columns if c.lower() in {"alt", "alt_m", "altitude", "altitude_m"}]
    alt_default = alt_candidates[0] if alt_candidates else None
    alt_col = st.selectbox("Altitude column (optional)", options=["(none)"] + col_options,
                           index=(0 if alt_default is None else (col_options.index(alt_default)+1)))
    time_candidates = [c for c in raw_df.columns if c.lower() in {"time", "time_uts", "time_sec", "t"}]
    time_default = time_candidates[0] if time_candidates else None
    time_col = st.selectbox("Time column (optional)", options=["(none)"] + col_options,
                            index=(0 if time_default is None else (col_options.index(time_default)+1)))

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
work_df[lon_col] = _lon_wrap(work_df[lon_col])

# Drop invalid rows
work_df = work_df[np.isfinite(work_df[lat_col]) & np.isfinite(work_df[lon_col])].copy()

if work_df.empty:
    st.error("No valid latitude/longitude rows found after parsing. "
             "If this is an HLMA export, ensure the file includes '*** data ***' and numeric rows after it.")
    st.stop()

# ---------------------------
# Sidebar — Filters
# ---------------------------
with st.sidebar:
    st.header("3) Filters")
    lat_min, lat_max = float(np.nanmin(work_df[lat_col])), float(np.nanmax(work_df[lat_col]))
    lon_min, lon_max = float(np.nanmin(work_df[lon_col])), float(np.nanmax(work_df[lon_col]))

    lat_rng = st.slider("Latitude range", min_value=lat_min, max_value=lat_max, value=(lat_min, lat_max))
    lon_rng = st.slider("Longitude range", min_value=lon_min, max_value=lon_max, value=(lon_min, lon_max))

    alt_rng = None
    if alt_col:
        if work_df[alt_col].notna().any():
            a_min, a_max = float(np.nanmin(work_df[alt_col])), float(np.nanmax(work_df[alt_col]))
            if a_min != a_max:
                alt_rng = st.slider("Altitude range (units as in file)", min_value=a_min, max_value=a_max, value=(a_min, a_max))

# Apply filters
flt = (work_df[lat_col].between(lat_rng[0], lat_rng[1])) & (work_df[lon_col].between(lon_rng[0], lon_rng[1]))
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

    def build_tooltip(row):
        parts = [f"lat: {row['lat']:.4f}", f"lon: {row['lon']:.4f}"]
        if not (np.isnan(row.get("alt", np.nan))):
            parts.append(f"alt: {row['alt']}")
        if time_col != "(none)":
            parts.append(f"time: {row.get(time_col)}")
        return " | ".join(parts)

    plot_df["tooltip"] = plot_df.apply(build_tooltip, axis=1)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[lon, lat]",
        get_radius=800,   # meters; tweak as desired
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
# Data Preview & Export + Debug
# ---------------------------
with st.expander("Preview parsed table"):
    st.dataframe(plot_df.head(500), use_container_width=True)

with st.expander("Parsing debug — first 20 non-comment lines after '*** data ***'"):
    try:
        sio_dbg = _open_text_from_upload(up)
        text_dbg = sio_dbg.getvalue()
        # Show only numeric rows after sentinel
        lines = text_dbg.splitlines()
        start_idx = 0
        for i, ln in enumerate(lines):
            if '*** data ***' in ln.lower():
                start_idx = i + 1
                break
        num_pat = re.compile(r'^\s*[+-]?\d')
        cleaned = [ln for ln in lines[start_idx:] if ln.strip() and ln.strip() != '...' and num_pat.match(ln)]
        st.code("\n".join(cleaned[:20]) or "(no numeric data lines found)", language="text")
    except Exception:
        pass

csv = plot_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered CSV",
    data=csv,
    file_name=(Path(up.name).stem + "_filtered.csv"),
    mime="text/csv",
)

st.caption(
    "Notes: This parser detects the HLMA '*** data ***' block and only parses numeric rows after it. "
    "Longitudes in 0..360 are wrapped to -180..180 for mapping. "
    "Adjust column picks in the sidebar if needed."
)
