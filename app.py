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


def _read_dat_to_df(uploaded) -> pd.DataFrame:
    """
    Robust .dat/.dat.gz parser that tolerates ragged rows & mixed delimiters.

    Strategy:
      1) Remove comment/blank lines and try sep=None (csv.Sniffer) with engine='python'.
      2) Fallback to regex split on whitespace OR comma (r"\\s+|,").
      3) Last resort: manual split; coerce each row to the most common column count.
    Also normalizes/creates column names and drops fully empty columns.
    """
    sio = _open_text_from_upload(uploaded)
    full_text = sio.getvalue()

    # Keep only data lines (no comments / blanks)
    data_lines = [ln for ln in full_text.splitlines()
                  if ln.strip() and not ln.lstrip().startswith('#')]
    if not data_lines:
        return pd.DataFrame(columns=["col_0", "col_1"])

    # Heuristic header detection: if many tokens are non-numeric, assume header
    first = data_lines[0].strip()
    toks = re.split(r"\s+|,", first)
    def _isnum(x):
        try:
            float(x)
            return True
        except Exception:
            return False
    non_num = sum(0 if _isnum(t) else 1 for t in toks if t != "")
    has_header = non_num >= max(1, len([t for t in toks if t != ""]) // 3)

    # Attempt 1: sep=None (Sniffer)
    df = None
    try:
        buf1 = io.StringIO("\n".join(data_lines))
        df = pd.read_csv(
            buf1,
            sep=None,                 # auto-detect delimiter
            engine="python",
            header=0 if has_header else None,
            on_bad_lines="skip",      # pandas >= 1.3
            skip_blank_lines=True,
        )
    except Exception:
        df = None

    # Attempt 2: regex delimiter (whitespace OR comma)
    if df is None or df.empty:
        try:
            buf2 = io.StringIO("\n".join(data_lines))
            df = pd.read_csv(
                buf2,
                sep=r"\s+|,",
                engine="python",
                header=0 if has_header else None,
                on_bad_lines="skip",
                skip_blank_lines=True,
            )
        except Exception:
            df = None

    # Attempt 3: manual split -> normalize to most-common width
    if df is None or df.empty:
        splitter = re.compile(r"[,\s]+")
        rows, widths = [], []
        for ln in data_lines:
            parts = [p for p in splitter.split(ln.strip()) if p != ""]
            if parts:
                rows.append(parts)
                widths.append(len(parts))
        if not rows:
            return pd.DataFrame(columns=["col_0", "col_1"])
        most_w = Counter(widths).most_common(1)[0][0]
        fixed = []
        for parts in rows:
            if len(parts) < most_w:
                parts = parts + [""] * (most_w - len(parts))
            elif len(parts) > most_w:
                parts = parts[:most_w]
            fixed.append(parts)
        df = pd.DataFrame(fixed)
        if has_header and len(df) >= 1:
            df.columns = [c if str(c).strip() != "" else f"col_{i}" for i, c in enumerate(df.iloc[0])]
            df = df.drop(df.index[0]).reset_index(drop=True)

    # Drop fully-empty columns
    if not df.empty:
        df = df.dropna(axis=1, how="all")

    # Normalize column names
    if df.columns.dtype == "int64" or any(isinstance(c, (int, np.integer)) for c in df.columns):
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
    else:
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
    st.caption("If your file has headers like 'lat'/'lon', I'll detect them. Otherwise, pick them below.")

if up is None:
    st.info("Upload an HLMA .dat (or .dat.gz) file in the sidebar to begin.\n\n"
            "Tip: comment lines starting with '#' are ignored.")
    st.stop()

# Parse (robust)
raw_df = _read_dat_to_df(up)

# Auto-detect likely latitude/longitude column names
lower_names = {str(c).lower(): str(c) for c in raw_df.columns}

cand_lat = [n for n in ["lat", "latitude", "lat_deg", "y", "phi", "col_1", "col_2"] if n in lower_names]
cand_lon = [n for n in ["lon", "longitude", "long", "lon_deg", "x", "lambda", "col_0", "col_3"] if n in lower_names]

lat_default = lower_names.get(cand_lat[0], None) if cand_lat else list(raw_df.columns)[0]
lon_default = lower_names.get(cand_lon[0], None) if cand_lon else (list(raw_df.columns)[1] if len(raw_df.columns) > 1 else list(raw_df.columns)[0])

with st.sidebar:
    col_options = list(raw_df.columns.astype(str))
    lat_col = st.selectbox("Latitude column", options=col_options,
                           index=(col_options.index(lat_default) if lat_default in col_options else 0))
    lon_col = st.selectbox("Longitude column", options=col_options,
                           index=(col_options.index(lon_default) if lon_default in col_options else min(1, len(col_options)-1)))
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
work_df[lon_col] = _lon_wrap(work_df[lon_col])

# Drop invalid rows
work_df = work_df[np.isfinite(work_df[lat_col]) & np.isfinite(work_df[lon_col])].copy()

if work_df.empty:
    st.error("No valid latitude/longitude rows found after parsing. Check the column mapping and try again.")
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
        get_radius=1000,
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

with st.expander("Parsing debug — first 20 non-comment lines"):
    try:
        sio_dbg = _open_text_from_upload(up)
        lines_dbg = [ln for ln in sio_dbg.getvalue().splitlines()
                     if ln.strip() and not ln.lstrip().startswith('#')]
        st.code("\n".join(lines_dbg[:20]) or "(no data lines)", language="text")
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
    "Notes: If your longitudes are 0..360 they are wrapped to -180..180 for mapping. "
    "If auto-detection picks wrong columns, fix them in the sidebar."
)
