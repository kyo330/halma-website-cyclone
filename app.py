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
# Page & header
# ---------------------------
st.set_page_config(page_title="Lightning — Fixed Area", layout="wide")
st.markdown("#### HLMA Website")

# ---------------------------
# HLMA helpers
# ---------------------------

def _is_gz(bytes_buf: bytes) -> bool:
    return len(bytes_buf) >= 2 and bytes_buf[0] == 0x1F and bytes_buf[1] == 0x8B

def _open_text_from_upload(uploaded) -> io.StringIO:
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
      - assign HLMA column names: time_uts, lat, lon, alt_m, chi2, nstations, p_dbw, mask
    """
    lines_all = s.splitlines()
    start_idx = None
    for i, ln in enumerate(lines_all):
        if '*** data ***' in ln.lower():
            start_idx = i + 1
            break
    if start_idx is None:
        return None

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

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=r"\s+",
        engine="python",
        header=None,
        on_bad_lines="skip",
        skip_blank_lines=True,
    )
    names_8 = ["time_uts", "lat", "lon", "alt_m", "chi2", "nstations", "p_dbw", "mask"]
    if df.shape[1] == 8:
        df.columns = names_8
    else:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        if df.shape[1] >= 4:
            df = df.rename(columns={
                df.columns[0]: "time_uts",
                df.columns[1]: "lat",
                df.columns[2]: "lon",
                df.columns[3]: "alt_m",
            })
    return df

def _generic_parse(s: str) -> pd.DataFrame:
    data_lines = [ln for ln in s.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    if not data_lines:
        return pd.DataFrame()

    df = None
    try:
        df = pd.read_csv(io.StringIO("\n".join(data_lines)),
                         sep=None, engine="python", header=None,
                         on_bad_lines="skip", skip_blank_lines=True)
    except Exception:
        df = None
    if df is None or df.empty:
        try:
            df = pd.read_csv(io.StringIO("\n".join(data_lines)),
                             sep=r"\s+|,", engine="python", header=None,
                             on_bad_lines="skip", skip_blank_lines=True)
        except Exception:
            df = None
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
    if df.columns.dtype == "int64" or any(isinstance(c, (int, np.integer)) for c in df.columns):
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
    return df

def _read_dat_to_df(uploaded) -> pd.DataFrame:
    sio = _open_text_from_upload(uploaded)
    text = sio.getvalue()
    df = _read_hlma_block(text)
    if df is None:
        df = _generic_parse(text)
    if df.empty:
        return df
    # Clean column names
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
# UI: upload + parse
# ---------------------------
with st.sidebar:
    st.header("Upload HLMA .dat/.dat.gz")
    up = st.file_uploader("Choose a .dat or .dat.gz file", type=["dat", "gz"], accept_multiple_files=False)

if up is None:
    st.info("Upload an HLMA export. Header + '*** data ***' handled automatically. Lines like '...' are skipped.")
    st.stop()

raw_df = _read_dat_to_df(up)

# pick columns (defaults favor HLMA names)
if set(["lat", "lon"]).issubset(raw_df.columns):
    lat_default, lon_default = "lat", "lon"
else:
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
    if lat_default is None and len(raw_df.columns) >= 2: lat_default = raw_df.columns[1]
    if lon_default is None and len(raw_df.columns) >= 3: lon_default = raw_df.columns[2]

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

# clean
work_df = raw_df.copy()
work_df[lat_col] = _coerce_numeric(work_df[lat_col])
work_df[lon_col] = _coerce_numeric(work_df[lon_col])
if alt_col != "(none)":
    work_df[alt_col] = _coerce_numeric(work_df[alt_col])
else:
    work_df["altitude_m"] = np.nan
    alt_col = "altitude_m"

work_df[lon_col] = _lon_wrap(work_df[lon_col])
work_df = work_df[np.isfinite(work_df[lat_col]) & np.isfinite(work_df[lon_col])].copy()

if work_df.empty:
    st.error("No valid latitude/longitude rows found after parsing. If this is an HLMA export, ensure '*** data ***' exists and rows follow it.")
    st.stop()

# ---------------------------
# filters
# ---------------------------
with st.sidebar:
    st.header("Filters")
    lat_min, lat_max = float(np.nanmin(work_df[lat_col])), float(np.nanmax(work_df[lat_col]))
    lon_min, lon_max = float(np.nanmin(work_df[lon_col])), float(np.nanmax(work_df[lon_col]))
    lat_rng = st.slider("Latitude range", min_value=lat_min, max_value=lat_max, value=(lat_min, lat_max))
    lon_rng = st.slider("Longitude range", min_value=lon_min, max_value=lon_max, value=(lon_min, lon_max))

    alt_rng = None
    if work_df[alt_col].notna().any():
        a_min, a_max = float(np.nanmin(work_df[alt_col])), float(np.nanmax(work_df[alt_col]))
        if a_min != a_max:
            alt_rng = st.slider("Altitude range (units as in file)", min_value=a_min, max_value=a_max, value=(a_min, a_max))

flt = (work_df[lat_col].between(lat_rng[0], lat_rng[1])) & (work_df[lon_col].between(lon_rng[0], lon_rng[1]))
if alt_rng is not None:
    flt &= work_df[alt_col].between(alt_rng[0], alt_rng[1])

plot_df = work_df.loc[flt, [lat_col, lon_col, alt_col] + ([time_col] if time_col != "(none)" else [])].copy()
plot_df.rename(columns={lat_col: "lat", lon_col: "lon", alt_col: "alt"}, inplace=True)

st.success(f"Parsed rows: {len(work_df):,} • Plotted rows: {len(plot_df):,}")

# ---------------------------
# map
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
        get_radius=800,
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
# preview / download
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
    "Notes: Parser detects HLMA '*** data ***' and reads only numeric rows after it; "
    "longitudes in 0..360 are wrapped to -180..180 for mapping."
)
