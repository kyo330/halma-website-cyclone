import io
import gzip
from pathlib import Path
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

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
    return len(bytes_buf) >= 2 and bytes_buf[0] == 0x1F and bytes_buf[1] == 0x8B

def _open_text_from_upload(uploaded) -> io.StringIO:
    raw = uploaded.read()
    if _is_gz(raw) or str(uploaded.name).lower().endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            text = gz.read().decode("utf-8", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    return io.StringIO(text)

def _detect_header_and_sep(sample_text: str):
    first_data_line = None
    for ln in sample_text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        first_data_line = s
        break

    has_header, sep, delim_ws = False, None, False
    if first_data_line is None:
        return has_header, sep, True

    if "," in first_data_line:
        sep = ","
        tokens = [t.strip() for t in first_data_line.split(",")]
    else:
        delim_ws = True
        tokens = first_data_line.split()

    def _is_num(tok: str) -> bool:
        try:
            float(tok); return True
        except Exception:
            return False
    non_num = sum(0 if _is_num(t) else 1 for t in tokens)
    has_header = non_num >= max(1, len(tokens) // 3)
    return has_header, sep, delim_ws

def _read_dat_to_df(uploaded) -> pd.DataFrame:
    sio = _open_text_from_upload(uploaded)
    sample = sio.getvalue()[:10_000]
    has_header, sep, delim_ws = _detect_header_and_sep(sample)

    sio.seek(0)
    try:
        if delim_ws:
            df = pd.read_csv(
                sio, comment="#", header=0 if has_header else None,
                delim_whitespace=True, engine="python",
            )
        else:
            df = pd.read_csv(
                sio, comment="#", header=0 if has_header else None,
                sep=sep, engine="python",
            )
    except Exception:
        sio.seek(0)
        df = pd.read_csv(
            sio, comment="#", header=0 if has_header else None,
            sep=r"\s+|,", engine="python",
        )

    if df.columns.dtype == "int64" or any(isinstance(c, (int, np.integer)) for c in df.columns):
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
    return df

def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _lon_wrap(lon: pd.Series) -> pd.Series:
    s = lon.copy()
    if s.max(skipna=True) > 180 and s.min(skipna=True) >= 0:
        s = ((s + 180) % 360) - 180
    return s

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
    st.caption("If your file has headers like 'lat'/'lon', I'll detect them. Otherwise, pick the columns below.")

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
cand_lat = [n for n in ["lat", "latitude", "lat_deg", "y", "phi", "col_1", "col_2"] if n in lower_names]
cand_lon = [n for n in ["lon", "longitude", "long", "lon_deg", "x", "lambda", "col_0", "col_3"] if n in lower_names]
lat_default = lower_names.get(cand_lat[0], None) if cand_lat else None
lon_default = lower_names.get(cand_lon[0], None) if cand_lon else None

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

work_df[lon_col] = _lon_wrap(work_df[lon_col])
work_df = work_df[np.isfinite(work_df[lat_col]) & np.isfinite(work_df[lon_col])].copy()

if work_df.empty:
    st.error("No valid latitude/longitude rows found after parsing. Check the column mapping and try again.")
    st.stop()

# ---------------------------
# Sidebar — Filters
# ---------------------------
with st.sidebar:
    st.header("3) Filters")
    lat_min = float(np.nanmin(work_df[lat_col])); lat_max = float(np.nanmax(work_df[lat_col]))
    lon_min = float(np.nanmin(work_df[lon_col])); lon_max = float(np.nanmax(work_df[lon_col]))
    lat_rng = st.slider("Latitude range", min_value=lat_min, max_value=lat_max, value=(lat_min, lat_max))
    lon_rng = st.slider("Longitude range", min_value=lon_min, max_value=lon_max, value=(lon_min, lon_max))

    alt_rng = None
    if alt_col:
        a_min = float(np.nanmin(work_df[alt_col])) if work_df[alt_col].notna().any() else 0.0
        a_max = float(np.nanmax(work_df[alt_col])) if work_df[alt_col].notna().any() else 0.0
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
# Map (Leaflet via HTML)
# ---------------------------
if plot_df.empty:
    st.warning("No points within the selected filters.")
else:
    # prepare data for JS
    rows = plot_df.to_dict(orient="records")
    data_json = json.dumps(rows)  # [{lat, lon, alt, <time?>}, ...]

    st_html(f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Leaflet Map</title>

  <!-- Leaflet -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <!-- MarkerCluster -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css" />
  <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>

  <!-- Heatmap -->
  <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>

  <style>
    html, body, #map {{ height: 700px; margin: 0; padding: 0; }}
    .summary {{
      position: absolute; top: 12px; left: 12px; z-index: 500;
      background: rgba(255,255,255,0.96); padding:8px 10px; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.2);
      font: 12px/1.2 system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="summary" id="summary">Rendering {len(rows)} points…</div>

  <script>
    const points = {data_json};

    // Center on mean lat/lon if available
    function mean(arr){ return arr.reduce((a,b)=>a+b,0)/Math.max(1,arr.length); }
    const lats = points.map(p => Number(p.lat)).filter(v => !Number.isNaN(v));
    const lons = points.map(p => Number(p.lon)).filter(v => !Number.isNaN(v));
    const center = [ (lats.length? mean(lats): 30.0), (lons.length? mean(lons): -95.0) ];

    const map = L.map('map').setView(center, 8);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
      {{ attribution: '© OpenStreetMap contributors' }}
    ).addTo(map);

    const cluster = L.markerClusterGroup({ disableClusteringAtZoom: 11 });

    function colorForAlt(a){ 
      if (a < 12000) return '#ffe633';
      if (a < 14000) return '#ffc300';
      if (a < 16000) return '#ff5733';
      return '#c70039';
    }

    const heatPts = [];
    let minLat=+90, maxLat=-90, minLon=+180, maxLon=-180;
    points.forEach(p => {{
      const lat = Number(p.lat), lon = Number(p.lon), alt = Number(p.alt);
      if (Number.isNaN(lat) || Number.isNaN(lon)) return;

      minLat = Math.min(minLat, lat); maxLat = Math.max(maxLat, lat);
      minLon = Math.min(minLon, lon); maxLon = Math.max(maxLon, lon);

      const c = colorForAlt(alt);
      const m = L.circleMarker([lat, lon], {{
        radius: 5,
        color: c, fillColor: c, fillOpacity: 0.9, opacity: 1, weight: 1
      }}).bindPopup(
        `<b>Alt:</b> ${isFinite(alt)? alt : 'n/a'} m<br>` +
        `<b>Lat/Lon:</b> ${lat.toFixed(4)}, ${lon.toFixed(4)}`
        {"+ (p.hasOwnProperty('"+time_col+"') && '"+time_col+"' !== '(none)' ? " + 
          "`<br><b>Time:</b> ${String(p['"+time_col+"'])}`" + " : '')" }
      );

      cluster.addLayer(m);

      // Heat intensity ~ scaled altitude (soft)
      const intensity = isFinite(alt) ? Math.max(0.3, Math.min(1, (alt-2000)/12000)) : 0.4;
      heatPts.push([lat, lon, intensity]);
    }});

    cluster.addTo(map);
    const heat = L.heatLayer(heatPts, {{ radius: 22, blur: 18, maxZoom: 11 }});
    // Toggle heat on by default? Comment out next line to start with markers only
    // heat.addTo(map);

    // Fit bounds to data (with padding)
    if (isFinite(minLat) && isFinite(maxLat) && isFinite(minLon) && isFinite(maxLon)) {{
      const pad = 0.05;
      map.fitBounds([[minLat - pad, minLon - pad], [maxLat + pad, maxLon + pad]]);
    }}

    document.getElementById('summary').innerHTML = 
      `<b>Rendered:</b> ${points.length} points<br>` +
      `<b>Extent:</b> lat ${minLat.toFixed(3)}…${maxLat.toFixed(3)}, lon ${minLon.toFixed(3)}…${maxLon.toFixed(3)}`;
  </script>
</body>
</html>
    """, height=720, scrolling=False)

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
    "Notes: Longitudes in 0..360 are wrapped to -180..180 for mapping. "
    "If auto-detection picks wrong columns, fix them in the sidebar."
)
