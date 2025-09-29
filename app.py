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
st.set_page_config(page_title="HLMA Lightning — .dat Upload & Leaflet Map", layout="wide")
st.title("⚡ HLMA Lightning — .dat Uploader & Map")
st.caption("Upload an HLMA .dat or .dat.gz file, parse robustly, and plot on a Leaflet map.")

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
      - assign HLMA column names (time_uts, lat, lon, alt_m, chi2, nstations, p_dbw, mask)
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
    """Generic robust parser for non-HLMA files."""
    data_lines = [ln for ln in s.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    if not data_lines:
        return pd.DataFrame()

    df = None
    try:
        df = pd.read_csv(io.StringIO("\n".join(data_lines)),
                         sep=None, engine="python",
                         header=None, on_bad_lines="skip",
                         skip_blank_lines=True)
    except Exception:
        df = None

    if df is None or df.empty:
        try:
            df = pd.read_csv(io.StringIO("\n".join(data_lines)),
                             sep=r"\s+|,", engine="python",
                             header=None, on_bad_lines="skip",
                             skip_blank_lines=True)
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

    if df.empty:
        return df

    # Normalize any unnamed columns
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
    st.info("Upload an HLMA .dat (or .dat.gz) file. Header + '*** data ***' are handled automatically; '...' lines are skipped.")
    st.stop()

# Parse
raw_df = _read_dat_to_df(up)

# Auto-detect likely latitude/longitude column names
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
    st.error("No valid latitude/longitude rows found after parsing. If this is an HLMA export, ensure the file includes '*** data ***' with numeric rows after it.")
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
    if work_df[alt_col].notna().any():
        a_min = float(np.nanmin(work_df[alt_col])); a_max = float(np.nanmax(work_df[alt_col]))
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
# Map (Leaflet via HTML; no f-strings)
# ---------------------------
if plot_df.empty:
    st.warning("No points within the selected filters.")
else:
    rows = plot_df.to_dict(orient="records")
    data_json = json.dumps(rows)                  # JSON string for injection
    time_field = time_col if time_col != "(none)" else ""  # pass empty when no time

    html_template = """
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

  <!-- Heatmap (optional) -->
  <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>

  <style>
    html, body, #map { height: 700px; margin: 0; padding: 0; }
    .summary {
      position: absolute; top: 12px; left: 12px; z-index: 500;
      background: rgba(255,255,255,0.96); padding:8px 10px; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.2);
      font: 12px/1.2 system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="summary" id="summary">Rendering __COUNT__ points…</div>

  <script>
    // Injected data
    const points = __DATA__;
    const timeField = __TIMEFIELD__;

    function mean(arr){ return arr.reduce(function(a,b){return a+b;},0)/Math.max(1,arr.length); }
    function colorForAlt(a){
      if (a < 12000) return '#ffe633';
      if (a < 14000) return '#ffc300';
      if (a < 16000) return '#ff5733';
      return '#c70039';
    }

    // Compute center
    var lats = [], lons = [];
    for (var i=0;i<points.length;i++){
      var p = points[i];
      var lat = Number(p.lat), lon = Number(p.lon);
      if (!isNaN(lat)) lats.push(lat);
      if (!isNaN(lon)) lons.push(lon);
    }
    var center = [(lats.length? mean(lats): 30.0), (lons.length? mean(lons): -95.0)];

    var map = L.map('map').setView(center, 8);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      { attribution: '© OpenStreetMap contributors' }
    ).addTo(map);

    var cluster = L.markerClusterGroup({ disableClusteringAtZoom: 11 });
    var heatPts = [];
    var minLat=+90, maxLat=-90, minLon=+180, maxLon=-180;

    for (var i=0;i<points.length;i++){
      var p = points[i];
      var lat = Number(p.lat), lon = Number(p.lon), alt = Number(p.alt);
      if (isNaN(lat) || isNaN(lon)) continue;

      if (lat < minLat) minLat = lat; if (lat > maxLat) maxLat = lat;
      if (lon < minLon) minLon = lon; if (lon > maxLon) maxLon = lon;

      var c = colorForAlt(alt);
      var popup = "<b>Alt:</b> " + (isFinite(alt)? alt : "n/a") + " m<br>"
                + "<b>Lat/Lon:</b> " + lat.toFixed(4) + ", " + lon.toFixed(4);
      if (timeField && (timeField in p)) {
        popup += "<br><b>Time:</b> " + String(p[timeField]);
      }

      var m = L.circleMarker([lat, lon], {
        radius: 5, color: c, fillColor: c, fillOpacity: 0.9, opacity: 1, weight: 1
      }).bindPopup(popup);

      cluster.addLayer(m);

      var intensity = isFinite(alt) ? Math.max(0.3, Math.min(1, (alt-2000)/12000)) : 0.4;
      heatPts.push([lat, lon, intensity]);
    }

    cluster.addTo(map);

    // Optional heat layer:
    // var heat = L.heatLayer(heatPts, { radius: 22, blur: 18, maxZoom: 11 }).addTo(map);

    if (isFinite(minLat) && isFinite(maxLat) && isFinite(minLon) && isFinite(maxLon)){
      var pad = 0.05;
      map.fitBounds([[minLat - pad, minLon - pad], [maxLat + pad, maxLon + pad]]);
    }

    document.getElementById('summary').innerHTML =
      "<b>Rendered:</b> " + points.length + " points<br>" +
      "<b>Extent:</b> lat " + minLat.toFixed(3) + "…" + maxLat.toFixed(3) +
      ", lon " + minLon.toFixed(3) + "…" + maxLon.toFixed(3);
  </script>
</body>
</html>
    """

    html = (
        html_template
        .replace("__COUNT__", str(len(rows)))
        .replace("__DATA__", data_json)
        .replace("__TIMEFIELD__", json.dumps(time_field))
    )

    st_html(html, height=720, scrolling=False)

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
    "Notes: HLMA parser reads only the numeric rows after '*** data ***' and skips '...'. "
    "Longitudes in 0..360 are wrapped to -180..180. Toggle a time column to see times in popups."
)
