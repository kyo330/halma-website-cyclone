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

# =========================
# Page config
# =========================
st.set_page_config(page_title="HLMA Lightning — Clusters + Storm Paths", layout="wide")
st.title("⚡ HLMA Lightning — Clusters + 🌪️ Storm Reports")
st.caption("Upload HLMA .dat/.dat.gz for lightning and an optional storm CSV. Lightning is clustered; storms overlay as paths with start/end markers.")

# =========================
# Helpers
# =========================

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

def _read_hlma_block(text: str) -> pd.DataFrame | None:
    """
    HLMA export format:
      - header block
      - a sentinel line: *** data ***
      - numeric rows (typically 8 columns): time_uts lat lon alt_m chi2 nstations p_dbw mask
    Keep only numeric rows after sentinel, skip literal '...'.
    """
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if '*** data ***' in ln.lower():
            start = i + 1
            break
    if start is None:
        return None  # not a recognized HLMA export layout

    num_line = re.compile(r'^\s*[+-]?\d')  # line starts with a number
    data_lines = []
    for ln in lines[start:]:
        s = ln.strip()
        if not s or s == '...':
            continue
        if not num_line.match(s):
            continue
        data_lines.append(ln)

    if not data_lines:
        return pd.DataFrame()

    # Use whitespace delimiter; tolerate ragged rows
    try:
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            sep=r"\s+",
            engine="python",
            header=None,
            on_bad_lines="skip",
            skip_blank_lines=True,
        )
    except TypeError:
        # pandas < 1.3 fallback
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            sep=r"\s+",
            engine="python",
            header=None,
            error_bad_lines=False,
            warn_bad_lines=False,
            skip_blank_lines=True,
        )

    # Name columns when 8 present; otherwise map best-effort first 4
    if df.shape[1] == 8:
        df.columns = ["time_uts", "lat", "lon", "alt_m", "chi2", "nstations", "p_dbw", "mask"]
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

def _generic_parse(text: str) -> pd.DataFrame:
    """Rugged parser for arbitrary .dat: strip comments, try multiple strategies, never crash."""
    data_lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not data_lines:
        return pd.DataFrame()

    # 1) csv.Sniffer (sep=None)
    try:
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            sep=None,
            engine="python",
            header=None,
            on_bad_lines="skip",
            skip_blank_lines=True,
        )
    except TypeError:
        # older pandas
        try:
            df = pd.read_csv(
                io.StringIO("\n".join(data_lines)),
                sep=None,
                engine="python",
                header=None,
                error_bad_lines=False,
                warn_bad_lines=False,
                skip_blank_lines=True,
            )
        except Exception:
            df = None
    except Exception:
        df = None

    # 2) regex delimiter (whitespace OR comma)
    if df is None or df.empty:
        try:
            df = pd.read_csv(
                io.StringIO("\n".join(data_lines)),
                sep=r"\s+|,",
                engine="python",
                header=None,
                on_bad_lines="skip",
                skip_blank_lines=True,
            )
        except TypeError:
            try:
                df = pd.read_csv(
                    io.StringIO("\n".join(data_lines)),
                    sep=r"\s+|,",
                    engine="python",
                    header=None,
                    error_bad_lines=False,
                    warn_bad_lines=False,
                    skip_blank_lines=True,
                )
            except Exception:
                df = None
        except Exception:
            df = None

    # 3) last resort: manual split normalized to most-common width
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
                parts += [""] * (most_w - len(parts))
            elif len(parts) > most_w:
                parts = parts[:most_w]
            fixed.append(parts)
        df = pd.DataFrame(fixed)

    # give generic names if pandas left them numeric
    if df.columns.dtype == "int64" or any(isinstance(c, (int, np.integer)) for c in df.columns):
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df

def _read_dat_to_df(uploaded) -> pd.DataFrame:
    """
    Unified, non-crashy entry point:
      1) Try HLMA-aware block reader (*** data ***).
      2) Else, generic robust parser.
    """
    sio = _open_text_from_upload(uploaded)
    text = sio.getvalue()

    df = _read_hlma_block(text)
    if df is None:
        df = _generic_parse(text)

    if df.empty:
        return df

    # Normalize any unnamed columns
    df.columns = [
        f"col_{i}" if (str(c).strip() == "" or str(c).lower().startswith("unnamed")) else str(c)
        for i, c in enumerate(df.columns)
    ]
    return df

def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _lon_wrap(lon: pd.Series) -> pd.Series:
    """Wrap longitudes to [-180, 180] if they look like [0, 360]."""
    s = lon.copy()
    if s.max(skipna=True) > 180 and s.min(skipna=True) >= 0:
        s = ((s + 180) % 360) - 180
    return s

# =========================
# Sidebar — Uploaders
# =========================
with st.sidebar:
    st.header("1) Upload HLMA .dat / .dat.gz")
    up_dat = st.file_uploader("Choose a .dat or .dat.gz file", type=["dat", "gz"], accept_multiple_files=False)

    st.header("2) Upload storm CSV (optional)")
    up_csv = st.file_uploader("CSV with storm paths (BEGIN_/END_ columns)", type=["csv"], accept_multiple_files=False)
    st.caption("Needs: BEGIN_LAT/LON, END_LAT/LON. Other fields (EF, length, width, etc.) are optional but shown if present.")

if up_dat is None:
    st.info("Upload an HLMA .dat/.dat.gz file to start. Then add a storm CSV to overlay.")
    st.stop()

# =========================
# Parse HLMA .dat
# =========================
raw_df = _read_dat_to_df(up_dat)

# Guess lat/lon defaults
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

# Clean lightning data
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
    st.error("No valid latitude/longitude rows found after parsing.")
    st.stop()

# =========================
# HLMA Filters
# =========================
with st.sidebar:
    st.header("3) Filters (Lightning)")
    lat_min = float(np.nanmin(work_df[lat_col])); lat_max = float(np.nanmax(work_df[lat_col]))
    lon_min = float(np.nanmin(work_df[lon_col])); lon_max = float(np.nanmax(work_df[lon_col]))
    lat_rng = st.slider("Latitude range", min_value=lat_min, max_value=lat_max, value=(lat_min, lat_max))
    lon_rng = st.slider("Longitude range", min_value=lon_min, max_value=lon_max, value=(lon_min, lon_max))

    alt_rng = None
    if work_df[alt_col].notna().any():
        a_min = float(np.nanmin(work_df[alt_col])); a_max = float(np.nanmax(work_df[alt_col]))
        if a_min != a_max:
            alt_rng = st.slider("Altitude range (units as in file)", min_value=a_min, max_value=a_max, value=(a_min, a_max))

flt = (work_df[lat_col].between(lat_rng[0], lat_rng[1])) & (work_df[lon_col].between(lon_rng[0], lon_rng[1]))
if alt_rng is not None:
    flt &= work_df[alt_col].between(alt_rng[0], alt_rng[1])

plot_df = work_df.loc[flt, [lat_col, lon_col, alt_col] + ([time_col] if time_col != "(none)" else [])].copy()
plot_df.rename(columns={lat_col: "lat", lon_col: "lon", alt_col: "alt"}, inplace=True)

# =========================
# Storm CSV (optional)
# =========================
storm_rows = []
if up_csv is not None:
    try:
        sdf = pd.read_csv(up_csv)
        req = {"BEGIN_LAT","BEGIN_LON","END_LAT","END_LON"}
        if not req.issubset(set(sdf.columns)):
            st.warning("Storm CSV missing required columns: BEGIN_LAT, BEGIN_LON, END_LAT, END_LON.")
        else:
            def safe_str(x):
                try:
                    return "" if (pd.isna(x)) else str(x)
                except Exception:
                    return ""
            for _, r in sdf.iterrows():
                try:
                    b_lat = float(r["BEGIN_LAT"]); b_lon = float(r["BEGIN_LON"])
                    e_lat = float(r["END_LAT"]);   e_lon = float(r["END_LON"])
                except Exception:
                    continue
                meta = {
                    "ef": safe_str(r.get("TOR_F_SCALE","")),
                    "len": safe_str(r.get("TOR_LENGTH","")),
                    "wid": safe_str(r.get("TOR_WIDTH","")),
                    "wfo": safe_str(r.get("WFO","")),
                    "state": safe_str(r.get("STATE_ABBR","")),
                    "begin_loc": safe_str(r.get("BEGIN_LOCATION","")),
                    "end_loc": safe_str(r.get("END_LOCATION","")),
                    "begin_time": safe_str(r.get("BEGIN_TIME","")),
                    "end_date": safe_str(r.get("END_DATE","")),
                    "end_time": safe_str(r.get("END_TIME","")),
                    "narr": safe_str(r.get("EVENT_NARRATIVE",""))
                }
                storm_rows.append({
                    "b_lat": b_lat, "b_lon": b_lon,
                    "e_lat": e_lat, "e_lon": e_lon,
                    "meta": meta
                })
    except Exception as e:
        st.warning(f"Could not read storm CSV: {e}")

st.success(f"Lightning parsed: {len(work_df):,} rows • Plotted: {len(plot_df):,} • Storm paths: {len(storm_rows)}")

# =========================
# Map (Leaflet with clustered lightning + zoomed-in detail)
# =========================
if plot_df.empty and not storm_rows:
    st.warning("Nothing to plot yet. Adjust filters or upload a storm CSV.")
else:
    lightning_json = json.dumps(plot_df.to_dict(orient="records"))   # [{lat,lon,alt,(time?)}]
    time_field = time_col if time_col != "(none)" else ""
    storm_json = json.dumps(storm_rows)                               # [{b_lat,b_lon,e_lat,e_lon,meta:{...}}]

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

  <!-- MarkerCluster (for lightning) -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css" />
  <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>

  <style>
    html, body, #map { height: 740px; margin: 0; padding: 0; }
    .legend {
      position: absolute; bottom: 16px; left: 16px; z-index: 500;
      background: rgba(255,255,255,0.97); padding:10px 12px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.2);
      font: 12px/1.2 system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; min-width: 240px;
    }
    .legend h4 { margin:0 0 6px 0; font-size: 13px; }
    .legend-item { display:flex; align-items:center; gap:8px; margin:4px 0; }
    .dot { width:12px; height:12px; border-radius:50%; border:1px solid rgba(0,0,0,0.25); display:inline-block; }
    .dot.low { background:#ffe633; }
    .dot.med { background:#ffc300; }
    .dot.high { background:#ff5733; }
    .dot.extreme { background:#c70039; }
    .line { width:22px; height:4px; background:#2e7d32; border:1px solid rgba(0,0,0,0.25); }
    .layers-box {
      position:absolute; top: 12px; left: 12px; z-index:600;
      background: rgba(255,255,255,0.96); padding:8px 10px; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.2);
      font: 12px/1.2 system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }
    .layers-box label { display:block; margin:4px 0; }

    .dot-icon { display:inline-block; width:12px; height:12px; }
    .leaflet-div-icon { background: transparent; border: none; } /* remove default box */
  </style>
</head>
<body>
  <div id="map"></div>

  <!-- Layer toggles -->
  <div class="layers-box">
    <label><input type="checkbox" id="toggleLightning" checked> Show lightning (cluster/detail)</label>
    <label><input type="checkbox" id="toggleStorms" __STORMCHECK__> Show storm reports</label>
  </div>

  <!-- Color legend -->
  <div class="legend" id="legend">
    <h4>Legend</h4>
    <div class="legend-item"><span class="dot low"></span> <span>&lt; 12 km</span></div>
    <div class="legend-item"><span class="dot med"></span> <span>12–14 km</span></div>
    <div class="legend-item"><span class="dot high"></span> <span>14–16 km</span></div>
    <div class="legend-item"><span class="dot extreme"></span> <span>&gt; 16 km</span></div>
    <div class="legend-item"><span class="line"></span> <span>Storm path (begin→end)</span></div>
  </div>

  <script>
    // Injected data
    const points = __LIGHTNING__;
    const timeField = __TIMEFIELD__;
    const storms = __STORMS__;

    function mean(arr){ return arr.reduce(function(a,b){return a+b;},0)/Math.max(1,arr.length); }
    function tierForAlt(a){
      if (a < 12000) return 'low';
      if (a < 14000) return 'med';
      if (a < 16000) return 'high';
      return 'extreme';
    }
    function colorForTier(t){
      if (t==='low') return '#ffe633';
      if (t==='med') return '#ffc300';
      if (t==='high') return '#ff5733';
      return '#c70039';
    }

    // Initial view: prefer lightning, fallback to storms
    var lats = [], lons = [];
    for (var i=0;i<points.length;i++){
      var p = points[i];
      var la = Number(p.lat), lo = Number(p.lon);
      if (!isNaN(la)) lats.push(la);
      if (!isNaN(lo)) lons.push(lo);
    }
    if (lats.length === 0 && storms.length > 0) {
      for (var i=0;i<storms.length;i++){
        lats.push(Number(storms[i].b_lat), Number(storms[i].e_lat));
        lons.push(Number(storms[i].b_lon), Number(storms[i].e_lon));
      }
    }
    var center = [(lats.length? mean(lats): 30.0), (lons.length? mean(lons): -95.0)];
    var map = L.map('map').setView(center, 8);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      { attribution: '© OpenStreetMap contributors' }
    ).addTo(map);

    // ---------- Lightning layers ----------
    // Clustered (zoomed out)
    var lightningCluster = L.markerClusterGroup({
      disableClusteringAtZoom: 12,         // switch at 12
      maxClusterRadius: 60,
      spiderfyOnMaxZoom: true,
      showCoverageOnHover: false
    });

    // Detail vector layer (zoomed in)
    var lightningDetail = L.layerGroup();

    function makeSVGIcon(color) {
      const svg = `
        <svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'>
          <circle cx='6' cy='6' r='4.5' fill='${color}' stroke='rgba(0,0,0,0.45)' stroke-width='1'/>
        </svg>`;
      return L.divIcon({
        html: svg,
        className: 'dot-icon',
        iconSize: [12,12],
        iconAnchor: [6,6]
      });
    }

    function renderLightning(){
      lightningCluster.clearLayers();
      lightningDetail.clearLayers();

      for (var i=0;i<points.length;i++){
        var p = points[i];
        var lat = Number(p.lat), lon = Number(p.lon), alt = Number(p.alt);
        if (isNaN(lat) || isNaN(lon)) continue;

        var tier = tierForAlt(alt);
        var color = colorForTier(tier);
        var popup = "<b>Alt:</b> " + (isFinite(alt)? alt : "n/a") + " m<br>"
                  + "<b>Lat/Lon:</b> " + lat.toFixed(4) + ", " + lon.toFixed(4);
        if (timeField && (timeField in p)) popup += "<br><b>Time:</b> " + String(p[timeField]);

        // Cluster marker (L.marker)
        var m = L.marker([lat, lon], { icon: makeSVGIcon(color), title: "Lightning" }).bindPopup(popup);
        lightningCluster.addLayer(m);

        // Detail marker (L.circleMarker) for crisp look when zoomed in
        var cm = L.circleMarker([lat, lon], {
          radius: 4.5, color: 'rgba(0,0,0,0.45)', weight: 1,
          fillColor: color, fillOpacity: 0.95
        }).bindPopup(popup);
        lightningDetail.addLayer(cm);
      }
    }
    renderLightning();

    // Start with clusters shown (detail auto when zoomed in)
    lightningCluster.addTo(map);

    // Flip between cluster/detail based on zoom
    function updateLightningVisibility(){
      var z = map.getZoom();
      if (z >= 12) {           // detail view
        if (map.hasLayer(lightningCluster)) map.removeLayer(lightningCluster);
        if (!map.hasLayer(lightningDetail)) map.addLayer(lightningDetail);
      } else {                 // clustered view
        if (map.hasLayer(lightningDetail)) map.removeLayer(lightningDetail);
        if (!map.hasLayer(lightningCluster)) map.addLayer(lightningCluster);
      }
    }
    map.on('zoomend', updateLightningVisibility);
    updateLightningVisibility();

    // ---------- Storm layer (paths + start/end markers) ----------
    var stormsGroup = L.layerGroup();
    function safe(x){ return (x===null || x===undefined) ? "" : String(x); }
    function buildStormPopup(meta, isStart){
      var label = isStart ? "<b>Start</b>" : "<b>End</b>";
      var ef = safe(meta.ef), len = safe(meta.len), wid = safe(meta.wid), wfo = safe(meta.wfo);
      var st = safe(meta.state), bl = safe(meta.begin_loc), el = safe(meta.end_loc);
      var bt = safe(meta.begin_time), ed = safe(meta.end_date), et = safe(meta.end_time);
      var narr = safe(meta.narr);
      return label + "<br>"
        + (ef? ("EF: " + ef + "<br>") : "")
        + (len? ("Length: " + len + " mi<br>") : "")
        + (wid? ("Width: " + wid + " yd<br>") : "")
        + (wfo? ("WFO: " + wfo + "<br>") : "")
        + (st? ("State: " + st + "<br>") : "")
        + (bl? ("Begin loc: " + bl + "<br>") : "")
        + (el? ("End loc: " + el + "<br>") : "")
        + (bt? ("Begin time: " + bt + "<br>") : "")
        + ((ed||et)? ("End: " + ed + " " + et + "<br>") : "")
        + (narr? ("<hr style='margin:6px 0'><div style='max-width:260px; white-space:normal'>" + narr + "</div>") : "");
    }
    function renderStorms(){
      stormsGroup.clearLayers();
      for (var i=0;i<storms.length;i++){
        var s = storms[i];
        var bl = [Number(s.b_lat), Number(s.b_lon)], el = [Number(s.e_lat), Number(s.e_lon)];
        if (!isFinite(bl[0]) || !isFinite(bl[1]) || !isFinite(el[0]) || !isFinite(el[1])) continue;

        L.polyline([bl, el], { color:'#2e7d32', weight:4, opacity:0.95 }).addTo(stormsGroup);
        L.circleMarker(bl, { radius: 6, color:'#2e7d32', fillColor:'#2e7d32', fillOpacity:0.95 })
          .bindPopup(buildStormPopup(s.meta, true)).addTo(stormsGroup);
        L.circleMarker(el, { radius: 6, color:'#b71c1c', fillColor:'#b71c1c', fillOpacity:0.95 })
          .bindPopup(buildStormPopup(s.meta, false)).addTo(stormsGroup);
      }
    }
    if (storms.length > 0){
      renderStorms();
      stormsGroup.addTo(map); // overlay on top of lightning
    }

    // Fit bounds to all data
    function fitData(){
      var have=false, minLat=+90, maxLat=-90, minLon=+180, maxLon=-180;
      for (var i=0;i<points.length;i++){
        var p = points[i]; var la = Number(p.lat), lo = Number(p.lon);
        if (!isNaN(la) && !isNaN(lo)){ have=true; if (la<minLat)minLat=la; if(la>maxLat)maxLat=la; if(lo<minLon)minLon=lo; if(lo>maxLon)maxLon=lo; }
      }
      for (var i=0;i<storms.length;i++){
        var s = storms[i];
        var bl = [Number(s.b_lat), Number(s.b_lon)], el = [Number(s.e_lat), Number(s.e_lon)];
        if (isFinite(bl[0]) && isFinite(bl[1])){ have=true; if (bl[0]<minLat)minLat=bl[0]; if(bl[0]>maxLat)maxLat=bl[0]; if(bl[1]<minLon)minLon=bl[1]; if(bl[1]>maxLon)maxLon=bl[1]; }
        if (isFinite(el[0]) && isFinite(el[1])){ have=true; if (el[0]<minLat)minLat=el[0]; if(el[0]>maxLat)maxLat=el[0]; if(el[1]<minLon)minLon=el[1]; if(el[1]>maxLon)maxLon=el[1]; }
      }
      if (have){
        var pad = 0.05;
        map.fitBounds([[minLat - pad, minLon - pad], [maxLat + pad, maxLon + pad]]);
      }
    }
    fitData();

    // Layer toggles
    var cbL = document.getElementById('toggleLightning');
    var cbS = document.getElementById('toggleStorms');

    function syncLightning(){
      var z = map.getZoom();
      if (!cbL.checked){
        if (map.hasLayer(lightningCluster)) map.removeLayer(lightningCluster);
        if (map.hasLayer(lightningDetail))  map.removeLayer(lightningDetail);
        return;
      }
      // respect zoom switch
      if (z >= 12) {
        if (map.hasLayer(lightningCluster)) map.removeLayer(lightningCluster);
        if (!map.hasLayer(lightningDetail)) map.addLayer(lightningDetail);
      } else {
        if (map.hasLayer(lightningDetail)) map.removeLayer(lightningDetail);
        if (!map.hasLayer(lightningCluster)) map.addLayer(lightningCluster);
      }
    }
    function syncStorms(){
      if (cbS.checked){ stormsGroup.addTo(map); }
      else { map.removeLayer(stormsGroup); }
    }
    cbL.addEventListener('change', syncLightning);
    cbS.addEventListener('change', syncStorms);
    map.on('zoomend', syncLightning);

    // Init
    syncLightning();
    syncStorms();
  </script>
</body>
</html>
    """

    html = (
        html_template
        .replace("__LIGHTNING__", lightning_json)
        .replace("__TIMEFIELD__", json.dumps(time_field))
        .replace("__STORMS__", storm_json)
        .replace("__STORMCHECK__", "checked" if len(storm_rows) > 0 else "")
    )

    st_html(html, height=780, scrolling=False)

# =========================
# Data Preview & Export
# =========================
with st.expander("Preview lightning table"):
    st.dataframe(plot_df.head(500), use_container_width=True)

if storm_rows:
    with st.expander("Preview storm rows"):
        st.dataframe(pd.DataFrame(storm_rows).head(500), use_container_width=True)

csv = plot_df.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered lightning CSV",
                   data=csv,
                   file_name=(Path(up_dat.name).stem + "_filtered.csv"),
                   mime="text/csv")

st.caption(
    "Lightning markers are clustered (zoomed out) and switch to round vector points at zoom ≥ 12. "
    "Storm reports overlay as green paths with start (green) and end (red) markers."
)
