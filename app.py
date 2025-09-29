# app.py
# Streamlit app to upload an HLMA/LYLOUT .dat file and plot points on a map
# Run with: streamlit run app.py
# Requirements: streamlit, pandas, pydeck (pip install streamlit pandas pydeck)

import io
import re
from datetime import datetime

import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title=".dat → Map (HLMA Lightning)", layout="wide")
st.title("HLMA .dat → Interactive Map")
st.caption(
    "Upload a LYLOUT/HLMA exported .dat file. I'll parse latitude/longitude automatically and plot the points."
)

# -------------------------------
# Helpers
# -------------------------------

def _looks_header(line: str) -> bool:
    """Heuristic: if the first non-comment line contains letters, treat as header."""
    if not line:
        return False
    line = line.strip()
    if not line or line.startswith("#"):
        return False
    return bool(re.search(r"[A-Za-z]", line))


def _read_text(file) -> str:
    """Read uploaded file-like to text (UTF-8 replace errors)."""
    if hasattr(file, "getvalue"):
        raw = file.getvalue()
    else:
        raw = file.read()
    # try utf-8, fall back to latin-1
    try:
        return raw.decode("utf-8")
    except Exception:
        return raw.decode("latin-1", errors="replace")


def _parse_dat(text: str) -> pd.DataFrame:
    """Parse .dat text into a DataFrame with best-effort header detection."""
    # Strip leading comment lines to inspect first data/header line
    sio = io.StringIO(text)
    first_non_comment = None
    for _ in range(50):  # look at most 50 lines for a non-comment
        pos = sio.tell()
        line = sio.readline()
        if not line:
            break
        if not line.lstrip().startswith("#") and line.strip():
            first_non_comment = line
            # rewind to the beginning for full read with pandas
            sio.seek(0)
            break
    if first_non_comment is None:
        # No data found
        return pd.DataFrame()

    has_header = _looks_header(first_non_comment)

    # Pandas read with whitespace separator; skip comment lines starting with '#'
    sio2 = io.StringIO(text)
    try:
        if has_header:
            df = pd.read_csv(
                sio2,
                sep=r"\s+",
                engine="python",
                comment="#",
                header=0,
            )
        else:
            df = pd.read_csv(
                sio2,
                sep=r"\s+",
                engine="python",
                comment="#",
                header=None,
            )
    except Exception:
        # Fallback: try comma-separated
        sio3 = io.StringIO(text)
        df = pd.read_csv(sio3, engine="python", comment="#", header=0 if has_header else None)

    # Drop empty columns that sometimes appear from ragged whitespace
    df = df.dropna(axis=1, how="all")
    return df


LAT_NAMES = ["lat", "latitude", "y_deg", "y", "lat_deg"]
LON_NAMES = ["lon", "long", "longitude", "x_deg", "x", "lng"]
ALT_NAMES = ["alt", "altitude", "height", "z", "agl", "msl"]
TIME_NAMES = ["time", "timestamp", "datetime", "utc", "date", "epoch", "msec", "nsec"]


def _find_coord_columns(df: pd.DataFrame):
    """Return (lat_col, lon_col, alt_col?, time_col?) using header names or value ranges."""
    cols = list(df.columns)
    lower = [str(c).strip().lower() for c in cols]

    def _match(candidates):
        for alias in candidates:
            for i, name in enumerate(lower):
                # exact or contains alias as token
                if name == alias or re.search(rf"\b{re.escape(alias)}\b", name):
                    return cols[i]
        return None

    lat_col = _match(LAT_NAMES)
    lon_col = _match(LON_NAMES)
    alt_col = _match(ALT_NAMES)
    time_col = _match(TIME_NAMES)

    # If not found by name, infer by numeric ranges
    if lat_col is None or lon_col is None:
        numeric = df.select_dtypes(include=["number"])  # only numeric columns
        cand_lat, cand_lon = None, None
        for c in numeric.columns:
            series = pd.to_numeric(numeric[c], errors="coerce")
            vmin, vmax = series.min(), series.max()
            if vmin is None or pd.isna(vmin) or pd.isna(vmax):
                continue
            if -90.5 <= vmin <= 90.5 and -90.5 <= vmax <= 90.5:
                cand_lat = cand_lat or c
            if -181 <= vmin <= 181 and -181 <= vmax <= 181:
                # Likely lon if range is wider than latitude
                if c != cand_lat:
                    cand_lon = cand_lon or c
        # Fallback heuristic: pick the two numeric columns with widest ranges in plausible bounds
        if lat_col is None and cand_lat is not None:
            lat_col = cand_lat
        if lon_col is None and cand_lon is not None:
            lon_col = cand_lon
        # If still not found and at least 2 numeric columns exist, try the first two
        if (lat_col is None or lon_col is None) and numeric.shape[1] >= 2:
            c1, c2 = numeric.columns[:2]
            lat_col = lat_col or c2  # many files are (lon, lat)
            lon_col = lon_col or c1

    return lat_col, lon_col, alt_col, time_col


def _coerce_time(series: pd.Series) -> pd.Series:
    """Try multiple time formats: ISO, epoch seconds/ms/us/ns."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        # Decide unit by magnitude
        # Rough thresholds
        # seconds ~ 1e9, ms ~ 1e12, us ~ 1e15, ns ~ 1e18
        maxv = s.max()
        unit = "s"
        if maxv > 1e14:
            unit = "ns"
        elif maxv > 1e11:
            unit = "ms"
        elif maxv > 1e8:
            unit = "s"
        try:
            return pd.to_datetime(s, unit=unit, errors="coerce", utc=True)
        except Exception:
            pass
    # Fallback to generic parser
    return pd.to_datetime(series, errors="coerce", utc=True)


# -------------------------------
# UI
# -------------------------------
left, right = st.columns([2, 1])

with left:
    uploaded = st.file_uploader("Upload .dat (or .txt/.csv)", type=["dat", "txt", "csv"], accept_multiple_files=False)

    if uploaded is not None:
        text = _read_text(uploaded)
        df = _parse_dat(text)
        if df.empty:
            st.error("Couldn't read any tabular data from the file. Make sure it's not empty and doesn't contain only comments.")
            st.stop()

        lat_col, lon_col, alt_col, time_col = _find_coord_columns(df)
        if lat_col is None or lon_col is None:
            st.warning(
                "I couldn't confidently detect latitude/longitude columns. "
                "Please use the selectors in the sidebar to manually pick them."
            )
        # Keep a copy for manual selection
        st.session_state["raw_df"] = df
        st.session_state["autodetected"] = {
            "lat": lat_col,
            "lon": lon_col,
            "alt": alt_col,
            "time": time_col,
        }

with right:
    st.subheader("Settings")
    if "raw_df" not in st.session_state:
        st.info("Upload a file to configure settings.")
    else:
        df = st.session_state["raw_df"].copy()
        detected = st.session_state["autodetected"]
        cols = list(df.columns)

        lat_choice = st.selectbox(
            "Latitude column",
            options=[None] + cols,
            index=(cols.index(detected["lat"]) + 1) if detected["lat"] in cols else 0,
        )
        lon_choice = st.selectbox(
            "Longitude column",
            options=[None] + cols,
            index=(cols.index(detected["lon"]) + 1) if detected["lon"] in cols else 0,
        )
        alt_choice = st.selectbox(
            "Altitude column (optional)",
            options=[None] + cols,
            index=(cols.index(detected["alt"]) + 1) if detected["alt"] in cols else 0,
        )
        time_choice = st.selectbox(
            "Time column (optional)",
            options=[None] + cols,
            index=(cols.index(detected["time"]) + 1) if detected["time"] in cols else 0,
        )

        size_px = st.slider("Point radius (meters)", 30, 2000, 150)
        opacity = st.slider("Opacity", 0.05, 1.0, 0.7)

        # Filters (only if columns exist)
        if alt_choice is not None:
            alt_vals = pd.to_numeric(df[alt_choice], errors="coerce")
            alt_min, alt_max = float(pd.Series(alt_vals).quantile(0.01)), float(pd.Series(alt_vals).quantile(0.99))
            r = st.slider("Altitude range", min_value=alt_min, max_value=alt_max, value=(alt_min, alt_max))
        else:
            r = None

        if time_choice is not None:
            times = _coerce_time(df[time_choice])
            tmin, tmax = times.min(), times.max()
            # Streamlit doesn't have a timezone-aware slider; show text filter
            t1 = st.text_input("Start time (ISO, optional)", value=str(tmin) if pd.notna(tmin) else "")
            t2 = st.text_input("End time (ISO, optional)", value=str(tmax) if pd.notna(tmax) else "")
        else:
            t1 = t2 = None

        # Apply selections & filters when both lat/lon are present
        if lat_choice is not None and lon_choice is not None:
            gdf = pd.DataFrame({
                "lat": pd.to_numeric(df[lat_choice], errors="coerce"),
                "lon": pd.to_numeric(df[lon_choice], errors="coerce"),
            })
            if alt_choice is not None:
                gdf["altitude"] = pd.to_numeric(df[alt_choice], errors="coerce")
            if time_choice is not None:
                gdf["time"] = _coerce_time(df[time_choice])

            # Drop rows with invalid coords
            gdf = gdf.dropna(subset=["lat", "lon"]).query("-90 <= lat <= 90 and -180 <= lon <= 180")

            # Filters
            if r is not None and "altitude" in gdf:
                gdf = gdf[(gdf["altitude"] >= r[0]) & (gdf["altitude"] <= r[1])]
            if t1 and t2 and "time" in gdf:
                try:
                    tstart = pd.to_datetime(t1, utc=True)
                    tend = pd.to_datetime(t2, utc=True)
                    gdf = gdf[(gdf["time"] >= tstart) & (gdf["time"] <= tend)]
                except Exception:
                    pass

            st.markdown("### Preview")
            st.dataframe(gdf.head(500))

            if not gdf.empty:
                # View state
                center = [float(gdf["lat"].mean()), float(gdf["lon"].mean())]
                view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=7, pitch=0)

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=gdf,
                    get_position="[lon, lat]",
                    get_radius=size_px,
                    get_fill_color="[0, 0, 0]",  # default; use Map style color
                    pickable=True,
                    opacity=opacity,
                    radius_min_pixels=1,
                )

                tooltip = {
                    "html": "<b>Lat:</b> {lat}<br/><b>Lon:</b> {lon}" + ("<br/><b>Alt:</b> {altitude}" if "altitude" in gdf else "") + ("<br/><b>Time:</b> {time}" if "time" in gdf else ""),
                    "style": {"backgroundColor": "white", "color": "black"},
                }

                rdeck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v9")
                st.pydeck_chart(rdeck, use_container_width=True)

                # Offer CSV download of the parsed subset
                csv = gdf.to_csv(index=False).encode("utf-8")
                st.download_button("Download parsed CSV", data=csv, file_name="parsed_points.csv", mime="text/csv")
            else:
                st.warning("No valid coordinate rows to plot after filtering.")
        else:
            st.info("Select the latitude and longitude columns to plot.")

st.markdown("---")
st.markdown(
    "**Notes**:\n\n"
    "• Comments beginning with '#' are ignored.\n\n"
    "• If column names aren't present, the app guesses lat/lon using numeric ranges.\n\n"
    "• Optional altitude/time columns (if detected or selected) enable simple filtering.\n\n"
    "• Map uses PyDeck/Deck.GL; drag to pan, scroll to zoom, hover for tooltips."
)
