# app.py
# Streamlit app to upload an HLMA/LYLOUT .dat file and plot points on a map
# Run: streamlit run app.py

import io
import re
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title=".dat → Map (HLMA Lightning)", layout="wide")
st.title("HLMA .dat → Interactive Map")
st.caption("Upload a LYLOUT/HLMA exported .dat file. I’ll parse latitude/longitude and plot the points.")

# -------------------------------
# Helpers
# -------------------------------

def _read_text(file) -> str:
    """Read uploaded file-like to text with tolerant decoding."""
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    try:
        return raw.decode("utf-8")
    except Exception:
        return raw.decode("latin-1", errors="replace")

def _parse_dat(text: str) -> pd.DataFrame:
    """
    Robust for NM Tech LMA 'Selected Data' exports:
      time_sec, lat, lon, alt_m, chi2, nstations, p_dbw, mask
    Preferred path: parse lines after the '*** data ***' marker.
    Fallback: generic whitespace parsing with on_bad_lines='skip'.
    """
    lines = text.splitlines()

    # Look for the explicit data marker first
    start_idx = None
    for i, line in enumerate(lines[:200]):
        if line.strip().lower().startswith("*** data ***"):
            start_idx = i + 1
            break

    if start_idx is not None:
        rows = []
        for ln in lines[start_idx:]:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            low = s.lower()
            if low.startswith("flash stats") or low.startswith("number of events"):
                break
            toks = s.split()
            # accept rows with >=7 tokens; pad/truncate to 8 fields
            if len(toks) >= 7:
                toks = (toks + [None]*8)[:8]
                rows.append(toks)

        if rows:
            df = pd.DataFrame(
                rows,
                columns=["time_sec","lat","lon","alt_m","chi2","nstations","p_dbw","mask"]
            )
            # coerce numerics
            for c in ["time_sec","lat","lon","alt_m","chi2","nstations","p_dbw"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df

    # Fallback path for “other” .dat styles
    # Heuristic: detect if first non-comment line looks like a header
    sio = io.StringIO(text)
    first_non_comment = None
    for _ in range(50):
        line = sio.readline()
        if not line:
            break
        if not line.lstrip().startswith("#") and line.strip():
            first_non_comment = line
            sio.seek(0)
            break

    if first_non_comment is None:
        return pd.DataFrame()

    has_letters = bool(re.search(r"[A-Za-z]", first_non_comment))

    sio2 = io.StringIO(text)
    try:
        df = pd.read_csv(
            sio2,
            sep=r"\s+",
            engine="python",
            comment="#",
            header=0 if has_letters else None,
            on_bad_lines="skip",
        )
    except Exception:
        sio3 = io.StringIO(text)
        df = pd.read_csv(
            sio3,
            engine="python",
            comment="#",
            header=0 if has_letters else None,
            on_bad_lines="skip",
        )
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
                if name == alias or re.search(rf"\b{re.escape(alias)}\b", name):
                    return cols[i]
        return None

    lat_col = _match(LAT_NAMES)
    lon_col = _match(LON_NAMES)
    alt_col = _match(ALT_NAMES)
    time_col = _match(TIME_NAMES)

    # If not found by name, infer by numeric ranges
    if lat_col is None or lon_col is None:
        numeric = df.select_dtypes(include=["number"])
        cand_lat, cand_lon = None, None
        for c in numeric.columns:
            s = pd.to_numeric(numeric[c], errors="coerce")
            vmin, vmax = s.min(), s.max()
            if pd.isna(vmin) or pd.isna(vmax):
                continue
            if -90.5 <= vmin <= 90.5 and -90.5 <= vmax <= 90.5:
                cand_lat = cand_lat or c
            if -181 <= vmin <= 181 and -181 <= vmax <= 181 and c != cand_lat:
                cand_lon = cand_lon or c
        if lat_col is None and cand_lat is not None:
            lat_col = cand_lat
        if lon_col is None and cand_lon is not None:
            lon_col = cand_lon
        if (lat_col is None or lon_col is None) and numeric.shape[1] >= 2:
            c1, c2 = numeric.columns[:2]
            lat_col = lat_col or c2  # common ordering is (lon, lat)
            lon_col = lon_col or c1

    return lat_col, lon_col, alt_col, time_col

def _coerce_time(series: pd.Series) -> pd.Series:
    """Try ISO or epoch (s/ms/us/ns)."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
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
    return pd.to_datetime(series, errors="coerce", utc=True)

# -------------------------------
# UI
# -------------------------------
left, right = st.columns([2, 1])

with left:
    uploaded = st.file_uploader(
        "Upload .dat (or .txt/.csv)", type=["dat", "txt", "csv"], accept_multiple_files=False
    )

    if uploaded is not None:
        text = _read_text(uploaded)
        try:
            df = _parse_dat(text)
        except Exception as e:
            st.error("Failed to parse file.")
            st.exception(e)  # show full traceback in the app
            st.stop()

        if df.empty:
            st.error("Couldn't read any tabular data from the file. Is it the 'Selected Data' export?")
            st.stop()

        # quick sanity print (helps confirm the change took effect)
        st.write("Parsed rows:", len(df), "Columns:", list(df.columns))

        lat_col, lon_col, alt_col, time_col = _find_coord_columns(df)
        if lat_col is None or lon_col is None:
            st.warning(
                "I couldn't confidently detect latitude/longitude columns. "
                "Use the selectors in the sidebar to pick them."
            )

        st.session_state["raw_df"] = df
        st.session_state["autodetected"] = {"lat": lat_col, "lon": lon_col, "alt": alt_col, "time": time_col}

with right:
    st.subheader("Settings")
    if "raw_df" not in st.session_state:
        st.info("Upload a file to configure settings.")
    else:
        df = st.session_state["raw_df"].copy()
        detected = st.session_state["autodetected"]
        cols = list(df.columns)

        lat_choice = st.selectbox(
            "Latitude column", options=[None] + cols,
            index=(cols.index(detected["lat"]) + 1) if detected["lat"] in cols else 0,
        )
        lon_choice = st.selectbox(
            "Longitude column", options=[None] + cols,
            index=(cols.index(detected["lon"]) + 1) if detected["lon"] in cols else 0,
        )
        alt_choice = st.selectbox(
            "Altitude column (optional)", options=[None] + cols,
            index=(cols.index(detected["alt"]) + 1) if detected["alt"] in cols else 0,
        )
        time_choice = st.selectbox(
            "Time column (optional)", options=[None] + cols,
            index=(cols.index(detected["time"]) + 1) if detected["time"] in cols else 0,
        )

        size_px = st.slider("Point radius (meters)", 30, 2000, 150)
        opacity = st.slider("Opacity", 0.05, 1.0, 0.7)

        if alt_choice is not None:
            alt_vals = pd.to_numeric(df[alt_choice], errors="coerce")
            alt_min, alt_max = float(pd.Series(alt_vals).quantile(0.01)), float(pd.Series(alt_vals).quantile(0.99))
            r = st.slider("Altitude range", min_value=alt_min, max_value=alt_max, value=(alt_min, alt_max))
        else:
            r = None

        if time_choice is not None:
            times = _coerce_time(df[time_choice])
            tmin, tmax = times.min(), times.max()
            t1 = st.text_input("Start time (ISO, optional)", value=str(tmin) if pd.notna(tmin) else "")
            t2 = st.text_input("End time (ISO, optional)", value=str(tmax) if pd.notna(tmax) else "")
        else:
            t1 = t2 = None

        if lat_choice is not None and lon_choice is not None:
            gdf = pd.DataFrame({
                "lat": pd.to_numeric(df[lat_choice], errors="coerce"),
                "lon": pd.to_numeric(df[lon_choice], errors="coerce"),
            })
            if alt_choice is not None:
                gdf["altitude"] = pd.to_numeric(df[alt_choice], errors="coerce")
            if time_choice is not None:
                gdf["time"] = _coerce_time(df[time_choice])

            gdf = gdf.dropna(subset=["lat", "lon"]).query("-90 <= lat <= 90 and -180 <= lon <= 180")

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
                center = [float(gdf["lat"].mean()), float(gdf["lon"].mean())]
                view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=7, pitch=0)

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=gdf,
                    get_position="[lon, lat]",
                    get_radius=size_px,
                    get_fill_color="[0, 0, 0]",
                    pickable=True,
                    opacity=opacity,
                    radius_min_pixels=1,
                )

                rdeck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/light-v9",
                )
                st.pydeck_chart(rdeck, use_container_width=True)

                csv = gdf.to_csv(index=False).encode("utf-8")
                st.download_button("Download parsed CSV", data=csv, file_name="parsed_points.csv", mime="text/csv")
            else:
                st.warning("No valid coordinate rows to plot after filtering.")
        else:
            st.info("Select the latitude and longitude columns to plot.")

st.markdown("---")
st.markdown(
    "**Notes**:\n\n"
    "• If column names aren't present, the app guesses lat/lon using numeric ranges.\n\n"
    "• Optional altitude/time columns (if selected) enable simple filtering.\n\n"
    "• Map uses PyDeck/Deck.GL; drag to pan, scroll to zoom, hover for tooltips."
)
