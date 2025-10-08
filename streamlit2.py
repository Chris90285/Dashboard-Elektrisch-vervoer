# ======================================================
# DASHBOARD: Laadpalen & Elektrische Voertuigen
# ======================================================

# ------------------- Imports --------------------------
# ------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import folium
import requests
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, FastMarkerCluster
import math
from typing import Tuple

# ------------------- Sidebar ---------------------------
# ------------------------------------------------------
with st.sidebar:
    st.markdown("## Laadpalen & Elektrische Voertuigen")
    st.markdown("---")

    page = st.selectbox(
        "Selecteer een pagina",
        [
            "⚡️ Laadpalen",
            "🚘 Voertuigen",
            "📊 Voorspellend model"
        ]
    )

    st.write("")
    st.info("🔋 OpenChargeMap Nederland API-data wordt per provincie geladen")
    st.markdown("---")
    st.write("Voor het laatst geüpdatet op:")
    st.write("*07 okt 2025*")

# ======================================================
#                   PAGINA-INDELING
# ======================================================

# ------------------- Helper functies ------------------
# ------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Bereken afstand in kilometers tussen twee GPS punten (haversine)."""
    R = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@st.cache_data(ttl=86400)
def get_laadpalen_data(lat: float, lon: float, radius: float, maxresults: int = 2000) -> pd.DataFrame:
    """
    Haal laadpalen op rond lat/lon binnen radius (km).
    Cached zodat meerdere view-interacties snel zijn.
    """
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json",
        "countrycode": "NL",
        "latitude": lat,
        "longitude": lon,
        "distance": radius,
        "maxresults": maxresults,
        "compact": True,
        "verbose": False,
        "key": "bbc1c977-6228-42fc-b6af-5e5f71be11a5"
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    df = pd.json_normalize(data)
    return df

def ensure_latlon(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            'AddressInfo.Latitude', 'AddressInfo.Longitude', 'AddressInfo.Title',
            'AddressInfo.AddressLine1', 'AddressInfo.Town', 'UsageCost', 'PowerKW'
        ])
    df = df.dropna(subset=['AddressInfo.Latitude', 'AddressInfo.Longitude'])
    # Flatten first connection object (if aanwezig)
    if 'Connections' in df.columns:
        connections = pd.json_normalize(
            df["Connections"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else {})
        )
        # Merge only if there's something to merge
        if not connections.empty:
            df = pd.concat([df.reset_index(drop=True), connections.reset_index(drop=True)], axis=1)
    df.reset_index(drop=True, inplace=True)
    return df

def estimate_radius_from_bounds(bounds: dict, center_lat: float, center_lon: float) -> float:
    """
    Probeer radius (km) te berekenen van bounds dict (zoals st_folium kan teruggeven).
    Als bounds niet beschikbaar, return None.
    """
    try:
        # st_folium kan bounds als {'northEast': {'lat': .., 'lng': ..}, 'southWest': {...}}
        ne = bounds.get("northEast") or bounds.get("ne") or bounds.get("NE")
        sw = bounds.get("southWest") or bounds.get("sw") or bounds.get("SW")
        if ne and sw:
            # radius ongeveer afstand van center naar NE hoek
            return max(0.5, haversine_km(center_lat, center_lon, ne['lat'], ne['lng']))
    except Exception:
        pass
    return None

def limit_by_distance(df: pd.DataFrame, center_lat: float, center_lon: float, max_points: int = 500) -> pd.DataFrame:
    """Beperk dataset tot de dichtstbijzijnde max_points t.o.v. center."""
    if df.empty:
        return df
    # veilige toegang
    lats = df['AddressInfo.Latitude'].astype(float)
    lons = df['AddressInfo.Longitude'].astype(float)
    dists = [haversine_km(center_lat, center_lon, lat, lon) for lat, lon in zip(lats, lons)]
    df = df.copy()
    df['_dist_km'] = dists
    df = df.sort_values('_dist_km').head(max_points).drop(columns=['_dist_km'])
    df.reset_index(drop=True, inplace=True)
    return df

# ------------------- Pagina 1 --------------------------
# ------------------------------------------------------
if page == "⚡️ Laadpalen":
    st.markdown("## Overzicht Laadpalen")
    st.write("Gebruik deze pagina voor een kort overzicht of KPI’s over laadpalen.")
    st.markdown("---")

    # ======================
    # 🔍 FILTER: Provincie
    # ======================
    provincies = {
        "Heel Nederland": [52.1, 5.3, 200],  # radius groot genoeg om heel NL te dekken
        "Groningen": [53.2194, 6.5665, 60],
        "Friesland": [53.1642, 5.7818, 60],
        "Drenthe": [52.9476, 6.6231, 60],
        "Overijssel": [52.4380, 6.5010, 60],
        "Flevoland": [52.5270, 5.5953, 60],
        "Gelderland": [52.0452, 5.8712, 60],
        "Utrecht": [52.0907, 5.1214, 60],
        "Noord-Holland": [52.5206, 4.7885, 60],
        "Zuid-Holland": [52.0116, 4.3571, 60],
        "Zeeland": [51.4940, 3.8497, 60],
        "Noord-Brabant": [51.5730, 5.0670, 60],
        "Limburg": [51.2490, 5.9330, 60],
    }

    provincie_keuze = st.selectbox("📍 Kies een provincie", provincies.keys(), index=0)
    center_lat, center_lon, radius_km = provincies[provincie_keuze]

    # ---------------------
    # Reset map state when provincie verandert
    # ---------------------
    if "last_provincie" not in st.session_state:
        st.session_state["last_provincie"] = None
    if "map_state" not in st.session_state:
        st.session_state["map_state"] = {"lat": center_lat, "lon": center_lon, "zoom": 8 if provincie_keuze == "Heel Nederland" else 10}

    if st.session_state["last_provincie"] != provincie_keuze:
        # Provincie is veranderd: reset map_state naar provincie's center en redelijk zoomlevel
        st.session_state["map_state"] = {
            "lat": center_lat,
            "lon": center_lon,
            "zoom": 8 if provincie_keuze == "Heel Nederland" else 10,
        }
        st.session_state["last_provincie"] = provincie_keuze

    # ---------------------
    # 🔌 API-call per regio (met caching)
    # ---------------------
    # We halen voor de provincie een 'overview' dataset (snelle weergave).
    with st.spinner(f"🔌 Laad overzicht laadpalen voor {provincie_keuze}..."):
        # overview radius: gebruik provincie radius_km maar limiet op maxresults om te voorkomen dat alles traag wordt
        try:
            overview_df = get_laadpalen_data(center_lat, center_lon, radius_km, maxresults=3000)
        except Exception as e:
            st.error(f"Fout bij ophalen data: {e}")
            overview_df = pd.DataFrame()
    overview_df = ensure_latlon(overview_df)

    # ---------------------
    # 🗺️ Kaart maken met "slim laden"
    # ---------------------
    st.markdown("### 🗺️ Kaartweergave")
    toon_popups = st.checkbox("Toon details bij laadpalen (kan trager zijn)", value=False)
    auto_load_on_zoom = st.checkbox("Laad automatisch detailpunten bij voldoende inzoomen", value=True)
    zoom_threshold = st.slider("Zoom threshold om automatisch details te laden", min_value=8, max_value=16, value=12)

    # Sla mapstate op (al eerder gezet)
    m = folium.Map(
        location=[st.session_state["map_state"]["lat"], st.session_state["map_state"]["lon"]],
        zoom_start=st.session_state["map_state"]["zoom"],
        tiles="OpenStreetMap"
    )

    # ---------------------
    # ⚡️ FAST OVERVIEW: geen popups, alleen snel cluster
    # ---------------------
    # FastMarkerCluster is snel voor een grote hoeveelheid punten.
    # We geven alleen lat/lon zodat de init render vlot is.
    overview_points = list(zip(overview_df["AddressInfo.Latitude"], overview_df["AddressInfo.Longitude"]))
    if len(overview_points) > 0:
        FastMarkerCluster(data=overview_points).add_to(m)
    else:
        folium.Marker([center_lat, center_lon], popup="Geen laadpalen gevonden voor deze provincie.").add_to(m)

    # Toon kaart & lees zoom-level en bounds
    map_data = st_folium(m, width=900, height=650)

    # Update session_state met laatste center/zoom
    if map_data:
        if "center" in map_data and map_data["center"]:
            st.session_state["map_state"]["lat"] = map_data["center"]["lat"]
            st.session_state["map_state"]["lon"] = map_data["center"]["lng"]
        if "zoom" in map_data and map_data["zoom"] is not None:
            st.session_state["map_state"]["zoom"] = map_data["zoom"]

    # ---------------------
    # 🧠 Detailmodus: enkel ophalen als ingezoomd of op knop gedrukt
    # ---------------------
    # Bepaal of we details moeten laden:
    load_details_now = False
    # 1) als gebruiker explicit op checkbox staat en op knop drukt
    if toon_popups:
        load_details_now = True

    # 2) of als auto_load_on_zoom en zoom >= threshold
    current_zoom = st.session_state["map_state"].get("zoom", 8)
    if auto_load_on_zoom and current_zoom >= zoom_threshold:
        # Als gebruiker auto-mode wil, force load (maar we blijven binnen limieten)
        load_details_now = True

    # 3) Of de gebruiker kan expliciet op knop klikken om details te laden
    if st.button("🔍 Laad detailpunten voor huidig gebied"):
        load_details_now = True

    if load_details_now:
        # Bepaal center en radius voor detail-oproep
        lat = st.session_state["map_state"]["lat"]
        lon = st.session_state["map_state"]["lon"]

        # Probeer radius te berekenen op basis van map bounds (indien beschikbaar)
        bounds = map_data.get("bounds") if isinstance(map_data, dict) else None
        radius_for_query = None
        if bounds:
            radius_for_query = estimate_radius_from_bounds(bounds, lat, lon)

        # fallback op zoom-based radius als bounds niet beschikbaar
        if radius_for_query is None:
            # eenvoudige mapping: hogere zoom -> kleinere radius
            if current_zoom >= 14:
                radius_for_query = 5
            elif current_zoom >= 12:
                radius_for_query = 15
            elif current_zoom >= 10:
                radius_for_query = 40
            else:
                radius_for_query = 80

        # Safety cap
        radius_for_query = min(max(radius_for_query, 1), 100)

        st.info(f"📡 Detaildata ophalen (straal ~{int(radius_for_query)} km rond huidige view)...")
        with st.spinner("🔍 Ophalen details..."):
            try:
                # We zetten een lagere maxresults voor detail-oproep zodat browser niet crasht
                detail_df = get_laadpalen_data(lat, lon, radius_for_query, maxresults=1500)
            except Exception as e:
                st.error(f"Fout bij ophalen detaildata: {e}")
                detail_df = pd.DataFrame()

        detail_df = ensure_latlon(detail_df)

        # Als er heel veel punten zijn: beperk tot N dichtstbijzijnde tot de user view center (prevents browser freeze)
        MAX_DETAIL_MARKERS = 700  # mocht nog omlaag als het te traag is bij jouw machine
        if len(detail_df) > MAX_DETAIL_MARKERS:
            detail_df = limit_by_distance(detail_df, lat, lon, max_points=MAX_DETAIL_MARKERS)
            st.warning(f"🧾 Er waren veel punten; we tonen nu de {MAX_DETAIL_MARKERS} dichtstbijzijnde laadpalen om performance te bewaren.")

        # Maak detailkaart met echte markers + popups (MarkerCluster voor betere UX)
        detail_map = folium.Map(location=[lat, lon], zoom_start=current_zoom, tiles="OpenStreetMap")
        marker_cluster = MarkerCluster().add_to(detail_map)

        for _, row in detail_df.iterrows():
            try:
                rlat = float(row["AddressInfo.Latitude"])
                rlon = float(row["AddressInfo.Longitude"])
            except Exception:
                continue

            title = row.get('AddressInfo.Title', 'Onbekend')
            addr = row.get('AddressInfo.AddressLine1', '')
            town = row.get('AddressInfo.Town', '')
            usage = row.get('UsageCost', 'N/B')
            power = row.get('PowerKW', 'N/B')

            popup_html = f"""
            <div style="font-size:12px;max-width:260px">
                <b>{title}</b><br/>
                {addr}<br/>
                {town}<br/>
                <b>Kosten:</b> {usage}<br/>
                <b>Vermogen:</b> {power} kW<br/>
            </div>
            """
            folium.Marker(
                location=[rlat, rlon],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color="green", icon="bolt", prefix="fa")
            ).add_to(marker_cluster)

        st.markdown("#### Detailweergave (markers met popups)")
        st_folium(detail_map, width=900, height=650)
    else:
        st.info("🔎 Zoom in of activeer 'Toon details' om individuele laadpalen met info te zien. Je kunt ook op 'Laad detailpunten...' klikken.")

# ------------------- Pagina 2 --------------------------
# ------------------------------------------------------
elif page == "🚘 Voertuigen":
    st.markdown("## Overzicht Elektrische Voertuigen")
    st.write("Gebruik deze pagina voor analyses over elektrische voertuigen.")
    st.markdown("---")
    st.write("🔧 Voeg hier je visualisaties toe.")

# ------------------- Pagina 3 --------------------------
# ------------------------------------------------------
elif page == "📊 Voorspellend model":
    st.markdown("## Voorspellend Model")
    st.write("Gebruik deze pagina voor modellen en prognoses over laad- en voertuiggedrag.")
    st.markdown("---")
    st.write("🔧 Voeg hier je voorspellend model of simulatie toe.")