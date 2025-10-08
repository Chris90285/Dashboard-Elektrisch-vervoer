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

# ------------------- Pagina 1 --------------------------
# ------------------------------------------------------
if page == "⚡️ Laadpalen":
    st.markdown("## Overzicht Laadpalen")
    st.write("Gebruik deze pagina voor een kort overzicht of KPI’s over laadpalen.")
    st.markdown("---")

    # ======================
    # 📍 FILTER: Provincie
    # ======================
    provincies = {
        "Heel Nederland": [52.1, 5.3, 200],
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

    # ======================
    # 🔌 API-call met caching
    # ======================
    @st.cache_data(ttl=86400)
    def get_laadpalen_data(lat: float, lon: float, radius: float) -> pd.DataFrame:
        url = "https://api.openchargemap.io/v3/poi/"
        params = {
            "output": "json",
            "countrycode": "NL",
            "latitude": lat,
            "longitude": lon,
            "distance": radius,
            "maxresults": 5000,
            "compact": True,
            "verbose": False,
            "key": "bbc1c977-6228-42fc-b6af-5e5f71be11a5"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.json_normalize(data)
        df = df.dropna(subset=['AddressInfo.Latitude', 'AddressInfo.Longitude'])
        return df

    with st.spinner(f"🔌 Laad laadpalen voor {provincie_keuze}..."):
        Laadpalen = get_laadpalen_data(center_lat, center_lon, radius_km)

    # ======================
    # 🌍 Kaartinstellingen
    # ======================
    DETAIL_ZOOM_LEVEL = 11  # 📌 Duidelijke drempel voor detailmodus

    if "map_state" not in st.session_state:
        st.session_state["map_state"] = {
            "lat": center_lat,
            "lon": center_lon,
            "zoom": 8 if provincie_keuze == "Heel Nederland" else 10,
        }

    # Als je een andere provincie kiest → herpositioneer kaart
    if provincie_keuze:
        st.session_state["map_state"]["lat"] = center_lat
        st.session_state["map_state"]["lon"] = center_lon
        st.session_state["map_state"]["zoom"] = 8 if provincie_keuze == "Heel Nederland" else 10

    # ======================
    # 🗺️ Kaart genereren
    # ======================
    m = folium.Map(
        location=[st.session_state["map_state"]["lat"], st.session_state["map_state"]["lon"]],
        zoom_start=st.session_state["map_state"]["zoom"],
        tiles="OpenStreetMap"
    )

    # Toon standaard snelle markers
    FastMarkerCluster(
        data=list(zip(Laadpalen["AddressInfo.Latitude"], Laadpalen["AddressInfo.Longitude"]))
    ).add_to(m)

    map_data = st_folium(m, width=800, height=600, returned_objects=["center", "zoom"])

    # ======================
    # 🧭 Zoom volgen & detailmodus
    # ======================
    if map_data and "zoom" in map_data:
        st.session_state["map_state"]["zoom"] = map_data["zoom"]
        st.session_state["map_state"]["lat"] = map_data["center"]["lat"]
        st.session_state["map_state"]["lon"] = map_data["center"]["lng"]

        if map_data["zoom"] >= DETAIL_ZOOM_LEVEL:
            st.info(f"🔍 Detailmodus actief (zoomniveau {map_data['zoom']})")

            with st.spinner("📡 Ophalen details in dit gebied..."):
                detail_data = get_laadpalen_data(
                    map_data["center"]["lat"], 
                    map_data["center"]["lng"], 
                    20  # radius bij detailmodus
                )

                detail_map = folium.Map(
                    location=[map_data["center"]["lat"], map_data["center"]["lng"]],
                    zoom_start=map_data["zoom"],
                    tiles="OpenStreetMap"
                )
                marker_cluster = MarkerCluster().add_to(detail_map)

                for _, row in detail_data.iterrows():
                    popup = f"""
                    <b>{row.get('AddressInfo.Title', 'Onbekend')}</b><br>
                    {row.get('AddressInfo.AddressLine1', '')}<br>
                    {row.get('AddressInfo.Town', '')}<br>
                    Kosten: {row.get('UsageCost', 'N/B')}<br>
                    Vermogen: {row.get('PowerKW', 'N/B')} kW
                    """
                    folium.Marker(
                        location=[row["AddressInfo.Latitude"], row["AddressInfo.Longitude"]],
                        popup=popup,
                        icon=folium.Icon(color="green", icon="bolt", prefix="fa")
                    ).add_to(marker_cluster)

                st_folium(detail_map, width=800, height=600)

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
