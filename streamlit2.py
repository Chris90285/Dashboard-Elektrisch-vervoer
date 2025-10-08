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
    st.markdown("## Kaart laadpalen")
    st.write("Op deze pagina is een kaart te zien met laadpalen in Nederland.")
    st.write("Klik op een laadpaal voor meer informatie.")
    st.markdown("---")

    # ======================
    # 🔍 FILTER: Provincie
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

    # ---------------------
    # 🔌 API-call met caching
    # ---------------------
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

    with st.spinner(f" Laad laadpalen voor {provincie_keuze}..."):
        df = get_laadpalen_data(center_lat, center_lon, radius_km)

        # 🔒 Filter: alleen laadpalen binnen de gekozen provincie
        if provincie_keuze != "Heel Nederland":
            Laadpalen = df[df["AddressInfo.StateOrProvince"].str.contains(provincie_keuze, case=False, na=False)]
        else:
            Laadpalen = df

    # ---------------------
    # Standaard aantal laadpalen
    # ---------------------
    MAX_DEFAULT = 300  

    # ---------------------
    # Kaart maken
    # ---------------------
    st.write(f"📍 Provincie: **{provincie_keuze}** — gevonden laadpalen: **{len(Laadpalen)}**")

    # Checkbox: alle punten met FastMarkerCluster 
    laad_alle = st.checkbox("Laad alle laadpalen (geen popups)", value=False)

    # Basemap
    if len(Laadpalen) == 0:
        st.warning("Geen laadpalen gevonden voor deze locatie/provincie.")
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap")
        st_folium(m, width=900, height=650)
    else:
        start_zoom = 8 if provincie_keuze == "Heel Nederland" else 10
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=start_zoom,
            tiles="OpenStreetMap"
        )

        if laad_alle:
            # ------------------------------------------------------------
            # SNEL: alle punten zonder popups
            # ------------------------------------------------------------
            coords = list(zip(Laadpalen["AddressInfo.Latitude"], Laadpalen["AddressInfo.Longitude"]))
            FastMarkerCluster(data=coords).add_to(m)
            st.info(f"Snelmodus: alle {len(coords)} laadpalen worden getoond (geen popups of individuele iconen voor performance).")
            # Let op: FastMarkerCluster gebruikt eenvoudige markers om performance te waarborgen.
        else:
            # ------------------------------------------------------------
            # DETAIL: toon alleen (MAX_DEFAULT) met popups
            # ------------------------------------------------------------
            if len(Laadpalen) > MAX_DEFAULT:
                subset_df = Laadpalen.sample(n=MAX_DEFAULT, random_state=1).reset_index(drop=True)
            else:
                subset_df = Laadpalen.reset_index(drop=True)

            marker_cluster = MarkerCluster().add_to(m)
            for _, row in subset_df.iterrows():
                lat = row["AddressInfo.Latitude"]
                lon = row["AddressInfo.Longitude"]
                popup = f"""
                <b>{row.get('AddressInfo.Title', 'Onbekend')}</b><br>
                {row.get('AddressInfo.AddressLine1', '')}<br>
                {row.get('AddressInfo.Town', '')}<br>
                Kosten: {row.get('UsageCost', 'N/B')}<br>
                Vermogen: {row.get('PowerKW', 'N/B')} kW
                """

                # ✅ Bliksem icoon voor laadpalen
                icon = folium.Icon(color="green", icon="bolt", prefix="fa")

                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup, max_width=300),
                    icon=icon
                ).add_to(marker_cluster)

            st.success(f"Detailmodus: {len(subset_df)} laadpalen met popups geladen.")

        # Render de kaart
        st_folium(m, width=900, height=650, returned_objects=["center", "zoom"])

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
