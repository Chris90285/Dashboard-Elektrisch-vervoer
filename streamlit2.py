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
import re  # 👈 nieuw: voor parsing van UsageCost

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
    st.write(f"Provincie: **{provincie_keuze}** — gevonden laadpalen: **{len(Laadpalen)}**")

    laad_alle = st.checkbox("Laad alle laadpalen (geen popups)", value=False)

    if len(Laadpalen) == 0:
        st.warning("Geen laadpalen gevonden voor deze locatie/provincie.")
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap")
        st_folium(m, width=900, height=650)
    else:
        start_zoom = 8 if provincie_keuze == "Heel Nederland" else 10
        m = folium.Map(location=[center_lat, center_lon], zoom_start=start_zoom, tiles="OpenStreetMap")

        if laad_alle:
            coords = list(zip(Laadpalen["AddressInfo.Latitude"], Laadpalen["AddressInfo.Longitude"]))
            FastMarkerCluster(data=coords).add_to(m)
            st.info(f"Snelmodus: {len(coords)} laadpalen zonder popups.")
        else:
            subset_df = Laadpalen.sample(n=min(len(Laadpalen), MAX_DEFAULT), random_state=1).reset_index(drop=True)
            marker_cluster = MarkerCluster().add_to(m)
            for _, row in subset_df.iterrows():
                lat, lon = row["AddressInfo.Latitude"], row["AddressInfo.Longitude"]
                popup = f"""
                <b>{row.get('AddressInfo.Title', 'Onbekend')}</b><br>
                {row.get('AddressInfo.AddressLine1', '')}<br>
                {row.get('AddressInfo.Town', '')}<br>
                Kosten: {row.get('UsageCost', 'N/B')}<br>
                Vermogen: {row.get('PowerKW', 'N/B')} kW
                """
                icon = folium.Icon(color="green", icon="bolt", prefix="fa")
                folium.Marker(location=[lat, lon], popup=folium.Popup(popup, max_width=300), icon=icon).add_to(marker_cluster)

            st.success(f"Detailmodus: {len(subset_df)} laadpalen met popups geladen.")

        st_folium(m, width=900, height=650, returned_objects=["center", "zoom"])
    st.markdown("<small>**Bron: openchargemap.org**</small>", unsafe_allow_html=True) 

    # ======================================================
    # ====== INTERACTIEVE GRAFIEK: Verdeling laadpalen =====
    # ======================================================
    st.markdown("---")
    st.subheader("📊 Analyse van laadpalen per provincie (alle data)")

    # Gebruik alle laadpalen (Heel Nederland) voor deze grafiek
    df_all = get_laadpalen_data(52.1, 5.3, 200)

    # ---- Data opschonen ----
    def parse_cost(value):
        """Haalt numerieke waarde uit kosten-string"""
        if isinstance(value, str):
            match = re.search(r"(\d+[\.,]?\d*)", value.replace(",", "."))
            return float(match.group(1)) if match else np.nan
        return np.nan

    df_all["UsageCostClean"] = df_all["UsageCost"].apply(parse_cost)
    df_all["PowerKW"] = pd.to_numeric(df_all["PowerKW"], errors="coerce")

    # ---- Aggregatie per provincie ----
    df_agg = (
        df_all.groupby("AddressInfo.StateOrProvince")
        .agg(
            Aantal_palen=("ID", "count"),
            Gemiddelde_kosten=("UsageCostClean", "mean"),
            Gemiddeld_vermogen=("PowerKW", "mean")
        )
        .reset_index()
        .rename(columns={"AddressInfo.StateOrProvince": "Provincie"})
        .sort_values("Aantal_palen", ascending=False)
    )

    keuze = st.selectbox(
        "📈 Kies wat je wilt visualiseren:",
        ["Aantal laadpalen", "Gemiddelde kosten per kWh", "Gemiddeld vermogen (kW)"]
    )

    if keuze == "Aantal laadpalen":
        y_col, title = "Aantal_palen", "Aantal laadpalen per provincie"
    elif keuze == "Gemiddelde kosten per kWh":
        y_col, title = "Gemiddelde_kosten", "Gemiddelde kosten per kWh per provincie"
    else:
        y_col, title = "Gemiddeld_vermogen", "Gemiddeld laadvermogen (kW) per provincie"

    fig = px.bar(
        df_agg,
        x="Provincie",
        y=y_col,
        color="Provincie",
        title=title,
        text_auto=True
    )
    fig.update_layout(xaxis_title="Provincie", yaxis_title="", showlegend=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

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
