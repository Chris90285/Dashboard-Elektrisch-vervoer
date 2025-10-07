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
# ------------------ Data inladen ----------------------
# ------------------------------------------------------
url = "https://api.openchargemap.io/v3/poi/"
params = {
    "output": "json",
    "countrycode": "NL",
    "maxresults": 200,
    "compact": True,
    "verbose": False,
    "key": "bbc1c977-6228-42fc-b6af-5e5f71be11a5"
}

response = requests.get(url, params=params)
data = response.json()

# Zet de JSON om naar DataFrame
Laadpalen = pd.json_normalize(data)

# Pak de eerste 'Connection' per laadpaal (anders krijg je nested JSON)
connections = pd.json_normalize(
    Laadpalen["Connections"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else {})
)
Laadpalen = pd.concat([Laadpalen, connections], axis=1)

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
    st.image("placeholder_afbeelding.png", use_container_width=True)
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

    # Gebruik de Laadpalen DataFrame (al gefilterd op Nederland)
    df = Laadpalen.dropna(subset=['AddressInfo.Latitude', 'AddressInfo.Longitude']).copy()

    # Basiskaart centreren op Nederland
    m = folium.Map(location=[52.1, 5.3], zoom_start=8, tiles="OpenStreetMap")

    # Voeg markers toe
    for _, row in df.iterrows():
        popup = f"""
        <b>{row.get('AddressInfo.Title', 'Onbekend')}</b><br>
        {row.get('AddressInfo.AddressLine1', '')}<br>
        {row.get('AddressInfo.Town', '')}<br>
        Kosten: {row.get('UsageCost', 'N/B')}<br>
        Vermogen: {row.get('PowerKW', 'N/B')} kW
        """
        folium.Marker(
            location=[row["AddressInfo.Latitude"], row["AddressInfo.Longitude"]],
            popup=folium.Popup(popup, max_width=300),
            icon=folium.Icon(color="green", icon="bolt", prefix="fa")
        ).add_to(m)

    # Toon kaart in Streamlit
    st_folium(m, width=700, height=600)



# ------------------- Pagina 2 --------------------------
# ------------------------------------------------------
elif page == "🚘 Voertuigen":
    st.markdown("##Overzicht Elektrische Voertuigen")
    st.write("Gebruik deze pagina voor analyses over elektrische voertuigen.")
    st.markdown("---")

    # 👉 Voeg hier je grafieken en analyses toe
    st.write("🔧 Voeg hier je visualisaties toe.")

# ------------------- Pagina 3 --------------------------
# ------------------------------------------------------
elif page == "📊 Voorspellend model":
    st.markdown("##Voorspellend Model")
    st.write("Gebruik deze pagina voor modellen en prognoses over laad- en voertuiggedrag.")
    st.markdown("---")

    # 👉 Voeg hier vergelijkingstabellen/grafieken of ML-modellen toe
    st.write("🔧 Voeg hier je voorspellend model of simulatie toe.")

# ======================================================

