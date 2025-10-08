# ======================================================
# DASHBOARD: Laadpalen & Elektrische Voertuigen
# ======================================================

# ------------------- Imports --------------------------
# ------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import folium
import requests
import re
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, FastMarkerCluster
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import warnings
import pickle
import io

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


# ------------------- Data inladen -----------------------
# -------------------------------------------------------
@st.cache_data
def load_data():
    df_auto = pd.read_csv("duitse_automerken_JA.csv")
    return df_auto

@st.cache_data(ttl=86400)
def get_laadpalen_data(lat: float, lon: float, radius: float) -> pd.DataFrame:
    """Haalt laadpalen binnen een straal op."""
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

@st.cache_data(ttl=86400)
def get_all_laadpalen_nederland() -> pd.DataFrame:
    """Haalt laadpalen van heel Nederland op (voor grafieken)."""
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json",
        "countrycode": "NL",
        "maxresults": 10000,
        "compact": True,
        "verbose": False,
        "key": "bbc1c977-6228-42fc-b6af-5e5f71be11a5"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.json_normalize(data)
    return df

# ✅ Dataframe inladen zodra de app start
df_auto = load_data()


# ======================================================
#                   PAGINA-INDELING
# ======================================================

# ------------------- Pagina 1 --------------------------
if page == "⚡️ Laadpalen":
    st.markdown("## Kaart laadpalen")
    st.write("Op deze pagina is een kaart te zien met laadpalen in Nederland.")
    st.write("Klik op een laadpaal voor meer informatie.")
    st.write("Gebruik het dropdown menu om verschillende provincies te bekijken.")
    st.markdown("---")

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

    with st.spinner(f" Laad laadpalen voor {provincie_keuze}..."):
        df = get_laadpalen_data(center_lat, center_lon, radius_km)
        df_all = get_all_laadpalen_nederland()

        if provincie_keuze != "Heel Nederland":
            Laadpalen = df[df["AddressInfo.StateOrProvince"].str.contains(provincie_keuze, case=False, na=False)]
        else:
            Laadpalen = df

    MAX_DEFAULT = 300  
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
            st.info(f"Snelmodus: alle {len(coords)} laadpalen getoond (geen popups).")
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

            st.success(f"{len(subset_df)} laadpalen met popups geladen.")
        st_folium(m, width=900, height=650, returned_objects=["center", "zoom"])

    st.markdown("<small>**Bron: openchargemap.org**</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📊 Verdeling laadpalen in Nederland")

    if len(df_all) > 0:
        # ✅ Verbeterde parse_cost functie
        def parse_cost(value):
            if isinstance(value, str):
                if "free" in value.lower() or "gratis" in value.lower():
                    return 0.0
                match = re.search(r"(\d+[\.,]?\d*)", value.replace(",", "."))
                return float(match.group(1)) if match else np.nan
            return np.nan

        df_all["UsageCostClean"] = df_all["UsageCost"].apply(parse_cost)

        # ✅ Onrealistische waarden uitsluiten (>2 €/kWh)
        df_all.loc[
            (df_all["UsageCostClean"] < 0) | (df_all["UsageCostClean"] > 2),
            "UsageCostClean"
        ] = np.nan

        if "PowerKW" in df_all.columns:
            df_all["PowerKW_clean"] = pd.to_numeric(df_all["PowerKW"], errors="coerce")
        elif "Connections.PowerKW" in df_all.columns:
            df_all["PowerKW_clean"] = pd.to_numeric(df_all["Connections.PowerKW"], errors="coerce")
        elif "Connections[0].PowerKW" in df_all.columns:
            df_all["PowerKW_clean"] = pd.to_numeric(df_all["Connections[0].PowerKW"], errors="coerce")
        else:
            df_all["PowerKW_clean"] = np.nan

        provincie_mapping = {
            "Groningen": "Groningen",
            "Friesland": "Friesland",
            "Fryslân": "Friesland",
            "Drenthe": "Drenthe",
            "Overijssel": "Overijssel",
            "Flevoland": "Flevoland",
            "Gelderland": "Gelderland",
            "Utrecht": "Utrecht",
            "Noord-Holland": "Noord-Holland",
            "North Holland": "Noord-Holland",
            "Zuid-Holland": "Zuid-Holland",
            "South Holland": "Zuid-Holland",
            "Zeeland": "Zeeland",
            "Noord-Brabant": "Noord-Brabant",
            "North Brabant": "Noord-Brabant",
            "Limburg": "Limburg"
        }

        df_all["Provincie"] = df_all["AddressInfo.StateOrProvince"].map(provincie_mapping)
        df_all = df_all[df_all["Provincie"].isin(list(provincies.keys()))]

        df_agg = (
            df_all.groupby("Provincie")
            .agg(
                Aantal_palen=("ID", "count"),
                Gemiddelde_kosten=("UsageCostClean", "mean"),
            )
            .reset_index()
            .sort_values("Aantal_palen", ascending=False)
        )

        keuze = st.selectbox(
            "📈 Kies welke verdeling je wilt zien:",
            ["Aantal laadpalen per provincie", "Gemiddelde kosten per provincie"]
        )

        if keuze == "Aantal laadpalen per provincie":
            fig = px.bar(df_agg, x="Provincie", y="Aantal_palen", title="Aantal laadpalen per provincie")
        elif keuze == "Gemiddelde kosten per provincie":
            fig = px.bar(df_agg, x="Provincie", y="Gemiddelde_kosten", title="Gemiddelde kosten per provincie (€ per kWh)")

        fig.update_layout(xaxis_title="Provincie", yaxis_title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Kon geen landelijke data laden voor de grafiek.")
