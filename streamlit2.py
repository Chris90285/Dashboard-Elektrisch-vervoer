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

            st.success(f"Detailmodus: {len(subset_df)} laadpalen met popups geladen.")
        st_folium(m, width=900, height=650, returned_objects=["center", "zoom"])

    st.markdown("<small>**Bron: openchargemap.org**</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📊 Verdeling laadpalen in Nederland")

    if len(df_all) > 0:
        def parse_cost(value):
            if isinstance(value, str):
                match = re.search(r"(\d+[\.,]?\d*)", value.replace(",", "."))
                return float(match.group(1)) if match else np.nan
            return np.nan

        df_all["UsageCostClean"] = df_all["UsageCost"].apply(parse_cost)

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


# ------------------- Pagina 2 --------------------------
elif page == "🚘 Voertuigen":
    st.markdown("## Overzicht Elektrische Voertuigen")
    st.write("Op deze pagina is informatie te vinden over elektrische auto's in Nederland.")
    st.markdown("---")


# ------------------- Pagina 3 --------------------------
elif page == "📊 Voorspellend model":
    st.markdown("## Voorspellend Model")
    st.write("Gebruik deze pagina voor modellen en prognoses over laad- en voertuiggedrag.")
    st.markdown("---")
    st.write("🔧 Voeg hier je voorspellend model of simulatie toe.")

    warnings.filterwarnings("ignore")

    # ---------- Instellingen ----------
    EINDDATUM = pd.Timestamp("2030-12-01")

    # ---------- Gebruik bestaande dataset ----------
    df_auto1 = df_auto.copy()

    # ---------- Type bepalen ----------
    def bepaal_type(merk, uitvoering):
        u = str(uitvoering).upper()
        m = str(merk).upper()
        if ("BMW I" in m or "PORSCHE" in m or
            u.startswith(("FA1FA1CZ","3EER","3EDF","3EDE","2EER","2EDF","2EDE",
                        "E11","0AW5","QE2QE2G1","QE1QE1G1","HE1HE1G1")) or
            "EV" in u or "FA1FA1MD" in u):
            return "Elektrisch"
        if "DIESEL" in u or "TDI" in u or "CDI" in u or "DPE" in u or u.startswith("D"):
            return "Diesel"
        return "Benzine"

    df_auto1["Type"] = df_auto1.apply(lambda r: bepaal_type(r.get("Merk",""), r.get("Uitvoering","")), axis=1)

    df_auto1["Datum eerste toelating"] = df_auto1["Datum eerste toelating"].astype(str).str.split(".").str[0]
    df_auto1["Datum eerste toelating"] = pd.to_datetime(df_auto1["Datum eerste toelating"], format="%Y%m%d", errors="coerce")

    df_auto2 = df_auto1.dropna(subset=["Datum eerste toelating"])
    df_auto2 = df_auto2[df_auto2["Datum eerste toelating"].dt.year > 2010]
    df_auto2["Maand"] = df_auto2["Datum eerste toelating"].dt.to_period("M").dt.to_timestamp()

    maand_counts = df_auto2.groupby(["Maand", "Type"]).size().unstack(fill_value=0).sort_index()
    if maand_counts.empty:
        raise SystemExit("⚠ Geen bruikbare data gevonden in dataset na 2010.")

    cumul_hist = maand_counts.cumsum()
    laatste_hist_maand = cumul_hist.index.max()
    forecast_start = laatste_hist_maand + pd.DateOffset(months=1)
    forecast_index = pd.date_range(start=forecast_start, end=EINDDATUM, freq="MS")
    h = len(forecast_index)

    forecast_median = pd.DataFrame(index=forecast_index)
    forecast_lower = pd.DataFrame(index=forecast_index)
    forecast_upper = pd.DataFrame(index=forecast_index)

    for col in maand_counts.columns:
        y = maand_counts[col].astype(float)
        
        if len(y) < 12:
            print(f"⚠ Te weinig data voor {col}, gebruik lineaire extrapolatie.")
            x = np.arange(len(y))
            m, b = np.polyfit(x, y, 1)
            future_x = np.arange(len(y), len(y) + h)
            future_pred = np.maximum(b + m * future_x, 0)
            conf_int = np.vstack([future_pred - y.std(), future_pred + y.std()]).T
        else:
            try:
                model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,0,12),
                                enforce_stationarity=False, enforce_invertibility=False)
                fit = model.fit(disp=False)
                pred = fit.get_forecast(steps=h)
                future_pred = np.maximum(pred.predicted_mean.values, 0)
                conf_int = pred.conf_int(alpha=0.05).values
            except Exception as e:
                print(f"⚠ Fout bij modelleren van {col}: {e}, val terug op lineair model.")
                x = np.arange(len(y))
                m, b = np.polyfit(x, y, 1)
                future_x = np.arange(len(y), len(y) + h)
                future_pred = np.maximum(b + m * future_x, 0)
                conf_int = np.vstack([future_pred - y.std(), future_pred + y.std()]).T
        
        last_cumul = cumul_hist[col].iloc[-1]
        cumul_forecast = last_cumul + np.cumsum(future_pred)
        cumul_lower = last_cumul + np.cumsum(np.maximum(conf_int[:,0], 0))
        cumul_upper = last_cumul + np.cumsum(np.maximum(conf_int[:,1], 0))
        
        forecast_median[col] = cumul_forecast
        forecast_lower[col] = cumul_lower
        forecast_upper[col] = cumul_upper

    plt.figure(figsize=(14,7))
    for col in cumul_hist.columns:
        plt.plot(cumul_hist.index, cumul_hist[col], linewidth=2, label=f"{col} (historisch)")
        plt.plot(forecast_index, forecast_median[col], linestyle="--", linewidth=2, label=f"{col} (SARIMAX voorspelling)")
        plt.fill_between(forecast_index, forecast_lower[col], forecast_upper[col], alpha=0.2)

    plt.title("Voertuigregistraties per brandstoftype — Historisch + SARIMAX-voorspelling tot 2030")
    plt.xlabel("Jaar")
    plt.ylabel("Aantal voertuigen")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
