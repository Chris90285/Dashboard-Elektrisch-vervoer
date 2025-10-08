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
        # ✅ Verbeterde parse_cost functie + filtering
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


# ------------------- Pagina 2 --------------------------
elif page == "🚘 Voertuigen":
    st.markdown("##Elektrische Voertuigen & laadtijden")
    st.write("Op deze pagina is informatie te vinden over elektrische auto's in Nederland.")
    st.markdown("---")

    data = pd.read_csv("duitse_automerken_JA.csv")

    # --- Definieer herkenningspatronen per type ---
    def bepaal_type(merk, uitvoering):
        u = str(uitvoering).upper()
        m = str(merk).upper()

        # Elektrisch
        if "BMW I" in m or "PORSCHE" in m or u.startswith("FA1FA1CZ") or u.startswith("3EER") or u.startswith("3EDF") or u.startswith("3EDE") or u.startswith("2EER") or u.startswith("2EDF") or u.startswith("2EDE") or u.startswith("E11") or u.startswith("0AW5") or u.startswith("QE2QE2G1") or u.startswith("QE1QE1G1") or u.startswith("HE1HE1G1") or "EV" in u or "FA1FA1MD" in u or "FA1FA1CZ" in u:
            return "Elektrisch"

        # Diesel
        if "DIESEL" in u or "TDI" in u or "CDI" in u or "DPE" in u or u.startswith("D"):
            return "Diesel"

        # Benzine (default)
        return "Benzine"

    # 🧩 Pas de functie toe op je DataFrame
    data["Type"] = data.apply(lambda row: bepaal_type(row["Merk"], row["Uitvoering"]), axis=1)


    # --- CATEGORIEËN CONTROLEREN ---
    # Verzekeren dat Type kolom Elektrisch/Hybride/Benzine/Diesel bevat
    st.write("Beschikbare brandstofcategorieën:", data["Type"].unique())


    # Zet om naar string en verwijder eventuele .0
    data["Datum eerste toelating"] = data["Datum eerste toelating"].astype(str).str.split(".").str[0]

    # Converteer naar datetime met format YYYYMMDD
    data["Datum eerste toelating"] = pd.to_datetime(
        data["Datum eerste toelating"], format="%Y%m%d", errors="coerce"
    )

    # Eventuele lege of niet-parseerbare datums verwijderen
    data = data.dropna(subset=["Datum eerste toelating"])

    # **Filteren op datum > 2010**
    data = data[data["Datum eerste toelating"].dt.year > 2010]

    # Nu kun je Maand kolom maken
    data["Maand"] = data["Datum eerste toelating"].dt.to_period("M").dt.to_timestamp()

    # Tel aantal voertuigen per maand en brandstoftype
    maand_aantal = data.groupby(["Maand", "Type"]).size().unstack(fill_value=0)

    # --- CUMULATIEVE SOM BEREKENEN ---
    cumulatief = maand_aantal.cumsum()

    # --- LIJNDIAGRAM MET STREAMLIT ---
    st.line_chart(cumulatief)

    # --- EXTRA INFO ---
    st.write("Data voorbeeld:", cumulatief.head())

   #-------------Grafiek Ann---------
    st.write("Analyseer laadsessies per uur en bekijk jaaroverzicht van totale geladen energie.")

    # ---- Bestand vast instellen ----
    file_path = "Charging_data.pkl"

    # ---- FUNCTIE: x-as als hele getallen zetten ----
    def force_integer_xaxis(fig):
        fig.update_xaxes(dtick=1)
        return fig

    # ---- DATA INLADEN ----
    try:
        ev_data = pd.read_pickle(file_path)
        ev_data.columns = (
            ev_data.columns.astype(str)
            .str.strip()
            .str.replace("\u200b", "", regex=False)
            .str.lower()
        )

        # ---- DATUMCONVERSIE EN KOLOMMEN TOEVOEGEN ----
        ev_data["start_time"] = pd.to_datetime(ev_data["start_time"], errors="coerce")
        ev_data["exit_time"] = pd.to_datetime(ev_data["exit_time"], errors="coerce")
        ev_data["hour"] = ev_data["start_time"].dt.hour
        ev_data["month"] = ev_data["start_time"].dt.to_period("M").astype(str)
        ev_data["year"] = ev_data["start_time"].dt.year
        ev_data = ev_data[ev_data["year"].notna()]
        ev_data["year"] = ev_data["year"].astype(int)

        # ---- FILTERS ----
        phase_options = ["Alle"] + [
            x for x in sorted(ev_data["n_phases"].dropna().unique()) if 0 <= x <= 6
        ]
        phase_choice = st.selectbox("Filter op aantal fasen (N_phases)", phase_options)


        ev_filtered = ev_data.copy()
        if phase_choice != "Alle":
            ev_filtered = ev_filtered[ev_filtered["n_phases"] == phase_choice]


        energy_col = "energy_delivered [kwh]"

        # ---- GRAFIEK 1: Laadsessies per uur van de dag ----
        st.subheader("Laadsessies per uur van de dag")
        hourly_counts = ev_filtered.groupby("hour").size().reset_index(name="Aantal laadsessies")
        fig1 = px.bar(hourly_counts, x="hour", y="Aantal laadsessies",
                    title="Aantal laadsessies per uur van de dag")
        fig1 = force_integer_xaxis(fig1)
        st.plotly_chart(fig1, use_container_width=True)

        # ---- GRAFIEK 2: Totaal geladen energie per maand ----
        st.subheader("Totaal geladen energie per maand")
        energy_by_month = (
            ev_filtered.groupby("month")[energy_col].sum().reset_index().sort_values("month")
        )
        fig2 = px.bar(energy_by_month, x="month", y=energy_col,
                    title="Totaal geladen energie per maand")
        fig2.update_xaxes(type='category')
        st.plotly_chart(fig2, use_container_width=True)

        # ---- GRAFIEK 3: Totaal geladen energie per jaar ----
        st.subheader("Totaal geladen energie per jaar")
        energy_by_year = (
            ev_filtered.groupby("year")[energy_col].sum().reset_index().sort_values("year")
        )
        fig3 = px.bar(energy_by_year, x="year", y=energy_col,
                    title="Totaal geladen energie per jaar")
        fig3 = force_integer_xaxis(fig3)
        st.plotly_chart(fig3, use_container_width=True)

        # ---- DATA BEKIJKEN ----
        with st.expander("📊 Bekijk gebruikte data"):
            st.dataframe(ev_filtered)

    except Exception as e:
        st.error(f"Er is een fout opgetreden bij het inlezen van `{file_path}`: {e}")


# ------------------- Pagina 3 --------------------------
elif page == "📊 Voorspellend model":
    st.markdown("## Voorspellend Model")
    st.write("Gebruik deze pagina voor modellen en prognoses over laad- en voertuiggedrag.")
    st.write("Hier is een voorspellend model te zien, wat de hoeveelheid type auto's voorspeld in Nederland.")
    st.write("")
    st.markdown("---")
    st.title("Voorspelling auto's in Nederland per brandstofcategorie")

    #-------Voorspellend model Koen-------

    warnings.filterwarnings("ignore")

    # ---------- Instellingen ----------
    EINDDATUM = pd.Timestamp("2030-12-01")


    # ---------- Kopie gebruiken ----------
    df_auto_kopie = df_auto.copy()

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

    df_auto_kopie["Type"] = df_auto_kopie.apply(
        lambda r: bepaal_type(r.get("Merk",""), r.get("Uitvoering","")), axis=1
    )

    # ---------- Datums opschonen ----------
    df_auto_kopie["Datum eerste toelating"] = df_auto_kopie["Datum eerste toelating"].astype(str).str.split(".").str[0]
    df_auto_kopie["Datum eerste toelating"] = pd.to_datetime(
        df_auto_kopie["Datum eerste toelating"], format="%Y%m%d", errors="coerce"
    )

    # ---------- Filteren en groeperen ----------
    df_auto_kopie2 = df_auto_kopie.dropna(subset=["Datum eerste toelating"])
    df_auto_kopie2 = df_auto_kopie2[df_auto_kopie2["Datum eerste toelating"].dt.year > 2010]
    df_auto_kopie2["Maand"] = df_auto_kopie2["Datum eerste toelating"].dt.to_period("M").dt.to_timestamp()

    maand_counts_charging = df_auto_kopie2.groupby(["Maand", "Type"]).size().unstack(fill_value=0).sort_index()
    if maand_counts_charging.empty:
        st.error("⚠ Geen bruikbare data gevonden in dataset na 2010.")
        st.stop()

    # ---------- Berekeningen ----------
    cumul_hist_charging = maand_counts_charging.cumsum()
    laatste_hist_maand = cumul_hist_charging.index.max()
    forecast_start = laatste_hist_maand + pd.DateOffset(months=1)
    forecast_index = pd.date_range(start=forecast_start, end=EINDDATUM, freq="MS")
    h = len(forecast_index)

    forecast_median_charging = pd.DataFrame(index=forecast_index)
    forecast_lower_charging = pd.DataFrame(index=forecast_index)
    forecast_upper_charging = pd.DataFrame(index=forecast_index)

    # ---------- SARIMAX voorspelling ----------
    for col in maand_counts_charging.columns:
        y = maand_counts_charging[col].astype(float)

        if len(y) < 12:
            st.warning(f"⚠ Te weinig data voor {col}, gebruik lineaire extrapolatie.")
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
                st.warning(f"⚠ Fout bij modelleren van {col}: {e}, val terug op lineair model.")
                x = np.arange(len(y))
                m, b = np.polyfit(x, y, 1)
                future_x = np.arange(len(y), len(y) + h)
                future_pred = np.maximum(b + m * future_x, 0)
                conf_int = np.vstack([future_pred - y.std(), future_pred + y.std()]).T

        last_cumul = cumul_hist_charging[col].iloc[-1]
        cumul_forecast = last_cumul + np.cumsum(future_pred)
        cumul_lower = last_cumul + np.cumsum(np.maximum(conf_int[:,0], 0))
        cumul_upper = last_cumul + np.cumsum(np.maximum(conf_int[:,1], 0))

        forecast_median_charging[col] = cumul_forecast
        forecast_lower_charging[col] = cumul_lower
        forecast_upper_charging[col] = cumul_upper

    # ---------- Plot ----------
    plt.figure(figsize=(14,7))
    for col in cumul_hist_charging.columns:
        plt.plot(cumul_hist_charging.index, cumul_hist_charging[col], linewidth=2, label=f"{col} (historisch)")
        plt.plot(forecast_index, forecast_median_charging[col], linestyle="--", linewidth=2, label=f"{col} (SARIMAX voorspelling)")
        plt.fill_between(forecast_index, forecast_lower_charging[col], forecast_upper_charging[col], alpha=0.2)

    plt.title("Voertuigregistraties per brandstoftype — Historisch + SARIMAX-voorspelling tot 2030")
    plt.xlabel("Jaar")
    plt.ylabel("Aantal voertuigen")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)