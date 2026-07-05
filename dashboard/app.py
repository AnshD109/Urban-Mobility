import matplotlib
matplotlib.use('Agg')

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Mobility ML Suite",
    page_icon="🚕",
    layout="wide",
)

# ── Model Loading (cached) ────────────────────────────────────────────────────
MODEL_DIR = Path("models")

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    errors = []
    model_files = [
        "eta_rf", "eta_lr", "eta_quantile", "eta_gbr",
        "feature_pipeline", "pricing_config",
        "maintenance_rf", "maintenance_aps", "maintenance_iso",
        "nlp_pipeline"
    ]
    for name in model_files:
        try:
            models[name] = joblib.load(MODEL_DIR / f"{name}.joblib")
        except Exception as e:
            errors.append(f"{name}: {e}")
    return models, errors

models, load_errors = load_models()

# ── Helper: run feature pipeline ─────────────────────────────────────────────
def run_feature_pipeline(raw_df, fp):
    """
    Manually apply the feature_pipeline dict to a raw input DataFrame.
    fp keys: features, target, global_mean, pu_enc, do_enc, borough_enc, route_enc
    """
    df = raw_df.copy()

    # Zone encodings
    pu_enc      = fp.get("pu_enc", {})
    do_enc      = fp.get("do_enc", {})
    borough_enc = fp.get("borough_enc", {})
    route_enc   = fp.get("route_enc", {})
    global_mean = fp.get("global_mean", 15.0)

    df["pu_zone_enc"]      = df["pickup_zone"].map(pu_enc).fillna(global_mean)
    df["do_zone_enc"]      = df["dropoff_zone"].map(do_enc).fillna(global_mean)
    df["pickup_borough_enc"]  = df["pickup_borough"].map(borough_enc).fillna(0)
    df["dropoff_borough_enc"] = df["dropoff_borough"].map(borough_enc).fillna(0)
    df["route_enc"]        = (df["pickup_zone"] + "_" + df["dropoff_zone"]).map(route_enc).fillna(global_mean)
    df["pickup_hour_int"]  = df["order_hour"]

    # Select final features
    feature_cols = fp.get("features", [
        "trip_distance", "route_enc", "pu_zone_enc", "pickup_hour_int",
        "is_rush_hour", "rain_x_dist", "wind_x_dist", "temp",
        "day_of_week", "month", "prcp", "wspd", "snow"
    ])
    # Keep only columns that exist
    available = [c for c in feature_cols if c in df.columns]
    return df[available]


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🚕 Mobility ML Suite")
page = st.sidebar.radio("Navigate", [
    "📊 ETA & Pricing",
    "🔧 Maintenance Risk",
    "💬 NLP Triage",
    "📈 EDA Summary",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Status**")
for m in ["eta_rf", "maintenance_rf", "nlp_pipeline"]:
    icon = "🟢" if m in models else "🔴"
    st.sidebar.markdown(f"{icon} {m}")

if load_errors:
    with st.sidebar.expander("⚠️ Load warnings"):
        for e in load_errors:
            st.caption(e)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ETA & Pricing
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 ETA & Pricing":
    st.title("📊 ETA & Pricing Estimator")
    st.markdown("Live ETA prediction and dynamic pricing using the trained Random Forest model.")

    col1, col2 = st.columns(2)
    with col1:
        pickup_zone     = st.text_input("Pickup Zone", "East Village")
        dropoff_zone    = st.text_input("Dropoff Zone", "Clinton East")
        pickup_borough  = st.selectbox("Pickup Borough",
                            ["Manhattan","Brooklyn","Queens","Bronx","Staten Island"])
        dropoff_borough = st.selectbox("Dropoff Borough",
                            ["Manhattan","Brooklyn","Queens","Bronx","Staten Island"])
        trip_distance   = st.slider("Trip Distance (km)", 0.5, 30.0, 2.7, step=0.1)

    with col2:
        order_hour  = st.slider("Hour of Day", 0, 23, 9)
        day_of_week = st.selectbox("Day of Week", list(range(7)),
                        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        month       = st.selectbox("Month", [1,2,3],
                        format_func=lambda m: ["Jan","Feb","Mar"][m-1])
        temp  = st.slider("Temperature (°C)", -10.0, 35.0, 5.0)
        prcp  = st.slider("Precipitation (mm)", 0.0, 20.0, 0.0)
        wspd  = st.slider("Wind Speed (m/s)", 0.0, 40.0, 10.0)
        snow  = st.slider("Snowfall (mm)", 0.0, 30.0, 0.0)

    if st.button("🚀 Get Quote", type="primary"):
        if "eta_rf" not in models or "feature_pipeline" not in models:
            st.error("❌ ETA models not found.")
        else:
            try:
                is_rush     = 1 if order_hour in range(7,10) or order_hour in range(17,20) else 0
                rain_x_dist = prcp * trip_distance
                wind_x_dist = wspd * trip_distance

                raw = pd.DataFrame([{
                    "pickup_zone":    pickup_zone,
                    "dropoff_zone":   dropoff_zone,
                    "pickup_borough": pickup_borough,
                    "dropoff_borough":dropoff_borough,
                    "trip_distance":  trip_distance,
                    "order_hour":     order_hour,
                    "day_of_week":    int(day_of_week),
                    "month":          month,
                    "temp":           temp,
                    "prcp":           prcp,
                    "wspd":           wspd,
                    "snow":           snow,
                    "is_rush_hour":   is_rush,
                    "rain_x_dist":    rain_x_dist,
                    "wind_x_dist":    wind_x_dist,
                }])

                fp = models["feature_pipeline"]
                X  = run_feature_pipeline(raw, fp)

                # eta_rf is a dict with 'model' key
                eta_p50 = float(models["eta_rf"]["model"].predict(X)[0])

                # Quantile bounds
                try:
                    q_dict  = models["eta_quantile"]
                    q_models = q_dict["models"]  # list of [p10_model, p90_model]
                    scaler  = q_dict.get("scaler")
                    Xs = scaler.transform(X) if scaler else X
                    eta_p10 = float(q_models[0].predict(Xs)[0])
                    eta_p90 = float(q_models[-1].predict(Xs)[0])
                except Exception:
                    eta_p10, eta_p90 = eta_p50*0.8, eta_p50*1.3

                # Pricing
                try:
                    cfg        = models["pricing_config"]
                    sc         = cfg.get("surge_config", {})
                    base_rate  = float(cfg.get("base_rate", 2.5))
                    base_min   = float(cfg.get("base_minimum", 3.0))
                    surge_rush = float(sc.get("rush_hour", 1.3))
                    surge_rain = float(sc.get("rain", 1.15))
                    surge_snow = float(sc.get("snow", 1.25))
                except Exception:
                    base_rate, base_min = 2.5, 3.0
                    surge_rush, surge_rain, surge_snow = 1.3, 1.15, 1.25

                surge = 1.0
                if is_rush:    surge *= surge_rush
                if prcp > 2.0: surge *= surge_rain
                if snow > 5.0: surge *= surge_snow
                surge = round(min(surge, 2.5), 2)
                price = round(max(base_min, base_rate * trip_distance) * surge, 2)

                st.success("✅ Quote generated!")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("ETA (median)", f"{eta_p50:.1f} min")
                c2.metric("ETA range",    f"{eta_p10:.0f}–{eta_p90:.0f} min")
                c3.metric("Surge",        f"{surge:.2f}×")
                c4.metric("Price",        f"${price:.2f}")

                fig, ax = plt.subplots(figsize=(6,1.8))
                ax.barh(["p10","p50","p90"],[eta_p10,eta_p50,eta_p90],
                        color=["#5BA4CF","#1A3A5C","#5BA4CF"])
                ax.set_xlabel("Minutes")
                ax.set_title("ETA Confidence Range")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Maintenance Risk
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Maintenance Risk":
    st.title("🔧 Vehicle Maintenance Risk Assessment")

    col1, col2 = st.columns(2)
    with col1:
        vehicle_id           = st.text_input("Vehicle ID", "V0042")
        odometer_km          = st.number_input("Odometer (km)", 0, 500000, 125000, step=1000)
        avg_speed_kmh        = st.slider("Avg Speed (km/h)", 0.0, 120.0, 52.0)
        max_speed_kmh        = st.slider("Max Speed (km/h)", 0.0, 200.0, 112.0)
        engine_temp_mean_c   = st.slider("Engine Temp Mean (°C)", 60.0, 140.0, 94.0)
        engine_temp_max_c    = st.slider("Engine Temp Max (°C)",  70.0, 160.0, 108.0)
        battery_voltage_mean = st.slider("Battery Voltage Mean (V)", 9.0, 15.0, 13.1)

    with col2:
        battery_voltage_min  = st.slider("Battery Voltage Min (V)", 8.0, 14.0, 11.8)
        vibration_mean_ms2   = st.slider("Vibration Mean (m/s²)", 0.0, 5.0, 0.9)
        vibration_max_ms2    = st.slider("Vibration Max (m/s²)",  0.0, 10.0, 2.1)
        brake_pressure_mean  = st.slider("Brake Pressure Mean (bar)", 0.0, 15.0, 7.8)
        hard_brakes_count    = st.number_input("Hard Brakes Count", 0, 100, 8)
        idle_hours           = st.number_input("Idle Hours", 0.0, 1000.0, 120.0)
        trips_count          = st.number_input("Trips Count", 0, 2000, 310)

    if st.button("🔍 Assess Risk", type="primary"):
        if "maintenance_rf" not in models:
            st.error("❌ Maintenance model not found.")
        else:
            try:
                mrf     = models["maintenance_rf"]
                model   = mrf["model"]
                scaler  = mrf.get("scaler")
                feats   = mrf.get("features", [
                    "odometer_km","avg_speed_kmh","max_speed_kmh",
                    "engine_temp_mean_c","engine_temp_max_c",
                    "battery_voltage_mean","battery_voltage_min",
                    "vibration_mean_ms2","vibration_max_ms2",
                    "brake_pressure_mean","hard_brakes_count",
                    "idle_hours","trips_count"
                ])

                raw = pd.DataFrame([{
                    "odometer_km":          odometer_km,
                    "avg_speed_kmh":        avg_speed_kmh,
                    "max_speed_kmh":        max_speed_kmh,
                    "engine_temp_mean_c":   engine_temp_mean_c,
                    "engine_temp_max_c":    engine_temp_max_c,
                    "battery_voltage_mean": battery_voltage_mean,
                    "battery_voltage_min":  battery_voltage_min,
                    "vibration_mean_ms2":   vibration_mean_ms2,
                    "vibration_max_ms2":    vibration_max_ms2,
                    "brake_pressure_mean":  brake_pressure_mean,
                    "hard_brakes_count":    int(hard_brakes_count),
                    "idle_hours":           float(idle_hours),
                    "trips_count":          int(trips_count),
                }])

                available = [c for c in feats if c in raw.columns]
                X = raw[available]
                if scaler:
                    X = scaler.transform(X)

                fail_prob  = float(model.predict_proba(X)[0][1])

                # Isolation Forest anomaly score
                iso_score = 0.0
                if "maintenance_iso" in models:
                    try:
                        iso     = models["maintenance_iso"]
                        iso_m   = iso["model"]
                        iso_s   = iso.get("scaler")
                        iso_f   = iso.get("features", available)
                        Xi      = raw[[c for c in iso_f if c in raw.columns]]
                        if iso_s: Xi = iso_s.transform(Xi)
                        iso_score = float(-iso_m.decision_function(Xi)[0])
                        iso_score = max(0.0, min(1.0, iso_score + 0.5))
                    except Exception:
                        iso_score = 0.0

                risk_score = round((fail_prob*0.7 + iso_score*0.3)*100, 1)

                if risk_score < 25:
                    level,icon,rec = "LOW",      "🟢","No immediate action. Schedule routine inspection in 3 months."
                elif risk_score < 50:
                    level,icon,rec = "MEDIUM",   "🟡","Monitor closely. Schedule preventive maintenance within 30 days."
                elif risk_score < 75:
                    level,icon,rec = "HIGH",     "🟠","Schedule maintenance within 7 days. Check engine and battery."
                else:
                    level,icon,rec = "CRITICAL", "🔴","Immediate inspection required. Take vehicle out of service."

                st.subheader(f"{icon} Risk Level: {level}")
                c1,c2 = st.columns(2)
                c1.metric("Risk Score",          f"{risk_score:.1f} / 100")
                c2.metric("Failure Probability", f"{fail_prob*100:.1f}%")
                st.info(f"**Recommendation:** {rec}")

                color = {"LOW":"#4CAF50","MEDIUM":"#FFC107","HIGH":"#FF5722","CRITICAL":"#B71C1C"}[level]
                fig, ax = plt.subplots(figsize=(6,1.2))
                ax.barh(["Risk"],[risk_score], color=color, height=0.5)
                ax.barh(["Risk"],[100-risk_score], left=risk_score, color="#e0e0e0", height=0.5)
                ax.set_xlim(0,100)
                ax.set_title(f"Risk Score: {risk_score}/100")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — NLP Triage
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 NLP Triage":
    st.title("💬 Incident & Feedback Triage")

    CATEGORY_ICONS = {
        "delay":"⏰","damage":"📦","wrong_order":"❌",
        "driver_conduct":"🚗","positive":"✅",
    }

    examples = [
        "My delivery was 45 minutes late due to heavy traffic.",
        "Package arrived completely crushed and broken.",
        "Received someone else's package entirely.",
        "Driver was rude and argumentative at the door.",
        "Excellent service! Delivery arrived 10 minutes early.",
    ]
    example = st.selectbox("Load example:", ["(type your own)"] + examples)
    default = "" if example == "(type your own)" else example
    text    = st.text_area("Incident / feedback text:", value=default, height=120)

    if st.button("🔍 Classify", type="primary") and text.strip():
        if "nlp_pipeline" not in models:
            st.error("❌ NLP model not found.")
        else:
            try:
                nlp_dict  = models["nlp_pipeline"]
                pipeline  = nlp_dict["pipeline"]
                labels    = nlp_dict.get("label_names", None)

                predicted  = pipeline.predict([text])[0]
                proba      = pipeline.predict_proba([text])[0]
                classes    = labels if labels is not None else pipeline.classes_
                confidence = float(proba.max())
                icon       = CATEGORY_ICONS.get(str(predicted), "")

                st.success(f"{icon} **{str(predicted).upper()}** (confidence: {confidence*100:.1f}%)")

                scores_df = pd.DataFrame({
                    "category":   classes,
                    "probability":proba
                }).sort_values("probability", ascending=True)

                fig, ax = plt.subplots(figsize=(6,3))
                ax.barh(scores_df["category"], scores_df["probability"],
                        color="steelblue", edgecolor="white")
                ax.set_xlim(0,1)
                ax.set_xlabel("Probability")
                ax.set_title("Category Scores")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            except Exception as e:
                st.error(f"❌ Classification error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDA Summary
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA Summary":
    st.title("📈 EDA Key Findings")
    st.markdown("Summary of key findings from `notebooks/01_EDA.ipynb` — 9.28M NYC TLC Q1-2024 trips.")

    st.subheader("Trip Duration Distribution")
    st.markdown("- Right-skewed; median ~12 min, mean ~15 min due to long airport/cross-borough trips.")
    st.markdown("- 90th percentile: ~28 min. Outlier trips >60 min capped during feature engineering.")

    st.subheader("Peak Hours")
    hours  = list(range(24))
    volume = [12,8,5,3,4,9,18,35,42,38,32,36,40,38,35,37,48,55,52,45,38,30,22,16]
    colors = ["#FF5722" if 17<=h<=19 or 7<=h<=9 else "#5BA4CF" for h in hours]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(hours, volume, color=colors, edgecolor="white")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Relative Trip Volume")
    ax.set_title("Trip Volume by Hour — Q1 2024 (rush hours highlighted)")
    ax.set_xticks(hours)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Weather Impact on Trip Duration")
    c1,c2 = st.columns(2)
    c1.metric("Rain impact", "+8%",  delta_color="inverse")
    c2.metric("Snow impact", "+18%", delta_color="inverse")

    st.subheader("Borough Distribution")
    borough_data = {"Manhattan":7_200_000,"Queens":900_000,"Brooklyn":150_000,
                    "Bronx":30_000,"Staten Island":6_200}
    fig2, ax2 = plt.subplots(figsize=(7,3))
    ax2.bar(borough_data.keys(), borough_data.values(), color="#5BA4CF", edgecolor="white")
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"{x/1e6:.1f}M" if x>=1e6 else f"{x/1e3:.0f}K"))
    ax2.set_title("Pickup Trips by Borough — Q1 2024")
    ax2.set_ylabel("Trips")
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.subheader("Top Features Correlated with ETA")
    feats = ["trip_distance","route_enc","pu_zone_enc","pickup_hour_int",
             "is_rush_hour","rain_x_dist","wind_x_dist","temp"]
    corrs = [0.72,0.58,0.51,0.31,0.28,0.18,0.12,-0.09]
    fig3, ax3 = plt.subplots(figsize=(7,4))
    colors3 = ["tomato" if c>0 else "steelblue" for c in corrs]
    ax3.barh(feats[::-1], corrs[::-1], color=colors3[::-1])
    ax3.axvline(0, color="black", lw=0.8)
    ax3.set_xlabel("Pearson r")
    ax3.set_title("Feature Correlation with eta_min")
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.subheader("Model Comparison")
    model_results = pd.DataFrame({
        "Model":     ["Linear Regression","Random Forest","Gradient Boosting","Quantile RF"],
        "MAE (min)": [5.72, 5.17, 5.31, 5.44],
        "R²":        [0.61, 0.74, 0.71, 0.68],
    })
    st.dataframe(
        model_results.style.highlight_min(subset=["MAE (min)"], color="#c8f7c5")
                           .highlight_max(subset=["R²"],        color="#c8f7c5"),
        use_container_width=True
    )