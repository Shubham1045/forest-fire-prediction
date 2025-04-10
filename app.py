import streamlit as st
import pickle
import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load model and scaler
try:
    ridge_model = pickle.load(open('notebooks/ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('notebooks/scaler.pkl', 'rb'))
    logging.info("Model and scaler loaded successfully.")
except FileNotFoundError:
    logging.error("Required model files not found.")
    st.error("Model or scaler files are missing. Please check the path.")
    st.stop()

# Streamlit app settings
st.set_page_config(page_title="Forest Fire FWI Predictor", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #111;
        }

        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }

        .stButton>button:hover {
            background-color: #ff1f1f;
        }

        .result-box {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 12px;
            border: 3px solid #007BFF;
            margin-top: 30px;
            color: #000000;
        }

        .result-title {
            font-size: 26px;
            font-weight: 700;
            color: #d00000;
        }

        .risk-level {
            font-size: 22px;
            font-weight: 600;
            margin-top: 10px;
        }

        .risk-safe { color: green; }
        .risk-moderate { color: orange; }
        .risk-high { color: #e67e22; }
        .risk-veryhigh { color: #e74c3c; }
        .risk-extreme { color: red; }
    </style>
""", unsafe_allow_html=True)

# App title and instructions
st.title("🔥 Forest Fire Weather Index Predictor")
st.markdown("Use this tool to assess the **risk of forest fire** based on weather and environmental data.")

# Input form
with st.form("input_form"):
    st.subheader("Enter Environmental Parameters")

    col1, col2 = st.columns(2)

    with col1:
        Temperature = st.number_input("🌡️ Temperature (°C)", value=29.0, step=0.1)
        RH = st.number_input("💧 Relative Humidity (%)", value=57.0, step=0.1)
        Ws = st.number_input("🌬️ Wind Speed (km/h)", value=18.0, step=0.1)
        Rain = st.number_input("🌧️ Rainfall (mm)", value=0.0, step=0.1)
        FFMC = st.number_input("🔥 FFMC (Fine Fuel Moisture Code)", value=65.7, step=0.1)

    with col2:
        DMC = st.number_input("🌲 DMC (Duff Moisture Code)", value=3.4, step=0.1)
        ISI = st.number_input("💨 ISI (Initial Spread Index)", value=1.3, step=0.1)
        Classes = st.number_input("📊 Classes (encoded)", value=0.0, step=1.0)
        Region = st.number_input("🗺️ Region (encoded)", value=0.0, step=1.0)

    submit = st.form_submit_button("🚀 Predict FWI")

# On form submission
if submit:
    try:
        # Prepare input
        input_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        input_df = pd.DataFrame(input_data, columns=[
            'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region'
        ])
        new_data_scaled = standard_scaler.transform(input_df)
        result = ridge_model.predict(new_data_scaled)[0]

        # Risk level classification
        low_threshold = 2.0
        moderate_threshold = 5.0
        high_threshold = 8.0
        very_high_threshold = 11.0

        if result < low_threshold:
            risk_level = "🟢 Safe"
            fire_message = "Low risk of fire. Conditions are stable."
        elif result < moderate_threshold:
            risk_level = "🟡 Moderate"
            fire_message = "Moderate risk. Stay alert."
        elif result < high_threshold:
            risk_level = "🟠 High"
            fire_message = "WARNING: Elevated risk of forest fire detected!"
        elif result < very_high_threshold:
            risk_level = "🔴 Very High"
            fire_message = "ALERT: High risk! Take precautions immediately!"
        else:
            risk_level = "⚫ Extreme"
            fire_message = "DANGER: Fire conditions detected! Emergency action required!"

        # Show results in a styled box
        st.markdown(f"""
            <div class="result-box">
                <div class="result-title">Predicted FWI: <span style="color:#d00000;">{result:.2f}</span></div>
                <div class="risk-level">{risk_level}</div>
                <p style="margin-top: 10px;">{fire_message}</p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        st.error(f"An error occurred during prediction: {e}")
