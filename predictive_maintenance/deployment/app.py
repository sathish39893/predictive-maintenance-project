import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="sathish39893/predictive-maintenance", filename="best_predictive_maintenance_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Predictive Maintenance App
st.title("Predictive Maintenance App")
st.write("The Predictive Maintenance App is an internal tool which predicts machine failure based on conditions")
st.write("Please enter the machine sensor data to check if the machine can failure or not")

# User input of sensor data
EngineRPM = st.number_input("Engine RPM", min_value=61, max_value=2239, value=650)
LubeOilPressure = st.number_input("Lube Oil Pressure (kPa)", min_value=0.003384, max_value=7.265566, value=4.055272)
FuelPressure = st.number_input("Fuel Pressure (kPa)", min_value=0.003187, max_value=21.138326, value=7.744973)
CoolantPressure = st.number_input("Coolant Pressure (kPa)", min_value=0.002483, max_value=7.478505, value=2.848840)
LubeOilTemperature = st.number_input("Lube oil Temerature (°C)", min_value=71.321974, max_value=89.580796, value=78.071691)
CoolantTemperature = st.number_input("Coolant Temperature (°C)", min_value=61.673325, max_value=195.527912, value=82.915411)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Engine rpm': EngineRPM,
    'Lub oil pressure': LubeOilPressure,
    'Fuel pressure': FuelPressure,
    'Coolant pressure': CoolantPressure,
    'lub oil temp': LubeOilTemperature,
    'Coolant temp': CoolantTemperature,
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "fail" if prediction == 1 else "not fail"
    st.write(f"Based on the information provided, the machine is likely to {result}.")
