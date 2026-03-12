import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained  
model = joblib.load('crop_model.pkl')

# App Title
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.title("🌾 Crop Yield Prediction System ")
st.markdown("Enter the soil and environmental details below to predict the estimated crop yield.")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    crop_type = st.selectbox("Select Crop Type", ['Wheat', 'Corn', 'Rice', 'Barley', 'Soybean', 'Cotton', 'Sugarcane', 'Tomato', 'Potato', 'Sunflower'])
    soil_type = st.selectbox("Select Soil Type", ['Peaty', 'Loamy', 'Sandy', 'Saline', 'Clay'])
    soil_ph = st.slider("Soil pH Level", 0.0, 14.0, 6.5)
    temp = st.number_input("Temperature (°C)", value=25.0)
    humidity = st.number_input("Humidity (%)", value=60.0)

with col2:
    wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
    n = st.number_input("Nitrogen (N)", value=50.0)
    p = st.number_input("Phosphorus (P)", value=40.0)
    k = st.number_input("Potassium (K)", value=30.0)
    soil_quality = st.slider("Soil Quality Index", 0.0, 100.0, 50.0)

# Prediction Logic
if st.button("Predict Yield"):
    # Create a dataframe for the input
    input_df = pd.DataFrame([[
        crop_type, soil_type, soil_ph, temp, humidity, 
        wind_speed, n, p, k, soil_quality
    ]], columns=['Crop_Type', 'Soil_Type', 'Soil_pH', 'Temperature', 'Humidity', 'Wind_Speed', 'N', 'P', 'K', 'Soil_Quality'])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display Result
    st.success(f"### Predicted Crop Yield 🌱 : {prediction:.2f} units")
    st.info("Note: This prediction is based on the Random Forest ML model.")