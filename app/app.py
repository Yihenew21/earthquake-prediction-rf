import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("🌍 Earthquake Magnitude Prediction")
st.write("Enter the earthquake parameters below to predict the magnitude.")

# Input fields
depth = st.number_input("Depth (km)", min_value=0.0, max_value=700.0, value=10.0, step=0.1)
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.01)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.01)
nst = st.number_input("Number of Stations (nst)", min_value=0, max_value=100, value=10, step=1)

# Predict button
if st.button("Predict Magnitude"):
    # Prepare input data
    X_new = np.array([[depth, latitude, longitude, nst]])
    X_new = scaler.transform(X_new)

    # Make prediction
    prediction = model.predict(X_new)
    st.success(f"🌟 Predicted Earthquake Magnitude: {prediction[0]:.2f}")

