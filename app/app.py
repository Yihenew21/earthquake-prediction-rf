import os
import streamlit as st
import pickle
import numpy as np
import plotly.express as px
import pandas as pd
import pydeck as pdk

# Set your Mapbox API token
os.environ["MAPBOX_ACCESS_TOKEN"] = "<pk.eyJ1IjoiYmlydWsyMSIsImEiOiJjbTZ3bGY2ZnAwaG1jMnFzNjIxMjFnYXFpIn0.ztG_q89N0vo6AztZCuDKKA>"

# Load trained model
with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize session state for latitude and longitude if they don't exist
if 'latitude' not in st.session_state:
    st.session_state.latitude = 0.0
if 'longitude' not in st.session_state:
    st.session_state.longitude = 0.0

# Streamlit UI Configuration
st.set_page_config(page_title="Earthquake Prediction", page_icon="🌍", layout="wide")

# Custom Styling for Mobile Responsiveness
st.markdown("""
    <style>
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            height: 50px;
            width: 100%;
        }
        .stTitle {
            text-align: center;
            color: #ff914d;
            font-size: 30px;
            font-weight: bold;
        }
        /* Responsive Design for mobile devices */
        @media (max-width: 768px) {
            .stTitle {
                font-size: 24px;
            }
            .stButton>button {
                font-size: 16px;
            }
        }
        /* Scrollable content */
        .block-container {
            overflow-y: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="stTitle">🌍 Earthquake Magnitude Prediction</p>', unsafe_allow_html=True)
st.write("Enter the earthquake parameters below to predict the magnitude.")

# Sidebar for Inputs
st.sidebar.header("🔧 Input Parameters")
depth = st.sidebar.number_input("🌎 Depth (km)", min_value=0.0, max_value=700.0, value=10.0, step=0.1, key="depth_input")
latitude = st.sidebar.number_input("📍 Latitude", min_value=-90.0, max_value=90.0, value=st.session_state.latitude, step=0.01, key="latitude_input")
longitude = st.sidebar.number_input("📍 Longitude", min_value=-180.0, max_value=180.0, value=st.session_state.longitude, step=0.01, key="longitude_input")
nst = st.sidebar.slider("📡 Number of Stations (nst)", min_value=0, max_value=100, value=10, step=1, key="nst_input")

# Back Button
if st.sidebar.button("🔙 Go Back to Prediction"):
    st.experimental_rerun()

# Interactive Map with Mapbox
st.sidebar.subheader("🌍 Select Location on Map")
map_center = [latitude, longitude]
map_zoom = 3  # Zoom level (adjust for better visualization)

# Create the map with pydeck using Mapbox
deck = pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=map_center[0], 
        longitude=map_center[1],
        zoom=map_zoom,
        pitch=0
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame({"lat": [latitude], "lon": [longitude]}),
            get_position=["lon", "lat"],
            get_radius=100000,
            get_fill_color=[255, 0, 0],
            radius_min_pixels=5,
        )
    ],
    map_style="mapbox://styles/mapbox/streets-v11",  # Mapbox style (use Mapbox access token)
    tooltip={"text": "Latitude: {lat}\nLongitude: {lon}"}
)

st.pydeck_chart(deck)

# Map click event to update latitude and longitude
if st.button("Click Map to Set Location"):
    # Simulating map click update
    # This part will need the logic to get lat, lon from the map click event, or from user selection.
    st.session_state.latitude = latitude  # Save updated latitude
    st.session_state.longitude = longitude  # Save updated longitude

    # Update the sidebar inputs to reflect the new location
    latitude = st.session_state.latitude
    longitude = st.session_state.longitude

    # Update the input fields with the new location
    st.sidebar.number_input("📍 Latitude", min_value=-90.0, max_value=90.0, value=latitude, step=0.01, key="latitude_input")
    st.sidebar.number_input("📍 Longitude", min_value=-180.0, max_value=180.0, value=longitude, step=0.01, key="longitude_input")

# Process Input and Prediction
if st.sidebar.button("🚀 Predict Magnitude"):
    try:
        X_new = np.array([[depth, latitude, longitude, nst]])
        X_new = scaler.transform(X_new)
        prediction = model.predict(X_new)[0]

        # Show Prediction Result
        st.success(f"🌟 **Predicted Earthquake Magnitude: {prediction:.2f}**")
        
        # Display Results as Gauge Chart
        fig = px.bar(x=["Predicted Magnitude"], y=[prediction], text=[f"{prediction:.2f}"],
                     color=["Magnitude"], color_discrete_sequence=["#FF4B4B"],
                     labels={"x": "Prediction", "y": "Magnitude"})
        fig.update_layout(yaxis=dict(range=[0, 10]), title="🌍 Earthquake Magnitude Prediction", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Display Alert if Magnitude > 5
        if prediction > 5:
            st.warning(f"⚠️ **Warning!** The predicted magnitude is high ({prediction:.2f}), indicating a potentially strong earthquake.")

    except Exception as e:
        st.error("⚠️ Error occurred: " + str(e))

# Feature Importance (Optional)
st.sidebar.subheader("📊 Feature Importance")
if st.sidebar.button("Show Feature Importance"):
    feature_names = ["Depth", "Latitude", "Longitude", "NST"]
    feature_importance = model.feature_importances_

    fig_imp = px.bar(x=feature_names, y=feature_importance, 
                     title="🔍 Feature Importance", color=feature_importance, 
                     labels={"x": "Features", "y": "Importance"},
                     color_continuous_scale="viridis")
    st.plotly_chart(fig_imp, use_container_width=True)

# Footer
st.markdown("""
    <hr>
    <center>🌍 Developed by YIHENEW ANIMUT ❤️ for Earthquake MAGNITUDE Prediction | github Username : @Yihenew21</center>
""", unsafe_allow_html=True)
