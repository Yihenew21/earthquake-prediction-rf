import streamlit as st
import pickle
import numpy as np
import plotly.express as px
import pandas as pd

# Load trained model
with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

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
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="stTitle">🌍 Earthquake Magnitude Prediction</p>', unsafe_allow_html=True)
st.write("Enter the earthquake parameters below to predict the magnitude.")

# Sidebar for Inputs
st.sidebar.header("🔧 Input Parameters")
depth = st.sidebar.number_input("🌎 Depth (km)", min_value=0.0, max_value=700.0, value=10.0, step=0.1)
latitude = st.sidebar.number_input("📍 Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.01)
longitude = st.sidebar.number_input("📍 Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.01)
nst = st.sidebar.slider("📡 Number of Stations (nst)", min_value=0, max_value=100, value=10, step=1)

# Interactive Map for Selecting Latitude and Longitude
st.sidebar.subheader("🌍 Select Location on Map")
location_data = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
selected_location = st.sidebar.map(location_data)

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
    <center>🌍 Developed BY @Yihenew Animut ❤️ for Earthquake Prediction | github : @Yihenew21</center>
""", unsafe_allow_html=True)
