 # Earthquake Prediction using Random Forest

## Overview
This project predicts earthquake magnitudes based on seismic features using a Random Forest model.

## Dataset
- Source: USGS Earthquake Catalog
- Features: Depth, Latitude, Longitude, Number of Stations (nst)
- Target: Earthquake Magnitude

## Installation

pip install -r requirements.txt


## Usage
1. **Train Model**

python src/train_model.py

2. **Make Predictions**

python src/predict.py

3. **Run API**

uvicorn deployment.api:app --reload

Then access the API at `http://127.0.0.1:8000/docs`

## Deployment
This project includes a FastAPI-based API for real-time earthquake predictions.

---