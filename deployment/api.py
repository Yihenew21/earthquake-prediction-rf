from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model and scaler
with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI SERVER STARTED SUCCESSFULLY!"}

class EarthquakeInput(BaseModel):
    depth: float
    latitude: float
    longitude: float
    nst: int

@app.post("/predict/")
def predict_magnitude(input_data: EarthquakeInput):
    X_new = np.array([[input_data.depth, input_data.latitude, input_data.longitude, input_data.nst]])
    X_new = scaler.transform(X_new)
    
    prediction = model.predict(X_new)
    return {"predicted_magnitude": round(prediction[0], 2)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

