import pickle
import numpy as np

# Load model and scaler
with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_earthquake(depth, latitude, longitude, nst):
    """
    Predict earthquake magnitude based on input features.
    """
    # Prepare input data
    X_new = np.array([[depth, latitude, longitude, nst]])
    X_new = scaler.transform(X_new)

    # Make prediction
    prediction = model.predict(X_new)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    prediction = predict_earthquake(10, 37.75, -122.45, 20)
    print(f"Predicted Magnitude: {prediction:.2f}")

