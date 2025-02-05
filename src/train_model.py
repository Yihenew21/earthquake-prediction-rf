import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import load_and_preprocess_data

# Load data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance: MSE={mse:.4f}, R2={r2:.4f}")

# Save trained model
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")

