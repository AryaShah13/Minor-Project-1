import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\ahsha\Desktop\crop_yield.csv")  # Ensure the dataset is in the same directory

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

# Handle missing values (if any)
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for col in ["crop", "season", "state"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df[["crop", "crop_year", "season", "state", "area", "annual_rainfall", "fertilizer"]].copy()
y = df["yield"]

# Standardize numerical features
numeric_features = ["crop_year", "area", "annual_rainfall", "fertilizer"]
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=7, random_state=42)
xgb.fit(X_train, y_train)

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Save models and encoders
joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(rf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# Function to predict yield
def predict_yield(crop, area, rainfall, state, fertilizer):
    # Load models and encoders
    xgb = joblib.load("xgb_model.pkl")
    rf = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")

    try:
        crop_encoded = label_encoders["crop"].transform([crop])[0]
        state_encoded = label_encoders["state"].transform([state])[0]
    except ValueError:
        return "Invalid crop or state name. Check your input."

    input_data = np.array([[crop_encoded, 2024, 0, state_encoded, area, rainfall, fertilizer]], dtype=float)
    input_data[:, [1, 4, 5, 6, 7]] = scaler.transform(input_data[:, [1, 4, 5, 6, 7]])

    # Predict using both models and average
    xgb_pred = xgb.predict(input_data)[0]
    rf_pred = rf.predict(input_data)[0]
    final_pred = (xgb_pred + rf_pred) / 2

    return round(final_pred, 2)

# Example usage
crop_input = input("Enter crop name: ").strip().lower()
area_input = float(input("Enter area (hectares): "))
rainfall_input = float(input("Enter annual rainfall (mm): "))

state_input = input("Enter state name: ").strip().lower()
fertilizer_input = float(input("Enter fertilizer amount (kg): "))

predicted_yield = predict_yield(crop_input, area_input, rainfall_input,state_input, fertilizer_input)
print(f"Predicted Yield: {predicted_yield} tons/hectare")

# Evaluate model
y_pred_xgb = xgb.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred = (y_pred_xgb + y_pred_rf) / 2  # Averaging ensemble

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
