import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset
df = pd.read_csv("C:/Users/ahsha./Desktop/mp/yeild prediction/crop_yield_cleaned.csv")

# Encode categorical features
label_encoders = {}
for col in ["Crop", "Season", "State"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save encoders
joblib.dump(label_encoders, "label_encoders.pkl")

# Select features and target
features = ["Crop", "Crop_Year", "Season", "State", "Area", "Annual_Rainfall", "Fertilizer"]
target = "Yield"
X = df[features]
y = df[target]

# Scale numerical features
scaler = StandardScaler()
X[["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer"]] = scaler.fit_transform(
    X[["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer"]])

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, "crop_yield_model.pkl")
print("Model saved as crop_yield_model.pkl")


# Load model and predict from user input
def predict_yield():
    model = joblib.load("crop_yield_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")

    def encode_or_add_new(label_encoder, value):
        if value not in label_encoder.classes_:
            new_classes = np.append(label_encoder.classes_, value)
            label_encoder.classes_ = new_classes
        return label_encoder.transform([value])[0]

    crop = input("Enter Crop: ")
    crop_year = int(input("Enter Crop Year: "))
    season = input("Enter Season: ")
    state = input("Enter State: ")
    area = float(input("Enter Area (in hectares): "))
    annual_rainfall = float(input("Enter Annual Rainfall (in mm): "))
    fertilizer = float(input("Enter Fertilizer Usage (in kg): "))

    # Encode categorical inputs dynamically
    crop = encode_or_add_new(label_encoders["Crop"], crop)
    season = encode_or_add_new(label_encoders["Season"], season)
    state = encode_or_add_new(label_encoders["State"], state)

    # Scale numerical inputs
    scaled_values = scaler.transform([[crop_year, area, annual_rainfall, fertilizer]])[0]
    crop_year, area, annual_rainfall, fertilizer = scaled_values

    # Create DataFrame with feature names
    input_data = pd.DataFrame([[crop, crop_year, season, state, area, annual_rainfall, fertilizer]],
                              columns=["Crop", "Crop_Year", "Season", "State", "Area", "Annual_Rainfall", "Fertilizer"])

    yield_prediction = model.predict(input_data)[0]
    print(f"Predicted Yield: {yield_prediction:.2f} ton")


# Run prediction function if script is executed
if __name__ == "__main__":
    predict_yield()
