from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ========================== Load Model & Encoders ========================== #

# File paths
model_path = "crop_yield_model.pkl"
encoder_path = "label_encoders.pkl"
scaler_path = "scaler.pkl"

# Check if files exist before loading
if not all(os.path.exists(path) for path in [model_path, encoder_path, scaler_path]):
    print("❌ Error: Model, encoder, or scaler files are missing. Please check file paths.")
    exit(1)

# Load trained model, encoders, and scaler
try:
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"❌ Error loading model or encoders: {e}")
    exit(1)  # Exit if models are not found

# ========================== Flask App Initialization ========================== #

app = Flask(__name__)

# ========================== Helper Function: Label Encoding ========================== #

def encode_or_add_new(label_encoder, value):
    """Encodes a categorical value, adding new classes if needed."""
    if value not in label_encoder.classes_:
        label_encoder.classes_ = np.append(label_encoder.classes_, value)
        label_encoder = LabelEncoder().fit(label_encoder.classes_)  # Re-fit encoder
    return label_encoder.transform([value])[0]

# ========================== Routes ========================== #

@app.route("/", methods=["GET"])
def home():
    """API Home Route"""
    return jsonify({"message": "🌱 Crop Yield Prediction API is Running!"})

@app.route("/predict_yield", methods=["POST"])
def predict_yield():
    """Predict Crop Yield Based on Input Data"""
    try:
        data = request.get_json()

        # Required Fields
        required_fields = ["Crop", "Crop_Year", "Season", "State", "Area", "Annual_Rainfall", "Fertilizer"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"❌ Missing field: {field}"}), 400

        # Convert & Encode Inputs
        crop = encode_or_add_new(label_encoders["Crop"], data["Crop"])
        crop_year = int(data["Crop_Year"])
        season = encode_or_add_new(label_encoders["Season"], data["Season"])
        state = encode_or_add_new(label_encoders["State"], data["State"])
        area = float(data["Area"])
        annual_rainfall = float(data["Annual_Rainfall"])
        fertilizer = float(data["Fertilizer"])

        # Scale numerical inputs (Crop Year, Area, Rainfall, Fertilizer)
        scaled_values = scaler.transform([[crop_year, area, annual_rainfall, fertilizer]])[0]
        crop_year, area, annual_rainfall, fertilizer = scaled_values

        # Prepare input data
        input_data = pd.DataFrame([[crop, crop_year, season, state, area, annual_rainfall, fertilizer]],
                                  columns=["Crop", "Crop_Year", "Season", "State", "Area", "Annual_Rainfall",
                                           "Fertilizer"])

        # Make Prediction
        yield_prediction = model.predict(input_data)[0]

        return jsonify({"Predicted_Yield": round(yield_prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================== Run Flask App ========================== #

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Runs on port 5000 (Remove debug=True in production)
