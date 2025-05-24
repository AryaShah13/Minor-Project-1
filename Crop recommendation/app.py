from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

try:
    with open("crop_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

try:
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("✅ Label encoder loaded successfully.")
except Exception as e:
    print(f"❌ Error loading label encoder: {e}")
    exit(1)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🌾 Crop Recommendation API is running!"})

@app.route("/predict", methods=["POST"])
def predict_crop():
    try:
        data = request.get_json()
        print("✅ Received Data:", data)

        required_fields = ["N", "P", "K", "temperature", "humidity", "rainfall"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        N = int(data["N"])
        P = int(data["P"])
        K = int(data["K"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        rainfall = float(data["rainfall"])

        input_features = np.array([[N, P, K, temperature, humidity, rainfall]])
        prediction = model.predict(input_features)[0]
        crop_name = label_encoder.inverse_transform([prediction])[0]

        confidence = 100.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_features)
            confidence = round(100 * float(np.max(proba)), 2)

        return jsonify({
            "recommended_crop": crop_name,
            "confidence": confidence
        })

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)