import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("C:/Users/ahsha/Desktop/Crop_recommendation_cleaned.csv")
features = ["N", "P", "K", "temperature", "humidity", "rainfall"]
X = df[features]
y = df["label"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

def predict_crop(N: int, P: int, K: int, temperature: float, humidity: float, rainfall: float) -> str:
    with open("crop_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    input_data = [[N, P, K, temperature, humidity, rainfall]]
    predicted_label = model.predict(input_data)[0]
    crop_name = label_encoder.inverse_transform([predicted_label])[0]

    return crop_name

if __name__ == "__main__":
    try:
        nitrogen_level = int(input("Enter nitrogen level: "))
        phosphorous_level = int(input("Enter phosphorous level: "))
        potassium_level = int(input("Enter potassium level: "))
        temperature_level = float(input("Enter temperature level (in Celsius): "))
        humidity_level = float(input("Enter humidity level: "))
        rainfall_level = float(input("Enter rainfall level (in mm): "))

        crop = predict_crop(
            nitrogen_level,
            phosphorous_level,
            potassium_level,
            temperature_level,
            humidity_level,
            rainfall_level
        )
        print("Recommended Crop:", crop)
    except ValueError:
        print("Invalid input. Please enter numeric values only.")