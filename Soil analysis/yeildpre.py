import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\ahsha\Desktop\crop_yield.csv")

# Encode categorical features
label_encoders = {}
for col in ['Crop', 'State']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target variable
X = df[['Crop', 'Area', 'Annual_Rainfall', 'State', 'Fertilizer']]
y = df['Yield']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
numeric_features = ['Area', 'Annual_Rainfall', 'Fertilizer']
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print model accuracy
print(f'Model Performance:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R2 Score: {r2:.4f}')


# Function to predict yield based on user input
def predict_yield():
    try:
        # Get user input
        crop = input("Enter Crop Name: ").strip()
        area = float(input("Enter Area (in hectares): "))
        rainfall = float(input("Enter Annual Rainfall (in mm): "))
        state = input("Enter State Name: ").strip()
        fertilizer = float(input("Enter Fertilizer Amount (in kg/ha): "))

        # Validate input
        if crop not in label_encoders['Crop'].classes_ or state not in label_encoders['State'].classes_:
            return "Invalid crop or state name. Check your input."

        # Encode categorical values
        crop_encoded = label_encoders['Crop'].transform([crop])[0]
        state_encoded = label_encoders['State'].transform([state])[0]

        # Prepare input data
        input_data = np.array([[crop_encoded, area, rainfall, state_encoded, fertilizer]], dtype=float)

        # Scale only numerical columns (Area, Annual_Rainfall, Fertilizer)
        input_data[:, [1, 2, 4]] = scaler.transform(input_data[:, [1, 2, 4]])

        # Predict yield
        predicted_yield = model.predict(input_data)[0]
        return f'Predicted Yield: {predicted_yield:.2f} tons per hectare'

    except Exception as e:
        return f"Error: {e}"


# Run the yield prediction
print(predict_yield())