import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained CNN model
model_path = "cn.h5"
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    exit()
model = load_model(model_path)

# Soil class labels (no label encoding)
class_labels = {
    0: "Black Soil",
    1: "Cinder Soil",
    2: "Laterite Soil",
    3: "Peat Soil",
    4: "Yellow Soil"
}

# Function to preprocess the image
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        exit()

# Function to predict soil type
def predict_soil(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)  # Get the class with the highest probability
    soil_type = class_labels.get(predicted_class, "Unknown Soil Type")  # Direct mapping

    print(f"Class Probabilities: {prediction[0]}")  # Debugging probabilities
    return predicted_class, soil_type

# Input image path
image_path = input("Enter image path: ").strip().replace('"', '').replace("'", "")

# Check if file exists
if not os.path.exists(image_path):
    print(f"Error: The file '{image_path}' does not exist.")
    exit()

# Make prediction
predicted_class, predicted_soil = predict_soil(image_path)
print(f"Predicted Class Index: {predicted_class}")
print(f"Predicted Soil Type: {predicted_soil}")
