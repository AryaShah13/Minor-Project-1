from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import traceback
import os

app = Flask(__name__)

# Load your trained model
MODEL_PATH = r"C:\Users\ahsha\Desktop\mp\minor project\cn.h5"  # Ensure the path is correct
model = tf.keras.models.load_model(MODEL_PATH)

# Preprocess image function
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Preprocessing Error: {e}")
        return None

@app.route('/')
def home():
    return "API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_path = "temp_image.jpg"
        file.save(img_path)

        img_array = preprocess_image(img_path)
        if img_array is None:
            return jsonify({'error': 'Image preprocessing failed'}), 500

        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction, axis=1)[0])

        return jsonify({'prediction': predicted_class})

    except Exception as e:
        print("An error occurred during prediction:")
        traceback.print_exc()  # Print full error trace in console
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
