# Streamlit App with Login, Signout, Sidebar Navigation and Soil Type Prediction
# Ensure you have cn.h5 model in the same directory

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the trained CNN model
MODEL_PATH = 'cn.h5'

@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Soil class labels (same as your AI model)
class_labels = {
    0: "Black Soil",
    1: "Cinder Soil",
    2: "Laterite Soil",
    3: "Peat Soil",
    4: "Yellow Soil"
}

# Simulated user database (Replace with a real database for production)
user_db = {
    "user1": "password1",
    "user2": "password2"
}

# Session state for login
def login(username, password):
    if username in user_db and user_db[username] == password:
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.success("Logged in successfully!")
    else:
        st.error("Invalid username or password")

def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = ""
    st.success("Signed out successfully!")

# Preprocess image function
def preprocess_image(image_data):
    try:
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img = img.resize((224, 224))  # Match model input size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Sidebar menu
def sidebar_menu():
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Go to", ("Home", "Upload Image", "Profile", "Sign Out"))
    return menu

# Main App Logic
def main_app():
    menu_choice = sidebar_menu()

    if menu_choice == "Home":
        st.title("🌱 Soil Type Prediction App")
        st.write(f"Welcome *{st.session_state['username']}*!")
        st.write("Use the sidebar to navigate through the app.")

    elif menu_choice == "Upload Image":
        st.title("Upload a Soil Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            img_data = uploaded_file.read()
            preprocessed_image = preprocess_image(img_data)

            if preprocessed_image is not None:
                prediction = model.predict(preprocessed_image)
                predicted_class = np.argmax(prediction)
                soil_type = class_labels.get(predicted_class, "Unknown Soil Type")

                st.subheader("Prediction Result")
                st.write(f"*Predicted Soil Type:* {soil_type}")
                st.write(f"*Class Probabilities:* {prediction[0]}")
            else:
                st.error("Failed to process the image.")
        else:
            st.info("Please upload an image file.")

    elif menu_choice == "Profile":
        st.title("User Profile")
        st.write(f"👤 Username: *{st.session_state['username']}*")
        st.write("Prediction history and settings can be shown here in future.")

    elif menu_choice == "Sign Out":
        logout()

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = ""

# Login page
if not st.session_state['logged_in']:
    st.title("🔐 Login to Soil Type Prediction App")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            login(username, password)
else:
    main_app()