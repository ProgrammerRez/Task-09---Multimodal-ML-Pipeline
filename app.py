import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tempfile import NamedTemporaryFile
import os

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Multimodal House Price Predictor")
st.title("Predict House Prices with Tabular + Image Data")
st.divider()

# -----------------------------
# Load Models and Pipeline
# -----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("models/model.joblib")
    pipeline = joblib.load("pipelines/preprocessor.joblib")
    return model, pipeline

model, tabular_pipeline = load_assets()

# -----------------------------
# Load ResNet50 for Image Embeddings
# -----------------------------
@st.cache_resource
def get_resnet_model(image_size=(224, 224)):
    base_model = ResNet50(weights="imagenet", include_top=False,
                          input_shape=(image_size[0], image_size[1], 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    return Model(inputs=base_model.input, outputs=x)

resnet_model = get_resnet_model()

# -----------------------------
# Image Preprocessing
# -----------------------------

def preprocess_image(uploaded_file, image_size=(224, 224)):
    """Converts uploaded image to NumPy array with batch dimension"""
    # Create a temp file that can be reopened
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name  # path to temp file

    try:
        img = load_img(tmp_path, target_size=image_size)
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)
    finally:
        # Cleanup temp file
        os.remove(tmp_path)

    return arr

def extract_image_embeddings(arr):
    return resnet_model.predict(arr, verbose=0)

# -----------------------------
# User Input Form
# -----------------------------
# Extract categories from pipeline
ordinal_encoder = tabular_pipeline.named_transformers_['cat'].named_steps['encoder']
categorical_cols = ['city']
categories_dict = {col: list(ordinal_encoder.categories_[i])
                   for i, col in enumerate(categorical_cols)}

# Streamlit form
with st.form("house_form"):
    st.header("Enter Tabular Features")
    
    bed = st.number_input("Bedrooms", min_value=0, value=3)
    bath = st.number_input("Bathrooms", min_value=0, value=2)
    sqft = st.number_input("Square Footage", min_value=100, value=1500)
    garage = st.number_input("Garage", min_value=0, value=1)
    
    # Dynamic selectbox
    city = st.selectbox("City", options=categories_dict['city'])

    uploaded_file = st.file_uploader("Upload House Image", type=["jpg","jpeg","png"])
    
    submit_button = st.form_submit_button("Predict Price")
# -----------------------------
# Prediction Logic
# -----------------------------
if submit_button:
    if uploaded_file is None:
        st.warning("Please upload an image to predict price.")
    else:
        # 1️⃣ Prepare tabular input
        total_rooms = bed + bath
        room_per_sqft = total_rooms / sqft

        input_data = pd.DataFrame({
            'bed': [bed],
            'bath': [bath],
            'sqft': [sqft],
            'garage': [garage],
            'TotalRooms': [total_rooms],
            'Room_per_Sqft': [room_per_sqft],
            'citi': [city]
        })

        X_tab_preprocessed = tabular_pipeline.transform(input_data)

        # 2️⃣ Prepare image input
        img_arr = preprocess_image(uploaded_file)
        img_features = extract_image_embeddings(img_arr)

        # 3️⃣ Combine features
        X_combined = np.hstack([X_tab_preprocessed, img_features])

        # 4️⃣ Predict
        pred_price = model.predict(X_combined)
        st.success(f"Predicted House Price: ${pred_price[0]:,.2f}")
