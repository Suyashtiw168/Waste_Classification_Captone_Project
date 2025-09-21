import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ======================
# 📌 1. Download model from Google Drive (if not already downloaded)
# ======================
MODEL_URL = "https://drive.google.com/uc?id=1pT_ktqFOrcgAG8uoDVYMdHZ3bkZj_xB5"  # 👈 apna file ID
MODEL_PATH = "waste_model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait ⏳"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ======================
# 📌 2. Load Model
# ======================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ======================
# 📌 3. Class Names
# ======================
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ======================
# 📌 4. Streamlit UI
# ======================
st.title("♻️ Waste Classification App")
st.write("Upload an image of waste and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    label = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.success(f"### ✅ Prediction: {label} ({confidence:.2f}%)")

