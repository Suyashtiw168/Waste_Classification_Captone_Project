

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ✅ Model download (Google Drive se)
model_path = "waste_model.keras"
if not os.path.exists(model_path):
    gdown.download("https://drive.google.com/uc?id=1pT_ktqFOrcgAG8uoDVYMdHZ3bkZj_xB5", model_path, quiet=False)

# ✅ Load trained model
model = tf.keras.models.load_model(model_path)

# ✅ Classes (same order as training)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("♻️ Waste Classification App")
st.write("Upload an image to classify the type of waste.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ✅ Preprocess exactly like training
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0   # normalize
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Predict
    predictions = model.predict(img_array)
    score = np.max(predictions)
    pred_class = class_names[np.argmax(predictions)]

    st.write(f"### ✅ Predicted: {pred_class} (Confidence: {score:.2f})")
