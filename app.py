import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Google Drive se model download (agar local me nahi hai)
MODEL_PATH = "waste_model_downloaded.keras"
if not os.path.exists(MODEL_PATH):
    file_id = "1pT_ktqFOrcgAG8uoDVYMdHZ3bkZj_xB5"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

# Model load
model = tf.keras.models.load_model(MODEL_PATH)

# Classes
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Streamlit UI
st.title("♻️ Waste Classification App")
st.write("Upload an image and I will predict the waste category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = np.array(image.resize((128, 128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1,128,128,3)

    # Prediction
    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    # Show results
    if confidence < 0.5:
        st.warning(f"⚠️ Model not confident (Confidence: {confidence:.2f}). Try another image.")
    else:
        st.success(f"✅ Predicted: {predicted_class} (Confidence: {confidence:.2f})")

        # Show all class probabilities
        st.subheader("🔍 All Class Probabilities")
        for cls, prob in zip(class_names, predictions[0]):
            st.write(f"- {cls}: {prob:.2f}")




