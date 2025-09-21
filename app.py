import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_model.h5")

model = load_model()

class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

st.title("♻️ Waste Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    confidence = np.max(preds)
    label = class_names[np.argmax(preds)]

    if confidence < 0.5:
        st.warning(f"⚠️ Model not confident (Confidence: {confidence:.2f}). Try another image.")
    else:
        st.success(f"✅ Predicted: {label} (Confidence: {confidence:.2f})")






