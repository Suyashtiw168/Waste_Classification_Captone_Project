import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import matplotlib.pyplot as plt

# -----------------------------
# 📌 1. Download Model from Drive
# -----------------------------
MODEL_URL = "https://drive.google.com/uc?id=1pT_ktqFOrcgAG8uoDVYMdHZ3bkZj_xB5"
MODEL_PATH = "waste_model.keras"

@st.cache_resource
def load_model():
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# Class labels (same as training)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# -----------------------------
# 📌 2. Streamlit UI
# -----------------------------
st.title("♻️ Waste Classification App")
st.write("Upload an image of waste material, and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # -----------------------------
    # 📌 3. Preprocess Image
    # -----------------------------
    img = img.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # -----------------------------
    # 📌 4. Prediction
    # -----------------------------
    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    # -----------------------------
    # 📌 5. Show Results
    # -----------------------------
    if confidence < 0.2:  # 👈 Lowered threshold
        st.warning(f"⚠️ Model not confident (Confidence: {confidence:.2f}). Try another image.")
    else:
        st.success(f"✅ Predicted: {predicted_class} (Confidence: {confidence:.2f})")

    # Show all class probabilities
    st.subheader("🔍 All Class Probabilities")
    for cls, prob in zip(class_names, predictions[0]):
        st.write(f"- {cls}: {prob:.2f}")

    # -----------------------------
    # 📌 6. Plot Probabilities (Bar Chart)
    # -----------------------------
    fig, ax = plt.subplots()
    ax.bar(class_names, predictions[0])
    ax.set_title("Class Probabilities")
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)




