import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# Download model from Google Drive if not exists
MODEL_PATH = "waste_model.keras"
FILE_ID = "1pT_ktqFOrcgAG8uoDVYMdHZ3bkZj_xB5"   # 👈 apna drive file id
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("♻ Waste Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((128,128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    confidence = np.max(preds)
    pred_class = class_names[np.argmax(preds)]

    # ✅ Confidence threshold check
    if confidence < 0.5:
        st.warning(f"⚠️ Model not confident (Confidence: {confidence:.2f}). Try another image.")
    else:
        st.success(f"✅ Predicted: {pred_class} (Confidence: {confidence:.2f})")

