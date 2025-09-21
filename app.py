import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ================================
# Download model (if not exists)
# ================================
MODEL_PATH = "waste_model.keras"
FILE_ID = "1pT_ktqFOrcgAG8uoDVYMdHZ3bkZj_xB5"   # 👈 apna Drive ka file id

if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# ================================
# Load model
# ================================
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ================================
# Streamlit UI
# ================================
st.title("♻ Waste Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)[0]

    # ✅ Top-2 predictions
    top2_idx = preds.argsort()[-2:][::-1]
    top2_classes = [(class_names[i], preds[i]) for i in top2_idx]

    # Show results
    st.subheader("Predictions:")
    for cls, conf in top2_classes:
        st.write(f"✅ {cls} (Confidence: {conf:.2f})")



