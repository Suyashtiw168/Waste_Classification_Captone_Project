import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# ✅ Class names aligned with training dataset order
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ✅ Google Drive model download
@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=1pT_ktqFOrcgAG8uoDVYMdHZ3bkZj_xB5"
    output = "waste_model.keras"
    gdown.download(url, output, quiet=False)
    model = tf.keras.models.load_model(output)
    return model

model = load_model()

st.title("♻️ Waste Classification App")
st.write("Upload an image of waste material and let the model classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))   # resize for model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    if confidence < 0.5:
        st.warning(f"⚠️ Model not confident (Confidence: {confidence:.2f}). Try another image.")
    else:
        st.success(f"✅ Predicted: {predicted_class} (Confidence: {confidence:.2f})")

    # Show top-3 predictions
    st.subheader("🔍 Top Predictions:")
    sorted_indices = np.argsort(predictions[0])[::-1]
    for i in range(3):
        st.write(f"{class_names[sorted_indices[i]]}: {predictions[0][sorted_indices[i]]:.2f}")




