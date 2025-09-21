import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# 📌 Load trained model
model = tf.keras.models.load_model("waste_model.keras")

# 📌 Class names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("♻️ Waste Classification App")
st.write("Upload an image of waste and the model will classify it!")

# 📌 Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128,128))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # ⚠️ no /255 if dataset not normalized
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]

    st.success(f"✅ Prediction: {pred_class}")

