import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/best_model.h5")
    return model

model = load_model()

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("Medical Image Abnormality Classifier")
st.write("Upload a X-ray image to predict if it's **Normal** or **Abnormal**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]
        label = "Abnormal" if prediction > 0.5 else "Normal"
        confidence = prediction if prediction > 0.5 else 1 - prediction

    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: {confidence:.2f}")

