import streamlit as st
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from PIL import Image

IMG_SIZE = 150

st.set_page_config(page_title="Flower Classification", layout="centered")
st.title("🌸 Flower Classification using CNN")
st.write("Upload a flower image to predict its class")

# Load model
@st.cache_resource
def load_assets():
    model = load_model("flower_cnn.h5")
    with open("classes.pkl","rb") as f:
        classes = pickle.load(f)
    return model, classes

model, classes = load_assets()

uploaded_file = st.file_uploader(
    "Upload Flower Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"🌼 Predicted Flower: **{classes[class_index]}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")