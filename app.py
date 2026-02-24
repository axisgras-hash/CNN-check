import streamlit as st
import numpy as np
import pickle
import os
import gdown
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------------------------------
# CONFIG
# ---------------------------------------
IMG_SIZE = 150
MODEL_PATH = "flower_cnn.h5"
CLASSES_PATH = "classes.pkl"

MODEL_URL = "https://drive.google.com/uc?id=1SMVQryvXOTZ_X1Loq3AyFnpp3nx7reOj"

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(page_title="Flower Classification", layout="centered")
st.title("🌸 Flower Classification using CNN")
st.write("Upload a flower image to predict its class")

# ---------------------------------------
# DOWNLOAD MODEL IF NOT EXISTS
# ---------------------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... please wait ⏳"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully ✅")

# ---------------------------------------
# LOAD MODEL & CLASSES (CACHED)
# ---------------------------------------
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    with open(CLASSES_PATH, "rb") as f:
        classes = pickle.load(f)
    return model, classes

model, classes = load_assets()

# ---------------------------------------
# FILE UPLOAD
# ---------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Flower Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------
# PREDICTION
# ---------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"🌼 Predicted Flower: **{classes[class_index]}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")
