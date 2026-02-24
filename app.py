import streamlit as st
import numpy as np
import pickle
import os
import gdown
from tensorflow.keras.models import load_model
from PIL import Image

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
IMG_SIZE = 150

MODEL_PATH = "flower_cnn.h5"
CLASSES_PATH = "classes.pkl"

MODEL_URL = "https://drive.google.com/uc?id=1SMVQryvXOTZ_X1Loq3AyFnpp3nx7reOj"
CLASSES_URL = "PASTE_YOUR_CLASSES_PKL_DRIVE_LINK_HERE"

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
st.set_page_config(page_title="Flower Classification", layout="centered")

st.title("🌸 Flower Classification using CNN")

st.info(
    "👉 You can **drag & drop** an image or click **Browse files**.\n\n"
    "📱 On some devices, a file picker popup may open automatically (browser behavior)."
)

# --------------------------------------------------
# DOWNLOAD MODEL & CLASSES IF NOT PRESENT
# --------------------------------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... please wait ⏳"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(CLASSES_PATH):
    with st.spinner("Downloading class labels... please wait ⏳"):
        gdown.download(CLASSES_URL, CLASSES_PATH, quiet=False)

# --------------------------------------------------
# LOAD MODEL & CLASSES (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    with open(CLASSES_PATH, "rb") as f:
        classes = pickle.load(f)
    return model, classes

model, classes = load_assets()

# --------------------------------------------------
# INPUT OPTIONS
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Flower Image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

camera_image = st.camera_input("📸 Or take a photo using camera")

# Decide which image to use
image_file = uploaded_file if uploaded_file is not None else camera_image

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Selected Image", use_column_width=True)

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"🌼 Predicted Flower: **{classes[class_index]}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")
