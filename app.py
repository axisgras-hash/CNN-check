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

st.caption(
    "📤 Drag & drop an image or use camera. "
    "On some devices, file picker popup is browser-controlled."
)

# --------------------------------------------------
# DOWNLOAD FILES IF NOT EXISTS
# --------------------------------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(CLASSES_PATH):
    with st.spinner("Downloading class labels..."):
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
# INPUT SECTION (SMALL LAYOUT)
# --------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📁 Upload Image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

with col2:
    camera_image = st.camera_input("📷 Camera (small view)")

# Decide input
image_file = uploaded_file if uploaded_file is not None else camera_image

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if image_file is not None:
    image = Image.open(image_file).convert("RGB")

    # 🔽 Reduce displayed image size
    st.image(
        image,
        caption="Selected Image",
        width=300   # <<< THIS CONTROLS DISPLAY SIZE
    )

    # Model preprocessing
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"🌼 Predicted Flower: **{classes[class_index]}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")
