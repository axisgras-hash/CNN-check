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
CONFIDENCE_THRESHOLD = 0.80  # below this → warn user

MODEL_PATH = "flower_cnn.h5"
CLASSES_PATH = "classes.pkl"

MODEL_URL = "https://drive.google.com/uc?id=1SMVQryvXOTZ_X1Loq3AyFnpp3nx7reOj"
CLASSES_URL = "PASTE_YOUR_CLASSES_PKL_DRIVE_LINK_HERE"

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
st.set_page_config(page_title="Flower Classification", layout="centered")

st.title("🌸 Flower Classification using CNN")
st.caption("Upload a flower image to predict its class")

# --------------------------------------------------
# SIDEBAR (PROFESSIONAL UX)
# --------------------------------------------------
st.sidebar.title("ℹ️ About This App")

st.sidebar.markdown(
    """
**Supported Flower Classes:**
- 🌼 Daisy  
- 🌾 Dandelion  
- 🌹 Roses  
- 🌻 Sunflowers  
- 🌷 Tulips  

⚠️ **Important Note**  
This model can predict **only the above flowers**.  
Uploading other flowers (e.g. *lily, lotus*) may give **incorrect predictions**.
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works:**\n"
    "- CNN trained on flower images\n"
    "- Predicts closest known class\n"
    "- Confidence shown for transparency"
)

# --------------------------------------------------
# DOWNLOAD MODEL & CLASSES
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
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Drag & drop or browse a flower image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Display image (controlled size)
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    predicted_class = classes[class_index]

    # --------------------------------------------------
    # SMART OUTPUT LOGIC
    # --------------------------------------------------
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            f"⚠️ Low confidence prediction.\n\n"
            f"Closest match: **{predicted_class}** ({confidence*100:.2f}%)\n\n"
            "This image may **not belong to the trained flower classes**."
        )
    else:
        st.success(f"🌼 Predicted Flower: **{predicted_class}**")
        st.info(f"Confidence: **{confidence * 100:.2f}%**")

