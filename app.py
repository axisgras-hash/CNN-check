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
IMG_SIZE_CNN = 150
IMG_SIZE_MN = 224
CONF_THRESHOLD = 0.65

CNN_MODEL_PATH = "flower_cnn.h5"
MN_MODEL_PATH = "flower_mobilenet.h5"
CLASSES_PATH = "classes.pkl"

# 🔗 GOOGLE DRIVE DIRECT DOWNLOAD LINKS
CNN_MODEL_URL = "https://drive.google.com/uc?id=19IKok1KShqVLFOlr9TuwTPfPaCl4vEwo"
MN_MODEL_URL  = "https://drive.google.com/uc?id=1-wc8WHs2Qu_KAB6_33645FQ88Pc5onxC"
CLASSES_URL   = "https://drive.google.com/uc?id=1LfEQLVZRDu6jAHzgYI9SULdRKMowaLvA"

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Flower Classification AI", layout="centered")

st.title("🌸 Flower Classification using Deep Learning")
st.caption("CNN vs Transfer Learning (MobileNetV2) – Seminar Demonstration")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("📌 About This App")
st.sidebar.markdown("""
### Models Used
- 🧪 **Custom CNN** (trained from scratch)
- 🚀 **MobileNetV2** (transfer learning)

### Supported Flower Classes
- Daisy  
- Dandelion  
- Roses  
- Sunflowers  
- Tulips  

⚠️ Uploading other flowers (e.g. Lily, Lotus) may give **approximate results**.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Why Two Models?
- CNN → learning fundamentals  
- MobileNet → better real-world accuracy  
- Final prediction → robustness & honesty
""")

# --------------------------------------------------
# DOWNLOAD FILES IF NOT PRESENT
# --------------------------------------------------
def download_if_needed(url, path):
    if not os.path.exists(path):
        gdown.download(url, path, quiet=False)

download_if_needed(CNN_MODEL_URL, CNN_MODEL_PATH)
download_if_needed(MN_MODEL_URL, MN_MODEL_PATH)
download_if_needed(CLASSES_URL, CLASSES_PATH)

# --------------------------------------------------
# LOAD MODELS (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_assets():
    cnn_model = load_model(CNN_MODEL_PATH)
    mn_model = load_model(MN_MODEL_PATH)
    with open(CLASSES_PATH, "rb") as f:
        classes = pickle.load(f)
    return cnn_model, mn_model, classes

cnn_model, mn_model, classes = load_assets()

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a flower image (JPG / PNG / WEBP)",
    type=["jpg", "jpeg", "png", "webp"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=320, caption="Uploaded Image")

    # ----- CNN PREDICTION -----
    img_cnn = image.resize((IMG_SIZE_CNN, IMG_SIZE_CNN))
    arr_cnn = np.expand_dims(np.array(img_cnn) / 255.0, axis=0)
    cnn_pred = cnn_model.predict(arr_cnn)[0]
    cnn_idx = np.argmax(cnn_pred)

    # ----- MobileNet PREDICTION -----
    img_mn = image.resize((IMG_SIZE_MN, IMG_SIZE_MN))
    arr_mn = np.expand_dims(np.array(img_mn) / 255.0, axis=0)
    mn_pred = mn_model.predict(arr_mn)[0]
    mn_idx = np.argmax(mn_pred)

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------
    st.subheader("🔍 Model-wise Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧪 Custom CNN")
        st.write(f"**{classes[cnn_idx]}**")
        st.write(f"Confidence: {cnn_pred[cnn_idx]*100:.2f}%")

    with col2:
        st.markdown("### 🚀 MobileNetV2")
        st.write(f"**{classes[mn_idx]}**")
        st.write(f"Confidence: {mn_pred[mn_idx]*100:.2f}%")

    # --------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------
    st.subheader("✅ Final Recommended Prediction")

    if mn_pred[mn_idx] >= CONF_THRESHOLD:
        final_class = classes[mn_idx]
        final_conf = mn_pred[mn_idx]
        source = "MobileNetV2 (Preferred)"
    else:
        final_class = classes[cnn_idx]
        final_conf = cnn_pred[cnn_idx]
        source = "Custom CNN (Fallback)"

    if final_conf < CONF_THRESHOLD:
        st.warning(
            f"Low confidence result.\n\n"
            f"Closest match: **{final_class}** ({final_conf*100:.2f}%)\n"
            "Image may not belong to trained classes."
        )
    else:
        st.success(f"🌼 **{final_class}**")
        st.info(f"Confidence: **{final_conf*100:.2f}%**")
        st.caption(f"Decision Source: {source}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
**Made by:** Ankit Mishra  
**Role:** Data Science & AI Trainer  

🔗 **LinkedIn:** https://www.linkedin.com/in/YOUR-LINKEDIN-ID  
🔗 **GitHub:** https://github.com/YOUR-GITHUB-USERNAME
""")


