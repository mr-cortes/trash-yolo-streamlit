import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------
# LOAD MODEL (cache so it loads once only)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # adjust if filename differs

model = load_model()

# ---------------------------------------------------------
# PAGE CONFIG & UI HEADER
# ---------------------------------------------------------
st.set_page_config(page_title="Scrap & Metal Detector", layout="wide")
st.title("🔩 Scrap & Metal Detector")
st.write("Upload an image or use camera to detect scrap and metal objects.")

# Sidebar settings
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# ---------------------------------------------------------
# INPUT HANDLING (Upload or Camera)
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("📸 Take a picture")

# Prioritize uploaded file, fallback to camera
image_source = uploaded_file if uploaded_file else camera_file

# ---------------------------------------------------------
# RUN YOLO & DISPLAY RESULTS
# ---------------------------------------------------------
if image_source:
    # Convert to PIL Image
    image = Image.open(image_source)
    st.image(image, caption="Input Image", use_container_width=True)

    if st.button("🔍 Detect Objects"):
        with st.spinner("Analyzing scrap..."):
            results = model.predict(image, conf=conf_threshold)

            # Convert annotated BGR numpy array → RGB PIL for Streamlit
            annotated_np = results[0].plot()
            annotated_img = Image.fromarray(annotated_np[:, :, ::-1])

            st.image(annotated_img, caption="Detected Objects", use_container_width=True)

            # Count detected objects
            boxes = results[0].boxes
            st.success(f"Detected {len(boxes)} objects!")
