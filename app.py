import os
os.environ["YOLO_VERBOSE"] = "False"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------
# 1️⃣ Load YOLO Model (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Make sure filename is correct

model = load_model()


# ---------------------------------------------------------
# 2️⃣ UI / Dashboard Config
# ---------------------------------------------------------
st.set_page_config(page_title="Scrap & Metal Detector", layout="wide")
st.title("🔩 Scrap & Metal Detector")
st.write("Upload or capture image. System will detect scrap/metal classifications.")


# Sidebar: Confidence slider
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)


# ---------------------------------------------------------
# 3️⃣ Input Methods: Upload or Camera
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("📸 Take Photo Now")

image_source = uploaded_file if uploaded_file else camera_file


# ---------------------------------------------------------
# 4️⃣ Detection + Output
# ---------------------------------------------------------
if image_source:
    # Convert buffer → PIL
    image = Image.open(image_source).convert("RGB")
    st.image(image, caption="📌 Input Image", use_container_width=True)

    # Detect button
    if st.button("🚀 RUN DETECTION"):
        with st.spinner("Detecting objects... Please wait..."):
            results = model.predict(image, conf=conf_threshold)

            # Annotated image conversion (avoid cv2 GUI dependency)
            annotated_np = results[0].plot()  # numpy BGR array
            annotated_img = Image.fromarray(annotated_np[:, :, ::-1])  # Convert BGR → RGB

            st.image(annotated_img, caption="🔍 Detection Results", use_container_width=True)

            # Counts
            count_objects = len(results[0].boxes)
            st.success(f"Detected {count_objects} object(s) in the image")

            # Optional: Display detection classes
            detected_classes = [model.names[int(c)] for c in results[0].boxes.cls]
            st.write("📌 Detected Classes:", detected_classes)
else:
    st.info("👆 Upload or capture an image to begin detection.")
