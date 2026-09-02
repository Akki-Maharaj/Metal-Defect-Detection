import os

import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.config import MODELS_DIR, CLASS_DISPLAY_NAMES
from src.predict import predict_image, load_artifacts
from src.defect_localization import draw_boxes

st.set_page_config(page_title="Metal Surface Defect Detector", layout="centered")

st.title("🔩 Metal Surface Defect Detector")
st.caption(
    "Classical computer vision pipeline: OpenCV preprocessing + hand-crafted "
    "features (HOG / LBP / GLCM / edges) + SVM classifier, with an unsupervised "
    "OpenCV localization overlay (Otsu threshold + morphology + contours)."
)

model_path = os.path.join(MODELS_DIR, "best_model.joblib")
if not os.path.exists(model_path):
    st.error(
        "No trained model found. Run `python -m src.train --data data/sample` "
        "(or `data/full` after downloading the dataset) first."
    )
    st.stop()

model, scaler, le = load_artifacts()

uploaded = st.file_uploader("Upload a grayscale metal surface image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    result = predict_image(img, model=model, scaler=scaler, le=le)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded image", use_container_width=True, clamp=True)
    with col2:
        vis = draw_boxes(result["preprocessed_image"], result["boxes"])
        st.image(vis, caption="Detected defect region(s)", use_container_width=True, channels="BGR")

    st.subheader(f"Prediction: {result['predicted_display_name']}")

    if result["class_probabilities"]:
        labels = [CLASS_DISPLAY_NAMES.get(c, c) for c in result["class_probabilities"]]
        values = list(result["class_probabilities"].values())
        order = np.argsort(values)[::-1]

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh([labels[i] for i in order], [values[i] for i in order], color="#3568b0")
        ax.set_xlabel("Probability")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        st.pyplot(fig)

    st.caption(f"{len(result['boxes'])} candidate defect region(s) found by the localization step.")
else:
    st.info("Upload an image, or try one from `data/sample/<class>/*.jpg`.")
