"""
app.py

Streamlit demo app for the NeuroAI Alzheimer's classification project.

WORKFLOW:
Upload MRI slice -> Prediction -> Confidence scores -> Grad-CAM overlay

DESIGN NOTE -- WHY NO LLM LAYER YET:
This first version is deliberately "deterministic only" -- confidence
scores and Grad-CAM come straight from the model's actual math, nothing
generated. This is the more important half of the trustworthy-AI story
this project is built around: it's verifiable, has no external API
dependency, and won't break during a live demo. A constrained LLM layer
(plain-language translation of these grounded outputs) is planned as a
follow-up addition, not a replacement for this core.

IMPORTANT: This is a research/educational prototype, NOT a diagnostic
tool. See the in-app disclaimer.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib as mpl

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.explainability import make_gradcam_heatmap, LAST_CONV_LAYER_NAME

CLASS_NAMES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
MODEL_PATH = "models/resnet_transfer_best.keras"


@st.cache_resource
def load_model():
    """
    Loads the trained model once and caches it across reruns -- Streamlit
    reruns the whole script on every interaction, so without caching this
    would reload the model (slow) on every single click.
    """
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(pil_image: Image.Image):
    """
    Takes a PIL image (from the file uploader) and preprocesses it exactly
    as the ResNet model expects: grayscale -> resize -> 3-channel ->
    ImageNet mean-centering. Mirrors src/explainability.py's preprocessing
    so predictions here match what evaluate.py measured.
    """
    img = pil_image.convert("L")  # force grayscale, regardless of upload format
    img = img.resize((176, 176))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.stack([img_array] * 3, axis=-1)  # 1 channel -> 3 channels

    display_img = img_array.copy()  # keep a clean 0-1 RGB copy for display

    img_resnet = img_array * 255.0
    img_resnet = tf.keras.applications.resnet50.preprocess_input(img_resnet)
    img_batch = np.expand_dims(img_resnet, axis=0)

    return img_batch, display_img


def overlay_heatmap(display_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    """
    Resizes the Grad-CAM heatmap to match the display image and blends
    them together using a "jet" colormap, matching explainability.py's
    visualization style.
    """
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (176, 176))
    heatmap_resized = tf.squeeze(heatmap_resized).numpy()

    jet = mpl.colormaps["jet"]
    heatmap_colored = jet(heatmap_resized)[..., :3]  # drop alpha channel from colormap

    overlaid = display_img * (1 - alpha) + heatmap_colored * alpha
    return np.clip(overlaid, 0, 1)


def main():
    st.set_page_config(page_title="NeuroAI: Alzheimer's MRI Classifier", layout="wide")

    st.title("Trustworthy NeuroAI: Alzheimer's MRI Classification")
    st.caption(
        "An explainable AI system for classifying Alzheimer's disease severity from brain MRI, "
        "with Grad-CAM biological validation of model attention."
    )

    st.warning(
        "**Research and educational prototype only.** This tool is NOT a diagnostic device and "
        "has not been clinically validated. Predictions should never be used for actual patient "
        "care decisions. See this project's README for known limitations, including a "
        "statistically limited Moderate Dementia test set (2 patients total in the source data)."
    )

    uploaded_file = st.file_uploader(
        "Upload a brain MRI slice (JPEG/PNG, axial view)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        img_batch, display_img = preprocess_image(pil_image)

        model = load_model()

        predictions = model.predict(img_batch)[0]
        predicted_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        heatmap, _ = make_gradcam_heatmap(img_batch, model, pred_index=predicted_idx)
        overlaid = overlay_heatmap(display_img, heatmap)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Uploaded Scan")
            st.image(display_img, use_container_width=True)
        with col2:
            st.subheader("Grad-CAM Attention")
            st.image(overlaid, use_container_width=True)
        with col3:
            st.subheader("Prediction")
            st.metric("Predicted Class", predicted_class)
            st.metric("Confidence", f"{confidence:.1%}")

            if predicted_class == "Moderate Dementia":
                st.caption(
                    "Note: this class's evaluation is based on a single-patient test set -- "
                    "treat this prediction with extra caution."
                )

        st.subheader("Full Confidence Breakdown")
        confidence_dict = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(confidence_dict)

        st.info(
            "**How to read the Grad-CAM overlay:** warmer colors (red/orange) indicate regions "
            "that most influenced this prediction. Compare against known Alzheimer's biomarkers "
            "(hippocampus, temporal lobe, ventricle size, cortical thinning) to judge whether the "
            "model's attention is anatomically plausible -- this project's own evaluation found "
            "*mixed* alignment, not consistent alignment. See the README's Explainability section "
            "for a full discussion."
        )

    else:
        st.info("Upload an MRI slice above to get started.")


if __name__ == "__main__":
    main()