"""
app.py

Streamlit demo app for the NeuroAI Alzheimer's classification project.

WORKFLOW:
Upload MRI slice -> Prediction -> Confidence scores -> Grad-CAM overlay
-> (optional) plain-language explanation

TWO-LAYER EXPLANATION ARCHITECTURE:
------------------------------------
Layer 1 (always shown, deterministic): confidence score + Grad-CAM
heatmap, computed directly from the model's own math. This is the more
important layer -- it's verifiable, has no external dependency, and is
the actual evidence behind the prediction.

Layer 2 (optional, on-demand): a constrained LLM call that TRANSLATES
the Layer 1 facts into plain language for a non-technical reader. The
LLM is explicitly restricted from adding new medical claims or reasoning
independently about the diagnosis -- it only restates grounded facts
already computed. This distinction (translation vs. independent
judgment) is the core trustworthy-AI pattern this project is built
around. Layer 2 is triggered by a button, not automatic, since each
call costs money and isn't needed for the app's core value.

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
import gdown
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.explainability import make_gradcam_heatmap

CLASS_NAMES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
MODEL_PATH = "models/resnet_transfer_best.keras"
GDRIVE_MODEL_FILE_ID = "1Jt_lXhbIl8gOZOFGD3ojEtB0ByHVtpTO"


@st.cache_resource
def load_model():
    """
    Loads the trained model once and caches it across reruns -- Streamlit
    reruns the whole script on every interaction, so without caching this
    would reload the model (slow) on every single click.

    Since the trained model file (.keras) is too large for GitHub and is
    excluded via .gitignore, it's hosted on Google Drive instead. If it's
    not already present locally (first run, or a fresh deployment), it's
    downloaded automatically before loading.
    """
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with st.spinner("Downloading trained model (first run only)..."):
            gdown.download(id=GDRIVE_MODEL_FILE_ID, output=MODEL_PATH, quiet=False)

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


def describe_heatmap_location(heatmap: np.ndarray) -> str:
    """
    Finds the peak-intensity location in the heatmap and describes it in
    plain, rough terms (e.g. "upper right", "lower center"). This is
    computed directly from the heatmap array -- deterministic, not
    LLM-generated -- so it's grounded evidence, same as the confidence
    score, rather than something the LLM has to guess or invent.
    """
    max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    row, col = max_pos
    n_rows, n_cols = heatmap.shape

    vertical = "upper" if row < n_rows / 2 else "lower"
    horizontal = "left" if col < n_cols / 2 else "right"

    return f"{vertical} {horizontal}"


def get_openai_api_key():
    """
    Checks Streamlit's secrets manager first (used once deployed),
    falling back to a local environment variable (used for local testing).
    """
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.environ.get("OPENAI_API_KEY")


def generate_llm_explanation(predicted_class: str, confidence: float, location_desc: str) -> str:
    """
    Generates a plain-language explanation using ONLY the grounded facts
    already computed (predicted_class, confidence, location_desc) -- the
    model is explicitly constrained to restate and contextualize these
    facts, not to reason independently about the diagnosis or invent
    additional clinical claims. This is the "constrained generation"
    layer described in the project README: deterministic evidence first,
    language model second, never the other way around.
    """
    client = OpenAI(api_key=get_openai_api_key())

    system_prompt = (
        "You are a plain-language assistant for a research prototype MRI classification tool. "
        "You will be given a predicted class, a confidence score, and a rough description of where "
        "the model's attention was concentrated on the scan. Your ONLY job is to restate these exact "
        "facts in one or two clear, accessible sentences for a non-technical reader. "
        "Do NOT add any new medical claims, diagnoses, or reasoning beyond what is given. "
        "Do NOT speculate about the patient's actual condition. "
        "Always include a brief reminder that this is a research prototype, not a diagnosis."
    )

    user_prompt = (
        f"Predicted class: {predicted_class}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Model attention concentrated in the: {location_desc} region of the scan\n\n"
        "Write a short, plain-language summary of these exact facts."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=150,
        temperature=0.3,  # low temperature -- consistent restatement, not creative variation
    )

    return response.choices[0].message.content


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

        st.divider()
        st.subheader("Plain-Language Explanation (Optional)")
        st.caption(
            "Generates a short, plain-language summary using only the grounded facts above -- "
            "the AI model is explicitly restricted from adding new medical claims or reasoning "
            "beyond what's already been computed."
        )
        if st.button("Generate plain-language explanation"):
            with st.spinner("Generating explanation..."):
                location_desc = describe_heatmap_location(heatmap)
                explanation = generate_llm_explanation(predicted_class, confidence, location_desc)
            st.success(explanation)

    else:
        st.info("Upload an MRI slice above to get started.")


if __name__ == "__main__":
    main()