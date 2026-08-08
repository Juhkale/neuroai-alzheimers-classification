"""
explainability.py

Implements Grad-CAM for the ResNet50 transfer learning model, and
generates a comparison grid of correct + misclassified predictions with
their heatmaps overlaid -- the core artifact for this project's
biological validation section.

WHY GRAD-CAM, AND WHY conv5_block3_out:
------------------------------------------
Grad-CAM answers "which pixels most influenced this prediction?" by
looking at gradients flowing back into the LAST convolutional layer --
this is the last point in the network that still has spatial structure
(i.e. still "knows" where things are in the image), before the fully
connected classification head discards that spatial information.

For our ResNet50 model, that layer is "conv5_block3_out" -- the final
output of ResNet50's last residual block, immediately before our custom
classification head (GlobalAveragePooling -> Dense -> Dropout -> Dense).

WHY WE MANUALLY RUN THE CLASSIFIER LAYERS:
----------------------------------------------
A loaded Keras Sequential model's `.output` attribute isn't reliably
available until the model has been called at least once in the current
session, which breaks the standard Grad-CAM pattern of building one
model with two outputs. Instead, we build a small functional model for
just the base ResNet (input -> last conv layer), then manually replay
the remaining classifier layers (model.layers[1:]) inside the same
GradientTape, so gradients still flow correctly end-to-end.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from src.preprocessing import CLASS_NAMES

REPORTS_DIR = "reports"
LAST_CONV_LAYER_NAME = "conv5_block3_out"


def load_and_preprocess_for_resnet(filepath: str):
    """
    Loads a single image and preprocesses it exactly as the ResNet model
    expects (grayscale -> 3-channel -> ImageNet mean-centering). Also
    returns a clean 0-1 RGB version of the image for display purposes.
    """
    img = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, (176, 176))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.grayscale_to_rgb(img)

    img_resnet = img * 255.0
    img_resnet = tf.keras.applications.resnet50.preprocess_input(img_resnet)
    img_array = tf.expand_dims(img_resnet, axis=0)

    display_img = tf.squeeze(img).numpy()
    return img_array, display_img


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER_NAME, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a single preprocessed image.

    img_array: shape (1, H, W, 3), already ResNet-preprocessed.
    pred_index: which class to explain. If None, explains the model's
        own top prediction.

    Returns: (heatmap as a 2D numpy array normalized to [0, 1], predicted class index)
    """
    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    conv_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    classifier_layers = model.layers[1:]

    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)

        x = conv_output
        for layer in classifier_layers:
            x = layer(x)
        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), int(pred_index)


def collect_example_predictions(model, test_manifest: pd.DataFrame):
    """
    Finds one correctly-predicted test image per class, plus one
    misclassified example, running Grad-CAM on each. Used to build the
    biological validation comparison grid.
    """
    class_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    examples = []
    misclassified_example = None

    for class_name in CLASS_NAMES:
        class_rows = test_manifest[test_manifest["class"] == class_name]
        for _, row in class_rows.iterrows():
            img_array, display_img = load_and_preprocess_for_resnet(row["filepath"])
            heatmap, pred_idx = make_gradcam_heatmap(img_array, model)
            true_idx = class_to_index[class_name]

            if pred_idx == true_idx:
                examples.append({
                    "filepath": row["filepath"], "true_class": class_name,
                    "pred_class": CLASS_NAMES[pred_idx], "heatmap": heatmap,
                    "display_img": display_img, "correct": True
                })
                break

            elif misclassified_example is None:
                misclassified_example = {
                    "filepath": row["filepath"], "true_class": class_name,
                    "pred_class": CLASS_NAMES[pred_idx], "heatmap": heatmap,
                    "display_img": display_img, "correct": False
                }

    if misclassified_example:
        examples.append(misclassified_example)

    return examples


def plot_examples_grid(examples: list, save_path: str = f"{REPORTS_DIR}/gradcam_examples_grid.png"):
    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))

    for i, ex in enumerate(examples):
        heatmap_resized = tf.image.resize(ex["heatmap"][..., tf.newaxis], (176, 176))
        heatmap_resized = tf.squeeze(heatmap_resized).numpy()

        axes[i, 0].imshow(ex["display_img"])
        axes[i, 0].set_title(f"True: {ex['true_class']}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(ex["display_img"])
        axes[i, 1].imshow(heatmap_resized, cmap="jet", alpha=0.4)
        status = "Correct" if ex["correct"] else "Misclassified"
        axes[i, 1].set_title(f"Predicted: {ex['pred_class']} ({status})")
        axes[i, 1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Grad-CAM examples grid saved to: {save_path}")


if __name__ == "__main__":
    resnet_model = tf.keras.models.load_model("models/resnet_transfer_best.keras")
    manifest = pd.read_csv("data/manifest_split.csv", dtype={"patient_id": str})
    test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)

    examples = collect_example_predictions(resnet_model, test_manifest)
    print(f"Collected {len(examples)} examples")
    for ex in examples:
        print(f"  True: {ex['true_class']}, Predicted: {ex['pred_class']}, Correct: {ex['correct']}")

    plot_examples_grid(examples)