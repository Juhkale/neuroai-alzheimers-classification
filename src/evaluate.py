"""
evaluate.py

Loads a trained model checkpoint and evaluates it on the TEST split --
data neither model has seen during training or validation. This is the
honest, final measure of performance.

IMPORTANT CAVEAT -- MODERATE DEMENTIA:
----------------------------------------
Per the project's documented hybrid approach, Moderate Dementia's test
set comes from a SINGLE patient (see dataset.py). Its precision/recall/F1
numbers below are reported for completeness, but should NOT be read with
the same statistical confidence as the other three classes -- a single
patient's scans cannot represent the class's true variability. This
script prints an explicit warning alongside that class's metrics for
this reason.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import build_dataset, CLASS_NAMES, SPLIT_MANIFEST

REPORTS_DIR = "reports"


def get_predictions(model, test_ds):
    """
    Runs the model on the full test set and returns (true_labels, predicted_labels).
    """
    predictions = model.predict(test_ds)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.concatenate([y.numpy() for x, y in test_ds])
    return true_labels, predicted_labels


def print_moderate_dementia_caveat():
    manifest = pd.read_csv(SPLIT_MANIFEST, dtype={"patient_id": str})
    moderate_test = manifest[
        (manifest["class"] == "Moderate Dementia") & (manifest["split"] == "test")
    ]
    n_patients = moderate_test["patient_id"].nunique()
    n_images = len(moderate_test)
    print(
        f"\n*** CAVEAT: Moderate Dementia's test metrics come from only "
        f"{n_patients} patient(s) ({n_images} images). This is far too few "
        f"to draw statistically reliable conclusions about the model's "
        f"true performance on this class -- treat this number as a "
        f"qualitative observation, not a validated metric. ***\n"
    )


def evaluate_model(model, test_ds, model_name: str):
    """
    Runs full evaluation: classification report + confusion matrix,
    printed and saved to reports/.
    """
    true_labels, predicted_labels = get_predictions(model, test_ds)

    print(f"\n{'=' * 60}")
    print(f"EVALUATION: {model_name}")
    print(f"{'=' * 60}")

    report = classification_report(
        true_labels, predicted_labels, target_names=CLASS_NAMES, digits=3
    )
    print(report)
    print_moderate_dementia_caveat()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = f"{REPORTS_DIR}/{model_name}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Evaluation: {model_name}\n\n")
        f.write(report)
    print(f"Classification report saved to: {report_path}")

    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    cm_path = f"{REPORTS_DIR}/{model_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print(f"Confusion matrix saved to: {cm_path}")

    return report, cm


if __name__ == "__main__":
    baseline_model = tf.keras.models.load_model("models/baseline_cnn_best.keras")
    baseline_test_ds = build_dataset("test", augment=False, three_channel=False)
    evaluate_model(baseline_model, baseline_test_ds, "baseline_cnn")

    resnet_model = tf.keras.models.load_model("models/resnet_transfer_best.keras")
    resnet_test_ds = build_dataset("test", augment=False, three_channel=True)
    evaluate_model(resnet_model, resnet_test_ds, "resnet_transfer")