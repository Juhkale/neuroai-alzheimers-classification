"""
train.py

Trains the baseline CNN on the OASIS Alzheimer's dataset using the data
pipeline from preprocessing.py.

KEY DESIGN DECISIONS:
------------------------
1. CLASS WEIGHTS: computed from the actual training set distribution,
   so the model is penalized more for mistakes on rare classes (like
   Moderate Dementia) rather than free to ignore them.
2. EARLY STOPPING: monitors validation loss, stops automatically if it
   stops improving, and restores the best-performing weights rather than
   whatever the final epoch happened to produce.
3. CHECKPOINTING: saves the best model to disk during training, so a
   Colab disconnect doesn't lose progress.
"""

import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.preprocessing import build_dataset, CLASS_NAMES, SPLIT_MANIFEST
from src.models import build_baseline_cnn

import pandas as pd

EPOCHS = 20
CHECKPOINT_PATH = "models/baseline_cnn_best.keras"


def compute_class_weights():
    """
    Reads the training split's labels and computes class weights using
    sklearn, so the model penalizes mistakes on rare classes (like
    Moderate Dementia) more heavily than common ones.
    """
    manifest = pd.read_csv(SPLIT_MANIFEST, dtype={"patient_id": str})
    train_df = manifest[manifest["split"] == "train"]

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=train_df["class"].map({name: i for i, name in enumerate(CLASS_NAMES)}).values
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    return class_weights_dict
    


def train():
    train_ds = build_dataset("train", augment=True)
    val_ds = build_dataset("val", augment=False)

    model = build_baseline_cnn()

    class_weights = compute_class_weights()
    print(f"Computed class weights: {class_weights}")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        monitor="val_loss",
        save_best_only=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint],
    )

    return model, history


if __name__ == "__main__":
    model, history = train()