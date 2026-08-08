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

import pandas as pd

import matplotlib.pyplot as plt

from src.models import build_baseline_cnn, build_resnet_transfer_model

import os

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

    max_weight = 2.0
    class_weights_dict = {k: min(v, max_weight) for k, v in class_weights_dict.items()}

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
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,        # halve the learning rate when triggered
        patience=3,        # wait 3 epochs of no improvement before reducing
        min_lr=1e-6         # don't shrink the learning rate below this floor
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint, reduce_lr],
    )

    return model, history


def plot_training_history(history, save_path="reports/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()


def train_resnet():
    """
    Trains the ResNet50 transfer learning model. Uses three_channel=True
    since ResNet50 requires RGB input -- see preprocessing.py's
    to_three_channel() for how that conversion happens.
    """
    train_ds = build_dataset("train", augment=True, three_channel=True)
    val_ds = build_dataset("val", augment=False, three_channel=True)

    model = build_resnet_transfer_model()

    class_weights = compute_class_weights()
    print(f"Computed class weights: {class_weights}")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/resnet_transfer_best.keras",
        monitor="val_loss", save_best_only=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint, reduce_lr],
    )

    return model, history


if __name__ == "__main__":
    model, history = train()
    plot_training_history(history)