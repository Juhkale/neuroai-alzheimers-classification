"""
preprocessing.py

Builds tf.data.Dataset pipelines (train/val/test) from manifest_split.csv.

KEY DESIGN DECISIONS (read before modifying):
------------------------------------------------
1. GRAYSCALE, FORCED: images are loaded with channels=1 regardless of how
   they're actually encoded in the JPEG file. MRI has no real color
   information, so this removes any redundant duplicate-channel data with
   zero information loss, and keeps the baseline CNN lean.

2. AUGMENTATION IS TRAIN-ONLY: val and test sets are NEVER augmented.
   Augmenting evaluation data would mean testing against artificially
   altered images rather than real ones, making the reported metrics
   meaningless. This is a common mistake worth explicitly avoiding.

3. AUGMENTATION IS DELIBERATELY MILD: small rotations, horizontal flip,
   slight zoom/brightness jitter only. We do NOT use aggressive
   augmentation like elastic deformation, because this task depends on
   real anatomical structure (hippocampus shape, ventricle size, cortical
   thinning) being preserved -- distorting that would undermine the
   biological validation / Grad-CAM interpretation work later.

4. RESNET-STYLE MODELS NEED 3 CHANNELS: this script keeps images as
   single-channel grayscale by default (used for the baseline CNN). A
   separate helper (`to_three_channel`) duplicates the single channel
   into 3 identical channels, for use only when feeding a pretrained
   RGB model (ResNet50, EfficientNet) in the transfer learning phase.
"""

import pandas as pd
import tensorflow as tf

IMG_SIZE = (176, 176)  # resize target -- close to source resolution (208x176), minimal distortion
BATCH_SIZE = 32
SPLIT_MANIFEST = "data/manifest_split.csv"

CLASS_NAMES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
# Augmentation layers must be created ONCE, at module level -- not inside
# augment_image() -- because tf.data traces that function as a tf.function,
# and TensorFlow requires any tf.Variable (which these layers create
# internally for their random seed state) to be created exactly once,
# not re-created on every call.
_rotation_layer = tf.keras.layers.RandomRotation(factor=0.04, seed=42)
_zoom_layer = tf.keras.layers.RandomZoom(height_factor=0.1, seed=42)

def load_and_preprocess_image(filepath: str, label: int):
    """
    Reads a JPEG from disk, decodes it as single-channel grayscale,
    resizes to IMG_SIZE, and normalizes pixel values to [0, 1].
    """
    image = tf.io.read_file(filepath)
    image = tf.io.decode_jpeg(image, channels=1)  # force grayscale, see module docstring
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def augment_image(image, label):
    """
    Mild, anatomy-preserving augmentation. Applied to TRAINING data only --
    see module docstring for why val/test must never be augmented.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Small random rotation (~10-15 degrees). tf.image doesn't have a
    # built-in rotation op, so we use a small-angle approximation via
    # tf.keras's RandomRotation layer, applied on the fly.
    image = _rotation_layer(image, training=True)
    image = _zoom_layer(image, training=True)

    # Clip back to valid [0, 1] range in case brightness/contrast pushed
    # values slightly out of bounds.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def to_three_channel(image, label):
    """
    Duplicates a single grayscale channel into 3 identical channels.
    Use this ONLY when feeding a pretrained RGB model (ResNet50,
    EfficientNet) -- the baseline CNN should stay single-channel.
    """
    image = tf.image.grayscale_to_rgb(image)
    return image, label


def build_dataset(split_name: str, augment: bool = False, three_channel: bool = False) -> tf.data.Dataset:
    """
    Builds a tf.data.Dataset for the given split ("train", "val", or "test").

    augment: should generally only be True for split_name="train".
    three_channel: set True when this dataset will feed a pretrained
        RGB model rather than the single-channel baseline CNN.
    """
    manifest = pd.read_csv(SPLIT_MANIFEST, dtype={"patient_id": str})
    split_df = manifest[manifest["split"] == split_name].copy()
    split_df["label"] = split_df["class"].map(CLASS_TO_INDEX)

    filepaths = split_df["filepath"].tolist()
    labels = split_df["label"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    if three_channel:
        ds = ds.map(to_three_channel, num_parallel_calls=tf.data.AUTOTUNE)

    if split_name == "train":
        ds = ds.shuffle(buffer_size=len(filepaths), seed=42)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    # Quick smoke test: build each split and print a batch's shape, so you
    # can confirm the pipeline works before plugging it into a real model.
    for split in ["train", "val", "test"]:
        ds = build_dataset(split, augment=(split == "train"))
        for images, labels in ds.take(1):
            print(f"{split}: batch image shape = {images.shape}, "
                  f"batch label shape = {labels.shape}")
        