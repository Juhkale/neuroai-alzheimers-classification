"""
prepare_dataset.py

Downloads the OASIS Alzheimer's Detection dataset (ninadaithal/imagesoasis)
and builds a patient-level, stratified, leakage-safe subsample for training.

WHY THIS SCRIPT EXISTS (read this before running):
----------------------------------------------------
The full dataset has ~86,000 images from 461 patients. That's too much to
train quickly on a free Colab GPU, and we don't need it all. But we can't
just randomly grab N images per class -- each patient contributes MANY
slices (a range of z-axis indices), so a naive random sample could still
leak the same patient into both our "sampled" train set and a held-out
test set later. This script solves that at the source: we select whole
PATIENTS, not individual images, so every split downstream (train/val/test)
is guaranteed to be leakage-free.

Run this once, at the start of the project, ideally in Colab where you have
disk space. It writes out a single manifest CSV that the rest of your
pipeline (preprocessing.py, dataset.py) will read from -- you should not
need to re-run this unless you change the sampling targets below.
"""

import os
import re
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

# -----------------------------------------------------------------------
# STEP 0: Config -- change these if you want a bigger/smaller subsample
# -----------------------------------------------------------------------
RANDOM_SEED = 42  # fixed seed = reproducible sample every time you re-run this

# Target number of IMAGES per class in the final subsample.
# Moderate Dementia has so few images to begin with (488) that we keep
# all of them -- see the README note on why we don't subsample this class.
TARGET_PER_CLASS = {
    "Non Demented": 1500,
    "Very mild Dementia": 1500,
    "Mild Dementia": 1500,
    "Moderate Dementia": None,  # None = keep every image in this class
}

OUTPUT_MANIFEST = "data/manifest.csv"  # where the final file list gets written


# -----------------------------------------------------------------------
# STEP 1: Download the dataset via kagglehub
# -----------------------------------------------------------------------
def download_dataset() -> str:
    """
    Downloads the dataset and returns the local root path.
    Requires a Kaggle account + API token configured in your environment
    (kagglehub will prompt you the first time if it's not set up).
    """
    import kagglehub
    path = kagglehub.dataset_download("ninadaithal/imagesoasis")
    print(f"Dataset downloaded to: {path}")
    return path


# -----------------------------------------------------------------------
# STEP 2: Parse patient ID out of each filename
# -----------------------------------------------------------------------
# Filenames look like: OAS1_0028_MR1_mpr-1_100.jpg
#                        ^^^^ ^^^^
#                      cohort patient_id
PATIENT_ID_PATTERN = re.compile(r"OAS1_(\d+)_MR\d+")


def extract_patient_id(filename: str) -> str:
    match = PATIENT_ID_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Could not parse patient ID from filename: {filename}")
    return match.group(1)


# -----------------------------------------------------------------------
# STEP 3: Build a full index of every image: filepath, class, patient_id
# -----------------------------------------------------------------------
def build_full_index(dataset_root: str) -> pd.DataFrame:
    """
    Walks the dataset's class folders and builds one row per image.
    This is the "ground truth" index -- we sample FROM this, we never
    modify the original downloaded files.
    """
    rows = []
    dataset_root = Path(dataset_root)

    # The class folder names as they appear on Kaggle's Data tab
    class_folders = [
        "Non Demented",
        "Very mild Dementia",
        "Mild Dementia",
        "Moderate Dementia",
    ]

    for class_name in class_folders:
        class_dir = dataset_root / "Data" / class_name
        if not class_dir.exists():
            # Fallback in case kagglehub nests the folder differently --
            # print what we found so you can adjust the path if needed.
            print(f"WARNING: expected folder not found: {class_dir}")
            continue

        for img_path in class_dir.glob("*.jpg"):
            patient_id = extract_patient_id(img_path.name)
            rows.append({
                "filepath": str(img_path),
                "filename": img_path.name,
                "class": class_name,
                "patient_id": patient_id,
            })

    df = pd.DataFrame(rows)
    print(f"Indexed {len(df)} total images across {df['patient_id'].nunique()} patients.")
    return df


# -----------------------------------------------------------------------
# STEP 4: Patient-level stratified subsampling
# -----------------------------------------------------------------------
# Max images to take from any single patient, for the majority classes.
# This forces the sample to draw from many patients instead of a few
# patients with lots of slices each -- patient diversity matters more
# than raw image count for generalization.
MAX_IMAGES_PER_PATIENT = 40


def subsample_by_patient(full_index: pd.DataFrame) -> pd.DataFrame:
    """
    For each class, shuffles patients and adds a CAPPED number of images
    per patient until the target image count is reached. This maximizes
    the number of unique patients represented in the sample, which matters
    more for generalization than raw image count does.

    Moderate Dementia is a special case: only 2 patients exist in the
    entire source dataset for this class, so capping wouldn't help --
    we keep every image and document this as a known limitation instead.
    """
    random.seed(RANDOM_SEED)
    sampled_frames = []

    for class_name, target in TARGET_PER_CLASS.items():
        class_df = full_index[full_index["class"] == class_name]

        if target is None:
            sampled_frames.append(class_df)
            print(f"{class_name}: kept all {len(class_df)} images "
                  f"({class_df['patient_id'].nunique()} patients) "
                  f"-- limited patient pool, see README limitations")
            continue

        patient_ids = class_df["patient_id"].unique().tolist()
        random.shuffle(patient_ids)

        selected_rows = []
        running_total = 0
        patients_used = 0

        for pid in patient_ids:
            if running_total >= target:
                break
            patient_rows = class_df[class_df["patient_id"] == pid]
            # Cap this patient's contribution -- take a random subset if
            # they have more than MAX_IMAGES_PER_PATIENT slices available.
            if len(patient_rows) > MAX_IMAGES_PER_PATIENT:
                patient_rows = patient_rows.sample(
                    n=MAX_IMAGES_PER_PATIENT, random_state=RANDOM_SEED
                )
            selected_rows.append(patient_rows)
            running_total += len(patient_rows)
            patients_used += 1

        class_sample = pd.concat(selected_rows, ignore_index=True)
        sampled_frames.append(class_sample)
        print(f"{class_name}: sampled {len(class_sample)} images "
              f"(target was {target}) from {patients_used} patients "
              f"(cap: {MAX_IMAGES_PER_PATIENT}/patient)")

    result = pd.concat(sampled_frames, ignore_index=True)
    return result


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == "__main__":
    dataset_root = download_dataset()
    full_index = build_full_index(dataset_root)
    subsample = subsample_by_patient(full_index)

    os.makedirs(os.path.dirname(OUTPUT_MANIFEST), exist_ok=True)
    subsample.to_csv(OUTPUT_MANIFEST, index=False)

    print("\nFinal subsample class distribution:")
    print(subsample["class"].value_counts())
    print(f"\nManifest written to: {OUTPUT_MANIFEST}")
    print("Next step: use this manifest to build your patient-level "
          "train/val/test split in dataset.py")