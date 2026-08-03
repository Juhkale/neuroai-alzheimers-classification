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
PATIENT_ID_PATTERN = re.compile(r"OAS1_(\d+)_MR1")


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
def subsample_by_patient(full_index: pd.DataFrame) -> pd.DataFrame:
    """
    For each class, randomly shuffles the PATIENTS (not images) and adds
    whole patients to the sample until we hit the target image count for
    that class. This guarantees:
      1. No patient is split -- all of a selected patient's slices in that
         class come along together.
      2. The final counts are close to (but may slightly exceed) the
         targets, since we add whole patients, not partial ones.
    """
    random.seed(RANDOM_SEED)
    sampled_frames = []

    for class_name, target in TARGET_PER_CLASS.items():
        class_df = full_index[full_index["class"] == class_name]

        if target is None:
            # Keep everything -- this is the Moderate Dementia case
            sampled_frames.append(class_df)
            print(f"{class_name}: kept all {len(class_df)} images "
                  f"({class_df['patient_id'].nunique()} patients)")
            continue

        patient_ids = class_df["patient_id"].unique().tolist()
        random.shuffle(patient_ids)

        selected_patients = []
        running_total = 0
        for pid in patient_ids:
            if running_total >= target:
                break
            n_images_for_patient = (class_df["patient_id"] == pid).sum()
            selected_patients.append(pid)
            running_total += n_images_for_patient

        class_sample = class_df[class_df["patient_id"].isin(selected_patients)]
        sampled_frames.append(class_sample)
        print(f"{class_name}: sampled {len(class_sample)} images "
              f"(target was {target}) from {len(selected_patients)} patients")

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