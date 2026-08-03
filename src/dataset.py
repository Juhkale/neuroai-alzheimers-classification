"""
dataset.py

Takes the subsampled manifest (data/manifest.csv) and splits it into
train / validation / test sets AT THE PATIENT LEVEL, so no patient's
images ever appear in more than one split. This is the same leakage
principle from prepare_dataset.py, applied one level further down the
pipeline.

SPLIT STRATEGY:
----------------
Standard split: 70% train / 15% val / 15% test, by patient, per class.
This ratio (rather than a more common 80/20) was chosen deliberately --
because we're splitting patients, not images, and most classes only have
20-38 unique patients, a 15% test/val split is needed to keep at least a
handful of patients in each split. See README for full reasoning.

SPECIAL CASE -- Moderate Dementia:
------------------------------------
This class only has 2 total patients in the source data. A three-way
split isn't statistically meaningful here. Per the project's documented
hybrid approach: one patient goes to train, one goes to test, and this
class gets NO validation set. Its test performance is reported but
explicitly flagged as a qualitative, single-patient result rather than a
statistically validated metric -- see the README's trustworthiness
discussion.

Run this after prepare_dataset.py. Output: data/manifest_split.csv, which
is the same manifest with an added "split" column (train/val/test).
"""

import pandas as pd
import random

RANDOM_SEED = 42
INPUT_MANIFEST = "data/manifest.csv"
OUTPUT_MANIFEST = "data/manifest_split.csv"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Classes handled by the standard 70/15/15 patient split.
# Moderate Dementia is deliberately excluded -- handled separately below.
STANDARD_SPLIT_CLASSES = ["Non Demented", "Very mild Dementia", "Mild Dementia"]


def split_patients_standard(patient_ids: list[str]) -> dict[str, str]:
    """
    Shuffles a list of patient IDs and divides them into train/val/test
    according to TRAIN_FRAC/VAL_FRAC/TEST_FRAC. Returns a dict mapping
    each patient_id -> split name.

    Uses simple index-based slicing after a seeded shuffle, rather than
    sklearn's train_test_split, because we want exact, easy-to-audit
    control over how small patient counts get divided (sklearn's
    stratify options don't handle single-class patient lists cleanly
    when counts are this small).
    """
    random.seed(RANDOM_SEED)
    shuffled = patient_ids.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * TRAIN_FRAC)
    n_val = round(n * VAL_FRAC)
    # Test gets whatever's left, avoiding rounding errors dropping a patient
    n_test = n - n_train - n_val

    # Safety net: with very small patient counts, rounding could zero out
    # a split. Guarantee at least 1 patient in val and test if we have
    # enough patients to spare from train.
    if n_val == 0 and n >= 3:
        n_val = 1
        n_train -= 1
    if n_test == 0 and n >= 3:
        n_test = 1
        n_train -= 1

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    mapping = {}
    for pid in train_ids:
        mapping[pid] = "train"
    for pid in val_ids:
        mapping[pid] = "val"
    for pid in test_ids:
        mapping[pid] = "test"
    return mapping


def split_moderate_dementia(patient_ids: list[str]) -> dict[str, str]:
    """
    Special case for the 2-patient Moderate Dementia class: one patient
    to train, one to test, no validation split. If this class ever has
    more than 2 patients in a future dataset version, this function
    should be revisited -- for now it assumes exactly 2.
    """
    if len(patient_ids) != 2:
        print(f"WARNING: expected exactly 2 patients for Moderate Dementia, "
              f"found {len(patient_ids)}. Splitting evenly train/test, "
              f"but double check this is still the right approach.")

    random.seed(RANDOM_SEED)
    shuffled = patient_ids.copy()
    random.shuffle(shuffled)

    mapping = {}
    midpoint = max(1, len(shuffled) // 2)
    for pid in shuffled[:midpoint]:
        mapping[pid] = "train"
    for pid in shuffled[midpoint:]:
        mapping[pid] = "test"
    return mapping


def build_split_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    all_patient_split_maps = {}

    for class_name in STANDARD_SPLIT_CLASSES:
        class_patients = manifest[manifest["class"] == class_name]["patient_id"].unique().tolist()
        mapping = split_patients_standard(class_patients)
        all_patient_split_maps.update(mapping)

    moderate_patients = manifest[manifest["class"] == "Moderate Dementia"]["patient_id"].unique().tolist()
    moderate_mapping = split_moderate_dementia(moderate_patients)
    all_patient_split_maps.update(moderate_mapping)

    manifest = manifest.copy()
    manifest["split"] = manifest["patient_id"].map(all_patient_split_maps)
    return manifest


def print_split_summary(manifest: pd.DataFrame) -> None:
    print("\n=== Split summary: image counts ===")
    print(manifest.groupby(["class", "split"]).size().unstack(fill_value=0))

    print("\n=== Split summary: unique patient counts ===")
    patient_counts = (
        manifest.groupby(["class", "split"])["patient_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    print(patient_counts)


if __name__ == "__main__":
    manifest = pd.read_csv(INPUT_MANIFEST, dtype={"patient_id": str})
    split_manifest = build_split_manifest(manifest)
    split_manifest.to_csv(OUTPUT_MANIFEST, index=False)

    print_split_summary(split_manifest)
    print(f"\nSplit manifest written to: {OUTPUT_MANIFEST}")
    print("Next step: build a PyTorch/TF Dataset class that reads this "
          "manifest and loads images by split.")