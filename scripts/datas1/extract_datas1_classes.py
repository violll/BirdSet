"""
schema_align.py
Derive the authoritative DataS1 target classes and write the canonical
DataS1 ebird-code mapping.

Source of truth for class derivation: test.ipynb
"""
import json

import pandas as pd
from transformers import EfficientNetForImageClassification
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]          # BirdSet/
MODEL_REF = "DBD-research-group/EfficientNet-B1-BirdSet-XCL"
CLEMENTS_XLSX = REPO_ROOT / "resources/ArcticBirdSounds/Clements_v2025-October-2025.xlsx"
ANNOTATIONS_DETAILS = REPO_ROOT / "data/DataS1/annotations_details.csv"
CLASS_MAPPING_OUT = REPO_ROOT / "resources/ebird_codes/DataS1_ebird_codes.json"


def load_model_label_space() -> list[str]:
    """Load the 9736-class XCL label space from the remote Hub model."""
    model = EfficientNetForImageClassification.from_pretrained(
        MODEL_REF,
        num_channels=1,
        ignore_mismatched_sizes=True,
    )
    return list(model.config.id2label.values())


def load_dataset_scientific_names() -> list[str]:
    """Load unique scientific names in DataS1 from annotations_details.csv."""
    if not ANNOTATIONS_DETAILS.exists():
        raise FileNotFoundError(f"Missing annotations_details.csv: {ANNOTATIONS_DETAILS}")
    df = pd.read_csv(ANNOTATIONS_DETAILS)
    if "scientific_name" not in df.columns:
        raise ValueError("annotations_details.csv must contain 'scientific_name' column")
    uniq = pd.unique(df.loc[:, "scientific_name"])
    return [s for s in uniq if isinstance(s, str) and s.strip()]


def build_target_classes() -> list[str]:
    """
    Derive TARGET_CLASSES exactly as in notebook cells 4-5:
    Intersection of model label space with DataS1 scientific names,
    translated to ebird codes via Clements Excel.
    """
    print("[1/4] Loading model label space...")
    model_labels = load_model_label_space()
    print(f"       Model classes: {len(model_labels)}")

    print("[2/4] Loading DataS1 scientific names...")
    sci_names = load_dataset_scientific_names()
    print(f"       Dataset scientific names: {len(sci_names)}")

    print(f"[3/4] Loading Clements (source: {CLEMENTS_XLSX}) ...")
    if not CLEMENTS_XLSX.exists():
        raise FileNotFoundError(f"Missing Clements Excel: {CLEMENTS_XLSX}")
    ebird_df = pd.read_excel(
        str(CLEMENTS_XLSX),
        usecols=[1, 7],
        index_col=1,
    )

    dataset_classes = []
    for label in sci_names:
        try:
            dataset_classes.append(ebird_df.loc[label, "species_code"])
        except Exception:
            print(f"       [WARN] Scientific name not in Clements: {label}")

    target_classes = sorted(
        set(dataset_classes).intersection(set(model_labels))
    )
    print(f"[4/4] TARGET_CLASSES (derived): {len(target_classes)} species")
    print(f"       Order: {target_classes}")

    return target_classes


def write_canonical_mappings(target_classes: list[str]) -> None:
    """Write DataS1 ebird code mapping in the existing BirdSet convention."""
    label2id = {ec: i for i, ec in enumerate(target_classes)}
    id2label = {str(i): ec for i, ec in enumerate(target_classes)}

    CLASS_MAPPING_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_MAPPING_OUT, "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)
    print(f"Wrote {CLASS_MAPPING_OUT}")


def main() -> None:
    target_classes = build_target_classes()
    write_canonical_mappings(target_classes)


if __name__ == "__main__":
    main()
