"""
validate_schema.py
Read-only validation for DataS1 dataset artifacts produced by schema alignment.
"""
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]          # BirdSet/
CLASS_MAPPING_JSON = REPO_ROOT / "resources/ebird_codes/DataS1_ebird_codes.json"
TRAIN_PARQUET = REPO_ROOT / "data/DataS1_DT_train/metadata-train.parquet"
TEST_PARQUET = REPO_ROOT / "data/DataS1/metadata.parquet"


def load_label_order() -> dict[str, int]:
    if not CLASS_MAPPING_JSON.exists():
        raise FileNotFoundError(f"Missing class mapping: {CLASS_MAPPING_JSON}")
    with open(CLASS_MAPPING_JSON) as f:
        data = json.load(f)
    return data.get("label2id", {})


def validate_train_schema(label_order: dict[str, int]) -> None:
    if not TRAIN_PARQUET.exists():
        print(f"[train] No train parquet at {TRAIN_PARQUET}, skipping.")
        return

    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"[train] Rows: {len(df)}, columns: {list(df.columns)}")

    if "ebird_code" not in df.columns:
        print("[train] [WARN] Missing 'ebird_code' column")
        return

    present = set(df["ebird_code"].dropna().unique().tolist())
    unknown = present - set(label_order.keys())
    if unknown:
        print(
            f"[train] [WARN] Train parquet has codes not in TARGET_CLASSES: "
            f"{sorted(unknown)[:10]}{'...' if len(unknown) > 10 else ''}"
        )
    else:
        print("[train] Train parquet class coverage looks good.")


def validate_test_schema() -> None:
    if not TEST_PARQUET.exists():
        print(f"[test ] No test parquet at {TEST_PARQUET}, skipping.")
        return

    ds = load_dataset("parquet", data_files=str(TEST_PARQUET), split="train")

    required = [
        "audio",
        "filepath",
        "start_time",
        "end_time",
        "ebird_code",
        "ebird_code_multilabel",
        "ebird_code_secondary",
    ]
    missing = [c for c in required if c not in ds.features]
    if missing:
        print(f"[test ] [WARN] Missing expected columns: {missing}")
    else:
        print(f"[test ] Required columns present.")

    print(f"[test ] Rows: {len(ds)}")
    print(f"[test ] Features:\n    {ds.features}")


def main() -> None:
    print("=" * 60)
    print("Validate Schema: DataS1 artifacts")
    print("=" * 60)

    label_order = load_label_order()
    validate_train_schema(label_order)
    print()
    validate_test_schema()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
