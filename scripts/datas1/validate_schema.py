"""
validate_schema.py
Read-only validation and comparison for DataS1 dataset artifacts
against canonical HF BirdSet HSN splits (train and test_5s).
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]          # BirdSet/


def load_hf_hsn_train():
    return load_dataset("DBD-research-group/BirdSet", "HSN", split="train")


def load_hf_hsn_test():
    return load_dataset("DBD-research-group/BirdSet", "HSN", split="test_5s")


def compare_schemas(split, train_test_splits) -> None:
    print(f"\n== {split} parquet validation/compare ==")
    if not train_test_splits[split]["datas1"].exists():
        print(f"[{split}] Missing parquet at {train_test_splits[split]['datas1']}, skipping.")
        return

    train_df = pd.read_parquet(train_test_splits[split]["datas1"])

    print(f"[{split}] Rows: {len(train_df)}, columns: {list(train_df.columns)}")

    # compare against HF reference
    try:
        hf = train_test_splits[split]["hf"]
        print(f"HF rows={len(hf)}")
        print("[HF sample - first row]")
        for k, v in hf[0].items():
            print(f"  {k}: {repr(v)[:200]}")
        print("[local sample - first row]")
        lcl = pd.read_parquet(train_test_splits[split]["datas1"]).iloc[0]
        for k, v in lcl.items():
            print(f"  {k}: {repr(v)[:200]}")
    except Exception as e:
        print(f"[{split}] [SKIP] HF reference load failed: {e}")

def view_ebird_code_multilabel_schema():
    hf = load_hf_hsn_test()
    df = hf.to_pandas()

    elem_counts = df["ebird_code_multilabel"].apply(
        lambda x: len(x)
    )
    print("\nebird_code_multilabel length counts:")
    print(elem_counts.value_counts().sort_index())
    print("\nNum empty:", int((elem_counts == 0).sum()))
    print("Num len=1:", int((elem_counts == 1).sum()))
    print("Num len>1:", int((elem_counts > 1).sum()))
    return elem_counts.value_counts()


def main() -> None:
    train_test_splits = {
        "train": {
            "datas1": REPO_ROOT / "data/DataS1_DT_train/metadata-train.parquet",
            "hf": load_hf_hsn_train()
        },
        "test_5s": {
            "datas1": REPO_ROOT / "data/DataS1/metadata.parquet",
            "hf": load_hf_hsn_test()
        }
    }

    print("=" * 60)
    print("Validate Schema + HF HSN compare: DataS1 artifacts")
    print("=" * 60)

    for split in ["train", "test_5s"]:
        compare_schemas(split, train_test_splits)
    view_ebird_code_multilabel_schema()

    print("\nDone.\n")


if __name__ == "__main__":
    main()
