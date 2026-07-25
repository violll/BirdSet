"""
validate_schema.py
Read-only validation and comparison for DataS1 dataset artifacts
against canonical HF BirdSet HSN splits (train and test_5s).
"""
from pathlib import Path

import json

import numpy as np
import pandas as pd
from datasets import load_dataset, Sequence, ClassLabel

REPO_ROOT = Path(__file__).resolve().parents[2]          # BirdSet/

CLASS_MAPPING_JSON = REPO_ROOT / "resources/ebird_codes/DataS1_ebird_codes.json"


def load_hf_hsn_train():
    return load_dataset("DBD-research-group/BirdSet", "HSN", split="train")


def load_hf_hsn_test():
    return load_dataset("DBD-research-group/BirdSet", "HSN", split="test_5s")


# Label-carrying columns whose population must mirror the HF HSN reference.
# NOTE: ebird_code_multilabel / ebird_code_secondary hold integer ClassLabel
# indices whose vocabulary differs from DataS1's by design, so we compare
# POPULATION only (never the integer values).
LABEL_COLS = ["ebird_code", "ebird_code_multilabel", "ebird_code_secondary"]


def _is_populated(v) -> bool:
    """A value counts as populated if it is not None/NaN and not an empty list/array.

    NOTE: list-valued parquet columns (e.g. ebird_code_multilabel) load as numpy
    arrays, not python lists. We must check for empty numpy arrays too, otherwise
    an empty np.array([]) slips past `isinstance(v, list)` and is wrongly counted
    as populated.
    """
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except (ValueError, TypeError):
        # pd.isna raises on array-likes (e.g. np.array([])); fall through to len check
        pass
    # handles both python list and numpy array (parquet Sequence columns)
    if isinstance(v, (list, np.ndarray)) and len(v) == 0:
        return False
    return True


def _popfrac(series) -> float:
    """Fraction of non-null / non-empty-list values in a pandas Series."""
    if len(series) == 0:
        return 0.0
    pop = sum(1 for v in series.tolist() if _is_populated(v))
    return float(pop) / len(series)


def compare_label_population(split, local_path, hf_dataset) -> None:
    """Per-label-column emptiness check (local parquet vs HF HSN reference).

    All-or-nothing: a column is either EMPTY (0% populated) or POPULATED
    (>=1 value) -- no 5% floor. Flags any column whose empty/not-empty
    verdict differs from the HF HSN reference. We compare POPULATION only
    (never the integer values), because ebird_code_multilabel /
    ebird_code_secondary hold ClassLabel indices whose vocabulary differs
    from DataS1's by design.
    """
    print(f"\n== {split}: label-column emptiness (local vs HF HSN) ==")
    if not local_path.exists():
        print(f"[{split}] Missing parquet at {local_path}, skipping.")
        return
    ldf = pd.read_parquet(local_path)

    n_hf = len(hf_dataset)
    hf_pop_cnt = {c: 0 for c in LABEL_COLS}
    for r in hf_dataset:
        for c in LABEL_COLS:
            if c in r and _is_populated(r.get(c)):
                hf_pop_cnt[c] += 1

    print(f"(HF rows={n_hf}; local rows={len(ldf)})")
    print(f"{'label column':<26}{'local':>10}{'hf':>10}  verdict")
    for c in LABEL_COLS:
        if c not in ldf.columns:
            print(f"{c:<26}{'ABSENT':>10}  local missing this column")
            continue
        loc_empty = _popfrac(ldf[c]) == 0.0
        hf_empty = (hf_pop_cnt[c] == 0)
        loc_str = "empty" if loc_empty else "populated"
        hf_str = "empty" if hf_empty else "populated"
        verdict = "OK" if loc_empty == hf_empty else "  <-- MISMATCH"
        print(f"{c:<26}{loc_str:>10}{hf_str:>10}  {verdict}")


def load_class_mapping() -> tuple[list[str], int]:
    """Load canonical DataS1 class mapping. Returns (ordered class names, num_classes)."""
    if not CLASS_MAPPING_JSON.exists():
        raise FileNotFoundError(f"Missing class mapping: {CLASS_MAPPING_JSON}")
    with open(CLASS_MAPPING_JSON) as f:
        data = json.load(f)
    id2label = {int(k): v for k, v in data["id2label"].items()}
    # rebuild canonical order exactly as build_datas1.py does
    target_classes = [id2label[i] for i in range(len(id2label))]
    return target_classes, len(target_classes)


def verify_classlabel_order(split, local_path, json_path) -> None:
    """Verify ebird_code_multilabel ClassLabel vocabulary order matches the JSON
    id2label canonical order. Guards the class-order hazard where a parquet built
    with an arbitrary order would silently map indices to the wrong species.
    Only meaningful when the column is stored as a ClassLabel (the test_5s split);
    train is a plain pandas parquet and is skipped.
    """
    print(f"\n== {split}: ebird_code_multilabel ClassLabel order vs JSON ==")
    if not local_path.exists():
        print(f"[{split}] Missing parquet at {local_path}, skipping.")
        return
    ds = load_dataset("parquet", data_files=str(local_path), split="train")
    feat = ds.features.get("ebird_code_multilabel")
    if not isinstance(feat, Sequence) or not isinstance(feat.feature, ClassLabel):
        print(f"[{split}] ebird_code_multilabel is not a ClassLabel in this parquet; "
              f"order check skipped (expected for train split).")
        return

    if not json_path.exists():
        print(f"[{split}] Missing JSON mapping {json_path}; cannot verify order.")
        return
    with open(json_path) as f:
        data = json.load(f)
    id2label = {int(k): v for k, v in data["id2label"].items()}
    json_order = [id2label[i] for i in range(len(id2label))]
    parquet_names = list(feat.feature.names)

    if parquet_names == json_order:
        print(f"[{split}] OK: ClassLabel order matches JSON id2label "
              f"({len(parquet_names)} classes).")
    else:
        print(f"[{split}] MISMATCH: ClassLabel order differs from JSON id2label.")
        for i, (a, b) in enumerate(zip(parquet_names, json_order)):
            if a != b:
                print(f"  first divergence at idx {i}: parquet={a!r} json={b!r}")
                break
        print(f"  same set of names? {set(parquet_names) == set(json_order)}")


def verify_label_index_validity(split, local_path, num_classes) -> None:
    """Verify every populated ebird_code_multilabel index lies in [0, num_classes-1].
    Catches corrupted/offset label indices. Reports out-of-range count and how many
    of the expected classes are actually present.
    """
    print(f"\n== {split}: ebird_code_multilabel index validity ==")
    if not local_path.exists():
        print(f"[{split}] Missing parquet at {local_path}, skipping.")
        return
    df = pd.read_parquet(local_path)
    if "ebird_code_multilabel" not in df.columns:
        print(f"[{split}] column absent; skipping.")
        return
    vals = df["ebird_code_multilabel"].tolist()
    out_of_range = 0
    present = set()
    total_entries = 0
    for v in vals:
        if not _is_populated(v):
            continue
        for idx in v:
            total_entries += 1
            i = int(idx)
            present.add(i)
            if num_classes is not None and (i < 0 or i >= num_classes):
                out_of_range += 1
    if num_classes is None:
        print(f"[{split}] num_classes unknown; range check skipped "
              f"({total_entries} label entries across {len(present)} classes).")
        return
    if out_of_range == 0:
        print(f"[{split}] OK: {total_entries} label entries, all in [0, {num_classes-1}]. "
              f"{len(present)}/{num_classes} classes present.")
    else:
        print(f"[{split}] MISMATCH: {out_of_range}/{total_entries} label entries out of "
              f"range [0, {num_classes-1}].")


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

    # canonical class mapping used by the extra integrity guards
    try:
        _, num_classes = load_class_mapping()
    except FileNotFoundError as e:
        print(f"[WARN] {e}; skipping class-order / index-validity guards.")
        num_classes = None

    print("=" * 60)
    print("Validate Schema + HF HSN compare: DataS1 artifacts")
    print("=" * 60)

    for split in ["train", "test_5s"]:
        compare_label_population(split, train_test_splits[split]["datas1"], train_test_splits[split]["hf"])
        if num_classes is not None:
            verify_label_index_validity(split, train_test_splits[split]["datas1"], num_classes)
            verify_classlabel_order(split, train_test_splits[split]["datas1"], CLASS_MAPPING_JSON)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
