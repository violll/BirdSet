"""
build_datas1.py
Data preparation pipeline for DataS1, translated from test.ipynb.

Steps:
1. Repair train parquet by dropping rows whose filepath is missing.
2. Build boxed annotations input CSV with reindexed ebird codes.
3. Clip annotations into 5-second segments.
4. Cast to Hugging Face dataset schema and write metadata.parquet.
5. Assemble final DatasetDict(train, test_5s).
"""
import json
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Audio, ClassLabel, Sequence, Features
from glob import glob
from tqdm.auto import tqdm
from opensoundscape.annotations import BoxedAnnotations

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]          # BirdSet/
CLASS_MAPPING_JSON = REPO_ROOT / "resources/ebird_codes/DataS1_ebird_codes.json"
CLEMENTS_XLSX = REPO_ROOT / "resources/ArcticBirdSounds/Clements_v2025-October-2025.xlsx"
ANNOTATIONS_DETAILS = REPO_ROOT / "data/DataS1/annotations_details.csv"
DATASET_ROOT = REPO_ROOT / "data/DataS1"
TRAIN_FULL_PARQUET = REPO_ROOT / "data/DataS1_DT_train/ogg/metadata-full.parquet"
TRAIN_PARQUET = REPO_ROOT / "data/DataS1_DT_train/metadata-train.parquet"
BOXED_ANNOTATIONS_CSV = REPO_ROOT / "data/DataS1/boxed_annotations_input_reindexed.csv"
TEST_PARQUET_OUT = REPO_ROOT / "data/DataS1/metadata.parquet"


def load_class_mapping() -> tuple[list[str], dict[str, int], dict[str, str]]:
    if not CLASS_MAPPING_JSON.exists():
        raise FileNotFoundError(f"Missing class mapping: {CLASS_MAPPING_JSON}")
    with open(CLASS_MAPPING_JSON) as f:
        data = json.load(f)
    id2label = {int(k): v for k, v in data["id2label"].items()}
    label2id = data["label2id"]
    target_classes = [id2label[i] for i in range(len(id2label))]
    return target_classes, label2id, id2label


def load_clements() -> pd.DataFrame:
    if not CLEMENTS_XLSX.exists():
        raise FileNotFoundError(f"Missing Clements Excel: {CLEMENTS_XLSX}")
    return pd.read_excel(
        str(CLEMENTS_XLSX),
        usecols=[1, 7],
        index_col=1,
    )


def load_annotations_details() -> pd.DataFrame:
    if not ANNOTATIONS_DETAILS.exists():
        raise FileNotFoundError(f"Missing annotations_details.csv: {ANNOTATIONS_DETAILS}")
    df = pd.read_csv(ANNOTATIONS_DETAILS)
    if "scientific_name" not in df.columns:
        raise ValueError("annotations_details.csv must contain 'scientific_name' column")
    return df


def repair_train_parquet(target_classes: list[str]) -> None:
    if not TRAIN_FULL_PARQUET.exists():
        raise FileNotFoundError(f"Missing train metadata-full parquet: {TRAIN_FULL_PARQUET}")

    df_all = pd.read_parquet(TRAIN_FULL_PARQUET)

    available = df_all[df_all["filepath"].map(lambda p: os.path.exists(p))]
    TRAIN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    available.to_parquet(TRAIN_PARQUET)
    print(f"Wrote repaired train parquet: {TRAIN_PARQUET} ({len(available)} rows)")


def build_boxed_annotations_csv(target_classes: list[str], ebird_df: pd.DataFrame, dataset_df: pd.DataFrame) -> None:
    audio_files = glob(os.path.join(DATASET_ROOT, "audio_annots/*.flac"))

    valid_f_audio = []
    valid_f_annot = []

    for audio_file in tqdm(audio_files):
        tag_file = audio_file.replace(".flac", "-tags.csv")
        if os.path.exists(tag_file):
            output_path = audio_file.replace("flac", "ogg")
            valid_f_audio.append(output_path)
            valid_f_annot.append(tag_file)

    formatted_annots = []
    for f in tqdm(valid_f_annot):
        df = pd.read_csv(f)
        df = df[df["tag"] != "UNKN"]
        if df.shape[0] == 0:
            continue

        df.rename(
            columns={
                "start": "start_time",
                "end": "end_time",
                "frequency_min": "low_f",
                "frequency_max": "high_f",
                "tag": "annotation",
                "file_name": "audio_file",
            },
            inplace=True,
        )

        df["audio_file"] = df["audio_file"].apply(
            lambda x: os.path.join(DATASET_ROOT, "audio_annots", x)
        )

        df_ebird_class_names = pd.merge(
            pd.merge(
                df,
                dataset_df[["tag", "scientific_name"]],
                left_on="annotation",
                right_on="tag",
            ),
            ebird_df.reset_index(),
            left_on="scientific_name",
            right_on="scientific name",
        )["species_code"]

        df["annotation"] = df_ebird_class_names.apply(
            lambda c: target_classes.index(c) if (not pd.isna(c) and c in target_classes) else pd.NA
        )

        df.drop(columns=["related", "overlap", "id"], inplace=True)
        formatted_annots.append(df[~df["annotation"].isna()])

    df_annots = pd.concat(formatted_annots, ignore_index=True)
    BOXED_ANNOTATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_annots.to_csv(BOXED_ANNOTATIONS_CSV)
    print(f"Wrote boxed annotations CSV: {BOXED_ANNOTATIONS_CSV} ({len(df_annots)} rows)")


def load_hf_hsn_test_features() -> Features:
    # provides access to canonical 29-column test_5s schema from HF BirdSet HSN
    return load_dataset(
        "DBD-research-group/BirdSet", "HSN", split="test_5s", streaming=True
    ).features.copy()


def clip_and_cast_test_dataset(target_classes: list[str]) -> Dataset:
    df_annots = pd.read_csv(BOXED_ANNOTATIONS_CSV, index_col=0)
    annots = BoxedAnnotations(df_annots)
    annots.audio_files = annots.df["audio_file"].unique()

    boxed_annots, _ = annots.clip_labels(
        clip_duration=5,
        clip_overlap=0,
        min_label_overlap=0.25,
        class_subset=[i for i in range(len(target_classes))],
        return_type="classes",
    )

    boxed_annots_df = boxed_annots.reset_index().rename(
        columns={"labels": "ebird_code_multilabel", "file": "filepath"}
    )

    # Load the canonical 29-column schema from HF BirdSet HSN test_5s
    features = load_hf_hsn_test_features()

    # pad any columns missing from the clipped df with None
    # so the later .cast() to the complete HF schema succeeds 
    for col in features:
        if col not in boxed_annots_df.columns:
            boxed_annots_df[col] = None
    features["ebird_code_multilabel"] = Sequence(ClassLabel(names=target_classes))

    dataset = (
        Dataset.from_pandas(boxed_annots_df)
        .cast_column("audio", Audio(sampling_rate=32000, mono=True))
        .cast(features)
    )
    return dataset


def assemble_dataset_dict(train: Dataset, test_5s: Dataset) -> DatasetDict:
    return DatasetDict({"train": train, "test_5s": test_5s})


def main() -> None:
    print("=" * 60)
    print("Build DataS1 dataset")
    print("=" * 60)

    target_classes, _, _ = load_class_mapping()
    ebird_df = load_clements()
    dataset_df = load_annotations_details()

    print("\nStep 1: Repair train parquet")
    repair_train_parquet(target_classes)

    print("\nStep 2: Build boxed annotations CSV")
    build_boxed_annotations_csv(target_classes, ebird_df, dataset_df)

    print("\nStep 3: Clip and cast test_5s dataset")
    test_5s_dataset = clip_and_cast_test_dataset(target_classes)

    # Save the built test_5s split
    TEST_PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)
    test_5s_dataset.to_parquet(str(TEST_PARQUET_OUT))
    print(f"Wrote test_5s parquet: {TEST_PARQUET_OUT} ({len(test_5s_dataset)} rows)")

    print("\nStep 4: Load train dataset")
    train_dataset = load_dataset("parquet", data_files=str(TRAIN_PARQUET), split="train")
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=32000, mono=True))

    print("\nStep 5: Assemble DatasetDict")
    dataset_dict = assemble_dataset_dict(train_dataset, test_5s_dataset)
    print(dataset_dict)

    print("\nDone.")


if __name__ == "__main__":
    main()
