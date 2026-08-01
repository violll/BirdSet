"""Stage: build - assemble final DataS1 test_5s dataset and train split.

Steps (folded from build_datas1.py / test.ipynb):
1. Build boxed annotations CSV with reindexed ebird codes (Clements Excel lookup)
2. Clip test annotations into 5s segments and cast to BirdSet dataset schema
3. Save TEST_PARQUET

Output: TEST_PARQUET (metadata.parquet)
"""
import os

import pandas as pd
from datasets import load_dataset, Dataset, Audio, ClassLabel, Sequence, Features
from glob import glob
from tqdm.auto import tqdm
from opensoundscape.annotations import BoxedAnnotations

from common import (
    TEST_DATASET_ROOT,
    TEST_PARQUET,
    load_clements,
    load_annotations_details,
)


def load_hf_hsn_test_features() -> Features:
    """Load canonical 29-column test_5s schema from HF BirdSet HSN (streaming)."""
    return load_dataset(
        "DBD-research-group/BirdSet", "HSN", split="test_5s", streaming=True
    ).features.copy()


def build_boxed_annotations_csv(label2id, ebird_df, dataset_df):
    """Build boxed annotations CSV with reindexed ebird codes."""
    audio_files = glob(os.path.join(str(TEST_DATASET_ROOT), "audio_annots/*.flac"))
    valid_f_annot = []

    for audio_file in tqdm(audio_files):
        tag_file = audio_file.replace(".flac", "-tags.csv")
        if os.path.exists(tag_file):
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
            lambda x: os.path.join(str(TEST_DATASET_ROOT), "audio_annots", os.path.basename(x))
        )

        # Map scientific name -> ebird_code -> int index (O(1) via label2id dict)
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
            lambda c: label2id[c] if (not pd.isna(c) and c in label2id) else pd.NA
        )
        df.drop(columns=["related", "overlap", "id"], inplace=True)
        formatted_annots.append(df[~df["annotation"].isna()])

    df_annots = pd.concat(formatted_annots, ignore_index=True)
    return BoxedAnnotations(df_annots)


def clip_and_cast_test_dataset(target_classes, annots):
    """Clip test audio into 5s segments, cast to HSN 29-column schema."""
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

    # Load canonical 29-column schema from HF BirdSet HSN test_5s
    features = load_hf_hsn_test_features()

    # Pad missing columns with None so .cast() succeeds
    for col in features:
        if col not in boxed_annots_df.columns:
            boxed_annots_df[col] = None

    # Class order hazard fix: cast against JSON-order target_classes (not set-intersection order)
    features["ebird_code_multilabel"] = Sequence(ClassLabel(names=target_classes))
    boxed_annots_df["audio"] = boxed_annots_df["filepath"]

    dataset = (
        Dataset.from_pandas(boxed_annots_df)
        .cast_column("audio", Audio(sampling_rate=32000, mono=True))
        .cast(features)
    )
    return dataset


def stage_build(target_classes, label2id):
    """Build final DataS1 test_5s dataset.

    1. Build boxed annotations CSV (Clements Excel lookup)
    2. Clip test audio into 5s segments, cast to HSN schema
    3. Save TEST_PARQUET (metadata.parquet)

    Does NOT export DatasetDict -- the datamodule manages its own cache.
    """
    print("\nStep 1: Build boxed annotations")
    ebird_df = load_clements()
    dataset_df = load_annotations_details()
    boxed_annots = build_boxed_annotations_csv(label2id, ebird_df, dataset_df)

    print("\nStep 2: Clip and cast test_5s dataset")
    test_5s_dataset = clip_and_cast_test_dataset(target_classes, boxed_annots)

    # Save the built test_5s split
    TEST_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    test_5s_dataset.to_parquet(str(TEST_PARQUET))
    print(f"Wrote test_5s parquet: {TEST_PARQUET} ({len(test_5s_dataset)} rows)")


    print(f"\nDone")
