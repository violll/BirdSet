#!/usr/bin/env python
"""
data_prep.py - Unified DataS1 preparation pipeline

Consolidates: extract_datas1_classes.py, extract_xcl_train.py, 
              build_datas1.py, validate_schema.py, extract_pretrained_model_weights.py

Usage:
    python scripts/datas1/data_prep.py --stages all
    python scripts/datas1/data_prep.py --stages download_train,build
    python scripts/datas1/data_prep.py --stages all --with-weights
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, Audio, ClassLabel, Sequence, Features, load_dataset
from pydub import AudioSegment
from tqdm.auto import tqdm
from opensoundscape.annotations import BoxedAnnotations

# ---------------------------------------------------------------------------
# Paths (PROJECT_ROOT resolves to repo root at runtime)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

CLASS_MAPPING_JSON = PROJECT_ROOT / "resources/ebird_codes/DataS1_ebird_codes.json"
CLEMENTS_XLSX = PROJECT_ROOT / "resources/ArcticBirdSounds/Clements_v2025-October-2025.xlsx"

TEST_DATASET_ROOT = PROJECT_ROOT / "data/DataS1"
ANNOTATIONS_DETAILS = PROJECT_ROOT / "data/DataS1/annotations_details.csv"

TRAIN_FULL_PARQUET = PROJECT_ROOT / "data/DataS1_DT_train/ogg/metadata-full.parquet"
TRAIN_PARQUET = PROJECT_ROOT / "data/DataS1_DT_train/metadata-train.parquet"
BOXED_ANNOTATIONS_CSV = PROJECT_ROOT / "data/DataS1/boxed_annotations_input_reindexed.csv"
TEST_PARQUET = PROJECT_ROOT / "data/DataS1/metadata.parquet"

WEIGHTS_OUT = PROJECT_ROOT / "resources/models/EfficientNet-B1-BirdSet-XCL.ckpt"
XCL_METADATA = PROJECT_ROOT / "data/xcl/XCL_metadata.parquet" 

# Test audio (OSF - manual download required)
TEST_AUDIO_DIR = TEST_DATASET_ROOT / "audio_annots"


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------
def load_class_mapping() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Load canonical DataS1 class mapping. Returns (target_classes, label2id, id2label)."""
    if not CLASS_MAPPING_JSON.exists():
        raise FileNotFoundError(f"Missing class mapping: {CLASS_MAPPING_JSON}")
    with open(CLASS_MAPPING_JSON) as f:
        data = json.load(f)
    id2label = {int(k): v for k, v in data["id2label"].items()}
    label2id = data["label2id"]
    target_classes = [id2label[i] for i in range(len(id2label))]
    return target_classes, label2id, id2label


def load_clements() -> pd.DataFrame:
    """Load Clements Excel with species_code index."""
    if not CLEMENTS_XLSX.exists():
        raise FileNotFoundError(f"Missing Clements Excel: {CLEMENTS_XLSX}")
    return pd.read_excel(str(CLEMENTS_XLSX), usecols=[1, 7], index_col=1)


# ---------------------------------------------------------------------------
# Class mapping generation
# ---------------------------------------------------------------------------
def _generate_class_mapping() -> bool:
    """Generate CLASS_MAPPING_JSON by running extract_datas1_classes.py.

    The ebird-code mapping is a *produced* artifact (intersection of the XCL
    model label space with DataS1 scientific names), not a static source file.
    Returns True if the JSON exists afterward. Requires `transformers`, which
    is available in the BirdSet/RunPod training env but NOT in the local
    comp0173 smoke-test env.
    """
    script = PROJECT_ROOT / "scripts/datas1/extract_datas1_classes.py"
    if not script.exists():
        print(f"  Cannot generate: {script} not found")
        return False
    print(f"  Generating class mapping via {script.name} ...")
    try:
        subprocess.run([sys.executable, str(script)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  Generation failed: {exc}")
        return False
    return CLASS_MAPPING_JSON.exists()


# ---------------------------------------------------------------------------
# Stage: preflight
# ---------------------------------------------------------------------------
def stage_preflight() -> bool:
    """Verify all required source files exist before proceeding.

    CLASS_MAPPING_JSON is a *generated* artifact (not a static source): if it
    is absent, attempt to (re)generate it via extract_datas1_classes.py before
    reporting preflight failure.
    """
    missing = []
    checks = [
        ("Clements Excel", CLEMENTS_XLSX),
        ("Test dataset directory", TEST_DATASET_ROOT),
        ("XCL metadata parquet", XCL_METADATA),
    ]
    for name, path in checks:
        if not path.exists():
            missing.append(f"{name}: {path}")

    # class mapping is generated, not assumed — regenerate when flagged missing
    if not CLASS_MAPPING_JSON.exists():
        if CLEMENTS_XLSX.exists() and ANNOTATIONS_DETAILS.exists():
            print("Class mapping missing — generating via extract_datas1_classes.py ...")
            if not _generate_class_mapping():
                missing.append(f"Class mapping JSON (generation failed): {CLASS_MAPPING_JSON}")
        else:
            missing.append(f"Class mapping JSON: {CLASS_MAPPING_JSON}")

    if missing:
        print("PREFLIGHT FAILED. Missing required files:")
        for m in missing:
            print(f"  - {m}")
        print("\nNote: ArcticBirdSounds test audio must be downloaded manually from OSF (https://osf.io/b9trx/overview)")
        return False
    print("✓ Preflight passed")
    return True


# ---------------------------------------------------------------------------
# Stage: download_train
# ---------------------------------------------------------------------------
def stage_download_train(target_classes: list[str]) -> None:
    """
    Download XCL audio files from xeno-canto and build metadata-full.parquet.
    
    Input: XCL metadata parquet at PROJECT_ROOT / "data/xcl/XCL_metadata.parquet"
    Output: TRAIN_FULL_PARQUET (metadata-full.parquet)
    """
    raise NotImplementedError("To be implemented")


# ---------------------------------------------------------------------------
# Stage: download_weights
# ---------------------------------------------------------------------------
def stage_download_weights() -> None:
    """
    Download EfficientNet-B1-BirdSet-XCL checkpoint from HuggingFace Hub.
    
    Input: HF model DBD-research-group/EfficientNet-B1-BirdSet-XCL
    Output: WEIGHTS_OUT (EfficientNet-B1-BirdSet-XCL.ckpt)
    """
    if WEIGHTS_OUT.exists():
        print(f"✓ Weights already at {WEIGHTS_OUT}")
        return
    raise NotImplementedError("To be implemented")


# ---------------------------------------------------------------------------
# Stage: build
# ---------------------------------------------------------------------------
def stage_build(target_classes: list[str]) -> None:
    """
    Build final DatasetDict:
    1. Clip test audio into 5s segments
    2. Cast to HF schema
    3. Assemble train/test splits
    
    Input: TRAIN_FULL_PARQUET, BOXED_ANNOTATIONS_CSV
    Output: TEST_PARQUET_OUT, TRAIN_PARQUET
    """
    raise NotImplementedError("To be implemented")


# ---------------------------------------------------------------------------
# Stage: validate
# ---------------------------------------------------------------------------
def stage_validate() -> None:
    """
    Validate DataS1 artifacts against HF HSN reference:
    - Label column population comparison
    - ClassLabel order verification
    - Index validity check
    """
    raise NotImplementedError("To be implemented")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
STAGES = ["preflight", "download_train", "download_weights", "build", "validate"]

def main():
    parser = argparse.ArgumentParser(description="DataS1 preparation pipeline")
    parser.add_argument(
        "--stages", 
        default="all",
        help="Comma-separated stages or 'all' (default: all)"
    )
    args = parser.parse_args()
    
    stage_list = [s.strip() for s in args.stages.split(",")]
    if "all" in stage_list:
        stage_list = STAGES
    
    # Preflight is always first
    if "preflight" not in stage_list:
        stage_list = ["preflight"] + stage_list
    
    for stage in stage_list:
        if stage not in STAGES:
            print(f"Unknown stage: {stage}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Running stage: {stage}")
        print("="*60)
        
        if stage == "preflight":
            if not stage_preflight():
                raise SystemExit(1)
        elif stage == "download_train":
            target_classes, _, _ = load_class_mapping()
            stage_download_train(target_classes)
        elif stage == "download_weights":
            stage_download_weights()
        elif stage == "build":
            target_classes, _, _ = load_class_mapping()
            stage_build(target_classes)
        elif stage == "validate":
            stage_validate()


if __name__ == "__main__":
    main()