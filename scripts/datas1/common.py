"""Shared utilities, path constants, and preflight checks for the DataS1 pipeline.

Imported by data_prep.py (CLI dispatcher) and individual stage modules.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (REPO_ROOT resolves to repo root at runtime)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))

CLASS_MAPPING_JSON = REPO_ROOT / "resources/ebird_codes/DataS1_ebird_codes.json"
CLEMENTS_XLSX = REPO_ROOT / "resources/ArcticBirdSounds/Clements_v2025-October-2025.xlsx"

TEST_DATASET_ROOT = REPO_ROOT / "data/DataS1"
ANNOTATIONS_DETAILS = REPO_ROOT / "data/DataS1/annotations_details.csv"

TRAIN_PARQUET = REPO_ROOT / "data/DataS1_DT_train/metadata-train.parquet"
BOXED_ANNOTATIONS_CSV = REPO_ROOT / "data/DataS1/boxed_annotations_input_reindexed.csv"
TEST_PARQUET = REPO_ROOT / "data/DataS1/metadata.parquet"

WEIGHTS_OUT = REPO_ROOT / "resources/models/EfficientNet-B1-BirdSet-XCL.ckpt"
XCL_METADATA = REPO_ROOT / "data/xcl/XCL_metadata.parquet"

# Test audio (ArcticBirdSounds — manual OSF download)
TEST_AUDIO_DIR = TEST_DATASET_ROOT / "audio_annots"

# HuggingFace Hub identifiers
XCL_HF_PATH = "DBD-research-group/BirdSet"
XCL_HF_NAME = "XCL"
XCL_WEIGHTS_HF_REPO = "DBD-research-group/EfficientNet-B1-BirdSet-XCL"
XCL_WEIGHTS_HF_FILE = "model.safetensors"


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


def load_annotations_details() -> pd.DataFrame:
    """Load DataS1 annotations_details.csv with scientific_name column."""
    if not ANNOTATIONS_DETAILS.exists():
        raise FileNotFoundError(f"Missing annotations_details.csv: {ANNOTATIONS_DETAILS}")
    df = pd.read_csv(ANNOTATIONS_DETAILS)
    if "scientific_name" not in df.columns:
        raise ValueError("annotations_details.csv must contain 'scientific_name' column")
    return df


# ---------------------------------------------------------------------------
# Class mapping generation (delegates to extract_datas1_classes.py via subprocess)
# ---------------------------------------------------------------------------
def _generate_class_mapping() -> bool:
    """Generate CLASS_MAPPING_JSON by running extract_datas1_classes.py.

    The ebird-code mapping is a *produced* artifact (intersection of the XCL
    model label space with DataS1 scientific names), not a static source file.
    Returns True if the JSON exists afterward. Requires `transformers`, which
    is available in the BirdSet/RunPod training env but NOT in the local
    comp0173 smoke-test env.
    """
    script = REPO_ROOT / "scripts/datas1/extract_datas1_classes.py"
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
    print("[OK] Preflight passed")
    return True
