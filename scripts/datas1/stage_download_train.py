"""Stage: download_train - prepare XCL training metadata for DataS1's 31 classes.

Modes ('--xcl-source'):
  xc  - download .ogg files from xeno-canto.org
  hf  - use full XCL dataset already cached by HF Hub 

Output: TRAIN_PARQUET (metadata-train.parquet)
"""
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
import requests
from pydub import AudioSegment
from tqdm.auto import tqdm

from common import (
    TRAIN_PARQUET,
    XCL_METADATA,
    XCL_HF_PATH,
    XCL_HF_NAME,
)


def _log_failure(message: str) -> None:
    tqdm.write(f"[FAILED] {message}")


def _download_xc_ogg(filepath: str) -> str:
    """Download a single .ogg from xeno-canto.org. Returns status string."""
    try:
        xc_id = filepath.split("XC")[-1].split(".")[0]
        if not xc_id.isdigit():
            _log_failure(f"skipped invalid xc_id for {filepath}")
            return "Skipped"
    except Exception as exc:
        _log_failure(f"skipped parse error for {filepath}: {exc}")
        return "Skipped"

    if os.path.exists(filepath):
        return "Exists"

    time.sleep(random.uniform(0.5, 1.0))
    try:
        url = f"https://xeno-canto.org/{xc_id}/download"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            audio = AudioSegment.from_file(BytesIO(r.content))
            audio.export(filepath, format="ogg")
            return "Downloaded"
        if r.status_code == 429:
            _log_failure(f"rate limit for XC{xc_id}")
            return "RateLimit"
        _log_failure(f"http_{r.status_code} for XC{xc_id}")
    except Exception as exc:
        _log_failure(f"failed XC{xc_id}: {exc}")
    return "Failed"


def _filter_xcl_metadata(target_classes: list[str], label2id: dict[str, int]) -> pd.DataFrame:
    """Load XCL metadata, filter to target classes, remap labels to int indices."""
    df = pd.read_parquet(XCL_METADATA)
    target_df = df[df["ebird_code"].isin(target_classes)].copy()
    print(f"Found {len(target_df)} samples for target classes")

    # Map ebird_code string -> int index
    target_df["ebird_code"] = target_df["ebird_code"].map(
        lambda c: label2id[c] if (not pd.isna(c) and c in label2id) else pd.NA
    )
    # NaN-safe multilabel mapping (ebird_code_multilabel is a numpy array of strings)
    target_df["ebird_code_multilabel"] = target_df["ebird_code_multilabel"].apply(
        lambda c: [label2id[name] if (not pd.isna(name) and name in label2id) else None
                   for name in c] if pd.notna(c) else []
    )
    return target_df


def _edit_filepaths(target_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Rename filepaths to output_dir and return the df."""
    recording_ids = target_df["filepath"].str.extract(r"(XC\d{4,6}\.ogg)")
    target_df["filepath"] = recording_ids[0].map(
        lambda p: os.path.join(output_dir, p) if pd.notna(p) else p
    )
    target_df["audio"] = target_df["filepath"]

    return target_df


def _download_xc_files(target_df: pd.DataFrame) -> None:
    """Download missing .ogg files from xeno-canto (rate-limited)."""
    missing = target_df[~target_df["filepath"].map(os.path.exists)]
    print(f"Downloading {len(missing)} files from xeno-canto...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_download_xc_ogg, row["filepath"]): idx
            for idx, row in missing.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res == "RateLimit":
                time.sleep(10)


def _write_train_parquet(target_df: pd.DataFrame) -> None:
    """Filter to existing files and write metadata-train.parquet."""
    available = target_df[target_df["filepath"].map(os.path.exists)]
    available.to_parquet(TRAIN_PARQUET)
    print(f"Wrote train parquet: {TRAIN_PARQUET} ({len(available)} rows)")
    print("Done!")


def stage_download_train(target_classes: list[str], label2id: dict[str, int], xcl_source: str = "xc") -> None:
    """Prepare XCL training data for the 31 target classes.

    xc mode: download .ogg files from xeno-canto.org 
    hf mode: use full XCL dataset cached by HF Hub

    Output: TRAIN_PARQUET (metadata-train.parquet)
    """
    if xcl_source == "hf":
        raise NotImplementedError(
            "hf mode requires XCL audio pre-downloaded via HF Hub. "
            "Download full XCL first, then implement path resolution."
        )

    # --- xc mode: download from xeno-canto ---
    output_dir = str(os.path.join(TRAIN_PARQUET.parent, "ogg"))
    os.makedirs(output_dir, exist_ok=True)

    target_df = _filter_xcl_metadata(target_classes, label2id)
    target_df = _edit_filepaths(target_df, output_dir)
    _download_xc_files(target_df)
    _write_train_parquet(target_df)
