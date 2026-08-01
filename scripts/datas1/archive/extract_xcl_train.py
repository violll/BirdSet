import pandas as pd
import requests
import os
import json
import time
import random
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from pydub import AudioSegment

# --- CONFIG ---
REPO_ROOT = Path(__file__).resolve().parents[2]          # BirdSet/
CLASS_MAPPING_JSON = REPO_ROOT / "resources/ebird_codes/DataS1_ebird_codes.json"
PARQUET_FILE = "/home/gil/comp0173/XCL_metadata.parquet" 
OUTPUT_DIR = REPO_ROOT / "data/DataS1_DT_train/ogg/"
OUTPUT_PARQUET = REPO_ROOT / "data/DataS1_DT_train/metadata-train.parquet"

with open(CLASS_MAPPING_JSON) as f:
    mapping = json.load(f)

TARGET_CLASSES = [mapping["id2label"][str(i)] for i in range(len(mapping["id2label"]))]

MAX_WORKERS = 4
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load offical metadata
df = pd.read_parquet(PARQUET_FILE)
target_df = df[df['ebird_code'].isin(TARGET_CLASSES)].copy()
print(f"Found {len(target_df)} samples for target classes")

# Metadata adjustments to match DT training format
recording_ids = target_df["filepath"].str.extract(r"(XC\d{4,6}\.ogg)")
target_df["ebird_code"] = target_df["ebird_code"].apply(
    lambda c: TARGET_CLASSES.index(c) if (not pd.isna(c) and c in TARGET_CLASSES) else pd.NA
)
target_df["ebird_code_multilabel"] = target_df["ebird_code_multilabel"].apply(
    lambda c: [TARGET_CLASSES.index(ebird_name) if (not pd.isna(ebird_name) and ebird_name in TARGET_CLASSES) else pd.NA for ebird_name in c]
)

target_df["filepath"] = recording_ids.map(
        lambda p: os.path.join(OUTPUT_DIR, os.path.basename(p))
    )
target_df["audio"] = target_df["filepath"]

# Save XCL filtered metadata
target_df.to_parquet(os.path.join(OUTPUT_DIR, "metadata-full.parquet"))


def log_failure(message):
    tqdm.write(f"[FAILED] {message}")


def download_hybrid(row_tuple):
    _, row = row_tuple
    filename = row['filepath']

    # extract id for XC source
    try:
        xc_id = filename.split("XC")[-1].split(".")[0]
        if not xc_id.isdigit():
            log_failure(f"skipped invalid xc_id for {filename}")
            return "Skipped"
    except Exception as exc:
        log_failure(f"skipped parse error for {filename}: {exc}")
        return "Skipped"

    if os.path.exists(filename):
        return "Exists"

    time.sleep(random.uniform(0.5, 1.0))
    try:
        url = f"https://xeno-canto.org/{xc_id}/download"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            audio = AudioSegment.from_file(BytesIO(r.content))
            audio.export(filename, format="ogg")

            return "Downloaded"

        if r.status_code == 429:
            log_failure(f"rate limit for XC{xc_id}")
            return "RateLimit"
        log_failure(f"http_{r.status_code} for XC{xc_id}")
    except Exception as exc:
        log_failure(f"failed XC{xc_id}: {exc}")
    return "Failed"


print(f"Downloading {len(target_df)} files...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(download_hybrid, r): r for r in target_df.iterrows()}
    for future in tqdm(as_completed(futures), total=len(target_df)):
        res = future.result()
        if res == "RateLimit":
            time.sleep(10)

# filter training df by availability
available = target_df[target_df["filepath"].map(os.path.exists)]
available.to_parquet(OUTPUT_PARQUET)
print(f"Wrote train parquet: {OUTPUT_PARQUET} ({len(available)} rows)")

print("Done!")