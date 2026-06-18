import pandas as pd
import requests
import os
import json
import time
import random
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from pydub import AudioSegment

# --- CONFIG ---
PARQUET_FILE = "/home/gil/comp0173/XCL_metadata.parquet"
TARGET_CLASSES = [
    "amgplo",
    "arcter",
    "baisan",
    "bkbplo",
    "brant",
    "batgod",
    "cangoo",
    "comrav",
    "dunlin",
    "gwfgoo",
    "kineid",
    "laplon",
    "lobdow",
    "lotduc",
    "lotjae",
    "pacloo",
    "pecsan",
    "pomjae",
    "pursan",
    "redpha1",
    "retloo",
    "rudtur",
    "sabgul",
    "sander",
    "semplo",
    "semsan",
    "snobun",
    "snogoo",
    "speeid",
    "tunswa",
    "whrsan"]

OUTPUT_DIR = "/home/gil/comp0173/BirdSet/data/DataS1_DT_train/ogg/"
MAX_WORKERS = 4
DEBUG = True
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load OFFICIAL Metadata
# This ensures our labels/splits are 100% consistent with BirdSet
df = pd.read_parquet(PARQUET_FILE)
target_df = df[df['ebird_code'].isin(TARGET_CLASSES)].copy()
print(target_df.shape)

# # change the filepath in the parquet file to match the actual path
# target_df["filepath"] = OUTPUT_DIR + target_df["filepath"].str.extract(r"(XC\d{4,6}.ogg)")

# # other metadata adjustments to match DT training format
# target_df["ebird_code"] = target_df["ebird_code"].apply(lambda c: TARGET_CLASSES.index(c) if (not pd.isna(c) and c in TARGET_CLASSES) else pd.NA)
# target_df["ebird_code_multilabel"] = target_df["ebird_code_multilabel"].apply(lambda c: [TARGET_CLASSES.index(ebird_name) if (not pd.isna(ebird_name) and ebird_name in TARGET_CLASSES) else pd.NA for ebird_name in c])
# target_df["audio"] = target_df["filepath"]

# print(f"Syncing {len(target_df)} files from Official Metadata...")

# # 2. Setup The Metadata Lock
# meta_path = os.path.join(OUTPUT_DIR, "metadata.txt")
# target_df.to_parquet(os.path.join(OUTPUT_DIR, "metadata-full.parquet"))   

def log_failure(message):
    print(f"[FAILED] {message}")




def download_hybrid(row_tuple):
    _, row = row_tuple
    filename = row['filepath']

    # Extract ID for Xeno-Canto Source
    try:
        xc_id = filename.split("XC")[-1].split(".")[0]
        if not xc_id.isdigit():
            log_failure(f"skipped invalid xc_id for {filename}")
            return "Skipped"
    except Exception as exc:
        log_failure(f"skipped parse error for {filename}: {exc}")
        return "Skipped"

    # save file
    save_name = f"XC{xc_id}.ogg"
    save_path = os.path.join(OUTPUT_DIR, save_name)

    # Resume Logic
    if os.path.exists(save_path):
        return "Exists"

    # Download
    time.sleep(random.uniform(0.5, 1.0))
    try:
        url = f"https://xeno-canto.org/{xc_id}/download"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            # Convert to ogg before saving.
            audio = AudioSegment.from_file(BytesIO(r.content))
            audio.export(save_path, format="ogg")

            # --- CRITICAL STEP ---
            # We save the OFFICIAL Hugging Face metadata, but update the 'file_name'
            # to point to our local audio. This bridges the gap.
            meta_entry = row.to_dict()
            meta_entry['file_name'] = save_name
            return json.dumps(meta_entry)

        if r.status_code == 429:
            log_failure(f"rate limit for XC{xc_id}")
            return "RateLimit"
        log_failure(f"http_{r.status_code} for XC{xc_id}")
    except Exception as exc:
        log_failure(f"failed XC{xc_id}: {exc}")
    return "Failed"

# print(f"Downloading {len(target_df)} files...")
# with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#     futures = {executor.submit(download_hybrid, r): r for r in target_df.iterrows()}
#     for future in tqdm(as_completed(futures), total=len(target_df)):
#         res = future.result()

#         if res == "RateLimit":
#             time.sleep(10)
