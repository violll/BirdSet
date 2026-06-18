import pandas as pd
import requests
import os
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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

OUTPUT_DIR = "/home/gil/comp0173/BirdSet/data/DataS1_DT_train/mp3/"
MAX_WORKERS = 4 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load OFFICIAL Metadata
# This ensures our labels/splits are 100% consistent with BirdSet
df = pd.read_parquet(PARQUET_FILE)
target_df = df[df['ebird_code'].isin(TARGET_CLASSES)].copy()

# change the filepath in the parquet file to match the actual path
target_df["filepath"] = "/home/gil/comp0173/BirdSet/data/DataS1_DT_train/ogg/" + target_df["filepath"]

# other metadata adjustments to match DT training format
target_df["ebird_code"] = target_df["ebird_code"].apply(lambda c: TARGET_CLASSES.index(c) if (not pd.isna(c) and c in TARGET_CLASSES) else pd.NA)
target_df["ebird_code_multilabel"] = target_df["ebird_code_multilabel"].apply(lambda c: [TARGET_CLASSES.index(ebird_name) if (not pd.isna(ebird_name) and ebird_name in TARGET_CLASSES) else pd.NA for ebird_name in c])
target_df["audio"] = target_df["filepath"]

print(f"Syncing {len(target_df)} files from Official Metadata...")

# 2. Setup The Metadata Lock
meta_path = os.path.join(OUTPUT_DIR, "metadata.parquet")

def download_hybrid(row_tuple):
    _, row = row_tuple
    filename = row['filepath']
    
    # Extract ID for Xeno-Canto Source
    try:
        xc_id = filename.split("XC")[-1].split(".")[0]
        if not xc_id.isdigit(): return "Skipped"
    except: return "Skipped"

    # save file
    save_name = f"XC{xc_id}.mp3"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    
    # Resume Logic
    if os.path.exists(save_path): return "Exists"

    # Download
    time.sleep(random.uniform(0.5, 1.0))
    try:
        r = requests.get(f"https://xeno-canto.org/{xc_id}/download", timeout=10)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(r.content)
            
            # --- CRITICAL STEP ---
            # We save the OFFICIAL Hugging Face metadata, but update the 'file_name'
            # to point to our local audio. This bridges the gap.
            meta_entry = row.to_dict()
            meta_entry['file_name'] = save_name 
            return json.dumps(meta_entry)
            
        elif r.status_code == 429: return "RateLimit"
    except: pass
    return "Failed"

# 3. Run
print(f"Downloading {len(target_df)} files...")
with open(meta_path, "a") as meta_f:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_hybrid, r): r for r in target_df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(target_df)):
            res = future.result()
            if res and res not in ["Exists", "Failed", "Skipped", "RateLimit"]:
                meta_f.write(res + "\n")
                meta_f.flush()
            elif res == "RateLimit": time.sleep(10)