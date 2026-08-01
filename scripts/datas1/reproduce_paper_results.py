#!/usr/bin/env python
"""
validate_paper_results.py - NeurIPS-24 BirdSet validation iterator

Runs XCL eval and all DT training

Sequence:
  1. XCL eval:  birdset/eval.py  experiment="birdset_neurips24/XCL/efficientnet.yaml"
  2. DT finetune+eval: birdset/train.py  experiment="birdset_neurips24/<DS>/DT/efficientnet.yaml"
                        for DS in HSN NBP NES PER POW SNE SSW UHH.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Repo root (script lives at <repo>/scripts/datas1/validate_paper_results.py)
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))

# Manually set HF_HOME
HF_HOME = REPO_ROOT / "data_birdset" / "hf_cache"
os.environ["HF_HOME"] = str(HF_HOME)

# Dataset cache set by BirdSet
DATASET_CACHE_ROOT = REPO_ROOT / "data_birdset"
BACKGROUND_NOISE_DIR = DATASET_CACHE_ROOT / "background_noise"

EVAL_PY = REPO_ROOT / "birdset" / "eval.py"
TRAIN_PY = REPO_ROOT / "birdset" / "train.py"

# Eight datasets from the BirdSet paper
# hf_name matches the dataset code for each (data_birdset/<hf_name>)
DT_DATASETS = ["HSN", "NBP", "NES", "PER", "POW", "SNE", "SSW", "UHH"]


def check_background_noise() -> bool:
    """Verify background noise files exist for augmentations; download if missing."""
    if BACKGROUND_NOISE_DIR.exists() and any(BACKGROUND_NOISE_DIR.iterdir()):
        n_files = sum(1 for _ in BACKGROUND_NOISE_DIR.iterdir())
        print(f"[OK]    Background noise: {BACKGROUND_NOISE_DIR} ({n_files} files)")
        return True

    # Auto-download if missing
    dl_script = REPO_ROOT / "resources" / "utils" / "download_background_noise.py"
    if not dl_script.exists():
        print(f"[FAIL]  Background noise missing and download script not found: {dl_script}")
        return False
    print(f"[WARN]  Background noise missing at {BACKGROUND_NOISE_DIR}")
    print(f"   Auto-downloading via {dl_script.name} ...")
    proc = subprocess.run([sys.executable, str(dl_script)], cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        print(f"[FAIL]  Background noise download failed (exit {proc.returncode})")
        return False
    n_files = sum(1 for _ in BACKGROUND_NOISE_DIR.iterdir()) if BACKGROUND_NOISE_DIR.exists() else 0
    print(f"[OK]    Background noise downloaded: {BACKGROUND_NOISE_DIR} ({n_files} files)")
    return True


def run_stage(label, script, experiment, dry_run=False):
    """Run one validation stage

    Returns True if the stage succeeded (or dry-run).
    """
    cmd = [sys.executable, str(script), f"experiment={experiment}"]
    print(f"\n{'=' * 72}")
    print(f"[{label}] {'[dry-run] ' if dry_run else ''}{' '.join(cmd)}")
    print(f"{'=' * 72}")
    if dry_run:
        return True
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser(
        description="Iterate NeurIPS-24 BirdSet validations"
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands, run nothing")
    ap.add_argument("--datasets", default=",".join(DT_DATASETS),
                    help="comma-separated DT datasets (default: all 8)")
    ap.add_argument("--skip-xcl", action="store_true",
                    help="skip the XCL base-model eval stage (useful after it has already passed)")
    ap.add_argument("--fail-fast", action="store_true",
                    help="abort on the first stage failure instead of continuing")
    args = ap.parse_args()
    dt_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    print(f"REPO_ROOT = {REPO_ROOT}")
    print(f"dataset cache root = {DATASET_CACHE_ROOT}")

    if not args.dry_run and not check_background_noise():
        print("\n[ERROR] Background noise required - aborting before any stage runs.")
        return 1

    ok = True

    # NOTE failures may have independent causes; continues by default
    # Override this behavior with --fail-fast to exit upon first failure

    # 1) base-model eval (XCL)
    if not args.skip_xcl:
        if not run_stage("XCL_eval", 
                         EVAL_PY, 
                         "birdset_neurips24/XCL/efficientnet.yaml",
                         args.dry_run):
            ok = False

            if args.fail_fast:
                print("[ERROR] --fail-fast: aborting")
                return 1
            else:
                print("[FAIL]  XCL base eval failed; continuing to DT stages")
    else:
        print("[INFO]  Skipping XCL base-model eval (--skip-xcl)")

    # 2) DT finetune+eval on each dataset
    for ds in dt_datasets:
        if not run_stage(f"DT_{ds}", 
                         TRAIN_PY, 
                         f"birdset_neurips24/{ds}/DT/efficientnet.yaml", 
                         args.dry_run):
            ok = False

            if args.fail_fast:
                print("[ERROR] --fail-fast: aborting")
                return 1
            else:
                print(f"[FAIL]  DT {ds} failed, continuing")

    print("\n[OK]    all validations done" if ok else "\n[FAIL]  one or more stages failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
