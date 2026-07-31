#!/usr/bin/env python
"""
validate_paper_results.py - NeurIPS-24 BirdSet validation iterator for RunPod.

Runs the validations in order, then evicts each dataset's download to keep disk
bounded across runs. Eviction deletes ONLY the dataset download; run OUTPUTS
(Hydra outputs/, checkpoints, metrics) are intentionally KEPT.

Why not HF_DATASETS_CACHE?  BirdSet's BaseDataModuleHF._load_data (and
DataS1DataModule._load_data) load data via
    load_dataset(path=hf_path, name=hf_name, cache_dir=<data_dir>)
where <data_dir> = PROJECT_ROOT/data_birdset/<hf_name>
(cf. birdset/datamodule/base_datamodule.py:269 and
 configs/paths/default.yaml:10).
It sets cache_dir explicitly, so redirecting the HF_DATASETS_CACHE env var
has NO effect. Eviction therefore targets data_birdset/<hf_name> directly.

Sequence:
  1. XCL base-model eval:  birdset/eval.py  experiment="birdset_neurips24/XCL/efficientnet.yaml"
       (hf_name = "XCL")
  2. DT finetune+eval on the 8 paper datasets:
      birdset/train.py  experiment="birdset_neurips24/<DS>/DT/efficientnet.yaml"
  for DS in HSN NBP NES PER POW SNE SSW UHH.
  Each stage: run -> rm -rf data_birdset/<hf_name> -> next.

Usage:
    python scripts/datas1/validate_paper_results.py
    python scripts/datas1/validate_paper_results.py --dry-run
    python scripts/datas1/validate_paper_results.py --datasets HSN,NBP
    python scripts/datas1/validate_paper_results.py --skip-xcl      # skip XCL baseline eval
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Repo root (script lives at <repo>/scripts/datas1/validate_paper_results.py).
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

# BirdSet downloads each dataset into data_birdset/<hf_name> via explicit
# cache_dir=<data_dir> in base_datamodule.py:269.
# This is the correct eviction target (NOT HF_DATASETS_CACHE).
DATASET_CACHE_ROOT = PROJECT_ROOT / "data_birdset"

EVAL_PY = PROJECT_ROOT / "birdset" / "eval.py"
TRAIN_PY = PROJECT_ROOT / "birdset" / "train.py"

# Eight seabird datasets from the BirdSet NeurIPS-24 paper;
# hf_name matches the dataset code for each (data_birdset/<hf_name>).
DT_DATASETS = ["HSN", "NBP", "NES", "PER", "POW", "SNE", "SSW", "UHH"]


def eviction_target(hf_name: str) -> Path:
    """data_birdset/<hf_name> directory that BirdSet downloads the dataset into."""
    return DATASET_CACHE_ROOT / hf_name


def run_stage(label, script, experiment, hf_name, dry_run=False):
    """Run one validation stage, then evict its dataset download.

    Disk cleanup deletes only the dataset download (data_birdset/<hf_name>);
    the run's HYDRA outputs/ and checkpoints are kept.
    Returns True if the stage succeeded (or dry-run).
    """
    cmd = [sys.executable, str(script), f"experiment={experiment}"]
    target = eviction_target(hf_name)
    print(f"\n{'=' * 72}")
    print(f"[{label}] {'[dry-run] ' if dry_run else ''}{' '.join(cmd)}")
    print(f"evict after run: {target}  (-> rm -rf; outputs kept)")
    print(f"{'=' * 72}")
    if dry_run:
        return True
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    shutil.rmtree(target, ignore_errors=True)  # evict THIS dataset download only
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser(
        description="Iterate NeurIPS-24 BirdSet validations with per-dataset eviction"
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands + eviction plan, run nothing")
    ap.add_argument("--datasets", default=",".join(DT_DATASETS),
                    help="comma-separated DT datasets (default: all 8)")
    ap.add_argument("--skip-xcl", action="store_true",
                    help="skip the XCL base-model eval stage (useful after it has already passed)")
    args = ap.parse_args()
    dt_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"dataset cache root = {DATASET_CACHE_ROOT}  "
          f"(per-dataset <hf_name> deleted after each run; outputs kept)")

    ok = True
    # 1) base-model eval (XCL) - cheap baseline, run first to de-risk DT runs.
    if not args.skip_xcl:
        if not run_stage("XCL_eval", EVAL_PY, "birdset_neurips24/XCL/efficientnet.yaml",
                         "XCL", args.dry_run):
            ok = False
            print("!! XCL base eval failed; continuing to DT stages")
    else:
        print("[skip-xcl] Skipping XCL base-model eval")

    # 2) DT finetune+eval on each dataset, evicting that dataset's download after.
    for ds in dt_datasets:
        if not run_stage(f"DT_{ds}", TRAIN_PY, f"birdset_neurips24/{ds}/DT/efficientnet.yaml",
                         ds, args.dry_run):
            ok = False
            print(f"!! DT {ds} failed; dataset evicted, continuing")

    print("\n✓ all validations done" if ok else "\n✗ one or more stages failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
