#!/usr/bin/env python
"""
data_prep.py - Unified DataS1 preparation pipeline (thin CLI dispatcher)

Consolidates: extract_datas1_classes.py, extract_xcl_train.py,
              build_datas1.py, validate_schema.py, extract_pretrained_model_weights.py

Stage implementations live in separate modules:
  common.py               -- shared constants, utilities, preflight
  stage_download_train.py -- XCL metadata filtering + xeno-canto download (xc mode) / HF cache (hf mode)
  stage_build.py          -- test audio clipping + HSN schema cast

Validation is not a pipeline stage -- run validate_schema.py standalone:
    python scripts/datas1/validate_schema.py

Usage:
    python scripts/datas1/data_prep.py --stages all
    python scripts/datas1/data_prep.py --stages download_train,build
    python scripts/datas1/data_prep.py --stages download_train --xcl-source xc
    python scripts/datas1/data_prep.py --stages download_train --xcl-source hf

--xcl-source: xc  = download audio from xeno-canto.org (default)
              hf  = use full XCL dataset from HF Hub cache (no per-file downloads)
"""
import argparse

from common import load_class_mapping, stage_preflight
from stage_download_train import stage_download_train
from stage_build import stage_build


STAGES = ["preflight", "download_train", "build"]


def main():
    parser = argparse.ArgumentParser(description="DataS1 preparation pipeline")
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages or 'all' (default: all)",
    )
    parser.add_argument(
        "--xcl-source",
        choices=["xc", "hf"],
        default="xc",
        help="XCL audio source: 'xc' = download from xeno-canto (default), "
             "'hf' = use full XCL from HF Hub cache (stub)",
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
        print(f"{'='*60}")

        if stage == "preflight":
            if not stage_preflight():
                raise SystemExit(1)
        elif stage == "download_train":
            target_classes, label2id, _ = load_class_mapping()
            stage_download_train(target_classes, label2id, xcl_source=args.xcl_source)
        elif stage == "build":
            target_classes, label2id, _ = load_class_mapping()
            stage_build(target_classes, label2id)


if __name__ == "__main__":
    main()
