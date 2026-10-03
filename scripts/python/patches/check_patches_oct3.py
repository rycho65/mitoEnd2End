#!/usr/bin/env python3
"""
check_patches.py: verify patches.py output against the original data.

For every image it checks that:
  - there is one folder per parent image with the expected number of patches
  - every patch is (H, W, 2)
  - stitching the patches back together gives exactly the raw TIFF and ilastik class

Usage:
    python check_patches.py --data data --out patches_out --patches 6
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="patches_out")
ap.add_argument("--patches", type=int, default=6)
ap.add_argument("--mask-class", type=int, default=0)
args = ap.parse_args()

failures, checked = 0, 0
for index_csv in sorted(Path(args.out).glob("*/patches_index.csv")):
    dataset = index_csv.parent.name
    rows = list(csv.DictReader(open(index_csv)))
    for raw_name in sorted({r["raw"] for r in rows}):
        group = [r for r in rows if r["raw"] == raw_name]
        try:
            raw = tifffile.imread(Path(args.data, dataset, "tiff", raw_name))
            with h5py.File(Path(args.data, dataset, "Ilastik", group[0]["mask"])) as f:
                mask = f["exported_data"][..., args.mask_class]

            assert len(group) == args.patches, (
                f"{len(group)} patches, expected {args.patches}"
            )
            folders = {r["patch"].split("/")[0] for r in group}
            assert len(folders) == 1, f"patches spread over folders {folders}"

            full = np.zeros(raw.shape + (2,), raw.dtype)
            for r in group:
                p = tifffile.imread(index_csv.parent / r["patch"])
                assert p.ndim == 3 and p.shape[-1] == 2, f"{r['patch']} shape {p.shape}"
                full[int(r["y0"]) : int(r["y1"]), int(r["x0"]) : int(r["x1"])] = p

            assert np.array_equal(full[..., 0], raw), "raw channel differs"
            assert np.array_equal(full[..., 1], mask), "mask channel differs"
            checked += 1
        except Exception as e:
            failures += 1
            print(f"FAIL {dataset}/{raw_name}: {e}")

if checked + failures == 0:
    sys.exit(
        f"No patches_index.csv found under {args.out}/*/ (did you run with --data?)"
    )
print(f"{checked} images OK, {failures} failed")
sys.exit(1 if failures else 0)
