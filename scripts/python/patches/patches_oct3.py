### patches.py


### input: paths: linux path for rawTIff and ilastikMasks
### parameters: rawTiff, ilastikMasks
### logic: split up each image pair (rawTiff, ilastikMasks) into a separate patches
# 
# return: folder with patches


### return folder shape:
### ret_folder/
###     PARENT_IMG/ (e.g., /1uM_FiNO2_24H/tiff/14_IMG0004_24h_1uM(_)FINO2_PKMO_PKMO.Confocal.ome.tif)
###         patch_0.tiff (2 channels, 1st channel is rawTiff, 2nd channel is ilastikMasks)
###                 (X, Y, 2)


### desired patch amount: 
## patches = 6 (only even)
## 
#!/usr/bin/env python3


"""
patches.py  (mitoEnd2End)

Split each (rawTiff, ilastikMask) image pair into N patches and save each patch
as a 2-channel TIFF:

    ret_folder/
        <PARENT_IMG>/            raw tif name without ".ome.tif", e.g.
                                 14_IMG0004_24h_1uM(_)FINO2_PKMO_PKMO.Confocal
            patch_0.tiff   (H, W, 2) uint16
                           [..., 0] = raw TIFF intensities
                           [..., 1] = ilastik probability for MASK_CLASS (0..65535 = 0..1)
            patch_1.tiff
            ...
        patches_index.csv   which source image / grid cell each patch came from

With --all, there is one more level: ret_folder/<dataset>/<PARENT_IMG>/patch_i.tiff

Data layout this expects (see repo README):
    data/<molarity_chemical_time>/
        Ilastik/<N>_<IMG#>.h5                 dataset "exported_data", (H, W, 2) uint16
        tiff/<N>_IMG<IMG#>_....ome.tif        (or the .tif files directly in the folder)

Pairing rule (same as scripts/jupyter/mask_viewing_oct-3.ipynb):
    "1_IMG0028_250nM_....ome.tif"  <->  "Ilastik/1_0028.h5"

Ilastik class 0 is the mitochondria class (it correlates positively with the raw
signal), so it is the default mask_class.

Command line (positional: argv[1] argv[2] argv[3] [argv[4]] [argv[5]])
-----------------------------------------------------------------------
    python patches.py <rawTiff> <ilastikMasks> <ret_folder> [n_patches=6] [mask_class=0]
    python patches.py data/250nM_FiNO2_6H/tiff data/250nM_FiNO2_6H/Ilastik patches_out 6

    # every dataset under data/ at once
    python patches.py --all <data_root> <ret_folder> [n_patches=6] [mask_class=0]
    python patches.py --all data patches_out 6

<rawTiff> / <ilastikMasks> can be folders or a single .tif / .h5 file.

From Python
-----------
    from patches import make_patches, make_all_patches
    make_patches("data/250nM_FiNO2_6H/tiff", "data/250nM_FiNO2_6H/Ilastik",
                 "patches_out", n_patches=6)
    make_all_patches("data", "patches_out", n_patches=6)

Requires: numpy, tifffile, h5py   (pip install numpy tifffile h5py)
"""

import csv
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile

TIFF_EXTS = {".tif", ".tiff"}
H5_DATASET = "exported_data"
RAW_RE = re.compile(r"^(\d+)_IMG(\d+)", re.IGNORECASE)  # 1_IMG0028_...
MASK_RE = re.compile(r"^(\d+)_(\d+)$")                  # 1_0028


# ------------------------------------------------------------------ helpers
def grid_shape(n_patches: int) -> tuple[int, int]:
    """Most square grid (rows, cols) with rows * cols == n_patches. 6 -> (2, 3)."""
    if n_patches < 2 or n_patches % 2 != 0:
        raise ValueError(f"n_patches must be an even number >= 2, got {n_patches}")
    rows = int(np.sqrt(n_patches))
    while n_patches % rows != 0:
        rows -= 1
    return rows, n_patches // rows


def load_raw(path: Path) -> np.ndarray:
    img = np.squeeze(tifffile.imread(path))
    if img.ndim != 2:
        raise ValueError(f"{path.name}: expected one 2D plane, got shape {img.shape}")
    return img


def load_mask(path: Path, mask_class: int) -> np.ndarray:
    """Return one ilastik class as a 2D array (H, W)."""
    if path.suffix.lower() == ".h5":
        with h5py.File(path, "r") as f:
            data = f[H5_DATASET][...]
    else:  # mask exported as TIFF instead of h5
        data = tifffile.imread(path)
    data = np.squeeze(data)
    if data.ndim == 3:  # (H, W, classes)
        if mask_class >= data.shape[-1]:
            raise ValueError(f"{path.name}: mask_class {mask_class} out of range "
                             f"(file has {data.shape[-1]} classes)")
        data = data[..., mask_class]
    if data.ndim != 2:
        raise ValueError(f"{path.name}: unexpected mask shape {data.shape}")
    return data


def to_dtype(arr: np.ndarray, dtype) -> np.ndarray:
    """Cast mask to the raw dtype. Float probabilities (0..1) are rescaled."""
    dtype = np.dtype(dtype)
    if arr.dtype == dtype:
        return arr
    if np.issubdtype(arr.dtype, np.floating) and np.issubdtype(dtype, np.integer):
        return np.round(np.clip(arr, 0, 1) * np.iinfo(dtype).max).astype(dtype)
    return arr.astype(dtype)


def raw_key(path: Path):
    m = RAW_RE.match(path.name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def mask_key(path: Path):
    m = MASK_RE.match(path.stem)
    return (int(m.group(1)), int(m.group(2))) if m else None


def list_files(path: Path, exts) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
    raise FileNotFoundError(path)


def pair_images(raw_path: Path, mask_path: Path) -> list[tuple[Path, Path]]:
    if raw_path.is_file() and mask_path.is_file():
        return [(raw_path, mask_path)]

    raws = list_files(raw_path, TIFF_EXTS)
    masks = {mask_key(m): m for m in list_files(mask_path, {".h5"} | TIFF_EXTS)
             if mask_key(m)}

    pairs, missing = [], []
    for raw in raws:
        key = raw_key(raw)
        if key is None:
            print(f"  WARNING: can't parse '<N>_IMG<#>' from {raw.name} -> skipped",
                  file=sys.stderr)
        elif key in masks:
            pairs.append((raw, masks[key]))
        else:
            missing.append(raw.name)
    if missing:
        print(f"  WARNING: no mask for {missing}", file=sys.stderr)
    return pairs


def split_into_patches(raw, mask, rows, cols):
    """Yield (r, c, bounds, patch). Patch edges are spread evenly so the whole
    image is covered; sizes differ by at most 1 px when it doesn't divide exactly."""
    h, w = raw.shape
    ys = np.linspace(0, h, rows + 1, dtype=int)
    xs = np.linspace(0, w, cols + 1, dtype=int)
    for r in range(rows):
        for c in range(cols):
            y0, y1, x0, x1 = ys[r], ys[r + 1], xs[c], xs[c + 1]
            yield r, c, (y0, y1, x0, x1), np.stack(
                [raw[y0:y1, x0:x1], mask[y0:y1, x0:x1]], axis=-1)


# --------------------------------------------------------------- main logic
def make_patches(raw_tiff, ilastik_masks, out_dir, n_patches=6, mask_class=0) -> Path:
    rows, cols = grid_shape(n_patches)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs = pair_images(Path(raw_tiff), Path(ilastik_masks))
    if not pairs:
        raise RuntimeError(f"No (rawTiff, ilastikMask) pairs found in "
                           f"{raw_tiff} / {ilastik_masks}")

    total, index_rows = 0, []
    for raw_p, mask_p in pairs:
        raw = load_raw(raw_p)
        mask = load_mask(mask_p, mask_class)
        if raw.shape != mask.shape:
            print(f"  WARNING: shape mismatch {raw_p.name} {raw.shape} vs "
                  f"{mask_p.name} {mask.shape} -> skipped", file=sys.stderr)
            continue
        mask = to_dtype(mask, raw.dtype)

        # one subfolder per parent image, patch numbering restarts at 0
        img_dir = out / parent_folder_name(raw_p)
        img_dir.mkdir(parents=True, exist_ok=True)

        for i, (r, c, (y0, y1, x0, x1), patch) in enumerate(
                split_into_patches(raw, mask, rows, cols)):
            name = f"patch_{i}.tiff"
            tifffile.imwrite(img_dir / name, patch,
                             photometric="minisblack", planarconfig="contig")
            index_rows.append([f"{img_dir.name}/{name}", raw_p.name, mask_p.name,
                               r, c, y0, y1, x0, x1, y1 - y0, x1 - x0])
            total += 1

    with open(out / "patches_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch", "raw", "mask", "row", "col",
                    "y0", "y1", "x0", "x1", "height", "width"])
        w.writerows(index_rows)

    print(f"  {len(pairs)} pairs -> {total} patches in {out}")
    return out


def parent_folder_name(raw_path: Path) -> str:
    """'14_IMG0004_..._PKMO.Confocal.ome.tif' -> '14_IMG0004_..._PKMO.Confocal'"""
    name = raw_path.name
    for ext in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return raw_path.stem


def find_datasets(data_root: Path):
    """Yield (name, raw_dir, mask_dir) for every dataset folder that has Ilastik/."""
    for d in sorted(p for p in data_root.iterdir() if p.is_dir()):
        mask_dir = d / "Ilastik"
        if not mask_dir.is_dir():
            continue
        raw_dir = d / "tiff" if (d / "tiff").is_dir() else d  # some sets have no tiff/
        yield d.name, raw_dir, mask_dir


def make_all_patches(data_root, out_dir, n_patches=6, mask_class=0) -> Path:
    """Run make_patches on every dataset folder under data_root.
    Output: out_dir/<dataset>/<PARENT_IMG>/patch_i.tiff"""
    datasets = list(find_datasets(Path(data_root)))
    if not datasets:
        raise RuntimeError(f"No dataset folders with an Ilastik/ subfolder in {data_root}")
    for name, raw_dir, mask_dir in datasets:
        print(name)
        make_patches(raw_dir, mask_dir, Path(out_dir) / name, n_patches, mask_class)
    return Path(out_dir)


def main(argv=None):
    """Positional API (argv[1], argv[2], ...):

        python patches.py <rawTiff> <ilastikMasks> <ret_folder> [n_patches] [mask_class]
        python patches.py --all <data_root> <ret_folder> [n_patches] [mask_class]
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    usage = ("usage:\n"
             "  python patches.py <rawTiff> <ilastikMasks> <ret_folder> [n_patches] [mask_class]\n"
             "  python patches.py --all <data_root> <ret_folder> [n_patches] [mask_class]")
    if not argv or argv[0] in ("-h", "--help"):
        print(usage)
        return 0

    try:
        if argv[0] == "--all":
            if not 3 <= len(argv) <= 5:
                raise ValueError("--all needs <data_root> <ret_folder> [n_patches] [mask_class]")
            data_root, out = argv[1], argv[2]
            n = int(argv[3]) if len(argv) > 3 else 6
            cls = int(argv[4]) if len(argv) > 4 else 0
            make_all_patches(data_root, out, n, cls)
        else:
            if not 3 <= len(argv) <= 5:
                raise ValueError("need <rawTiff> <ilastikMasks> <ret_folder> [n_patches] [mask_class]")
            raw, masks, out = argv[0], argv[1], argv[2]
            n = int(argv[3]) if len(argv) > 3 else 6
            cls = int(argv[4]) if len(argv) > 4 else 0
            make_patches(raw, masks, out, n, cls)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}\n\n{usage}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())