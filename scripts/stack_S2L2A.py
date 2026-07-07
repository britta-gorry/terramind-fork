#!/usr/bin/env python3
"""
stack_S2L2A.py
==============
Stack Sentinel-2 L2A bands from a .SAFE folder into a single 12-band GeoTIFF
that matches TerraMind's expected S2L2A band order exactly.

TerraMind S2L2A band order (12 bands, B10 excluded):
PRETRAINED_BANDS = {
    'untok_sen2l2a@224': [
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR",
        "SWIR_1",
        "SWIR_2",
    ],}


EXCLUDED: B10 (Cirrus, 60m)
These are not in the TerraMind S2L2A tokenizer.

Usage:
    pixi run python scripts/stack_S2L2A.py \
        --safe_dir  data/S2B_MSIL2A_....SAFE \
        --output    data/stacked/my_scene.tif

    # Process multiple SAFE folders at once:
    pixi run python scripts/stack_S2L2A.py \
        --safe_dir  data/scene1.SAFE data/scene2.SAFE \
        --output    data/stacked/
"""
from __future__ import annotations

import torch
from terratorch import FULL_MODEL_REGISTRY
from terratorch.models.backbones.terramind.model.terramind_register import v1_pretraining_mean, v1_pretraining_std

import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
import rioxarray

# ──────────────────────────────────────────────────────────────────────────────
# TERRAMIND S2L2A BAND SPECIFICATION
# Exactly 12 bands, in this order, excluding B01 and B10.
# ──────────────────────────────────────────────────────────────────────────────
S2L2A_BANDS = [
    ("B01", "R60m", "*_B01_60m.jp2"),   # 0: Coastal Aerosol
    ("B02", "R10m", "*_B02_10m.jp2"),   # 1: Blue
    ("B03", "R10m", "*_B03_10m.jp2"),   # 2: Green
    ("B04", "R10m", "*_B04_10m.jp2"),   # 3: Red
    ("B05", "R20m", "*_B05_20m.jp2"),   # 4: Red Edge 1
    ("B06", "R20m", "*_B06_20m.jp2"),   # 5: Red Edge 2
    ("B07", "R20m", "*_B07_20m.jp2"),   # 6: Red Edge 3
    ("B08", "R10m", "*_B08_10m.jp2"),   # 7: NIR Broad
    ("B8A", "R20m", "*_B8A_20m.jp2"),   # 8: NIR Narrow
    ("B09", "R60m", "*_B09_60m.jp2"),   # 9: Water Vapour
    ("B11", "R20m", "*_B11_20m.jp2"),   # 10: SWIR 1
    ("B12", "R20m", "*_B12_20m.jp2"),   # 11: SWIR 2
]

# TerraMind's published normalisation stats for S2L2A (z-score: subtract mean, divide std)
# These are in units of ESA reflectance (0–10000 scale).
# TerraMind applies these internally when standardize=True, but we print them
# so you can verify your pixel value range is in the right ballpark.
# S2L2A_MEAN = torch.tensor([1390.458, 1503.317, 1718.197, 1853.91,  2199.1,
#                             2779.975, 2987.011, 3083.234, 3132.22,  3162.988,
#                             2424.884, 1857.648], dtype=torch.float32)
# S2L2A_STD  = torch.tensor([2106.761, 2141.107, 2038.973, 2134.138, 2085.321,
#                             1889.926, 1820.257, 1871.918, 1753.829, 1797.379,
#                             1434.261, 1334.311], dtype=torch.float32)
S2L2A_MEAN = torch.tensor(v1_pretraining_mean['untok_sen2l2a@224'])
S2L2A_STD = torch.tensor(v1_pretraining_std['untok_sen2l2a@224'])

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
def _log(msg): logging.getLogger().info(msg)


# ──────────────────────────────────────────────────────────────────────────────
# FIND GRANULE
# ──────────────────────────────────────────────────────────────────────────────
def find_granule_img_dir(safe_dir: Path) -> Path:
    """
    Find the IMG_DATA directory inside a .SAFE folder.
    SAFE folder structure:
      <name>.SAFE/GRANULE/<granule_id>/IMG_DATA/R10m/ R20m/ R60m/
    """
    hits = sorted(safe_dir.glob("GRANULE/*/IMG_DATA"))
    if not hits:
        _log(f"  ERROR: No GRANULE/*/IMG_DATA found in {safe_dir}")
        sys.exit(1)
    if len(hits) > 1:
        _log(f"  WARNING: Multiple granules found, using first: {hits[0]}")
    return hits[0]


def find_band_path(img_data_dir: Path, res_subdir: str, glob_suffix: str, band_name: str) -> Path:
    """Find the JP2 file for a single band."""
    pattern = str(img_data_dir / res_subdir / glob_suffix)
    hits = glob.glob(pattern)
    if not hits:
        _log(f"  ERROR: Could not find {band_name} at {pattern}")
        sys.exit(1)
    return Path(hits[0])


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATE PIXEL VALUES
# ──────────────────────────────────────────────────────────────────────────────
def validate_pixel_values(stack: np.ndarray, band_names: list[str]):
    """
    Print per-band statistics so you can verify values are in the expected
    ESA reflectance range (roughly 0-10000, occasionally up to ~15000 for snow).
    TerraMind normalisation stats are computed on this scale.
    """
    _log("\n  Band value statistics (expect roughly 0-10000 for valid S2L2A):")
    _log(f"  {'Band':<6}  {'Min':>8}  {'P5':>8}  {'Median':>8}  {'P95':>8}  {'Max':>8}  {'Status'}")
    _log("  " + "─" * 68)

    all_ok = True
    for i, bname in enumerate(band_names):
        band = stack[i].astype(np.float32)
        # Mask nodata (0 is common nodata in S2)
        valid = band[band > 0]
        if len(valid) == 0:
            _log(f"  {bname:<6}  WARNING: all zero or nodata")
            all_ok = False
            continue
        p5   = float(np.percentile(valid, 5))
        p50  = float(np.percentile(valid, 50))
        p95  = float(np.percentile(valid, 95))
        vmin = float(valid.min())
        vmax = float(valid.max())

        # Flag bands where values look wrong
        if vmax < 100:
            status = "⚠ values too low — may be 0-1 scale already"
            all_ok = False
        elif vmax > 50000:
            status = "⚠ values very high — check for scaling issues"
            all_ok = False
        elif p95 < 100:
            status = "⚠ 95th percentile too low"
            all_ok = False
        else:
            status = "✓"

        _log(f"  {bname:<6}  {vmin:>8.0f}  {p5:>8.0f}  {p50:>8.0f}  {p95:>8.0f}  {vmax:>8.0f}  {status}")

    if all_ok:
        _log("  All bands look correct for TerraMind S2L2A input.")
    else:
        _log("  ⚠ Some bands look suspicious. Review before running generation/reconstruction.")

    # Compare to TerraMind mean values
    _log("\n  Comparison to TerraMind normalisation means (z-score sanity check):")
    _log(f"  {'Band':<6}  {'Your Median':>12}  {'TerraMind Mean':>15}  {'Ratio':>8}")
    _log("  " + "─" * 50)
    for i, bname in enumerate(band_names):
        band = stack[i].astype(np.float32)
        valid = band[band > 0]
        if len(valid) == 0:
            continue
        median = float(np.percentile(valid, 50))
        tm_mean = S2L2A_MEAN[i]
        ratio = median / tm_mean if tm_mean != 0 else float("nan")
        flag = "✓" if 0.3 < ratio < 3.0 else "⚠ large deviation"
        _log(f"  {bname:<6}  {median:>12.0f}  {tm_mean:>15.1f}  {ratio:>8.2f}  {flag}")


# ──────────────────────────────────────────────────────────────────────────────
# STACK ONE SAFE FOLDER
# ──────────────────────────────────────────────────────────────────────────────
def stack_safe(safe_dir: Path, output_path: Path):
    """
    Stack one .SAFE folder into a 12-band GeoTIFF aligned to 10m resolution.
    """
    _log(f"\n{'='*60}")
    _log(f"  Processing: {safe_dir.name}")
    _log(f"  Output:     {output_path}")

    img_data_dir = find_granule_img_dir(safe_dir)
    _log(f"  IMG_DATA:   {img_data_dir}")

    # Find all band paths in the correct TerraMind order
    band_paths = []
    band_names = []
    for band_name, res_subdir, glob_suffix in S2L2A_BANDS:
        p = find_band_path(img_data_dir, res_subdir, glob_suffix, band_name)
        band_paths.append(p)
        band_names.append(band_name)
        _log(f"  Found {band_name}: {p.name}")

    # Use B02 (10m) as the reference grid — all bands reproject to match this
    ref_path = band_paths[0]   # B02 is index 0
    ref = rioxarray.open_rasterio(ref_path, masked=True)
    target_shape = ref.rio.shape   # (H, W) at 10m
    target_crs   = ref.rio.crs
    target_transform = ref.rio.transform()
    H, W = target_shape
    _log(f"\n  Reference grid (B02 at 10m): {W} x {H} pixels, CRS={target_crs}")

    # Load, reproject, and collect each band
    stacked = []
    for i, (path, bname) in enumerate(zip(band_paths, band_names)):
        band_da = rioxarray.open_rasterio(path, masked=True)
        if band_da.rio.shape != target_shape:
            # Reproject to match 10m grid using bilinear resampling
            # (bilinear is better than nearest-neighbour for continuous spectral values)
            band_da = band_da.rio.reproject_match(ref, resampling=Resampling.bilinear)
        # Extract as numpy [H, W]
        arr = band_da.values.squeeze().astype(np.float32)
        stacked.append(arr)
        _log(f"  Stacked {bname}  shape={arr.shape}  range=[{arr.min():.0f}, {arr.max():.0f}]")

    stack_array = np.stack(stacked, axis=0)  # [12, H, W]
    _log(f"\n  Final stack shape: {stack_array.shape}  (expected [12, H, W])")

    # Validate pixel values
    validate_pixel_values(stack_array, band_names)

    # Write output GeoTIFF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=12,
        dtype="float32",
        crs=target_crs,
        transform=target_transform,
        compress="lzw",          # lossless compression, smaller file
    ) as dst:
        dst.write(stack_array)
        # Write band names as metadata so you can always check
        for i, bname in enumerate(band_names):
            dst.update_tags(i + 1, band_name=bname)

    size_mb = output_path.stat().st_size / 1e6
    _log(f"\n  ✓ Saved: {output_path}  ({size_mb:.1f} MB)")
    _log(f"  Band order in file:")
    for i, bname in enumerate(band_names):
        _log(f"    Band {i+1}: {bname}")

    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Stack Sentinel-2 L2A bands from .SAFE folders in TerraMind order.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--safe_dir", type=Path, nargs="+", required=True,
        help="One or more .SAFE directory paths to process.")
    parser.add_argument("--output", type=Path, required=True,
        help="Output path. If processing multiple SAFE folders, provide a directory. "
             "If processing one, provide a full .tif filename.")
    args = parser.parse_args()

    safe_dirs = args.safe_dir
    output    = args.output

    for safe_dir in safe_dirs:
        if not safe_dir.exists():
            _log(f"ERROR: .SAFE directory not found: {safe_dir}")
            sys.exit(1)

        # Determine output path
        if len(safe_dirs) > 1 or output.suffix != ".tif":
            # output is a directory — derive filename from SAFE folder name
            output.mkdir(parents=True, exist_ok=True)
            stem = safe_dir.stem.replace(".SAFE", "")
            out_path = output / f"{stem}.tif"
        else:
            out_path = output

        stack_safe(safe_dir, out_path)

    _log("\nAll done.")


if __name__ == "__main__":
    main()
