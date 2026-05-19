"""
explore_datamodule.py
=====================
Breaks open a TerraTorch GenericMultiModalDataModule and shows you exactly
what data looks like at every level:
  1. On-disk file structure (dynamically probed)
  2. A single raw .tif file (before any processing)
  3. A single batch from the dataloader (after normalisation)
  4. A visual sanity-check image saved to disk

Usage:
  pixi run python explore_datamodule.py

Key fix vs previous version:
  train_transform must be actual albumentations objects, NOT dicts with
  class_path keys. Dicts are YAML-only syntax — TerraTorch's CLI resolves
  them for you. When calling Python directly, pass instances.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # HPC-safe: no display needed
import matplotlib.pyplot as plt

# ── CONFIGURE PATHS ──────────────────────────────────────────────────────────
DATASET_ROOT = "sen1floods11_v1.1"   # folder containing data/ and splits/
OUTPUT_DIR   = "output/exploration"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: ON-DISK FILE STRUCTURE (dynamically probed)
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: ON-DISK FILE STRUCTURE")
print("=" * 60)

data_root   = os.path.join(DATASET_ROOT, "data")
splits_root = os.path.join(DATASET_ROOT, "splits")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("  (rasterio not available — file metadata will be skipped)")

# Discover all subdirectories under data/
subdirs = []
if os.path.isdir(data_root):
    subdirs = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])
    print(f"\n  Subdirectories found under {data_root}/")
    for subdir in subdirs:
        subpath  = os.path.join(data_root, subdir)
        tif_files = sorted([f for f in os.listdir(subpath) if f.endswith(".tif")])
        print(f"\n    [{subdir}]  ({len(tif_files)} .tif files)")
        if tif_files and HAS_RASTERIO:
            with rasterio.open(os.path.join(subpath, tif_files[0])) as src:
                print(f"      Example file:  {tif_files[0]}")
                print(f"      Shape (B,H,W): ({src.count}, {src.height}, {src.width})")
                print(f"      Dtype:         {src.dtypes[0]}")
                print(f"      Resolution:    {src.res} metres")
                print(f"      CRS:           {src.crs}")
        elif tif_files:
            print(f"      Example file:  {tif_files[0]}")
else:
    print(f"  [NOT FOUND] {data_root}")

# Discover split files
if os.path.isdir(splits_root):
    split_files = sorted([f for f in os.listdir(splits_root) if f.endswith(".txt")])
    print(f"\n  Split files found under {splits_root}/")
    for sf in split_files:
        with open(os.path.join(splits_root, sf)) as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"    {sf}: {len(lines)} scenes  "
              f"(first: '{lines[0]}', last: '{lines[-1]}')")
else:
    print(f"  [NOT FOUND] {splits_root}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: RAW .TIF FILE PROPERTIES (rasterio, before any processing)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: RAW .TIF FILE PROPERTIES (before any processing)")
print("=" * 60)

if HAS_RASTERIO and subdirs:
    try:
        split_path = os.path.join(splits_root, "flood_train_data.txt")
        with open(split_path) as f:
            first_scene = f.readline().strip()
        print(f"\n  Using first training scene: '{first_scene}'")

        for subdir in subdirs:
            subpath = os.path.join(data_root, subdir)
            candidates = sorted([
                fn for fn in os.listdir(subpath)
                if first_scene in fn and fn.endswith(".tif")
            ])
            if not candidates:
                print(f"\n  [{subdir}]  No file found matching '{first_scene}'")
                continue

            tif_path = os.path.join(subpath, candidates[0])
            print(f"\n  [{subdir}]  {candidates[0]}")
            with rasterio.open(tif_path) as src:
                data = src.read()
                print(f"    Shape (bands, H, W):  {data.shape}")
                print(f"    Dtype:                {data.dtype}")
                print(f"    Nodata value:         {src.nodata}")
                print(f"    Value range:          [{data.min():.2f}, {data.max():.2f}]")
                # If single-band small integers, it's a label mask
                if data.shape[0] == 1 and data.max() <= 10:
                    unique, counts = np.unique(data, return_counts=True)
                    print(f"    Unique values:        {dict(zip(unique.tolist(), counts.tolist()))}")
                    print(f"    (typical: 0=background, 1=class, -1=nodata)")
                else:
                    print(f"    Per-band means:       "
                          f"{np.mean(data, axis=(1, 2)).round(1).tolist()}")
    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()
else:
    print("  Skipped (rasterio unavailable or no data subdirs found)")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: DATAMODULE BATCH (after normalisation)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: DATAMODULE BATCH (after normalisation & transforms)")
print("=" * 60)

# ── KEY FIX ─────────────────────────────────────────────────────────────────
# When using the YAML CLI, transform entries like:
#   - class_path: albumentations.pytorch.transforms.ToTensorV2
# are resolved automatically by TerraTorch's YAML parser.
#
# When calling Python directly you MUST pass actual albumentations objects.
# Passing a dict causes: "dict object has no attribute available_keys".
# ────────────────────────────────────────────────────────────────────────────
import albumentations as A
from albumentations.pytorch import ToTensorV2

explore_transform = A.Compose(
    [ToTensorV2()],
    is_check_shapes=False,   # needed for multi-band remote sensing data
)

try:
    import torch
    from terratorch.datamodules import GenericMultiModalDataModule

    datamodule = GenericMultiModalDataModule(
        task="segmentation",
        batch_size=2,        # small batch for inspection
        num_workers=0,       # 0 = main process (safer for interactive sessions)
        modalities=["S2L1C", "S1GRD"],
        # Note: rgb_modality is deprecated. Use dict form for rgb_indices:
        rgb_indices={"S2L1C": [3, 2, 1]},

        train_data_root={
            "S2L1C": f"{DATASET_ROOT}/data/S2L1CHand",
            "S1GRD":  f"{DATASET_ROOT}/data/S1GRDHand",
        },
        train_label_data_root=f"{DATASET_ROOT}/data/LabelHand",
        val_data_root={
            "S2L1C": f"{DATASET_ROOT}/data/S2L1CHand",
            "S1GRD":  f"{DATASET_ROOT}/data/S1GRDHand",
        },
        val_label_data_root=f"{DATASET_ROOT}/data/LabelHand",
        test_data_root={
            "S2L1C": f"{DATASET_ROOT}/data/S2L1CHand",
            "S1GRD":  f"{DATASET_ROOT}/data/S1GRDHand",
        },
        test_label_data_root=f"{DATASET_ROOT}/data/LabelHand",

        train_split=f"{DATASET_ROOT}/splits/flood_train_data.txt",
        val_split=f"{DATASET_ROOT}/splits/flood_valid_data.txt",
        test_split=f"{DATASET_ROOT}/splits/flood_test_data.txt",

        image_grep={"S2L1C": "*_S2Hand.tif", "S1GRD": "*_S1Hand.tif"},
        label_grep="*_LabelHand.tif",

        no_label_replace=-1,
        no_data_replace=0,
        num_classes=2,

        means={
            "S2L1C": [2357.089, 2137.385, 2018.788, 2082.986, 2295.651,
                      2854.537, 3122.849, 3040.560, 3306.481, 1473.847,
                      506.070,  2472.825, 1838.929],
            "S1GRD": [-12.599, -20.293],
        },
        stds={
            "S2L1C": [1624.683, 1675.806, 1557.708, 1833.702, 1823.738,
                      1733.977, 1732.131, 1679.732, 1727.260, 1024.687,
                      442.165,  1331.411, 1160.419],
            "S1GRD": [5.195, 5.890],
        },

        # Pass actual albumentations objects, not dicts
        train_transform=explore_transform,
    )

    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print("\n  Batch contents:")
    print(f"  {'Key':<12}  {'Shape':<30}  {'dtype':<12}  {'min':>10}  {'max':>10}")
    print(f"  {'-'*80}")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            valid = val[val != -1] if key == "label" else val
            vmin = valid.min().item() if valid.numel() > 0 else float("nan")
            vmax = val.max().item()
            print(f"  {key:<12}  {str(tuple(val.shape)):<30}  {str(val.dtype):<12}  "
                  f"{vmin:>10.4f}  {vmax:>10.4f}")
        else:
            print(f"  {key:<12}  {type(val)}")

    print("\n  Per-band statistics for sample [0]  "
          "(values are normalised — expect roughly -3 to +3):")
    for modality in ["S2L1C", "S1GRD"]:
        if modality not in batch:
            continue
        t = batch[modality][0]   # (C, H, W)
        print(f"\n  {modality} — {t.shape[0]} bands, "
              f"spatial size {t.shape[1]}×{t.shape[2]}")
        print(f"  {'Band':<6}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
        for b in range(t.shape[0]):
            bd = t[b]
            print(f"  {b:<6}  {bd.mean().item():>8.4f}  {bd.std().item():>8.4f}  "
                  f"{bd.min().item():>8.4f}  {bd.max().item():>8.4f}")

    if "label" in batch:
        lbl = batch["label"][0]
        print(f"\n  Label — shape {tuple(lbl.shape)}, dtype {lbl.dtype}")
        for val_id, name in [(-1, "nodata/cloud"), (0, "other"), (1, "flood")]:
            count = (lbl == val_id).sum().item()
            pct   = 100 * count / lbl.numel()
            print(f"    {val_id:>2}  ({name:<14}): {count:>7} pixels  ({pct:.1f}%)")


    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4: VISUALISATION
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SECTION 4: VISUALISATION")
    print("=" * 60)

    def norm_display(arr):
        lo, hi = arr.min(), arr.max()
        return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)

    s2  = batch["S2L1C"][0]   # (13, H, W) normalised
    s1  = batch["S1GRD"][0]   # ( 2, H, W) normalised
    lbl = batch["label"][0]   # (H, W)

    rgb         = norm_display(s2[[3, 2, 1]].permute(1, 2, 0).numpy())
    vv          = norm_display(s1[0].numpy())
    lbl_display = np.where(lbl.numpy() == -1, np.nan, lbl.float().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Sen1Floods11 — single training sample (post-normalisation)",
                 fontsize=13)

    axes[0].imshow(rgb)
    axes[0].set_title("S2L1C false-colour (bands 3-2-1)")
    axes[0].axis("off")

    im1 = axes[1].imshow(vv, cmap="gray")
    axes[1].set_title("S1GRD — VV band (normalised)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    cmap = matplotlib.colormaps.get_cmap("RdYlBu_r").resampled(2)
    im2  = axes[2].imshow(lbl_display, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title("Label  (0=other, 1=flood,  grey=nodata)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, ticks=[0, 1])

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "sample_batch_visualisation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out_path}")

except Exception as e:
    import traceback
    print(f"\n  Error: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Exploration complete.")
print("=" * 60)
