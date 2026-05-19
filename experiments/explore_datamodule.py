"""
explore_datamodule.py
=====================
Breaks open a TerraTorch GenericMultiModalDataModule and shows you exactly
what data looks like at every level:
  1. On-disk file structure
  2. A single raw .tif file (before any processing)
  3. A single batch from the dataloader (after normalisation and augmentation)
  4. A visual sanity-check image saved to disk

Usage (HPC interactive session or notebook):
  python explore_datamodule.py

Requires your conda environment: conda activate terramind
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # HPC-safe: no display needed
import matplotlib.pyplot as plt

# ── 1. CONFIGURE PATHS ──────────────────────────────────────────────────────
# Edit these to match your HPC paths
DATASET_ROOT = "sen1floods11_v1.1"          # folder containing data/ and splits/
OUTPUT_DIR   = "output/exploration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 2. ON-DISK STRUCTURE AUDIT ───────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: ON-DISK FILE STRUCTURE")
print("=" * 60)

modality_dirs = {
    "S2L1C":  os.path.join(DATASET_ROOT, "data", "S2L1CHand"),
    "S1GRD":  os.path.join(DATASET_ROOT, "data", "S1GRDHand"),
    "Labels": os.path.join(DATASET_ROOT, "data", "LabelHand"),
}

for name, path in modality_dirs.items():
    if not os.path.isdir(path):
        print(f"  [MISSING] {name}: {path}")
        continue
    files = sorted([f for f in os.listdir(path) if f.endswith(".tif")])
    print(f"\n  {name}  ({path})")
    print(f"    Total .tif files: {len(files)}")
    print(f"    First 3 files:    {files[:3]}")
    print(f"    Last  3 files:    {files[-3:]}")

# Check a split file
for split_name in ["flood_train_data.txt", "flood_valid_data.txt", "flood_test_data.txt"]:
    split_path = os.path.join(DATASET_ROOT, "splits", split_name)
    if os.path.isfile(split_path):
        with open(split_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"\n  Split '{split_name}': {len(lines)} scenes")
        print(f"    First entry: '{lines[0]}'")
    else:
        print(f"\n  [MISSING] {split_path}")

# ── 3. RAW .TIF INSPECTION (rasterio) ────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2: RAW .TIF FILE PROPERTIES (before any processing)")
print("=" * 60)

try:
    import rasterio

    # Pick the first scene from the train split
    split_path = os.path.join(DATASET_ROOT, "splits", "flood_train_data.txt")
    with open(split_path) as f:
        first_scene = f.readline().strip()

    tif_targets = {
        "S2L1C":  os.path.join(DATASET_ROOT, "data", "S2L1CHand",  f"{first_scene}_S2Hand.tif"),
        "S1GRD":  os.path.join(DATASET_ROOT, "data", "S1GRDHand",  f"{first_scene}_S1Hand.tif"),
        "Labels": os.path.join(DATASET_ROOT, "data", "LabelHand",  f"{first_scene}_LabelHand.tif"),
    }

    for name, tif_path in tif_targets.items():
        print(f"\n  [{name}]  {tif_path}")
        if not os.path.isfile(tif_path):
            # Try finding any matching file if name pattern differs
            folder = os.path.dirname(tif_path)
            candidates = [f for f in os.listdir(folder) if first_scene in f]
            if candidates:
                tif_path = os.path.join(folder, candidates[0])
                print(f"    (Using: {candidates[0]})")
            else:
                print(f"    [FILE NOT FOUND — check scene name in split file]")
                continue

        with rasterio.open(tif_path) as src:
            data = src.read()   # shape: (bands, height, width)
            print(f"    Shape (bands, H, W): {data.shape}")
            print(f"    Dtype:               {data.dtype}")
            print(f"    CRS:                 {src.crs}")
            print(f"    Pixel resolution:    {src.res} metres")
            print(f"    Nodata value:        {src.nodata}")
            print(f"    Value range:         [{data.min():.3f}, {data.max():.3f}]")
            if name != "Labels":
                print(f"    Per-band means:      {np.mean(data, axis=(1,2)).round(1).tolist()}")
            else:
                unique, counts = np.unique(data, return_counts=True)
                print(f"    Unique label values: {dict(zip(unique.tolist(), counts.tolist()))}")
                print(f"    (0=other/non-flood, 1=flood, -1=nodata/cloud)")

except ImportError:
    print("  rasterio not available — skipping raw .tif inspection")
except Exception as e:
    print(f"  Error during .tif inspection: {e}")

# ── 4. DATAMODULE BATCH INSPECTION ───────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3: DATAMODULE BATCH (after normalisation & transforms)")
print("=" * 60)

try:
    import torch
    from terratorch.datamodules import GenericMultiModalDataModule

    # ── Build the same datamodule as your YAML config ──
    # (Adjust batch_size=1 for quick exploration; num_workers=0 avoids fork issues
    #  in interactive sessions)
    datamodule = GenericMultiModalDataModule(
        task="segmentation",
        batch_size=2,           # small batch for inspection
        num_workers=0,          # 0 = run in main process (safer for interactive use)
        modalities=["S2L1C", "S1GRD"],
        rgb_modality="S2L1C",
        rgb_indices=[3, 2, 1],

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

        # No augmentation for exploration — just ToTensor
        train_transform=[
            {"class_path": "albumentations.pytorch.transforms.ToTensorV2"}
        ],
    )

    datamodule.setup("fit")

    # Pull a single batch from the train dataloader
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print("\n  Batch is a Python dict with the following keys and shapes:")
    print(f"  {'Key':<12}  {'Type':<25}  {'Shape':<30}  {'dtype':<12}  ['min', 'max']")
    print(f"  {'-'*90}")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            vmin = val[val != -1].min().item() if key == "label" else val.min().item()
            vmax = val.max().item()
            print(f"  {key:<12}  {str(type(val).__name__):<25}  {str(tuple(val.shape)):<30}  {str(val.dtype):<12}  [{vmin:.3f}, {vmax:.3f}]")
        else:
            print(f"  {key:<12}  {str(type(val))}")

    # ── Per-band statistics for one image in the batch ──
    print("\n  Per-band statistics for sample [0] in this batch:")
    for modality in ["S2L1C", "S1GRD"]:
        if modality not in batch:
            continue
        t = batch[modality][0]   # shape: (C, H, W)
        print(f"\n  {modality} — shape {tuple(t.shape)}")
        print(f"  {'Band':<6}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
        for b in range(t.shape[0]):
            bdata = t[b]
            print(f"  {b:<6}  {bdata.mean().item():>8.4f}  {bdata.std().item():>8.4f}  "
                  f"{bdata.min().item():>8.4f}  {bdata.max().item():>8.4f}")

    # ── Label summary ──
    if "label" in batch:
        lbl = batch["label"][0]
        print(f"\n  Label — shape {tuple(lbl.shape)}, dtype {lbl.dtype}")
        for val_id, name in [(-1, "nodata/cloud"), (0, "other/non-flood"), (1, "flood")]:
            count = (lbl == val_id).sum().item()
            pct = 100 * count / lbl.numel()
            print(f"    value {val_id:>2} ({name:<16}): {count:>7} pixels  ({pct:.1f}%)")

    # ── 5. VISUALISATION ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SECTION 4: VISUALISATION")
    print("=" * 60)

    # Show: S2 false-colour RGB | S1 VV band | label mask
    s2   = batch["S2L1C"][0]   # (13, H, W)  — normalised
    s1   = batch["S1GRD"][0]   # (2,  H, W)  — normalised
    lbl  = batch["label"][0]   # (H, W)

    # S2 false-colour: bands 3,2,1 (R,G,B in L1C band ordering)
    rgb = s2[[3, 2, 1]].permute(1, 2, 0).numpy()
    # Clip and rescale for display (normalised values can be negative)
    rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6), 0, 1)

    # S1 VV band
    vv = s1[0].numpy()
    vv_display = np.clip((vv - vv.min()) / (vv.max() - vv.min() + 1e-6), 0, 1)

    # Label: map -1→NaN for display
    lbl_np = lbl.float().numpy()
    lbl_display = np.where(lbl_np == -1, np.nan, lbl_np)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Sen1Floods11 — single training sample (post-normalisation)", fontsize=13)

    axes[0].imshow(rgb)
    axes[0].set_title("S2L1C false-colour (bands 3-2-1)")
    axes[0].axis("off")

    im1 = axes[1].imshow(vv_display, cmap="gray")
    axes[1].set_title("S1GRD — VV band (normalised)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    cmap = plt.cm.get_cmap("RdYlBu_r", 2)
    im2 = axes[2].imshow(lbl_display, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title("Label  (0=other, 1=flood,  grey=nodata)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, ticks=[0, 1])

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "sample_batch_visualisation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Visualisation saved to: {out_path}")

except Exception as e:
    import traceback
    print(f"\n  Error during datamodule inspection: {e}")
    traceback.print_exc()

# ── 6. DATASET STRUCTURE SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5: DATASET STRUCTURE COMPARISON (for reference)")
print("=" * 60)

summary = """
  DATASET          TASK          LAYOUT                           LABEL FORMAT
  ─────────────────────────────────────────────────────────────────────────────
  Sen1Floods11     Segmentation  data/<MODALITY>Hand/             Separate LabelHand/ folder
                                 splits/flood_*.txt               Value: 0/1/-1 (binary)
                                 Multimodal: S2L1C (13ch) +
                                             S1GRD (2ch)

  BurnScars        Segmentation  data/ (flat)                     *_mask.tif alongside image
                   (binary)      No split files — you split       Value: 0=background, 1=burn
                                 programmatically                 Unimodal: HLS 6-band

  EuroSAT          Classification  data/<ClassName>/              No mask — folder name IS label
                                 No split files                   Unimodal: S2 13-band
                                 (class-folder layout)

  Sen4Map          Segmentation  HDF5 files (not .tif)            Multi-temporal monthly stacks
  (monthly)        or regression Monthly composites per pixel     Access via h5py or terratorch
                                 Very different loader needed     loader, not GenericMultiModal
  ─────────────────────────────────────────────────────────────────────────────

  YOUR FUTURE DRONE DATA (Antarctic MSI):
    - Will likely arrive as GeoTIFF stacks (n_bands, H, W) at ~2-10cm GSD
    - You will need to write a split .txt file (or CSV) of tile filenames
    - Labels will be hand-annotated masks (same pixel layout as image)
    - The GenericMultiModalDataModule OR a custom TerraTorch Dataset class
      can handle this — whichever is less YAML-config effort
"""
print(summary)

print("=" * 60)
print("Exploration complete.")
print("=" * 60)
