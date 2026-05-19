"""
terramind_inspect_infer.py
══════════════════════════════════════════════════════════════════════════════
TerraMind — Batch Inspection & Zero-Shot Inference
══════════════════════════════════════════════════════════════════════════════

PURPOSE
───────
This script does four things in one workflow:

  1. DATAMODULE  — instantiates the same TerraTorch DataModule that your YAML
                   configs use, so you can see the Python equivalent of every
                   YAML key.

  2. INSPECTION  — deeply prints shapes, dtypes, value ranges, and metadata at
                   every step of the pipeline (loading → normalisation →
                   augmentation → collation).  The output is designed so you
                   and your supervisors can look at it and answer questions like:
                     • Does my data need to be square?
                     • What band order does the model expect?
                     • How does the mask label encoding work?
                     • What do I need to change for HDF5, shapefiles, or
                       different band sets?

  3. MANUAL LOAD — builds the same batch tensor from raw .tif files yourself
                   (Johan's approach, extended), so you see exactly what the
                   DataModule does under the hood.  Both paths produce the same
                   shape and can be fed to the model.

  4. INFERENCE   — runs zero-shot segmentation with the pretrained TerraMind
                   backbone (no fine-tuned checkpoint required).  Results will
                   be poor without fine-tuning — that is expected.  The point
                   is to prove the pipeline is end-to-end correct and that your
                   data can be read.

OUTPUT
──────
  output/inspect_infer_<TIMESTAMP>/
    ├── inspection_log.txt          # full printed log saved to file
    ├── batch_sample_<i>.png        # RGB | GT mask | prediction (one per sample)
    └── manual_sample.png           # same plot for the manually-built tensor

HPC NOTES
─────────
  • matplotlib is set to "Agg" — no display required, safe on compute nodes
  • No GPU is required (inference will be slow on CPU; fine for inspection)
  • No SLURM job needed for a quick smoke-test; submit as a job for the full
    dataset

HOW TO RUN
──────────
  python terramind_inspect_infer.py

  To switch dataset, change DATASET below.
  To test without ground-truth labels, set WITH_GROUND_TRUTH = False.
──────────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — CONFIGURATION BLOCK
# Only change values in this block.  Nothing else in the script needs editing.
# ─────────────────────────────────────────────────────────────────────────────

# ── Dataset selection ────────────────────────────────────────────────────────
# "burnscars"    — HLS BurnScars (6-band, single modality, GenericNonGeo)
# "sen1floods11" — Sen1Floods11  (13+2 band, dual modality, GenericMultiModal)
# DATASET = "burnscars"
DATASET = "sen1floods11"
DATASET = "aspa135_sample"  # sample data from aspa135 altum drone imagery

# ── Ground-truth availability ────────────────────────────────────────────────
# True  → load mask labels, print mask stats, show GT vs prediction comparison
# False → skip labels entirely (use this when you have imagery but no masks)
WITH_GROUND_TRUTH = True

# ── How many samples to inspect and visualise ────────────────────────────────
N_SAMPLES = 1

# ── Model variant ────────────────────────────────────────────────────────────
# "terramind_v1_small"  — lighter, faster to load, good for smoke tests
# "terramind_v1_base"   — matches your YAML configs
BACKBONE = "terramind_v1_small"

# ── Data roots  ──────────────────────────────────────────────────────────────
# These mirror the paths in your YAML configs.
# Relative paths are relative to wherever you run the script from.

BURNSCARS_CFG = dict(
    data_root   = "hls_burn_scars/data",
    split_root  = "hls_burn_scars/splits",
    train_split = "hls_burn_scars/splits/train.txt",
    val_split   = "hls_burn_scars/splits/val.txt",
    test_split  = "hls_burn_scars/splits/test.txt",
    img_grep    = "*_merged.tif",
    label_grep  = "*.mask.tif",
    dataset_bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    rgb_indices = [2, 1, 0],          # RED, GREEN, BLUE (0-indexed)
    means = [0.0333, 0.0570, 0.0589, 0.2323, 0.1973, 0.1194],
    stds  = [0.0227, 0.0268, 0.0400, 0.0779, 0.0871, 0.0724],
    backbone_modality = "S2L2A",
    backbone_bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    num_classes = 2,
    class_names = ["Other", "Burn scar"],
)

SEN1FLOODS_CFG = dict(
    data_root_s2 = "sen1floods11_v1.1/data/S2L1CHand",
    data_root_s1 = "sen1floods11_v1.1/data/S1GRDHand",
    label_root   = "sen1floods11_v1.1/data/LabelHand",
    split_root   = "sen1floods11_v1.1/splits",
    train_split  = "sen1floods11_v1.1/splits/flood_train_data.txt",
    val_split    = "sen1floods11_v1.1/splits/flood_valid_data.txt",
    test_split   = "sen1floods11_v1.1/splits/flood_test_data.txt",
    img_grep_s2  = "*_S2Hand.tif",
    img_grep_s1  = "*_S1Hand.tif",
    label_grep   = "*_LabelHand.tif",
    rgb_indices  = {"S2L1C": [3, 2, 1]},
    means = dict(
        S2L1C=[2357.089, 2137.385, 2018.788, 2082.986, 2295.651,
               2854.537, 3122.849, 3040.560, 3306.481, 1473.847, 506.070,
               2472.825, 1838.929],
        S1GRD=[-12.599, -20.293],
    ),
    stds = dict(
        S2L1C=[1624.683, 1675.806, 1557.708, 1833.702, 1823.738,
               1733.977, 1732.131, 1679.732, 1727.26, 1024.687, 442.165,
               1331.411, 1160.419],
        S1GRD=[5.195, 5.890],
    ),
    num_classes = 2,
    class_names = ["Other", "Flood"],
)

ASPA135_CFG = dict(
    data_root   = "sample_data/aspa135-arthur-data",
    # split_root  = "aspa135_sample/splits",
    # train_split = "aspa135_sample/splits/train.txt",
    # val_split   = "aspa135_sample/splits/val.txt",
    test_split  = "sample_data/aspa135-arthur-data", # no split, just one sample
    # img_grep    = "*_merged.tif",
    # label_grep  = "*.mask.tif",
    # edited bands but unedited stats (simple test following BurnScars config)
    dataset_bands = ["BLUE", "GREEN", "RED", "RED_EDGE", "NIR", "THERMAL"],
    rgb_indices = [2, 1, 0],          # RED, GREEN, BLUE (0-indexed)
    means = [0.0333, 0.0570, 0.0589, 0.2323, 0.1973, 0.1194],
    stds  = [0.0227, 0.0268, 0.0400, 0.0779, 0.0871, 0.0724],
    backbone_modality = "S2L2A",
    backbone_bands = ["BLUE", "GREEN", "RED", "RED_EDGE", "NIR", "THERMAL"],
    num_classes = 2,
    class_names = ["Other", "Vegetation"],
)

# ── Number of segmentation classes ──────────────────────────────────────────
# BurnScars:    2  (Others / Burned)
# Sen1Floods11: 2  (Others / Flood)
# ASPA135_CFG: 2  (Others / Vegetation)
if DATASET == "burnscars":
    NUM_CLASSES = BURNSCARS_CFG["num_classes"]
    CLASS_NAMES = BURNSCARS_CFG["class_names"]
elif DATASET == "sen1floods11":
    NUM_CLASSES = SEN1FLOODS_CFG["num_classes"]
    CLASS_NAMES = SEN1FLOODS_CFG["class_names"]
elif DATASET == "aspa135_sample":
    NUM_CLASSES = ASPA135_CFG["num_classes"]
    CLASS_NAMES = ASPA135_CFG["class_names"]
# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_BASE = "output/inspect_infer"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SETUP
# Imports, output directory, log file, and a helper print function.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import datetime
import textwrap
import importlib

# HPC-safe backend — must be set BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import torch
import terratorch
import rasterio


# ── Create timestamped output directory ──────────────────────────────────────
_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"{OUTPUT_BASE}_{DATASET}_{_ts}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Tee: write everything to both stdout and a log file ──────────────────────
_log_path = os.path.join(OUTPUT_DIR, "inspection_log.txt")
_log_file = open(_log_path, "w")

class _Tee:
    """Mirrors stdout to a file simultaneously."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()

sys.stdout = _Tee(sys.__stdout__, _log_file)

def banner(title: str):
    """Print a prominent section header."""
    line = "═" * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")

def sub(title: str):
    """Print a sub-section header."""
    print(f"\n  ── {title} " + "─" * max(0, 60 - len(title)))

def inspect_tensor(name: str, t, indent: int = 4):
    """
    Print shape, dtype, min/max/mean/std for a numpy array or torch tensor.
    This is the core inspection helper — you will see it called everywhere.
    """
    pad = " " * indent
    try:
        arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    except Exception:
        arr = np.asarray(t)

    shape_str = str(tuple(arr.shape))
    dtype_str = str(arr.dtype)

    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        stats = (f"min={finite.min():.4f}  max={finite.max():.4f}  "
                 f"mean={finite.mean():.4f}  std={finite.std():.4f}")
    else:
        stats = "no finite values"

    nan_count  = int(np.isnan(arr).sum())
    inf_count  = int(np.isinf(arr).sum())
    flag = ""
    if nan_count or inf_count:
        flag = f"  ⚠️  NaN={nan_count} Inf={inf_count}"

    print(f"{pad}{name}")
    print(f"{pad}  shape : {shape_str}  dtype : {dtype_str}")
    print(f"{pad}  values: {stats}{flag}")


banner("TerraMind — Batch Inspection & Zero-Shot Inference")
print(f"  Dataset          : {DATASET}")
print(f"  Backbone         : {BACKBONE}")
print(f"  With ground truth: {WITH_GROUND_TRUTH}")
print(f"  Samples to show  : {N_SAMPLES}")
print(f"  Output directory : {OUTPUT_DIR}")
print(f"  Log file         : {_log_path}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — LIBRARY VERSIONS
# Print every key library version so HPC environment issues are easy to debug.
# ─────────────────────────────────────────────────────────────────────────────

banner("SECTION 2 — Library Versions & Hardware")

from importlib.metadata import version

_libs = [
    "torch", "torchvision", "lightning",
    "rasterio", "numpy", "albumentations", "matplotlib",
]

lib = "terratorch"
ver = version("terratorch")
print(f"  {lib:<18} {ver}")

for lib in _libs:
    try:
        m = importlib.import_module(lib)
        ver = getattr(m, "__version__", "?")
        print(f"  {lib:<18} {ver}")
    except ImportError:
        print(f"  {lib:<18} NOT INSTALLED")

print(f"\n  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device    : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version   : {torch.version.cuda}")
else:
    print("  Running on CPU (fine for inspection; slow for training)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATAMODULE (TerraTorch Python API)
#
# This mirrors exactly what your YAML config does — just in Python instead.
# Each keyword argument here corresponds to a key in the YAML init_args block.
# A comment next to every argument shows the matching YAML key.
#
# WHY THIS MATTERS
# ─────────────────
# Understanding this section means you know how to adapt the DataModule to:
#   • Different datasets (new data_root, split_root, img_grep)
#   • Different band sets (dataset_bands, means, stds)
#   • Different label formats (label_grep, no_label_replace)
#   • HDF5 or other file types (requires a custom Dataset class instead,
#     but the DataModule structure stays the same)
# ─────────────────────────────────────────────────────────────────────────────

banner("SECTION 3 — DataModule (Python API = YAML config in code)")

from terratorch.datamodules import (
    GenericNonGeoSegmentationDataModule,
    GenericMultiModalDataModule,
)

if DATASET == "burnscars":
    C = BURNSCARS_CFG
    sub("Instantiating GenericNonGeoSegmentationDataModule")
    print("""
  This is the same class used in your terramind_v1_base_burnscars.yaml.
  Each argument below maps directly to a key under data: → init_args: in the YAML.
    """)

    datamodule = GenericNonGeoSegmentationDataModule(
        # ── YAML: batch_size ──────────────────────────────────────────────────
        batch_size    = 4,          # small for inspection; 8 in training YAML

        # ── YAML: num_workers ─────────────────────────────────────────────────
        num_workers   = 0,          # 0 = no multiprocessing (safer for inspect)

        # ── YAML: dataset_bands ───────────────────────────────────────────────
        # Named spectral bands in the order they appear in the .tif file.
        # The model uses these names to select from pretrained band embeddings.
        dataset_bands = C["dataset_bands"],

        # ── YAML: rgb_indices ─────────────────────────────────────────────────
        # Which band indices (0-based) to use as R, G, B in preview images.
        rgb_indices   = C["rgb_indices"],

        # ── YAML: train/val/test_data_root ────────────────────────────────────
        train_data_root = C["data_root"],
        val_data_root   = C["data_root"],
        test_data_root  = C["data_root"],

        # ── YAML: train/val/test_split ────────────────────────────────────────
        # Text file: one image filename stem per line.
        train_split = C["train_split"],
        val_split   = C["val_split"],
        test_split  = C["test_split"],

        # ── YAML: img_grep ────────────────────────────────────────────────────
        img_grep    = C["img_grep"],

        # ── YAML: label_grep ──────────────────────────────────────────────────
        label_grep  = C["label_grep"],

        # ── YAML: means / stds ───────────────────────────────────────────────
        # Per-channel normalisation statistics (computed over the training set).
        # During __getitem__ TerraTorch applies: (pixel - mean) / std
        # These numbers must match the YAML — wrong stats = garbage features.
        means = C["means"],
        stds  = C["stds"],

        # ── YAML: no_data_replace / no_label_replace ──────────────────────────
        # Pixels with the no-data sentinel in the image → replaced with 0.
        # Pixels with the no-data sentinel in the mask  → replaced with -1.
        # -1 labels are ignored by the Dice/CE loss (ignore_index in YAML).
        no_data_replace  = 0,
        no_label_replace = -1,

        # ── YAML: num_classes ─────────────────────────────────────────────────
        num_classes = NUM_CLASSES,

        # ── No augmentation for inspection ───────────────────────────────────
        # The YAML training config uses albumentations.D4 (random 90° rotations
        # + flips).  We skip that here so images are in their original form.
        # train_transform = [D4, ToTensorV2]
        # val_transform   = [ToTensorV2]          ← default (no spatial aug)
    )

    sub("Setting up DataModule (reads split files, builds Dataset objects)")
    datamodule.setup("fit")   # "fit" prepares train + val datasets

    val_loader = datamodule.val_dataloader()
    print(f"\n  Val dataset size : {len(datamodule.val_dataset)} samples")
    print(f"  Batch size       : {datamodule.batch_size}")
    print(f"  Batches in val   : {len(val_loader)}")

elif DATASET == "sen1floods11":
    C = SEN1FLOODS_CFG
    sub("Instantiating GenericMultiModalDataModule")
    print("""
  This is the class used in terramind_v1_base_sen1floods11.yaml.
  It handles MULTIPLE image modalities in a single batch (S2L1C + S1GRD).
    """)

    datamodule = GenericMultiModalDataModule(
        task        = "segmentation",
        batch_size  = 4,
        num_workers = 0,
        modalities  = ["S2L1C", "S1GRD"],
        rgb_indices = C["rgb_indices"],

        train_data_root = {"S2L1C": C["data_root_s2"], "S1GRD": C["data_root_s1"]},
        val_data_root   = {"S2L1C": C["data_root_s2"], "S1GRD": C["data_root_s1"]},
        test_data_root  = {"S2L1C": C["data_root_s2"], "S1GRD": C["data_root_s1"]},
        train_label_data_root = C["label_root"],
        val_label_data_root   = C["label_root"],
        test_label_data_root  = C["label_root"],
        train_split = C["train_split"],
        val_split   = C["val_split"],
        test_split  = C["test_split"],
        image_grep  = {"S2L1C": C["img_grep_s2"], "S1GRD": C["img_grep_s1"]},
        label_grep  = C["label_grep"],
        means       = C["means"],
        stds        = C["stds"],
        no_data_replace  = 0,
        no_label_replace = -1,
        num_classes = NUM_CLASSES,
    )

    datamodule.setup("fit")
    val_loader = datamodule.val_dataloader()
    print(f"\n  Val dataset size : {len(datamodule.val_dataset)} samples")
    print(f"  Batches in val   : {len(val_loader)}")

elif DATASET == "aspa135_sample":
    C = ASPA135_CFG
    sub("Instantiating GenericNonGeoSegmentationDataModule")
    print("""
  This is the same class used in terramind_v1_base_burnscars.yaml, but slightly 
  adapted for ASPA135 sample data.
  Each argument below maps directly to a key under data: → init_args: in the YAML.
    """)

    datamodule = GenericNonGeoSegmentationDataModule(
        # ── YAML: batch_size ──────────────────────────────────────────────────
        batch_size    = 4,          # small for inspection; 8 in training YAML

        # ── YAML: num_workers ─────────────────────────────────────────────────
        num_workers   = 0,          # 0 = no multiprocessing (safer for inspect)

        # ── YAML: dataset_bands ───────────────────────────────────────────────
        # Named spectral bands in the order they appear in the .tif file.
        # The model uses these names to select from pretrained band embeddings.
        dataset_bands = C["dataset_bands"],

        # ── YAML: rgb_indices ─────────────────────────────────────────────────
        # Which band indices (0-based) to use as R, G, B in preview images.
        rgb_indices   = C["rgb_indices"],

        # ── YAML: train/val/test_data_root ────────────────────────────────────
        # train_data_root = C["data_root"],
        # val_data_root   = C["data_root"],
        test_data_root  = C["data_root"],

        # ── YAML: train/val/test_split ────────────────────────────────────────
        # Text file: one image filename stem per line.
        # train_split = C["train_split"],
        # val_split   = C["val_split"],
        test_split  = C["test_split"],

        # # ── YAML: img_grep ────────────────────────────────────────────────────
        # img_grep    = C["img_grep"],

        # # ── YAML: label_grep ──────────────────────────────────────────────────
        # label_grep  = C["label_grep"],

        # ── YAML: means / stds ───────────────────────────────────────────────
        # Per-channel normalisation statistics (computed over the training set).
        # During __getitem__ TerraTorch applies: (pixel - mean) / std
        # These numbers must match the YAML — wrong stats = garbage features.
        means = C["means"],
        stds  = C["stds"],

        # ── YAML: no_data_replace / no_label_replace ──────────────────────────
        # Pixels with the no-data sentinel in the image → replaced with 0.
        # Pixels with the no-data sentinel in the mask  → replaced with -1.
        # -1 labels are ignored by the Dice/CE loss (ignore_index in YAML).
        no_data_replace  = 0,
        no_label_replace = -1,

        # ── YAML: num_classes ─────────────────────────────────────────────────
        num_classes = NUM_CLASSES,

        # ── No augmentation for inspection ───────────────────────────────────
        # The YAML training config uses albumentations.D4 (random 90° rotations
        # + flips).  We skip that here so images are in their original form.
        # train_transform = [D4, ToTensorV2]
        # val_transform   = [ToTensorV2]          ← default (no spatial aug)
    )

    sub("Setting up DataModule (reads split files, builds Dataset objects)")
    datamodule.setup("fit")   # "fit" prepares train + val datasets

    val_loader = datamodule.val_dataloader()
    print(f"\n  Val dataset size : {len(datamodule.val_dataset)} samples")
    print(f"  Batch size       : {datamodule.batch_size}")
    print(f"  Batches in val   : {len(val_loader)}")
          

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — BATCH INSPECTION
#
# This is the "start inspecting from here" section mentioned by your advisor.
# We pull ONE batch and print everything about it.
#
# WHAT IS A BATCH?
# ─────────────────
# A batch is a Python dict with (at minimum):
#   "image"  → the normalised pixel data, ready to feed into the model
#   "mask"   → integer class labels (if WITH_GROUND_TRUTH)
#
# For GenericNonGeo  (BurnScars):
#   batch["image"]  is a dict {"S2L2A": Tensor[B, C, H, W]}
#
# For GenericMultiModal (Sen1Floods11):
#   batch["image"]  is a dict {"S2L1C": Tensor[B,13,H,W], "S1GRD": Tensor[B,2,H,W]}
#
# B = batch size, C = number of channels, H = height, W = width
# ─────────────────────────────────────────────────────────────────────────────

banner("SECTION 4 — Batch Inspection")

sub("Fetching one batch from the val DataLoader")
batch = next(iter(val_loader))

print(f"\n  Top-level batch keys: {list(batch.keys())}")
print(f"  Batch size (B)      : inferred from image tensor below")

# ── 4a. Image tensor(s) ──────────────────────────────────────────────────────
sub("batch['image'] — normalised pixel tensor(s)")

raw_image = batch["image"]

# Normalise to a consistent structure: always a dict {modality_name: tensor}
# GenericNonGeoSegmentationDataModule (BurnScars) returns a plain tensor.
# GenericMultiModalDataModule (Sen1Floods11) returns a dict of tensors.
# Everything downstream uses `images` (the dict form) so both paths work.

if isinstance(raw_image, torch.Tensor):
    # ── BurnScars / single-modality case ─────────────────────────────────────
    # batch["image"] is directly [B, C, H, W].
    # We wrap it in a dict so the rest of the script is consistent.
    if DATASET == "burnscars":
        modality_name = BURNSCARS_CFG["backbone_modality"]   # "S2L2A"
    elif DATASET == "sen1floods11":
        modality_name = "image"   # fallback label
    images = {modality_name: raw_image}
    print(f"""
  batch['image'] is a plain tensor (GenericNonGeoSegmentationDataModule).
  This is how BurnScars batches arrive — one tensor, no modality dict.
  Wrapped into dict {{'{modality_name}': tensor}} for consistent handling below.
    """)

elif isinstance(raw_image, torch.Tensor):
    # ── ASPA135 sample data case (also single modality) ────────────────────────
    modality_name = ASPA135_CFG["backbone_modality"]
    images = {modality_name: raw_image}
    print(f"""
  batch['image'] is a plain tensor (GenericNonGeoSegmentationDataModule).
  This is how BurnScars batches arrive — one tensor, no modality dict.
  Wrapped into dict {{'{modality_name}': tensor}} for consistent handling below.
    """)

elif isinstance(raw_image, dict):
    # ── Sen1Floods11 / multi-modality case ───────────────────────────────────
    # batch["image"] is already {"S2L1C": tensor, "S1GRD": tensor, ...}
    images = raw_image
    print(f"""
  batch['image'] is a dict (GenericMultiModalDataModule).
  This is how Sen1Floods11 batches arrive — one tensor per modality.
  Modalities: {list(images.keys())}
    """)

else:
    raise TypeError(f"Unexpected type for batch['image']: {type(raw_image)}")

# ── Print properties for every modality tensor ───────────────────────────────
print(f"  Modalities in this batch: {list(images.keys())}")

for modality, tensor in images.items():
    B, C, H, W = tensor.shape
    print()
    print(f"    Modality : '{modality}'")
    print(f"    Shape    : [B={B}, C={C}, H={H}, W={W}]")
    print(f"               B = batch size  (samples packed for GPU)")
    print(f"               C = spectral channels (bands)")
    print(f"               H = image height in pixels")
    print(f"               W = image width  in pixels")
    print(f"    dtype    : {tensor.dtype}")
    inspect_tensor(f"values", tensor, indent=6)
    print(f"      → Is image square?      {'Yes ✓' if H == W else f'No  (H={H}, W={W})'}")
    print(f"      → Is H divisible by 16? {'Yes ✓' if H % 16 == 0 else f'No — needs padding ({H} mod 16 = {H%16})'}")
    print(f"      → Is W divisible by 16? {'Yes ✓' if W % 16 == 0 else f'No — needs padding ({W} mod 16 = {W%16})'}")
    print(f"      → Memory per batch:     {tensor.numel() * 4 / 1e6:.2f} MB (float32)")
    print(f"""
      Values are z-score normalised: (raw_pixel - mean) / std
      So they are roughly centred around 0 with std ≈ 1.
      Large outliers (outside ±5) usually indicate no-data pixels.
    """)

# ── 4b. Mask tensor ──────────────────────────────────────────────────────────
if WITH_GROUND_TRUTH and "mask" in batch:
    sub("batch['mask'] — ground-truth label tensor")

    mask = batch["mask"]
    B_m, H_m, W_m = mask.shape
    print(f"""
  Shape: [B={B_m}, H={H_m}, W={W_m}]
  Note: NO channel dimension — each pixel is ONE integer label.
  dtype: {mask.dtype}  (integer, not float)

  Label values:
     0  → class 0  ("{CLASS_NAMES[0]}" — background)
     1  → class 1  ("{CLASS_NAMES[1]}" — foreground)
    -1  → ignored  (no-data / cloud / invalid; excluded from loss)
    """)

    inspect_tensor("mask values", mask, indent=4)

    mask_np = mask.numpy()
    unique_vals, counts = np.unique(mask_np, return_counts=True)
    total_pixels = mask_np.size
    print(f"\n      Label distribution across this batch ({B_m} images × {H_m}×{W_m} pixels):")
    for v, cnt in zip(unique_vals, counts):
        if v == -1:
            label = "ignored"
        elif 0 <= v < len(CLASS_NAMES):
            label = CLASS_NAMES[v]
        else:
            label = "unknown"
        print(f"        label {v:4d}  ({label:<10})  {cnt:>10,} pixels  "
              f"({100 * cnt / total_pixels:.1f}%)")

else:
    print("\n  ⚠  No mask in batch (WITH_GROUND_TRUTH=False or dataset has no labels).")
    mask    = None
    mask_np = None

# ── 4c. Filename list ─────────────────────────────────────────────────────────
sub("batch['filename'] — source file names")

if "filename" in batch:
    fnames = batch["filename"]
    print(f"""
  type  : {type(fnames).__name__}
  length: {len(fnames)}  (one entry per sample in the batch)
  
  Each entry tells you which file on disk a sample came from.
  Useful for:
    • Tracing a bad prediction back to its source image
    • Verifying the split file is loading the right scenes
    • Matching predictions to geographic coordinates later
    """)
    for idx, fn in enumerate(fnames):
        print(f"    [{idx}] {fn}")

# ── 4d. Any remaining metadata keys ──────────────────────────────────────────
sub("Other keys in the batch (metadata)")
known_keys = {"image", "mask", "filename"}
extra_keys = [k for k in batch.keys() if k not in known_keys]
if extra_keys:
    for key in extra_keys:
        val = batch[key]
        print(f"  '{key}': {type(val).__name__}  →  {val}")
else:
    print("  (No extra keys beyond image / mask / filename)")

# ── 4e. Reference table for adapting to other data formats ───────────────────
sub("Adapting to different data types — reference notes")
print(textwrap.dedent("""
  ┌────────────────────┬────────────────────────────────────────────────────┐
  │ Format / change    │ What to update                                     │
  ├────────────────────┼────────────────────────────────────────────────────┤
  │ GeoTIFF (.tif)     │ Nothing — this is the default format ✓             │
  │ HDF5 (.h5/.hdf5)   │ Write a custom torch.utils.data.Dataset using h5py │
  │                    │ and pass it directly to DataLoader                 │
  │ Shapefile labels   │ Rasterise first with rasterio.features.rasterize() │
  │                    │ then point label_grep at the output .tif           │
  │ Non-square images  │ Any H and W divisible by 16 works.                 │
  │                    │ Pad to the nearest multiple of 16 if needed.       │
  │ Different bands    │ Update dataset_bands + means/stds lists            │
  │ Hyperspectral      │ Same as above; backbone_bands picks the subset     │
  │                    │ of pretrained band embeddings to use               │
  └────────────────────┴────────────────────────────────────────────────────┘
"""))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MANUAL TENSOR BUILDING
#
# Johan's example approach, extended with full inspection logging.
# This shows you what the DataModule does under the hood, step by step.
#
# WHY BUILD MANUALLY?
# ────────────────────
# • Helps you understand the pipeline completely
# • Lets you feed arbitrary data (e.g. your drone images) without
#   needing to restructure into the split/grep format
# • Useful for one-off inference on a single image with no split file
# ─────────────────────────────────────────────────────────────────────────────

banner("SECTION 5 — Manual Tensor Building (Johan's approach, annotated)")

def read_tif_all_bands(path: str) -> np.ndarray:
    """
    Read ALL bands from a GeoTIFF into a float32 numpy array.

    Returns: array of shape [C, H, W]
      C = number of bands in the file
      H = height in pixels
      W = width in pixels

    Also prints: CRS, resolution, nodata value, dtype, shape.
    This information is useful for understanding your data's coordinate
    system and spatial resolution before feeding it to the model.
    """
    with rasterio.open(path) as src:
        print(f"\n    File : {os.path.basename(path)}")
        print(f"      Bands      : {src.count}  (C dimension)")
        print(f"      Height     : {src.height} px  (H dimension)")
        print(f"      Width      : {src.width} px   (W dimension)")
        print(f"      Dtype      : {src.dtypes[0]}")
        print(f"      CRS        : {src.crs}")
        print(f"      Resolution : {src.res[0]:.4f} × {src.res[1]:.4f}  (pixel size in CRS units)")
        print(f"      No-data    : {src.nodata}")
        print(f"      Bounds     : {src.bounds}")
        arr = src.read().astype(np.float32)  # shape: [C, H, W]
    return arr

def normalise_image(arr_chw: np.ndarray,
                    means: list,
                    stds: list,
                    no_data_replace: float = 0.0) -> np.ndarray:
    """
    Apply per-channel z-score normalisation: (pixel - mean) / std

    This is exactly what GenericNonGeoSegmentationDataModule does internally.
    The means and stds must be the same values as in your YAML config.

    Parameters
    ----------
    arr_chw          : float32 array [C, H, W]
    means            : list of C means (one per band)
    stds             : list of C stds  (one per band)
    no_data_replace  : value to substitute for NaN/Inf (default 0)

    Returns: normalised float32 array [C, H, W]
    """
    assert len(means) == arr_chw.shape[0], (
        f"Band count mismatch: image has {arr_chw.shape[0]} bands "
        f"but {len(means)} means provided"
    )
    means_np = np.array(means, dtype=np.float32).reshape(-1, 1, 1)
    stds_np  = np.array(stds,  dtype=np.float32).reshape(-1, 1, 1)
    normalised = (arr_chw - means_np) / (stds_np + 1e-8)
    # Replace NaN/Inf (from no-data pixels) with 0
    normalised = np.nan_to_num(normalised,
                               nan=no_data_replace,
                               posinf=no_data_replace,
                               neginf=no_data_replace)
    return normalised

def to_bchw_tensor(arr_chw: np.ndarray) -> "torch.Tensor":
    """
    Convert [C, H, W] numpy array → [1, C, H, W] torch float32 tensor.

    The leading dimension (B=1) is the batch dimension.
    When you use a DataLoader, it stacks multiple samples: [B, C, H, W].
    """
    return torch.from_numpy(arr_chw).unsqueeze(0)  # [C,H,W] → [1,C,H,W]


# ── Find a real image from the split file ────────────────────────────────────
sub("Locating a real image file from the split file")

if DATASET == "burnscars":
    split_path = BURNSCARS_CFG["val_split"] # use val split for small size
    data_root  = BURNSCARS_CFG["data_root"]
    _img_suffix = "_merged.tif"
    _lbl_suffix = ".mask.tif"
elif DATASET == "sen1floods11":
    split_path = SEN1FLOODS_CFG["val_split"] # use val split for small size
    data_root  = SEN1FLOODS_CFG["data_root_s2"]
    _img_suffix = "_S2Hand.tif"
    _lbl_suffix = "_LabelHand.tif"

_manual_img_path = None
_manual_lbl_path = None

if os.path.isfile(split_path):
    with open(split_path) as f:
        stems = [ln.strip() for ln in f if ln.strip()]
    print(f"  Split file       : {split_path}")
    print(f"  Total stems      : {len(stems)}")
    print(f"  First 3 stems    : {stems[:3]}")

    for stem in stems:
        candidate_img = os.path.join(data_root, stem + _img_suffix)
        if os.path.isfile(candidate_img):
            _manual_img_path = candidate_img
            if DATASET == "burnscars":
                candidate_lbl = os.path.join(data_root, stem + _lbl_suffix)
            else:
                candidate_lbl = os.path.join(
                    SEN1FLOODS_CFG["label_root"], stem + _lbl_suffix)
            if os.path.isfile(candidate_lbl):
                _manual_lbl_path = candidate_lbl
            break
else:
    print(f"  ⚠  Split file not found: {split_path}")
    print("     Skipping manual tensor section (set correct paths in Section 0)")

if _manual_img_path:
    sub("Step 1 — Read raw pixel values from GeoTIFF")
    raw_arr = read_tif_all_bands(_manual_img_path)     # [C, H, W]
    inspect_tensor("raw_arr (before normalisation)", raw_arr, indent=4)

    sub("Step 2 — Apply z-score normalisation (same as DataModule)")
    if DATASET == "burnscars":
        norm_arr = normalise_image(raw_arr,
                                   BURNSCARS_CFG["means"],
                                   BURNSCARS_CFG["stds"])
    else:
        norm_arr = normalise_image(raw_arr,
                                   SEN1FLOODS_CFG["means"]["S2L1C"],
                                   SEN1FLOODS_CFG["stds"]["S2L1C"])
    inspect_tensor("norm_arr (after normalisation)", norm_arr, indent=4)
    print("""
    Note: values are now roughly in [-3, 3].  This is what the model sees.
    Extreme values (outside ±5) usually indicate a no-data pixel or sensor
    artefact — they will be suppressed by the nan_to_num call above.
    """)

    sub("Step 3 — Convert to [B=1, C, H, W] torch tensor")
    manual_tensor = to_bchw_tensor(norm_arr)           # [1, C, H, W]
    inspect_tensor("manual_tensor", manual_tensor, indent=4)
    print(f"""
    This tensor is equivalent to batch['image']['{BURNSCARS_CFG.get('backbone_modality', 'S2L2A')}'][0:1]
    from the DataLoader batch above.  They should have the same shape and
    similar value distributions.
    """)

    if WITH_GROUND_TRUTH and _manual_lbl_path:
        sub("Step 4 — Read ground-truth mask")
        with rasterio.open(_manual_lbl_path) as src:
            lbl_arr = src.read(1)   # [H, W]  — integer labels
        inspect_tensor("label mask (raw)", lbl_arr, indent=4)
        unique_l, cnts_l = np.unique(lbl_arr, return_counts=True)
        print(f"    Unique label values: {unique_l.tolist()}")
        print("    (Values matching your no_label_replace=-1 are ignored at training time)")
else:
    manual_tensor = None
    print("  (Manual tensor section skipped — no image file found)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MODEL LOADING (ZERO-SHOT)
#
# We instantiate TerraMind from scratch using only the pretrained weights.
# No fine-tuned checkpoint is required.
#
# ARCHITECTURE OVERVIEW
# ──────────────────────
# SemanticSegmentationTask
#   └── EncoderDecoderFactory
#         ├── BACKBONE  (terramind_v1_base)
#         │     A Vision Transformer pretrained on multi-modal satellite data.
#         │     Accepts named spectral bands; you choose a subset of the
#         │     pretrained bands to use (backbone_bands in YAML).
#         │     Outputs patch tokens: one 768-d vector per 16×16 pixel patch.
#         │
#         ├── NECK  (SelectIndices → ReshapeTokensToImage →
#         │          LearnedInterpolateToPyramidal)
#         │     Converts transformer tokens back to spatial feature maps at
#         │     multiple scales — this is what the decoder needs.
#         │
#         └── DECODER  (UNetDecoder)
#               Upsamples multi-scale features to the original image size
#               and outputs a per-pixel class probability map.
#
# ─────────────────────────────────────────────────────────────────────────────

banner("SECTION 6 — Model Loading (Zero-Shot, Pretrained Backbone)")

import terratorch.tasks
from terratorch.registry import BACKBONE_REGISTRY

# terramind_v1_base, terramind_v1_small

sub("Building SemanticSegmentationTask (mirrors model: section of your YAML)")

if DATASET == "burnscars":
    _modality   = BURNSCARS_CFG["backbone_modality"]    # "S2L2A"
    _bands      = {_modality: BURNSCARS_CFG["backbone_bands"]}
    _modalities = [_modality]
else:
    _modality   = None
    _bands      = {}
    _modalities = ["S2L1C", "S1GRD"]

print(f"""
  Backbone         : {BACKBONE}
  Modalities       : {_modalities}
  Bands subset     : {_bands if _bands else '(all pretrained bands)'}

  The 'backbone_bands' parameter is how TerraMind uses partial band sets.
  If your dataset has only 4 of the 13 S2 bands, list just those 4 here.
  The model will load the corresponding learned embeddings and skip the rest.
  This is how adaptation to different sensor configurations works.
""")

model = terratorch.tasks.SemanticSegmentationTask(
    model_factory = "EncoderDecoderFactory",
    model_args = dict(
        backbone             = BACKBONE,
        backbone_pretrained  = True,       # ← download/use pretrained weights
        backbone_modalities  = _modalities,
        # backbone_bands: pass named bands to use a subset of pretrained bands.
        # Omit this key entirely to use ALL pretrained bands for a modality.
        **({"backbone_bands": _bands} if _bands else {}),
        backbone_merge_method = "mean",    # how multi-modality tokens are merged

        # ── NECK ─────────────────────────────────────────────────────────────
        # These convert ViT patch tokens → spatial feature maps for the decoder
        necks = [
            dict(name="SelectIndices",
                 # Pick 4 evenly-spaced transformer layers to use.
                 # base  model (12 layers): [2, 5, 8, 11]
                 # large model (24 layers): [5, 11, 17, 23]
                 indices=[2, 5, 8, 11]),
            dict(name="ReshapeTokensToImage", remove_cls_token=False),
            dict(name="LearnedInterpolateToPyramidal"),
        ],

        # ── DECODER ──────────────────────────────────────────────────────────
        decoder          = "UNetDecoder",
        decoder_channels = [512, 256, 128, 64],

        head_dropout = 0.1,
        num_classes  = NUM_CLASSES,
    ),
    loss         = "dice",
    ignore_index = -1,
    freeze_backbone = False,
    freeze_decoder  = False,
    class_names = CLASS_NAMES,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)
model.eval()

sub("Model summary")
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters     : {n_params:>12,}")
print(f"  Trainable parameters : {n_trainable:>12,}")
print(f"  Device               : {DEVICE}")
print(f"""
  Note on zero-shot quality:
  The backbone has pretrained representations of satellite imagery, but the
  decoder head is RANDOMLY INITIALISED.  Zero-shot segmentation will produce
  noise-like predictions.  This is expected and is not an error.  The point
  is to confirm: (1) the model loads without error, and (2) your data flows
  through the pipeline to a valid output tensor.
""")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — ZERO-SHOT INFERENCE
#
# We run the model on the batch from the DataLoader AND on the manually
# built tensor, and inspect the output shape at every step.
#
# OUTPUT SHAPE GUIDE
# ───────────────────
# model(images) returns a ModelOutput object with an .output attribute.
# .output has shape [B, NUM_CLASSES, H, W]:
#   B           = batch size
#   NUM_CLASSES = 2 (Others / Burned)
#   H, W        = same as input image (UNetDecoder upsamples back to original)
# Each [H, W] slice is a RAW LOGIT (before softmax/sigmoid).
# Higher logit → model thinks that class is more likely at that pixel.
#
# To get a hard prediction (one class per pixel):
#   preds = torch.argmax(output, dim=1)   → shape [B, H, W]
# ─────────────────────────────────────────────────────────────────────────────

banner("SECTION 7 — Zero-Shot Inference")

def run_inference(images_dict: dict, model, device) -> "torch.Tensor":
    """
    Run one forward pass and return the raw logit tensor [B, NUM_CLASSES, H, W].

    Parameters
    ----------
    images_dict : dict mapping modality name → float32 tensor [B, C, H, W]
    model       : SemanticSegmentationTask (eval mode)
    device      : torch.device

    Returns
    -------
    logits : float32 tensor [B, NUM_CLASSES, H, W]
    """
    images_on_device = {k: v.to(device) for k, v in images_dict.items()}
    with torch.no_grad():
        output = model(images_on_device)
    # output is a ModelOutput dataclass; .output holds the logit tensor
    return output.output.cpu()

# ── 7a. Inference on DataLoader batch ───────────────────────────────────────
sub("Running inference on DataLoader batch")

logits = run_inference(images, model, DEVICE)
inspect_tensor("logits (raw model output)", logits, indent=4)
print(f"""
    Shape: [B={logits.shape[0]}, NUM_CLASSES={logits.shape[1]}, H={logits.shape[2]}, W={logits.shape[3]}]
    Each of the {logits.shape[1]} class channels contains one logit per pixel.
    Logit interpretation:
      large positive → model leans toward this class
      large negative → model leans away from this class
      near zero      → uncertain
    (With random decoder weights these will be nearly uniform — expected.)
""")

probs = torch.softmax(logits, dim=1)   # → [B, NUM_CLASSES, H, W], sum to 1 along class dim
inspect_tensor("probs (after softmax)", probs, indent=4)

preds = torch.argmax(logits, dim=1)   # → [B, H, W]
inspect_tensor("preds (argmax = predicted class per pixel)", preds, indent=4)

# ── 7b. Inference on manually built tensor ───────────────────────────────────
if manual_tensor is not None:
    sub("Running inference on manually built tensor")
    if DATASET == "burnscars":
        manual_images = {"S2L2A": manual_tensor}
    else:
        manual_images = {"S2L1C": manual_tensor}

    manual_logits = run_inference(manual_images, model, DEVICE)
    inspect_tensor("manual_logits", manual_logits, indent=4)
    manual_preds = torch.argmax(manual_logits, dim=1)
    inspect_tensor("manual_preds", manual_preds, indent=4)
    print("""
    If you see the same shape as the DataLoader inference above, your
    manually built tensor is compatible with the model.  ✓
    """)
else:
    manual_preds = None
    manual_logits = None

# ── 7c. Metrics (if ground truth available) ──────────────────────────────────
if WITH_GROUND_TRUTH and mask_np is not None:
    sub("Basic metrics on DataLoader batch (zero-shot — expect low values)")
    print("""
    mIoU (mean Intersection over Union) is the primary metric used in your
    YAML configs (val/mIoU).  Here we compute a rough version manually.
    """)
    preds_np  = preds.numpy()            # [B, H, W]
    # Flatten and ignore -1 pixels
    flat_pred  = preds_np.ravel()
    flat_true  = mask_np.ravel()
    valid_mask = flat_true >= 0

    iou_per_class = []
    for cls in range(NUM_CLASSES):
        tp = ((flat_pred == cls) & (flat_true == cls) & valid_mask).sum()
        fp = ((flat_pred == cls) & (flat_true != cls) & valid_mask).sum()
        fn = ((flat_pred != cls) & (flat_true == cls) & valid_mask).sum()
        denom = tp + fp + fn
        iou = float(tp) / float(denom) if denom > 0 else float("nan")
        iou_per_class.append(iou)
        print(f"    IoU class {cls} ({CLASS_NAMES[cls]:<10}): {iou:.4f}")
    miou = np.nanmean(iou_per_class)
    print(f"    mIoU (mean)         : {miou:.4f}")
    print(f"""
    These zero-shot numbers are expected to be low (~0.3–0.5 or even random).
    After fine-tuning on 100 epochs the paper reports mIoU ≈ 0.85+ for BurnScars.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — VISUALISATION
#
# Saves a PNG for each sample showing:
#   Left   : RGB composite (from rgb_indices bands, before normalisation)
#   Centre : Ground-truth mask         (if WITH_GROUND_TRUTH=True)
#   Right  : Predicted mask
#
# The colour scheme uses a two-class palette:
#   Black (0) → Not burned / Other
#   Red   (1) → Burn scar / Flood
#   Grey (-1) → No data / No data
# ─────────────────────────────────────────────────────────────────────────────

banner("SECTION 8 — Visualisation")

# Colour map for 2-class segmentation masks
if DATASET == "burnscars":
    CLASS_COLOURS = {
        -1: [0.10, 0.10, 0.10],   # black — No data
         0: [0.80, 0.80, 0.80],   # grey  — Not burned
         1: [0.75, 0.15, 0.05],   # red   — Burn scar
    }
else:
    CLASS_COLOURS = {
        -1: [0.10, 0.10, 0.10],   # black — No data
         0: [0.80, 0.80, 0.80],   # grey  — Other
         1: [0.08, 0.35, 0.75],   # blue  — Flood
    }

def mask_to_rgb(mask_hw: np.ndarray) -> np.ndarray:
    """Convert [H, W] integer mask → [H, W, 3] RGB float image for display."""
    rgb = np.zeros((*mask_hw.shape, 3), dtype=np.float32)
    for val, colour in CLASS_COLOURS.items():
        rgb[mask_hw == val] = colour
    return rgb

def denorm_to_rgb(tensor_chw: np.ndarray,
                  means: list,
                  stds: list,
                  rgb_idx: list) -> np.ndarray:
    """
    Reverse z-score normalisation, extract RGB bands, and stretch for display.

    The key fix vs the original: percentile stretch is applied PER CHANNEL,
    not across all channels at once.  Applying a single global stretch to
    [H, W, 3] picks one scalar p2 and p98 for the whole array.  If the red
    band is much brighter than blue, the blue channel gets crushed to near-zero
    and the image shows a strong colour cast (often yellow or cyan).
    Stretching each channel independently removes the cast and gives a result
    that looks like what you'd see in a GIS viewer.

    Parameters
    ----------
    tensor_chw : normalised float32 array [C, H, W]  — from the DataLoader
    means, stds: per-band lists used to normalise (from YAML / config block)
    rgb_idx    : list of 3 band indices selecting [R, G, B] for display
                 e.g. [2, 1, 0] means band-2=R, band-1=G, band-0=B

    Returns
    -------
    uint8 array [H, W, 3] ready for imshow
    """
    # Step 1: undo z-score normalisation → back to original pixel value range
    means_np = np.array(means, dtype=np.float32).reshape(-1, 1, 1)
    stds_np  = np.array(stds,  dtype=np.float32).reshape(-1, 1, 1)
    raw = tensor_chw * stds_np + means_np          # [C, H, W], original units

    # Step 2: select the three bands for display
    rgb_bands = raw[rgb_idx]                        # [3, H, W]

    # Step 3: per-channel percentile stretch
    # Each band is stretched independently from its own p2→p98 range to [0,1].
    # This is what GIS tools (QGIS, ArcGIS, EO Browser) do by default.
    stretched = np.zeros_like(rgb_bands)            # [3, H, W]
    for c in range(3):
        band = rgb_bands[c]                         # [H, W]
        p2  = np.percentile(band, 2)
        p98 = np.percentile(band, 98)
        stretched[c] = np.clip((band - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)

    # Step 4: [3, H, W] → [H, W, 3] for matplotlib imshow
    return (stretched.transpose(1, 2, 0) * 255).astype(np.uint8)

legend_patches = [
    mpatches.Patch(color=CLASS_COLOURS[0],  label=CLASS_NAMES[0]),
    mpatches.Patch(color=CLASS_COLOURS[1],  label=CLASS_NAMES[1]),
    mpatches.Patch(color=CLASS_COLOURS[-1], label="No data"),
]

if DATASET == "burnscars":
    _vis_means    = BURNSCARS_CFG["means"]
    _vis_stds     = BURNSCARS_CFG["stds"]
    _vis_rgb_idx  = BURNSCARS_CFG["rgb_indices"]
    _vis_modality = "S2L2A"
else:
    _vis_means    = SEN1FLOODS_CFG["means"]["S2L1C"]
    _vis_stds     = SEN1FLOODS_CFG["stds"]["S2L1C"]
    _vis_rgb_idx  = SEN1FLOODS_CFG["rgb_indices"]["S2L1C"]
    _vis_modality = "S2L1C"

sub(f"Saving {N_SAMPLES} sample plot(s) to {OUTPUT_DIR}")

# Pull filenames from the batch if they exist, for titles and file names.
# batch["filename"] has different structures depending on the DataModule:
#   GenericNonGeoSegmentation (BurnScars)   → a plain list of strings
#   GenericMultiModal         (Sen1Floods11) → a dict {modality: list_of_strings}
# Normalise to a plain list in both cases.
raw_fnames = batch.get("filename", None)

if raw_fnames is None:
    # No filename key at all — fall back to numbered labels
    batch_filenames = [None] * logits.shape[0]
elif isinstance(raw_fnames, dict):
    # Multi-modal: pick filenames from the primary visual modality.
    # All modalities refer to the same scene, so any key gives the same stems.
    first_key = next(iter(raw_fnames))
    batch_filenames = list(raw_fnames[first_key])
else:
    # Single-modality: already a plain list
    batch_filenames = list(raw_fnames)

n_cols = 3 if (WITH_GROUND_TRUTH and mask_np is not None) else 2

for i in range(min(N_SAMPLES, logits.shape[0])):
    img_tensor   = images[_vis_modality][i].cpu().numpy()   # [C, H, W] normalised
    pred_i       = preds.numpy()[i]                          # [H, W]
    fname        = batch_filenames[i]                        # string or None

    rgb_display  = denorm_to_rgb(img_tensor, _vis_means, _vis_stds, _vis_rgb_idx)
    pred_display = mask_to_rgb(pred_i)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(rgb_display)
    axes[0].set_title("RGB (de-normalised)", fontsize=11)
    axes[0].axis("off")

    col = 1
    if WITH_GROUND_TRUTH and mask_np is not None:
        gt_display = mask_to_rgb(mask_np[i])
        axes[col].imshow(gt_display)
        axes[col].set_title("Ground Truth", fontsize=11)
        axes[col].axis("off")
        col += 1

    axes[col].imshow(pred_display)
    axes[col].set_title("Prediction (zero-shot)", fontsize=11)
    axes[col].axis("off")

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    # ── Title: include filename so the figure is traceable ───────────────────
    fname_display = os.path.basename(fname) if fname else f"sample_{i}"
    plt.suptitle(
        f"{fname_display}\n{DATASET}  |  {BACKBONE}  (zero-shot)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    # ── Output filename: use the image stem so it matches the source file ─────
    # Strip path and extension, replace characters that are unsafe in filenames.
    stem = os.path.splitext(os.path.basename(fname))[0] if fname else f"sample_{i}"
    safe_stem = stem.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{safe_stem}_pred_fig.png")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}  (source: {fname_display})")

# ── Manual tensor visualisation ──────────────────────────────────────────────
if manual_tensor is not None and manual_preds is not None:
    sub("Saving manual tensor plot")
    img_manual = manual_tensor[0].numpy()   # [C, H, W]
    pred_manual = manual_preds.numpy()[0]   # [H, W]
    rgb_m  = denorm_to_rgb(img_manual, _vis_means, _vis_stds, _vis_rgb_idx)
    pred_m = mask_to_rgb(pred_manual)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    axes[0].imshow(rgb_m)
    axes[0].set_title("RGB (manual load)", fontsize=12)
    axes[0].axis("off")
    col = 1
    if WITH_GROUND_TRUTH and _manual_lbl_path:
        with rasterio.open(_manual_lbl_path) as src:
            lbl_vis = src.read(1).astype(np.int32)
        gt_m = mask_to_rgb(lbl_vis)
        axes[col].imshow(gt_m)
        axes[col].set_title("Ground Truth", fontsize=12)
        axes[col].axis("off")
        col += 1
    axes[col].imshow(pred_m)
    axes[col].set_title("Prediction (zero-shot)", fontsize=12)
    axes[col].axis("off")
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(f"Manual Load  |  {DATASET}  |  {BACKBONE}  (zero-shot)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "manual_sample.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — ADAPTATION CHECKLIST
#
# A printed summary of what needs to change when you move to new data.
# Designed to be readable in a supervisor meeting without opening the code.
# ─────────────────────────────────────────────────────────────────────────────

banner("SECTION 9 — Adaptation Checklist for New Data")

print(textwrap.dedent(f"""
  Based on the inspection above, here is what must change when you move
  to a new dataset (e.g. Antarctic drone imagery).

  ┌─────────────────────────────┬──────────────────────────────────────────┐
  │ What changes                │ Where to change it                       │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ File paths & split files    │ Section 0 config block (this script)     │
  │                             │ data: init_args in YAML config           │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ Band names & count          │ dataset_bands list (DataModule)          │
  │                             │ backbone_bands dict (model_args in YAML) │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ Normalisation statistics    │ means & stds lists                       │
  │                             │ Compute from your own training tiles     │
  │                             │ using: np.mean(arr) / np.std(arr)        │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ Number of classes           │ NUM_CLASSES (top of this script)         │
  │                             │ num_classes in YAML model_args           │
  │                             │ CLASS_NAMES list (both here and in YAML) │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ Spatial resolution          │ No model change needed.                  │
  │ (drone vs satellite)        │ ViT processes any H×W divisible by 16.  │
  │                             │ Pad to nearest multiple of 16 if needed. │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ Hyperspectral (many bands)  │ Choose a subset of bands that match      │
  │                             │ pretrained band names, OR map to the     │
  │                             │ closest spectral equivalent.             │
  │                             │ Update backbone_bands accordingly.       │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ HDF5 / non-GeoTIFF format   │ Write a custom torch.utils.data.Dataset  │
  │                             │ class using h5py to __getitem__.         │
  │                             │ Pass it directly to DataLoader instead   │
  │                             │ of using GenericNonGeoSegmentation...    │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ Shapefile labels            │ Rasterise first using rasterio:          │
  │                             │   rasterio.features.rasterize()          │
  │                             │ Then treat the output .tif as label_grep │
  └─────────────────────────────┴──────────────────────────────────────────┘

  Current run output: {OUTPUT_DIR}/
    inspection_log.txt — full log of this session
    batch_sample_*.png — DataLoader batch visualisations
    manual_sample.png  — manually loaded tensor visualisation

  Next steps:
    1. Fine-tune with your YAML config:
         terratorch fit --config terramind_v1_base_burnscars.yaml
    2. Once fine-tuned, load the checkpoint:
         model = SemanticSegmentationTask.load_from_checkpoint(...)
    3. Replace DataLoader batch with your own imagery tensor (Section 5)
    4. Adapt dataset_bands / means / stds for your drone sensor
"""))

banner("Done")
print(f"  Log saved to : {_log_path}")
sys.stdout = sys.__stdout__
_log_file.close()
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
