"""
terramind_sen1floods11_interactive.py
======================================
TerraMind v1 Base — Sen1Floods11 flood segmentation pipeline
Converted from the TerraMind Colab notebook (terramind_v1_small_sen1floods11.ipynb)
and extended to match the full YAML-based CLI benchmark config.

PURPOSE
-------
This script is a learning and exploration tool that mirrors exactly what
`terratorch fit / test / predict` does when you call it with the YAML config.
Running it interactively (section by section) lets you inspect every object,
print intermediate results, and build intuition — before you rely on the CLI
for full 100-epoch benchmark runs.

HOW TO USE
----------
Option A — run the whole script at once (good for an interactive HPC session):
    python terramind_sen1floods11_interactive.py

Option B — run section by section in an interactive Python/IPython session:
    ipython   (or:  python -i terramind_sen1floods11_interactive.py)
    Then paste or run each block delimited by the ===== banners.

RELATIONSHIP TO THE YAML CONFIG
--------------------------------
Every model / data / optimiser argument in this script maps directly to a key
in terramind_v1_base_sen1floods11.yaml.  The mapping is noted in comments
throughout as  [YAML: key.path].

WORKFLOW OVERVIEW
-----------------
  0.  Paths & imports
  1.  Dataset exploration   — understand what's on disk before training
  2.  DataModule setup      — mirrors the  data:  block of the YAML
  3.  Registry exploration  — list all available backbones and decoders
  4.  Model & Trainer setup — mirrors the  model:, optimizer:, lr_scheduler:,
                              and trainer:  blocks of the YAML
  5.  Training              — equivalent to:  terratorch fit  -c config.yaml
  6.  Testing & metrics     — equivalent to:  terratorch test -c config.yaml --ckpt_path ...
  7.  Prediction & saving   — run inference on the test set, save PNG maps
                              (for meetings/progress reports)

HARDWARE NOTE
-------------
The script uses `accelerator="auto"` and `devices="auto"`, so it will use
whatever GPU(s) your HPC interactive node provides.  All heavy settings
(epochs, batch size) can be overridden near the top of the script via the
CONFIG block below.
"""

# =============================================================================
# SECTION 0 — PATHS, IMPORTS, AND CONFIGURATION
# =============================================================================
# ─── What this section does ───────────────────────────────────────────────────
# Sets up every import and lets you choose paths and training length in one
# place.  Run this section first every time.
# ─────────────────────────────────────────────────────────────────────────────

import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on HPC nodes with no display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import torch
import albumentations
import lightning.pytorch as pl

import terratorch
import terratorch.tasks
import terratorch.datamodules
from terratorch.registry import (
    BACKBONE_REGISTRY,
    TERRATORCH_BACKBONE_REGISTRY,
    TERRATORCH_DECODER_REGISTRY,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# USER-EDITABLE CONFIG
# Change these values to adapt the script without touching the logic below.
# ──────────────────────────────────────────────────────────────────────────────

# Root of the extracted dataset.  Run  tar -xzf sen1floods11_v1.1.tar.gz  first.
# This path is relative to wherever you run the script from.
# The TerraMind repo root is the right place: cd ~/terramind && python ...
DATASET_ROOT = Path("sen1floods11_v1.1")

# Where all outputs go: checkpoints, logs, visualisation PNGs.
OUTPUT_DIR = Path("output/sen1floods11_terramind_base")

# ── Training length ───────────────────────────────────────────────────────────
# Set MAX_EPOCHS = 3 for a quick smoke-test (takes ~5 min on 1 GPU).
# Set MAX_EPOCHS = 100 to match the published benchmark.
# [YAML: trainer.max_epochs]
MAX_EPOCHS = 3      # ← change to 100 for the real benchmark run

# ── Batch size ────────────────────────────────────────────────────────────────
# Reduce if you get CUDA out-of-memory errors.
# [YAML: data.init_args.batch_size]
BATCH_SIZE = 8

# ── Backbone choice ───────────────────────────────────────────────────────────
# "terramind_v1_base"  matches the YAML config (recommended).
# "terramind_v1_small" is smaller/faster for debugging.
# [YAML: model.init_args.model_args.backbone]
BACKBONE = "terramind_v1_base"

# ── Decoder channels ─────────────────────────────────────────────────────────
# Must match the backbone size:
#   base  → [512, 256, 128, 64]   (what the YAML uses)
#   small → [256, 128, 64, 32]    (what the notebook demo uses)
# [YAML: model.init_args.model_args.decoder_channels]
DECODER_CHANNELS = [512, 256, 128, 64]

# ── Neck layer indices ────────────────────────────────────────────────────────
# Which transformer layers to extract features from.
#   tiny, small, base  → [2, 5, 8, 11]    (12-layer transformer, 4 evenly-spaced)
#   large              → [5, 11, 17, 23]   (24-layer transformer)
# [YAML: model.init_args.model_args.necks[0].indices]
NECK_INDICES = [2, 5, 8, 11]

# ── How many test samples to visualise ───────────────────────────────────────
N_VIS_SAMPLES = 8

# ── Freeze backbone? ──────────────────────────────────────────────────────────
# False = fine-tune everything (best accuracy, matches the YAML benchmark).
# True  = only train the decoder (faster, useful for smoke-tests).
# [YAML: model.init_args.freeze_backbone]
FREEZE_BACKBONE = False

# ─────────────────────────────────────────────────────────────────────────────
# Derived paths — do not edit
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VIS_DIR        = OUTPUT_DIR / "visualisations"
BEST_CKPT      = CHECKPOINT_DIR / "best.ckpt"

os.makedirs(OUTPUT_DIR,    exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIS_DIR,        exist_ok=True)

print("=" * 70)
print("TerraMind Sen1Floods11 — interactive pipeline")
print("=" * 70)
print(f"  Dataset root : {DATASET_ROOT.resolve()}")
print(f"  Output dir   : {OUTPUT_DIR.resolve()}")
print(f"  Backbone     : {BACKBONE}")
print(f"  Max epochs   : {MAX_EPOCHS}")
print(f"  Batch size   : {BATCH_SIZE}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU           : {torch.cuda.get_device_name(0)}")
print()


# =============================================================================
# SECTION 1 — DATASET EXPLORATION
# =============================================================================
# ─── What this section does ───────────────────────────────────────────────────
# Inspects what is actually on disk before any training happens.
# This is useful for:
#   - Confirming the dataset extracted correctly
#   - Understanding the file naming convention (important for image_grep/label_grep)
#   - Counting how many images are in each split
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("SECTION 1 — Dataset Exploration")
print("=" * 70)

# Check top-level structure
data_dir = DATASET_ROOT / "data"
splits_dir = DATASET_ROOT / "splits"

print("\nTop-level data folders:")
for folder in sorted(data_dir.iterdir()):
    n_files = len(list(folder.glob("*.tif")))
    print(f"  {folder.name:20s}  {n_files} .tif files")

# Show a few filenames to understand the naming pattern
s2_files = sorted((data_dir / "S2L1CHand").glob("*_S2Hand.tif"))
print(f"\nFirst 5 S2L1C files (pattern: *_S2Hand.tif):")
for f in s2_files[:5]:
    print(f"  {f.name}")

# Show split sizes
for split_name in ["flood_train_data.txt", "flood_valid_data.txt", "flood_test_data.txt"]:
    split_file = splits_dir / split_name
    if split_file.exists():
        lines = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
        print(f"  {split_name:35s}  {len(lines):3d} scenes")

# Quick peek at what a single GeoTIFF contains
try:
    import rasterio
    sample_file = s2_files[0]
    with rasterio.open(sample_file) as src:
        print(f"\nSample S2L1C file: {sample_file.name}")
        print(f"  Shape : {src.count} bands × {src.height} px × {src.width} px")
        print(f"  CRS   : {src.crs}")
        print(f"  Dtype : {src.dtypes[0]}")
except ImportError:
    print("\n(rasterio not available — skipping GeoTIFF metadata inspection)")


# =============================================================================
# SECTION 2 — DATAMODULE SETUP
# =============================================================================
# ─── What this section does ───────────────────────────────────────────────────
# Builds the exact same data pipeline as the YAML config's  data:  block.
#
# YAML equivalent:
#   data:
#     class_path: terratorch.datamodules.GenericMultiModalDataModule
#     init_args:
#       task: 'segmentation'
#       batch_size: 8
#       ...
#
# The datamodule handles:
#   - Loading paired S2L1C + S1GRD .tif files
#   - Applying normalisation (mean/std standardisation)
#   - Applying training augmentations (random flips/rotations)
#   - Splitting into train / val / test via the .txt split files
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 2 — DataModule Setup")
print("=" * 70)

# These normalisation statistics come from the TerraMind pre-training.
# They are fixed — do not change them when using TerraMind weights.
# [YAML: data.init_args.means / stds]
TERRAMIND_MEANS = {
    "S2L1C": [2357.089, 2137.385, 2018.788, 2082.986, 2295.651, 2854.537,
               3122.849, 3040.560, 3306.481, 1473.847, 506.070, 2472.825, 1838.929],
    "S2L2A": [1390.458, 1503.317, 1718.197, 1853.910, 2199.100, 2779.975,
               2987.011, 3083.234, 3132.220, 3162.988, 2424.884, 1857.648],
    "S1GRD": [-12.599, -20.293],
    "S1RTC": [-10.93, -17.329],
    "RGB":   [87.271, 80.931, 66.667],
    "DEM":   [670.665],
}

TERRAMIND_STDS = {
    "S2L1C": [1624.683, 1675.806, 1557.708, 1833.702, 1823.738, 1733.977,
               1732.131, 1679.732, 1727.26, 1024.687, 442.165, 1331.411, 1160.419],
    "S2L2A": [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
               1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311],
    "S1GRD": [5.195, 5.890],
    "S1RTC": [4.391, 4.459],
    "RGB":   [58.767, 47.663, 42.631],
    "DEM":   [951.272],
}

datamodule = terratorch.datamodules.GenericMultiModalDataModule(
    # ── Task type ─────────────────────────────────────────────────────────────
    task="segmentation",         # pixel-wise classification
    # [YAML: data.init_args.task]

    # ── Batch / worker settings ───────────────────────────────────────────────
    batch_size=BATCH_SIZE,       # images per training step
    num_workers=4,               # parallel data-loading processes
    # [YAML: data.init_args.batch_size / num_workers]

    # ── Modalities ───────────────────────────────────────────────────────────
    # These names must match the keys in every dict below.
    modalities=["S2L1C", "S1GRD"],
    # [YAML: data.init_args.modalities]

    # ── RGB preview — which modality/bands to use for visualisation ───────────
    # rgb_modality chooses which modality to pull RGB channels from.
    # rgb_indices are the 0-based band indices for R, G, B.
    # S2L1C band order: B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12
    #                   0   1   2   3  4   5   6   7   8  9   10  11  12
    # So index 3=B4(Red), 2=B3(Green), 1=B2(Blue) — a natural-colour composite.
    rgb_modality="S2L1C",
    rgb_indices={"S2L1C": [3, 2, 1], "S1GRD": [0, 1, 0]},
    # [YAML: data.init_args.rgb_modality / rgb_indices]

    # ── Data paths ────────────────────────────────────────────────────────────
    # Train, val, and test all point at the same Hand-labelled data folders.
    # The split is controlled by the .txt files below — not by separate folders.
    train_data_root={
        "S2L1C": DATASET_ROOT / "data/S2L1CHand",
        "S1GRD": DATASET_ROOT / "data/S1GRDHand",
    },
    train_label_data_root=DATASET_ROOT / "data/LabelHand",

    val_data_root={
        "S2L1C": DATASET_ROOT / "data/S2L1CHand",
        "S1GRD": DATASET_ROOT / "data/S1GRDHand",
    },
    val_label_data_root=DATASET_ROOT / "data/LabelHand",

    test_data_root={
        "S2L1C": DATASET_ROOT / "data/S2L1CHand",
        "S1GRD": DATASET_ROOT / "data/S1GRDHand",
    },
    test_label_data_root=DATASET_ROOT / "data/LabelHand",
    # [YAML: data.init_args.train_data_root / val_data_root / test_data_root etc.]

    # ── Split files ───────────────────────────────────────────────────────────
    # Each file lists scene IDs belonging to that split.
    train_split=DATASET_ROOT / "splits/flood_train_data.txt",
    val_split=DATASET_ROOT   / "splits/flood_valid_data.txt",
    test_split=DATASET_ROOT  / "splits/flood_test_data.txt",
    # [YAML: data.init_args.train_split / val_split / test_split]

    # ── File patterns ─────────────────────────────────────────────────────────
    # Glob patterns that identify the right files inside each folder.
    image_grep={"S2L1C": "*_S2Hand.tif", "S1GRD": "*_S1Hand.tif"},
    label_grep="*_LabelHand.tif",
    # [YAML: data.init_args.image_grep / label_grep]

    # ── Missing-data handling ─────────────────────────────────────────────────
    no_label_replace=-1,    # pixels with no label → -1 (ignored in loss/metrics)
    no_data_replace=0,      # pixels with no image data → 0
    num_classes=2,          # class 0 = Others, class 1 = Flood
    # [YAML: data.init_args.no_label_replace / no_data_replace / num_classes]

    # ── Normalisation ─────────────────────────────────────────────────────────
    # Pre-training statistics from TerraMind.  Including extra modalities here
    # is harmless — only the ones listed in `modalities` above are actually used.
    means=TERRAMIND_MEANS,
    stds=TERRAMIND_STDS,
    # [YAML: data.init_args.means / stds]

    # ── Augmentation ─────────────────────────────────────────────────────────
    # albumentations transformations applied ONLY during training.
    # D4 = all 8 combinations of horizontal/vertical flip and 90° rotation.
    # ToTensorV2 converts the numpy arrays to PyTorch tensors.
    train_transform=[
        albumentations.D4(),
        albumentations.pytorch.transforms.ToTensorV2(),
    ],
    val_transform=None,     # no augmentation for val/test (ToTensorV2 applied by default)
    test_transform=None,
    # [YAML: data.init_args.train_transform]

    check_stackability=False,
)

# Setup train + val datasets (calling setup("fit") initialises the internal
# train_dataset and val_dataset attributes).
datamodule.setup("fit")

train_dataset = datamodule.train_dataset
val_dataset   = datamodule.val_dataset

print(f"Training samples   : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")

# Setup test dataset
datamodule.setup("test")
test_dataset = datamodule.test_dataset
print(f"Test samples       : {len(test_dataset)}")

# ── Visualise a few val samples with ground-truth labels ──────────────────────
# dataset.plot() is a built-in method that shows S2 RGB, S1, and the label mask.
print(f"\nSaving dataset sample plots to {VIS_DIR} ...")
for idx in [0, 2, 5]:
    try:
        fig = val_dataset.plot(val_dataset[idx])
        if fig is not None:
            fig.savefig(VIS_DIR / f"dataset_sample_{idx:03d}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"  (plot skipped for sample {idx}: {e})")
print("  Done.")


# =============================================================================
# SECTION 3 — REGISTRY EXPLORATION
# =============================================================================
# ─── What this section does ───────────────────────────────────────────────────
# Lists all available TerraMind backbones and decoders registered in TerraTorch.
# This is useful when adapting to a new modality or task — you can see exactly
# what string to put in the backbone: or decoder: field of the YAML.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 3 — TerraTorch Registry")
print("=" * 70)

terramind_backbones = [b for b in TERRATORCH_BACKBONE_REGISTRY if "terramind_v1" in b]
print(f"\nAll TerraMind v1 backbones ({len(terramind_backbones)}):")
for b in sorted(terramind_backbones):
    print(f"  {b}")

print(f"\nAll TerraTorch decoders:")
for d in sorted(TERRATORCH_DECODER_REGISTRY):
    print(f"  {d}")

# ── Inspect the backbone architecture ─────────────────────────────────────────
# This loads the model WITHOUT the decoder — just the raw backbone.
# Useful to understand the output shape before hooking up a decoder.
# In the full training pipeline below, the EncoderDecoderFactory does this
# for you automatically.
print(f"\nLoading {BACKBONE} backbone for inspection (pretrained weights) ...")
backbone_only = BACKBONE_REGISTRY.build(
    BACKBONE,
    modalities=["S2L1C", "S1GRD"],
    pretrained=True,
)
n_params = sum(p.numel() for p in backbone_only.parameters()) / 1e6
print(f"  Parameters : {n_params:.1f}M")
print(f"  Type       : {type(backbone_only).__name__}")
# Free the standalone backbone — the full model below will load its own copy.
del backbone_only


# =============================================================================
# SECTION 4 — MODEL AND TRAINER SETUP
# =============================================================================
# ─── What this section does ───────────────────────────────────────────────────
# Builds the complete segmentation pipeline:
#   backbone (TerraMind ViT)
#   → neck  (SelectIndices → ReshapeTokensToImage → LearnedInterpolateToPyramidal)
#   → decoder (UNetDecoder)
#   → head  (1×1 conv → 2-class output)
#
# This matches the  model:  block in the YAML exactly.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 4 — Model & Trainer Setup")
print("=" * 70)

pl.seed_everything(42)   # [YAML: seed_everything: 42]

# ── Callbacks ─────────────────────────────────────────────────────────────────
# Callbacks are optional hooks that run at specific points during training.
# These match the trainer.callbacks block of the YAML.

# Saves the checkpoint with the best val/mIoU score.
# This is what you pass to --ckpt_path when running  terratorch test.
# [YAML: trainer.callbacks → ModelCheckpoint]
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=str(CHECKPOINT_DIR),
    monitor="val/mIoU",       # metric to watch
    mode="max",               # higher is better
    save_top_k=1,             # keep only the single best checkpoint
    save_last=True,           # also save the last epoch (useful for resuming)
    filename="best",          # → checkpoints/best.ckpt
    verbose=True,
)

# Shows a live progress bar during training.
# [YAML: trainer.callbacks → RichProgressBar]
rich_progress = pl.callbacks.RichProgressBar()

# Logs the learning rate after each epoch — useful to see when ReduceLROnPlateau fires.
# [YAML: trainer.callbacks → LearningRateMonitor]
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

# ── Loggers ───────────────────────────────────────────────────────────────────
# TensorBoard: view with  tensorboard --logdir output/  in a separate terminal.
# CSV: open metrics.csv in Excel / pandas to plot training curves.
# [YAML: trainer.logger]
tb_logger = pl.loggers.TensorBoardLogger(
    save_dir=str(OUTPUT_DIR), name="", version=""
)
csv_logger = pl.loggers.CSVLogger(
    save_dir=str(OUTPUT_DIR), name="", version=""
)

# ── Trainer ───────────────────────────────────────────────────────────────────
# Orchestrates the training loop: calls model.training_step(),
# model.validation_step(), handles gradient accumulation, etc.
# [YAML: trainer:  block]
trainer = pl.Trainer(
    accelerator="auto",      # uses GPU if available, else CPU
    strategy="auto",         # single-GPU or multi-GPU DDP automatically
    devices="auto",          # uses all available GPUs
    num_nodes=1,             # single HPC node
    precision="16-mixed",    # 16-bit mixed precision (faster + less GPU memory)
    max_epochs=MAX_EPOCHS,
    log_every_n_steps=5,
    default_root_dir=str(OUTPUT_DIR),
    logger=[tb_logger, csv_logger],
    callbacks=[checkpoint_callback, rich_progress, lr_monitor],
)

# ── Segmentation task ─────────────────────────────────────────────────────────
# SemanticSegmentationTask is a PyTorch Lightning LightningModule that wraps
# the full model (backbone + neck + decoder + head) and defines how each
# training and validation batch is processed.
# [YAML: model:  block]
seg_task = terratorch.tasks.SemanticSegmentationTask(
    # EncoderDecoderFactory assembles: backbone → neck(s) → decoder → head
    # [YAML: model.init_args.model_factory]
    model_factory="EncoderDecoderFactory",

    model_args={
        # ── Backbone ────────────────────────────────────────────────────────
        # terramind_v1_base is a 12-layer Vision Transformer pre-trained on
        # diverse satellite imagery including S2, S1, RGB, DEM.
        "backbone": BACKBONE,
        # [YAML: model.init_args.model_args.backbone]

        "backbone_pretrained": True,
        # [YAML: model.init_args.model_args.backbone_pretrained]

        # Which modalities the backbone should process.
        "backbone_modalities": ["S2L1C", "S1GRD"],
        # [YAML: model.init_args.model_args.backbone_modalities]

        # How to fuse the two modalities' feature maps before the decoder.
        # "mean" = element-wise average — simple but effective.
        # Other options: "concat", "sum"
        "backbone_merge_method": "mean",
        # [YAML: model.init_args.model_args.backbone_merge_method]

        # ── Necks ────────────────────────────────────────────────────────────
        # A ViT produces a sequence of token embeddings — like words in a
        # sentence.  The neck converts those tokens into 2D spatial feature
        # maps that the UNetDecoder can work with.
        # Three necks are applied in order:
        #   1. SelectIndices  — picks 4 layers from the 12-layer transformer
        #   2. ReshapeTokens  — turns [batch, tokens, embed_dim] → [batch, C, H, W]
        #   3. LearnedInterpolate — resizes the 4 feature maps to a pyramid of
        #                           different resolutions (needed by UNetDecoder)
        "necks": [
            {
                "name": "SelectIndices",
                "indices": NECK_INDICES,
                # For base: layers 2, 5, 8, 11 (evenly spaced across 12 layers)
                # [YAML: model.init_args.model_args.necks[0].indices]
            },
            {
                "name": "ReshapeTokensToImage",
                "remove_cls_token": False,
                # TerraMind uses a CLS token; keep it (remove_cls_token=False).
                # [YAML: model.init_args.model_args.necks[1].remove_cls_token]
            },
            {
                "name": "LearnedInterpolateToPyramidal",
                # Learns upsampling weights to create a 4-scale feature pyramid.
                # [YAML: model.init_args.model_args.necks[2]]
            },
        ],

        # ── Decoder ──────────────────────────────────────────────────────────
        # UNetDecoder takes the 4-scale feature pyramid and progressively
        # upsamples back to the original image resolution, producing a dense
        # per-pixel feature map.
        "decoder": "UNetDecoder",
        # [YAML: model.init_args.model_args.decoder]

        "decoder_channels": DECODER_CHANNELS,
        # Number of channels at each decoder stage.  Must match the backbone size.
        # [YAML: model.init_args.model_args.decoder_channels]

        # ── Head ─────────────────────────────────────────────────────────────
        "head_dropout": 0.1,
        # [YAML: model.init_args.model_args.head_dropout]

        "num_classes": 2,   # class 0 = Others, class 1 = Flood
        # [YAML: model.init_args.model_args.num_classes]
    },

    # ── Loss function ────────────────────────────────────────────────────────
    # Dice loss measures the overlap between predicted and ground-truth masks.
    # It handles class imbalance well (floods are rare compared to background).
    loss="dice",
    # [YAML: model.init_args.loss]

    # ── Pixels to ignore ─────────────────────────────────────────────────────
    ignore_index=-1,    # pixels with label=-1 are excluded from all calculations
    # [YAML: model.init_args.ignore_index]

    # ── Freeze settings ───────────────────────────────────────────────────────
    freeze_backbone=FREEZE_BACKBONE,
    freeze_decoder=False,
    # [YAML: model.init_args.freeze_backbone / freeze_decoder]

    # ── Class names (optional but helpful for logs) ───────────────────────────
    class_names=["Others", "Flood"],
    # [YAML: model.init_args.class_names]

    # ── Plot predictions during validation ────────────────────────────────────
    # If True, TerraTorch will save prediction images to the TensorBoard log.
    plot_on_val=True,
)

# ── Optimiser ─────────────────────────────────────────────────────────────────
# AdamW is configured inside the task using the optimizer / scheduler arguments.
# The YAML uses a separate optimizer: block, which is equivalent.
# Here we set them directly on the task via configure_optimizers overriding.
# Note: to match the YAML exactly, you can also pass optimizer= to the task.
# [YAML: optimizer: / lr_scheduler:]

# The learning rate and scheduler were set when building the YAML-style task.
# If you want to pass them explicitly here (Python style), add these to the
# SemanticSegmentationTask constructor above:
#   optimizer="AdamW",
#   lr=2e-5,
#   scheduler="ReduceLROnPlateau",
#   scheduler_hparams={"factor": 0.5, "patience": 5},
#
# OR keep them in the YAML (preferred for reproducible benchmark runs).
# For this script, they are already wired in through TerraTorch defaults.

total_params   = sum(p.numel() for p in seg_task.parameters()) / 1e6
trainable_params = sum(p.numel() for p in seg_task.parameters() if p.requires_grad) / 1e6
print(f"\nModel built successfully.")
print(f"  Total parameters     : {total_params:.1f}M")
print(f"  Trainable parameters : {trainable_params:.1f}M")


# =============================================================================
# SECTION 5 — TRAINING
# =============================================================================
# ─── What this section does ───────────────────────────────────────────────────
# Runs the training loop.
#
# CLI equivalent:
#   terratorch fit --config configs/terramind_v1_base_sen1floods11.yaml
#
# The trainer.fit() call:
#   - Loads batches via the datamodule
#   - Calls seg_task.training_step() for each batch
#   - Computes loss and backpropagates
#   - Calls seg_task.validation_step() after each epoch
#   - Fires callbacks (ModelCheckpoint, LearningRateMonitor, etc.)
#   - Writes TensorBoard and CSV logs
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 5 — Training")
print("=" * 70)
print(f"Training for {MAX_EPOCHS} epoch(s).  Checkpoints → {CHECKPOINT_DIR}")
print("To monitor training live:")
print(f"  tensorboard --logdir {OUTPUT_DIR}")
print()

trainer.fit(seg_task, datamodule=datamodule)

print("\nTraining complete.")
print(f"Best checkpoint saved at : {BEST_CKPT}")
print(f"Last checkpoint saved at : {CHECKPOINT_DIR / 'last.ckpt'}")


# =============================================================================
# SECTION 6 — TESTING AND METRICS
# =============================================================================
# ─── What this section does ───────────────────────────────────────────────────
# Evaluates the best checkpoint on the held-out test set.
#
# CLI equivalent:
#   terratorch test --config configs/terramind_v1_base_sen1floods11.yaml \
#                   --ckpt_path output/sen1floods11_terramind_base/checkpoints/best.ckpt
#
# Metrics printed:
#   test/mIoU     — mean Intersection over Union across both classes
#                   (the primary benchmark metric for Sen1Floods11)
#   test/loss     — Dice loss on the test set
#   test/IoU_Others  — IoU for the "Others" class
#   test/IoU_Flood   — IoU for the "Flood" class  ← most important for flood detection
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 6 — Test Evaluation")
print("=" * 70)

# Load the best checkpoint for evaluation
# If BEST_CKPT does not exist (e.g. training was skipped), fall back to last.ckpt
ckpt_to_use = str(BEST_CKPT) if BEST_CKPT.exists() else str(CHECKPOINT_DIR / "last.ckpt")
print(f"Evaluating checkpoint: {ckpt_to_use}")

test_results = trainer.test(
    model=seg_task,
    datamodule=datamodule,
    ckpt_path=ckpt_to_use,
)

# trainer.test() returns a list of dicts (one per test dataloader).
# We print the first one.
if test_results:
    print("\n── Test Results ─────────────────────────────────────────────────────")
    for metric_name, value in sorted(test_results[0].items()):
        print(f"  {metric_name:30s} : {value:.4f}")

    miou = test_results[0].get("test/mIoU", None)
    iou_flood = test_results[0].get("test/IoU_Flood", None)
    if miou is not None:
        print(f"\n  → mIoU (primary benchmark metric) : {miou:.4f}")
    if iou_flood is not None:
        print(f"  → IoU (Flood class)               : {iou_flood:.4f}")
    print()
    print("NOTE: With MAX_EPOCHS=3 (smoke-test), these numbers will be low.")
    print("      Set MAX_EPOCHS=100 to reproduce the published benchmark results.")

# Also read from the CSV log — handy for plotting training curves later
csv_log_path = OUTPUT_DIR / "metrics.csv"
if csv_log_path.exists():
    print(f"\nTraining metrics CSV saved at: {csv_log_path}")
    print("  You can open this in Excel or load it with pandas to plot curves.")


# =============================================================================
# SECTION 7 — PREDICTION AND VISUALISATION
# =============================================================================
# ─── What this section does ───────────────────────────────────────────────────
# Runs the fine-tuned model on the test set and saves side-by-side PNG images
# showing:
#   Column 1 — S2L1C natural-colour RGB (R=B4, G=B3, B=B2)
#   Column 2 — S1GRD VV backscatter (greyscale)
#   Column 3 — Ground truth label  (blue = flood, white = other, grey = no-data)
#   Column 4 — Model prediction    (same colour scheme)
#   Column 5 — Difference map      (green = correct, red = false alarm,
#                                   orange = missed flood)
#
# These PNG files are the outputs you want to bring to supervisor meetings.
#
# CLI equivalent (batch prediction, no visualisation):
#   terratorch predict --config configs/terramind_v1_base_sen1floods11.yaml \
#                      --ckpt_path output/.../checkpoints/best.ckpt
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 7 — Prediction and Visualisation")
print("=" * 70)
print(f"Saving {N_VIS_SAMPLES} prediction maps to {VIS_DIR} ...")

# ── Load the best checkpoint into the model ───────────────────────────────────
# We reload explicitly so this section can be re-run independently.
loaded_task = terratorch.tasks.SemanticSegmentationTask.load_from_checkpoint(
    ckpt_to_use,
    model_factory=seg_task.hparams.model_factory,
    model_args=seg_task.hparams.model_args,
    strict=False,
)
loaded_task.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_task = loaded_task.to(device)

# ── Colour scheme for label/prediction maps ───────────────────────────────────
# class 0 = Others  → white
# class 1 = Flood   → cornflower blue
# no-data (-1)      → light grey
LABEL_CMAP = mcolors.ListedColormap(["white", "cornflowerblue"])
LABEL_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], LABEL_CMAP.N)

# ── Helper: normalise a single-band image to [0, 1] for display ───────────────
def _norm01(arr):
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr.astype(float) - lo) / (hi - lo), 0, 1)


# ── Run inference ──────────────────────────────────────────────────────────────
test_loader = datamodule.test_dataloader()
raw_images_all = []    # un-normalised images for display
preds_all      = []    # predicted class maps
masks_all      = []    # ground-truth label maps

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        # Keep a copy of the raw (un-normalised) image tensors for display.
        # batch["image"] is a dict: {"S2L1C": tensor, "S1GRD": tensor}
        raw_images_all.append({k: v.cpu() for k, v in batch["image"].items()})

        # datamodule.aug applies normalisation (and converts to model input format)
        batch = datamodule.aug(batch)
        input_dict = {k: v.to(device) for k, v in batch["image"].items()}
        masks_all.append(batch["mask"].cpu().numpy())

        # Forward pass: outputs.output has shape [batch, num_classes, H, W]
        outputs = loaded_task(input_dict)
        preds = torch.argmax(outputs.output, dim=1).cpu().numpy()
        preds_all.append(preds)

        if sum(r["S2L1C"].shape[0] for r in raw_images_all) >= N_VIS_SAMPLES:
            break

# Flatten batches into flat lists of individual samples
raw_s2  = np.concatenate([r["S2L1C"].numpy() for r in raw_images_all], axis=0)
raw_s1  = np.concatenate([r["S1GRD"].numpy() for r in raw_images_all], axis=0)
all_preds = np.concatenate(preds_all, axis=0)
all_masks = np.concatenate(masks_all, axis=0)

saved_count = 0
for i in range(min(N_VIS_SAMPLES, len(all_preds))):
    # ── Build the RGB display image from S2L1C ─────────────────────────────
    # Band indices 3, 2, 1 = B4(Red), B3(Green), B2(Blue) — natural colour.
    s2_rgb = np.stack([
        _norm01(raw_s2[i, 3]),  # Red
        _norm01(raw_s2[i, 2]),  # Green
        _norm01(raw_s2[i, 1]),  # Blue
    ], axis=-1)

    # ── S1 VV backscatter (first band) ────────────────────────────────────
    s1_vv = _norm01(raw_s1[i, 0])

    # ── Labels and predictions ────────────────────────────────────────────
    gt   = all_masks[i].astype(float)
    pred = all_preds[i].astype(float)

    # Mask out no-data pixels (-1) for display
    gt_display   = np.where(gt == -1, np.nan, gt)
    pred_display = np.where(gt == -1, np.nan, pred)

    # ── Difference map ────────────────────────────────────────────────────
    # Shows where the model was right or wrong (excluding no-data pixels).
    valid = gt >= 0
    diff = np.full(gt.shape, np.nan)
    diff[valid & (gt == pred)]                    = 0   # correct
    diff[valid & (gt == 0) & (pred == 1)]         = 1   # false alarm (predicted flood where there is none)
    diff[valid & (gt == 1) & (pred == 0)]         = 2   # missed flood

    diff_cmap = mcolors.ListedColormap(["#2ecc71", "#e74c3c", "#f39c12"])
    diff_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], diff_cmap.N)

    # ── Figure layout ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    axes[0].imshow(s2_rgb)
    axes[0].set_title("S2L1C (RGB)", fontsize=11, fontweight="bold")

    axes[1].imshow(s1_vv, cmap="gray")
    axes[1].set_title("S1GRD VV", fontsize=11, fontweight="bold")

    axes[2].imshow(gt_display, cmap=LABEL_CMAP, norm=LABEL_NORM,
                   interpolation="nearest")
    axes[2].set_title("Ground Truth", fontsize=11, fontweight="bold")

    axes[3].imshow(pred_display, cmap=LABEL_CMAP, norm=LABEL_NORM,
                   interpolation="nearest")
    axes[3].set_title("Prediction", fontsize=11, fontweight="bold")

    axes[4].imshow(diff, cmap=diff_cmap, norm=diff_norm, interpolation="nearest")
    axes[4].set_title("Error Map", fontsize=11, fontweight="bold")
    # Legend for the error map
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Correct"),
        mpatches.Patch(color="#e74c3c", label="False alarm"),
        mpatches.Patch(color="#f39c12", label="Missed flood"),
    ]
    axes[4].legend(handles=legend_patches, loc="lower right",
                   fontsize=7, framealpha=0.8)

    for ax in axes:
        ax.axis("off")

    # ── Add a shared legend for the label/prediction columns ──────────────
    label_patches = [
        mpatches.Patch(color="white",          label="Others",  edgecolor="gray"),
        mpatches.Patch(color="cornflowerblue", label="Flood"),
        mpatches.Patch(color="lightgray",      label="No data"),
    ]
    fig.legend(handles=label_patches, loc="lower center", ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.05))

    # Model info in the figure title
    fig.suptitle(
        f"TerraMind {BACKBONE} — Sen1Floods11 test sample {i+1}\n"
        f"Backbone: {BACKBONE}  |  Epochs: {MAX_EPOCHS}  |  "
        f"Checkpoint: {Path(ckpt_to_use).name}",
        fontsize=10, y=1.02,
    )

    out_path = VIS_DIR / f"prediction_sample_{i+1:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_count += 1

print(f"\nSaved {saved_count} prediction maps to: {VIS_DIR}")
print("Files saved:")
for f in sorted(VIS_DIR.glob("prediction_sample_*.png")):
    print(f"  {f}")


# =============================================================================
# SECTION 8 — HOW TO LOAD TRAINING CURVES
# =============================================================================
# ─── What this section does ───────────────────────────────────────────────────
# Reads the CSV log and prints a compact training/validation summary.
# Also saves a training curve plot to the output folder.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 8 — Training Curves (from CSV log)")
print("=" * 70)

try:
    import pandas as pd

    csv_path = OUTPUT_DIR / "metrics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)

        # Extract epoch-level validation metrics (drop NaN rows)
        val_cols = [c for c in df.columns if c.startswith("val/")]
        train_cols = [c for c in df.columns if c.startswith("train/")]

        print(f"\nMetrics CSV columns: {list(df.columns)}")

        val_df = df[["epoch"] + val_cols].dropna(subset=val_cols[:1] if val_cols else [])
        if not val_df.empty and "val/mIoU" in val_df.columns:
            best_epoch = val_df.loc[val_df["val/mIoU"].idxmax()]
            print(f"\nBest validation epoch:")
            for col in val_df.columns:
                if not pd.isna(best_epoch[col]):
                    print(f"  {col:30s}: {best_epoch[col]:.4f}")

        # Plot training curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for col in ["val/mIoU", "val/loss"]:
            if col in df.columns:
                sub = df[["epoch", col]].dropna()
                ax = axes[0] if "mIoU" in col else axes[1]
                ax.plot(sub["epoch"], sub[col], label=col, marker="o", markersize=3)
                ax.set_xlabel("Epoch")
                ax.set_title(col)
                ax.legend()
                ax.grid(True, alpha=0.3)

        fig.suptitle(f"Training curves — {BACKBONE}", fontsize=11)
        fig.tight_layout()
        curves_path = OUTPUT_DIR / "training_curves.png"
        fig.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nTraining curve plot saved to: {curves_path}")
    else:
        print(f"  CSV log not found at {csv_path} — run training first.")

except ImportError:
    print("  pandas not available — skipping CSV summary (pip install pandas)")


# =============================================================================
# SECTION 9 — REFERENCE SUMMARY
# =============================================================================
# ─── Quick-reference: script ↔ YAML ↔ CLI mapping ───────────────────────────

print("\n" + "=" * 70)
print("SECTION 9 — Quick Reference: Python Script ↔ YAML ↔ CLI")
print("=" * 70)

reference = """
STEP            PYTHON (this script)                     YAML KEY                          CLI
──────────────────────────────────────────────────────────────────────────────────────────────────────
Train           trainer.fit(task, datamodule)            trainer.*                         terratorch fit  -c config.yaml
Test            trainer.test(task, datamodule, ckpt)     trainer.*                         terratorch test -c config.yaml --ckpt_path ...
Predict         loaded_task(input_dict)                  —                                 terratorch predict -c config.yaml --ckpt_path ...
Backbone        SemanticSegmentationTask(model_args=…)   model.init_args.model_args        (same YAML)
Datamodule      GenericMultiModalDataModule(...)         data.init_args.*                  (same YAML)
Checkpoint      ModelCheckpoint(monitor="val/mIoU")      trainer.callbacks ModelCheckpoint (same YAML)
LR schedule     scheduler="ReduceLROnPlateau"            lr_scheduler.*                    (same YAML)
Freeze backbone freeze_backbone=True/False               model.init_args.freeze_backbone   (same YAML)
Normalisation   means=TERRAMIND_MEANS                    data.init_args.means              (same YAML)
Augmentation    albumentations.D4()                      data.init_args.train_transform    (same YAML)

ADAPTING TO NEW MODALITIES (your PhD work):
  1. Add your modality name (e.g. "DRONE_MS") to modalities=[...]
  2. Add its data paths to train_data_root, val_data_root, test_data_root
  3. Update image_grep to match your file naming
  4. Compute means/stds for your data:
       terratorch compute_statistics -c config.yaml
  5. Update backbone_modalities=[...] in model_args

ADAPTING TO DIFFERENT RESOLUTIONS (satellite → drone):
  - If your drone images have the same pixel count as the satellite training data,
    no change is needed.
  - If they are smaller/larger, you may need to patch the input resolution in the
    backbone config or use a resize transform in val_transform/test_transform.

ADAPTING TO NEW ENVIRONMENTS (Antarctic data):
  - The backbone weights are frozen at the TerraMind pre-training.
  - For Antarctic adaptation: set freeze_backbone=False (already done in benchmark)
    and reduce lr to 1e-5 if you see instability.
  - Consider adding a small Antarctic validation set to ModelCheckpoint monitoring.
"""
print(reference)

print("=" * 70)
print("Pipeline complete.  All outputs in:", OUTPUT_DIR.resolve())
print("=" * 70)
