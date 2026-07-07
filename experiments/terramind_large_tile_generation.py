#!/usr/bin/env python
"""
terramind_large_tile_generation.py
Converted from: large_tile_generation.ipynb

PURPOSE:
  Generates modalities from a large satellite tile (e.g. 2000×2000 px) by
  sliding a 256×256 px window across the image using TerraTorch's
  tiled_inference function, then stitching the results with overlap blending.

  TerraMind was pre-trained on 224×224 patches.  Passing larger inputs
  directly can cause OOM errors or degraded quality. tiled_inference is the
  correct approach for any image larger than ~512×512.

RELEVANCE TO YOUR RESEARCH:
  Your drone imagery is large and at fine spatial resolution — tiled_inference
  is exactly the mechanism you'll use to run TerraMind over full drone scenes.
  This script is your template for that workflow.  The crop/stride parameters
  control the sliding window; reducing stride increases overlap (smoother
  boundaries) at the cost of more compute.

HPC USAGE:
  Interactive:  python terramind_large_tile_generation.py
  Via PBS:      qsub submit_large_tile.pbs

PREREQUISITES:
  - plotting_utils.py in the same directory (from TerraMind repo: notebooks/)
  - Large tile .tif files.  Download on a login node (see DATA DOWNLOAD below).
    Compute nodes typically have no internet access.

OUTPUT:
  output/large_tile_YYYYMMDD_HHMM/
    run.log                  — full terminal output
    input_rgb.png            — false-colour RGB of the clipped input
    generations_panel.png    — input + all generated modalities side-by-side
    <mod>.png                — individual figure per generated modality
"""

# ─────────────────────────────────────────────────────────────────────────────
# MUST come before any other matplotlib import
import matplotlib
matplotlib.use("Agg")
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Path to the large tile .tif file.
# DATA DOWNLOAD (run on a login node that has internet):
#
#   from huggingface_hub import hf_hub_download
#   hf_hub_download(repo_id='ibm-esa-geospatial/Examples',
#                   filename='S2L2A/Santiago.tif',
#                   repo_type='dataset', local_dir='examples/')
#
# Use Santiago or Singapore (comment/uncomment as in the notebook):
# TILE_PATH = "examples/S2L2A/Santiago.tif"
# TILE_PATH = "examples/S2L2A/Singapore_2025-01-09.tif"
TILE_PATH = "data/S2B_MSIL2A_20230324T013549_N0510_R088_T49DDG_20240803T170113.SAFE/S2B_MSIL2A_20230324T013549_N0510_R088_T49DDG_20240803T170113.tif"

# Optional spatial crop: [row_start, row_end] — set to None to use the full tile.
# The notebook cropped row 500:1500 to speed up inference on a 2000×2000 tile.
# None here uses the full extent; adjust for your available GPU memory.
# ROW_CROP = (500, 1500)   # set to None to disable
ROW_CROP = None

# Which modalities to generate (same options as terramind_generation.py)
OUTPUT_MODALITIES = ["S1GRD", "LULC"]

# Tiled inference parameters (see tiled_inference docstring)
#   crop    — window size in pixels (must be a multiple of 16)
#   stride  — step between windows; smaller = more overlap = smoother edges
#   batch_size — how many tiles to process per GPU forward pass (reduce if OOM)
TILE_CROP    = 256
TILE_STRIDE  = 192
TILE_BATCH   = 16

# Diffusion timesteps per tile
TIMESTEPS = 10

# Model name
MODEL_NAME = "terramind_v1_base_generate"

# ─────────────────────────────────────────────────────────────────────────────


import os
import sys
import logging
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

# ── Timestamped output directory ─────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_DIR = os.path.join("output", f"large_tile_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = os.path.join(OUTPUT_DIR, "run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

log.info("=" * 70)
log.info("TerraMind Large-Tile Generation (tiled_inference)")
log.info(f"  Script:            terramind_large_tile_generation.py")
log.info(f"  Model:             {MODEL_NAME}")
log.info(f"  Tile path:         {TILE_PATH}")
log.info(f"  Row crop:          {ROW_CROP}")
log.info(f"  Output modalities: {OUTPUT_MODALITIES}")
log.info(f"  crop/stride/batch: {TILE_CROP}/{TILE_STRIDE}/{TILE_BATCH}")
log.info(f"  Diffusion steps:   {TIMESTEPS}")
log.info(f"  Output dir:        {OUTPUT_DIR}")
log.info("=" * 70)

# ── Try plotting helpers ──────────────────────────────────────────────────────
try:
    from plotting_utils import plot_s2, plot_modality
    HAS_PLOT_UTILS = True
    log.info("plotting_utils.py found — using TerraMind plot helpers.")
except ImportError:
    HAS_PLOT_UTILS = False
    log.warning("plotting_utils.py not found — using minimal fallback plots.")

def _fallback_plot_s2(data, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    t = data[0] if data.ndim == 4 else data
    # Move to cpu for numpy conversion if needed
    if hasattr(t, 'cpu'):
        t = t.cpu()
    rgb = t[[3, 2, 1]].permute(1, 2, 0).numpy()
    rgb = np.clip(rgb / 2000.0, 0, 1)
    ax.imshow(rgb)
    ax.axis("off")

def _fallback_plot_modality(mod, data, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    t = data[0] if data.ndim == 4 else data
    if hasattr(t, 'cpu'):
        t = t.cpu()
    t = t.numpy()
    if t.shape[0] == 1:
        ax.imshow(t[0], cmap="viridis")
    elif t.shape[0] == 2:
        ax.imshow(t[0], cmap="gray")
    else:
        ax.imshow(t.argmax(axis=0), cmap="tab10")
    ax.axis("off")
    ax.set_title(mod, fontsize=8)

if not HAS_PLOT_UTILS:
    plot_s2 = _fallback_plot_s2
    plot_modality = _fallback_plot_modality

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

log.info(f"Device: {device}")
if device == "cuda":
    log.info(f"  GPU:  {torch.cuda.get_device_name(0)}")
    log.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────────────────────────────────────
# Check input file exists
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.isfile(TILE_PATH):
    log.error(
        f"Tile file not found: {TILE_PATH}\n"
        f"  Download it on a login node (needs internet):\n\n"
        f"    from huggingface_hub import hf_hub_download\n"
        f"    hf_hub_download(\n"
        f"        repo_id='ibm-esa-geospatial/Examples',\n"
        f"        filename='S2L2A/Santiago.tif',\n"
        f"        repo_type='dataset',\n"
        f"        local_dir='examples/'\n"
        f"    )\n"
    )
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Load data  (notebook code — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

import rioxarray as rxr

log.info(f"Loading tile: {TILE_PATH}")
data = rxr.open_rasterio(TILE_PATH).values   # shape: (C, H, W)

log.info(f"Tile shape after load: {data.shape}  (channels, H, W)")
log.info(f"Value range: min={data.min():.1f}  max={data.max():.1f}")

# Optionally reduce image size (notebook: data = data[:, 500:1500])
if ROW_CROP is not None:
    r0, r1 = ROW_CROP
    data = data[:, r0:r1]
    log.info(f"Cropped to rows {r0}:{r1}  → new shape: {data.shape}")

# Build input tensor and add batch dimension  (notebook line — unchanged)
input_tensor = torch.tensor(data, dtype=torch.float, device=device).unsqueeze(0)
log.info(f"Input tensor shape: {input_tensor.shape}  (batch, C, H, W)")

# ─────────────────────────────────────────────────────────────────────────────
# Visualise and save input
# ─────────────────────────────────────────────────────────────────────────────

fig_in, ax_in = plt.subplots(1, 1, figsize=(10, 10))
plot_s2(data, ax=ax_in)
ax_in.set_title(f"S2L2A input — {os.path.basename(TILE_PATH)}", fontsize=10)
fig_in.tight_layout()
input_path = os.path.join(OUTPUT_DIR, "input_rgb.png")
fig_in.savefig(input_path, dpi=150, bbox_inches="tight")
plt.close(fig_in)
log.info(f"Saved input figure: {input_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Build model  (notebook code — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

log.info(f"Building model: {MODEL_NAME}")
log.info("  (Downloading weights if not cached — may take a minute)")

from terratorch.registry import FULL_MODEL_REGISTRY

model = FULL_MODEL_REGISTRY.build(
    MODEL_NAME,
    modalities=["S2L2A"],
    output_modalities=OUTPUT_MODALITIES,
    pretrained=True,
    standardize=True,
    timesteps=TIMESTEPS,   # Number of diffusion steps (set at build time for tiled_inference)
)

model = model.to(device)
model.eval()
log.info("Model built.")

# ─────────────────────────────────────────────────────────────────────────────
# Tiled inference  (notebook code — unchanged)
#
# tiled_inference slides a window of size `crop` across the input with
# step `stride`, runs the model on each tile, and blends the overlapping
# output regions.  The forward function must return a single tensor, so we
# concatenate all generated modalities along the channel dimension.
# ─────────────────────────────────────────────────────────────────────────────

from terratorch.tasks.tiled_inference import tiled_inference

def model_forward(x):
    # Run chained generation for all output modalities
    generated = model(x)
    # TerraTorch tiled inference expects a tensor output from model forward.
    # We concatenate all generations along the channel dimension.
    out = torch.concat([generated[m] for m in OUTPUT_MODALITIES], dim=1)
    return out

log.info(
    f"Running tiled_inference:  crop={TILE_CROP}  stride={TILE_STRIDE}  "
    f"batch_size={TILE_BATCH}  timesteps={TIMESTEPS}"
)
log.info("  This may take several minutes on a large tile …")

pred = tiled_inference(
    model_forward,
    input_tensor,
    crop=TILE_CROP,
    stride=TILE_STRIDE,
    batch_size=TILE_BATCH,
    verbose=True,
)
pred = pred.squeeze(0)  # Remove batch dim  (notebook line — unchanged)

log.info(f"tiled_inference output shape: {pred.shape}  (C_total, H, W)")

# ─────────────────────────────────────────────────────────────────────────────
# Unstack output modalities  (notebook code — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

# Number of channels per modality
num_channels_all = {
    "S2L2A": 12, "S1GRD": 2, "S1RTC": 2, "DEM": 1,
    "LULC": 10, "NDVI": 1,
}
num_channels = {m: num_channels_all[m] for m in OUTPUT_MODALITIES}
start_idx = np.cumsum([0] + list(num_channels.values()))

# Split up the stacked bands into each modality  (notebook line — unchanged)
generated = {
    m: pred[i : i + c].cpu()
    for m, i, c in zip(OUTPUT_MODALITIES, start_idx, num_channels.values())
}

if "LULC" in generated:
    # Get LULC classes  (notebook line — unchanged)
    generated["LULC"] = generated["LULC"].argmax(dim=0)

log.info("Generated modalities:")
for mod, val in generated.items():
    log.info(f"  {mod}: shape={val.shape}  min={val.min():.3f}  max={val.max():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot and save  (replaces plt.show())
# ─────────────────────────────────────────────────────────────────────────────

# Combined panel  (same layout as notebook)
n_plots = len(generated) + 1
fig_all, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

plot_s2(input_tensor, ax=axes[0])
axes[0].set_title("Input (S2L2A)", fontsize=9)

for i, (mod, value) in enumerate(generated.items()):
    plot_modality(mod, value, ax=axes[i + 1])
    axes[i + 1].set_title(f"Generated {mod}", fontsize=9)

crop_label = f"rows {ROW_CROP[0]}:{ROW_CROP[1]}" if ROW_CROP else "full tile"
fig_all.suptitle(
    f"TerraMind large-tile generation — {os.path.basename(TILE_PATH)} "
    f"({crop_label}) — crop={TILE_CROP} stride={TILE_STRIDE}",
    fontsize=9,
)
fig_all.tight_layout()
panel_path = os.path.join(OUTPUT_DIR, "generations_panel.png")
fig_all.savefig(panel_path, dpi=150, bbox_inches="tight")
plt.close(fig_all)
log.info(f"Saved generation panel: {panel_path}")

# Individual modality figures
for mod, value in generated.items():
    fig_m, ax_m = plt.subplots(1, 1, figsize=(8, 8))
    plot_modality(mod, value, ax=ax_m)
    ax_m.set_title(f"Generated {mod} (tiled inference)", fontsize=10)
    fig_m.tight_layout()
    mod_path = os.path.join(OUTPUT_DIR, f"{mod}.png")
    fig_m.savefig(mod_path, dpi=150, bbox_inches="tight")
    plt.close(fig_m)
    log.info(f"Saved modality figure: {mod_path}")

# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 70)
log.info("Done.")
log.info(f"All outputs saved to: {OUTPUT_DIR}/")
log.info(
    "NOTE: tiled inference on large tiles can look 'patchy' at tile boundaries.\n"
    "  Reducing stride (e.g. 128 instead of 192) increases overlap and smooths\n"
    "  boundaries at the cost of more compute."
)
log.info("=" * 70)
