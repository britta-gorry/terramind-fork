#!/usr/bin/env python
"""
terramind_generation.py
Converted from: terramind_generation.ipynb

PURPOSE:
  Runs TerraMind's generative capability — given a Sentinel-2 L2A image as
  input, the model generates other modalities (e.g. S1GRD radar, DEM
  elevation, LULC land-cover classes) using a diffusion process.

  This is the 'single-example, single-model' version — the simplest entry
  point to TerraMind generation. See the other scripts for large-tile
  (tiled_inference) and any-to-any variants.

RELEVANCE TO YOUR RESEARCH:
  The generative model can synthesise missing modalities for your drone
  scenes (e.g. approximate SAR or DEM) without labels, supporting
  unsupervised and zero-shot use-cases. Antarctic scenes are out-of-domain
  for the model, so these generations will degrade gracefully — inspecting
  HOW they degrade tells you about domain gap.

HPC USAGE:
  Interactive:  python terramind_generation.py
  Via PBS:      qsub submit_generation.pbs

PREREQUISITES:
  - plotting_utils.py must be in the same directory (copy from
    TerraMind repo: notebooks/plotting_utils.py)
  - Example .tif files must exist (see EXAMPLES_DIR below).
    Download on a login node: python -c "from huggingface_hub import ..."
    or run the notebook once on a machine with internet access.

OUTPUT:
  output/generation_YYYYMMDD_HHMM/
    run.log          — full terminal output
    rgb_input.png    — false-colour RGB of the S2L2A input
    generations.png  — side-by-side panel: input + all generated modalities
    <mod>.png        — one saved figure per generated modality
"""

# ─────────────────────────────────────────────────────────────────────────────
# MUST come before any other matplotlib import — disables the display
# backend so the script works on HPC nodes that have no screen.
import matplotlib
matplotlib.use("Agg")
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Path to the TerraMind examples/ directory.
# The notebook used '../examples' (relative to notebooks/); adjust for your HPC
# working directory.
EXAMPLES_DIR = "examples"

# Which of the five bundled S2L2A examples to use (0–4).
EXAMPLE_ID = 1

# Which modalities to generate. The model supports:
#   S1GRD  — Sentinel-1 SAR (2 bands)
#   S1RTC  — Sentinel-1 RTC SAR (2 bands)
#   DEM    — Digital Elevation Model (1 band)
#   LULC   — Land-Use / Land-Cover classes (10-class softmax output)
#   NDVI   — Normalised Difference Vegetation Index (1 band)
# Remove any modality to skip it and reduce GPU memory usage.
OUTPUT_MODALITIES = ['S1GRD', 'DEM', 'LULC']

# Number of diffusion timesteps.
# More steps → better quality but slower. 10 is a good starting point.
# Try 5 for a quick smoke-test, 50 for higher quality.
TIMESTEPS = 10

# Model size. Options: 'terramind_v1_base_generate'
#                      'terramind_v1_large_generate'  (needs more GPU RAM)
MODEL_NAME = 'terramind_v1_base_generate'

# ─────────────────────────────────────────────────────────────────────────────


import os
import sys
import logging
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

# ── Output directory — timestamped so each run is clearly labelled ──────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_DIR = os.path.join("output", f"generation_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logging — writes to both the terminal and a log file ────────────────────
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
log.info("TerraMind Generation")
log.info(f"  Script:           terramind_generation.py")
log.info(f"  Model:            {MODEL_NAME}")
log.info(f"  Input modality:   S2L2A (example {EXAMPLE_ID})")
log.info(f"  Output modalities:{OUTPUT_MODALITIES}")
log.info(f"  Diffusion steps:  {TIMESTEPS}")
log.info(f"  Output dir:       {OUTPUT_DIR}")
log.info("=" * 70)

# ── Try to import TerraMind's plotting helpers ───────────────────────────────
# Copy notebooks/plotting_utils.py from the TerraMind repo to your working
# directory. If it's absent, a minimal fallback is used instead.
try:
    from plotting_utils import plot_s2, plot_modality
    HAS_PLOT_UTILS = True
    log.info("plotting_utils.py found — using TerraMind plot helpers.")
except ImportError:
    HAS_PLOT_UTILS = False
    log.warning(
        "plotting_utils.py not found. Using minimal fallback plots.\n"
        "  Copy notebooks/plotting_utils.py from the TerraMind repo for "
        "richer visualisations."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Minimal plot helpers (used only if plotting_utils.py is not available)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_plot_s2(data, ax=None):
    """Display an S2L2A tensor [B,C,H,W] or [C,H,W] as a false-colour RGB."""
    if ax is None:
        fig, ax = plt.subplots()
    t = data[0] if data.ndim == 4 else data   # remove batch dim if present
    # Bands 3,2,1 = Red, Green, Blue in S2L2A (0-indexed)
    rgb = t[[3, 2, 1]].permute(1, 2, 0).cpu().numpy()
    rgb = np.clip(rgb / 2000.0, 0, 1)
    ax.imshow(rgb)
    ax.axis("off")


def _fallback_plot_modality(mod, data, ax=None):
    """Minimal modality visualisation — grayscale or RGB depending on bands."""
    if ax is None:
        fig, ax = plt.subplots()
    t = data[0] if data.ndim == 4 else data
    t = t.cpu().numpy()
    if t.shape[0] == 1:
        ax.imshow(t[0], cmap="viridis")
    elif t.shape[0] == 2:
        # e.g. S1GRD — show first band (VV)
        ax.imshow(t[0], cmap="gray")
    else:
        # e.g. LULC softmax — argmax to class map
        ax.imshow(t.argmax(axis=0), cmap="tab10")
    ax.axis("off")
    ax.set_title(mod, fontsize=8)


if not HAS_PLOT_UTILS:
    plot_s2 = _fallback_plot_s2
    plot_modality = _fallback_plot_modality

# ─────────────────────────────────────────────────────────────────────────────
# Device selection  (unchanged from notebook)
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
# Example file paths  (unchanged from notebook — local path list + colab URLs)
# ─────────────────────────────────────────────────────────────────────────────

# local_examples = [
#     os.path.join(EXAMPLES_DIR, "S2L2A", "38D_378R_2_3.tif"),
#     os.path.join(EXAMPLES_DIR, "S2L2A", "282D_485L_3_3.tif"),
#     os.path.join(EXAMPLES_DIR, "S2L2A", "433D_629L_3_1.tif"),
#     os.path.join(EXAMPLES_DIR, "S2L2A", "637U_59R_1_3.tif"),
#     os.path.join(EXAMPLES_DIR, "S2L2A", "609U_541L_3_0.tif"),
# ]

# ── Check that the chosen example exists ────────────────────────────────────
# example_path = local_examples[EXAMPLE_ID]
example_path = "data/hls_burn_scars/data/subsetted_512x512_HLS.S30.T10SEH.2020285.v1.4_merged.tif"

# if not os.path.isfile(example_path):
#     log.error(
#         f"Example file not found: {example_path}\n"
#         f"  Download TerraMind example data from Hugging Face on a login node:\n"
#         f"    python -c \"\n"
#         f"    from huggingface_hub import hf_hub_download\n"
#         f"    hf_hub_download(repo_id='ibm-esa-geospatial/Examples',\n"
#         f"                    filename='S2L2A/38D_378R_2_3.tif',\n"
#         f"                    repo_type='dataset', local_dir='{EXAMPLES_DIR}/')\n"
#         f"    \"\n"
#         f"  Or copy the examples/ folder from the TerraMind repo."
#     )
#     sys.exit(1)

log.info(f"Loading example: {example_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Load data  (unchanged from notebook)
# ─────────────────────────────────────────────────────────────────────────────

import rioxarray as rxr

data = rxr.open_rasterio(example_path)
# Convert to shape [B, C, 224, 224]  — notebook line preserved exactly
data = torch.Tensor(data.values).unsqueeze(0)   # device='cpu' implicit

log.info(f"Input tensor shape: {data.shape}  (batch, channels, H, W)")
log.info(f"Input value range:  min={data.min():.1f}  max={data.max():.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# Visualise and save the S2L2A input  (replaces plt.show())
# ─────────────────────────────────────────────────────────────────────────────

fig_in, ax_in = plt.subplots(1, 1, figsize=(6, 6))
plot_s2(data, ax=ax_in)
ax_in.set_title(f"S2L2A input — example {EXAMPLE_ID}", fontsize=10)
fig_in.tight_layout()
input_fig_path = os.path.join(OUTPUT_DIR, "rgb_input.png")
fig_in.savefig(input_fig_path, dpi=150, bbox_inches="tight")
plt.close(fig_in)
log.info(f"Saved input figure: {input_fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Build model  (unchanged from notebook)
# ─────────────────────────────────────────────────────────────────────────────

log.info(f"Building model: {MODEL_NAME}")
log.info(f"  Output modalities: {OUTPUT_MODALITIES}")
log.info("  (Downloading weights from Hugging Face if not cached — may take a minute)")

from terratorch.registry import FULL_MODEL_REGISTRY

model = FULL_MODEL_REGISTRY.build(
    MODEL_NAME,
    modalities=["S2L2A"],            # Define the input
    output_modalities=OUTPUT_MODALITIES,  # Define the output
    pretrained=True,
    standardize=True,  # If standardize=True, you don't need to normalise yourself.
)

model = model.to(device)
model.eval()

log.info("Model built and moved to device.")

# Show standardisation values (informational — useful when standardize=False)
from terratorch.models.backbones.terramind.model.terramind_register import (
    v1_pretraining_mean, v1_pretraining_std,
)
log.info("Pre-training standardisation keys available:")
for k in v1_pretraining_mean:
    log.info(f"  {k}")

# ─────────────────────────────────────────────────────────────────────────────
# Run inference  (unchanged from notebook)
# ─────────────────────────────────────────────────────────────────────────────

log.info(f"Running generation with {TIMESTEPS} diffusion timesteps …")

input_tensor = data.to(device)
with torch.no_grad():
    generated = model(input_tensor, verbose=True, timesteps=TIMESTEPS)

log.info("Generation complete.")
for mod, val in generated.items():
    log.info(f"  {mod}: shape={val.shape}  min={val.min():.3f}  max={val.max():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot and save all generations  (replaces plt.show())
# ─────────────────────────────────────────────────────────────────────────────

# ── Combined panel (same layout as notebook) ─────────────────────────────────
n_plots = len(generated) + 1
fig_all, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

plot_s2(input_tensor, ax=axes[0])
axes[0].set_title("Input (S2L2A)", fontsize=9)

for i, (mod, value) in enumerate(generated.items()):
    plot_modality(mod, value, ax=axes[i + 1])
    axes[i + 1].set_title(f"Generated {mod}", fontsize=9)

fig_all.suptitle(
    f"TerraMind generation — example {EXAMPLE_ID} — {TIMESTEPS} steps",
    fontsize=11,
)
fig_all.tight_layout()
panel_path = os.path.join(OUTPUT_DIR, "generations_panel.png")
fig_all.savefig(panel_path, dpi=150, bbox_inches="tight")
plt.close(fig_all)
log.info(f"Saved generation panel: {panel_path}")

# ── Individual figure per modality ────────────────────────────────────────────
for mod, value in generated.items():
    fig_m, ax_m = plt.subplots(1, 1, figsize=(5, 5))
    plot_modality(mod, value, ax=ax_m)
    ax_m.set_title(f"Generated {mod}", fontsize=10)
    fig_m.tight_layout()
    mod_path = os.path.join(OUTPUT_DIR, f"{mod}.png")
    fig_m.savefig(mod_path, dpi=150, bbox_inches="tight")
    plt.close(fig_m)
    log.info(f"Saved modality figure: {mod_path}")

# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 70)
log.info("Done.")
log.info(f"All outputs saved to: {OUTPUT_DIR}/")
log.info("=" * 70)
