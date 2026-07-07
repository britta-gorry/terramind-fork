#!/usr/bin/env python
"""
terramind_any_to_any_generation.py
Converted from: terramind_any_to_any_generation.ipynb

PURPOSE:
  Demonstrates TerraMind's any-to-any generation capability.  For each
  modality in the input set, the model is run with THAT SINGLE MODALITY as
  input and all other modalities as targets.  The result is an N×N grid
  showing: rows = input modality, columns = generated modality.

  This comprehensively answers the question "how well can TerraMind cross
  from each modality to every other?"

RELEVANCE TO YOUR RESEARCH:
  This is directly relevant to your domain-adaptation challenge of working
  with drone multispectral/hyperspectral data that has no satellite
  counterpart.  You can use any-to-any generation to:
    • Synthesise missing modalities (e.g. SAR or DEM) for Antarctic scenes
    • Assess how out-of-domain your drone data is (if generations are poor,
      the domain gap is large)
    • Generate pseudo-labelled training data without annotation

  WARNING: This script runs N separate model initialisations (one per input
  modality) and performs N×(N-1) generations.  It is GPU-memory intensive
  and slow.  Start with 2–3 modalities and add more once it works.

HPC USAGE:
  Interactive:  python terramind_any_to_any_generation.py
  Via PBS:      qsub submit_any_to_any.pbs

PREREQUISITES:
  - plotting_utils.py in the same directory (from TerraMind repo: notebooks/)
  - All modality .tif files for the chosen example.
    The TerraMind examples/ directory must contain sub-folders:
      S2L2A/, S1RTC/, DEM/, LULC/, NDVI/
    Download on a login node (see the notebook for URLs or use hf_hub_download).

OUTPUT:
  output/any_to_any_YYYYMMDD_HHMM/
    run.log                   — full terminal output
    any_to_any_<file>.png     — the full N×N generation grid
    <input>_to_<output>.png   — individual panel for each (input, output) pair
"""

# ─────────────────────────────────────────────────────────────────────────────
# MUST come before any other matplotlib import
import matplotlib
matplotlib.use("Agg")
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Parent directory containing one sub-folder per modality
#  (i.e. EXAMPLES_DIR/S2L2A/*.tif, EXAMPLES_DIR/S1RTC/*.tif, etc.)
EXAMPLES_DIR = "examples"

# Which of the bundled example files to use.
# All modalities must have a file with this same filename.
EXAMPLE_FILE = "38D_378R_2_3.tif"

# Modalities to include in the any-to-any experiment.
# Removing modalities saves GPU time and memory.
# Full set: ['S2L2A', 'S1RTC', 'DEM', 'LULC', 'NDVI']
# NOTE: S1GRD is NOT in the bundled any-to-any examples; S1RTC is used instead.
MODALITIES = ["S2L2A", "S1RTC", "DEM", "LULC", "NDVI"]

# Number of diffusion timesteps per generation call
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
OUTPUT_DIR = os.path.join("output", f"any_to_any_{timestamp}")
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
log.info("TerraMind Any-to-Any Generation")
log.info(f"  Script:          terramind_any_to_any_generation.py")
log.info(f"  Model:           {MODEL_NAME}")
log.info(f"  Example file:    {EXAMPLE_FILE}")
log.info(f"  Modalities:      {MODALITIES}")
log.info(f"  Diffusion steps: {TIMESTEPS}")
log.info(f"  Num experiments: {len(MODALITIES)}  (one per input modality)")
log.info(f"  Output dir:      {OUTPUT_DIR}")
log.info("=" * 70)

# ── Plotting helpers ──────────────────────────────────────────────────────────
try:
    from plotting_utils import plot_modality
    HAS_PLOT_UTILS = True
    log.info("plotting_utils.py found — using TerraMind plot helpers.")
except ImportError:
    HAS_PLOT_UTILS = False
    log.warning("plotting_utils.py not found — using minimal fallback plots.")

def _fallback_plot_modality(mod, data, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    t = data[0] if data.ndim == 4 else data
    if hasattr(t, 'cpu'):
        t = t.cpu()
    t = t.numpy() if hasattr(t, 'numpy') else t
    if t.ndim == 2:
        ax.imshow(t, cmap="tab10")
    elif t.shape[0] == 1:
        ax.imshow(t[0], cmap="viridis")
    elif t.shape[0] == 2:
        ax.imshow(t[0], cmap="gray")
    else:
        ax.imshow(t.argmax(axis=0), cmap="tab10")
    ax.axis("off")

if not HAS_PLOT_UTILS:
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
# Load data  (notebook code — unchanged, error handling added)
# ─────────────────────────────────────────────────────────────────────────────

import rioxarray as rxr
from terratorch.registry import FULL_MODEL_REGISTRY

# Define modalities  (notebook line — unchanged)
modalities = MODALITIES

log.info("Loading input data …")

# Check files exist before loading
missing = []
for m in modalities:
    path = os.path.join(EXAMPLES_DIR, m, EXAMPLE_FILE)
    if not os.path.isfile(path):
        missing.append(path)

if missing:
    log.error(
        "Missing example files — download them on a login node:\n"
        + "\n".join(f"  {p}" for p in missing)
        + "\n\n"
        "  Each modality sub-folder (S2L2A, S1RTC, DEM, LULC, NDVI) must\n"
        "  contain a .tif file with the same filename.\n"
        "  Use hf_hub_download or copy from the TerraMind repo examples/."
    )
    sys.exit(1)

# Load each modality  (notebook dict comprehension — unchanged)
data = {m: rxr.open_rasterio(os.path.join(EXAMPLES_DIR, m, EXAMPLE_FILE)) for m in modalities}

# Tensor with shape [B, C, 224, 224]  (notebook block — unchanged)
data = {
    k: torch.Tensor(v.values).unsqueeze(0)   # [B, C, H, W]
    for k, v in data.items()
}

log.info("Input tensors loaded:")
for m, t in data.items():
    log.info(f"  {m}: shape={t.shape}  min={t.min():.2f}  max={t.max():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Any-to-any generation  (notebook loop — unchanged, logging added)
# ─────────────────────────────────────────────────────────────────────────────

outputs = {}   # outputs[input_modality] = {output_modality: tensor, ...}

for m in modalities:
    log.info(f"\n── Input modality: {m} ─────────────────────────────────────────")

    # Determine output modalities for this run  (notebook lines — unchanged)
    out_modalities = modalities[:]
    out_modalities.remove(m)

    # Init model  (notebook block — unchanged)
    log.info(f"  Building model with output modalities: {out_modalities}")
    model = FULL_MODEL_REGISTRY.build(
        MODEL_NAME,
        modalities=[m],
        output_modalities=out_modalities,
        pretrained=True,
        standardize=True,
    )
    model = model.to(device)
    model.eval()

    # Run generation  (notebook block — unchanged)
    input_t = data[m].clone().to(device)
    log.info(f"  Running generation ({TIMESTEPS} timesteps) …")
    with torch.no_grad():
        generated = model(input_t, verbose=True, timesteps=TIMESTEPS)

    outputs[m] = generated
    log.info(f"  Generated modalities:")
    for out_mod, val in generated.items():
        log.info(f"    {out_mod}: shape={val.shape}  min={val.min():.3f}  max={val.max():.3f}")

    # Free GPU memory between experiments
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        log.info(f"  GPU cache cleared. Free VRAM: "
                 f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved")

# ─────────────────────────────────────────────────────────────────────────────
# Plot any-to-any grid  (notebook code — unchanged, plt.show() → savefig)
# ─────────────────────────────────────────────────────────────────────────────

log.info("\nGenerating any-to-any grid figure …")

n_mod = len(modalities)
# Grid: rows = input modality, columns = [input itself] + [all other modalities]
fig, axes = plt.subplots(nrows=n_mod, ncols=n_mod + 1, figsize=(12, 10))

# Column headers  (notebook lines — unchanged)
axes[0][0].set_title("Input", fontsize=8)
for i, m in enumerate(modalities):
    axes[0][i + 1].set_title(m, fontsize=8)

# Plot inputs on the diagonal  (notebook loop — unchanged)
for (m, inp), ax_row in zip(data.items(), axes):
    plot_modality(m, inp, ax=ax_row[0])
    for a in ax_row:
        a.axis("off")

# Row labels
for i, m in enumerate(modalities):
    axes[i][0].set_ylabel(f"Input: {m}", fontsize=7, rotation=90, labelpad=4)

# Fill in generations  (notebook loop — unchanged)
for k, m_output in enumerate(outputs.values()):
    for m, out in m_output.items():
        j = modalities.index(m) + 1
        plot_modality(m, out, ax=axes[k][j])

fig.suptitle(
    f"TerraMind any-to-any generation — {EXAMPLE_FILE} — {TIMESTEPS} steps",
    fontsize=10,
)
plt.tight_layout()

# Save  (replaces plt.savefig(...) + plt.show() from notebook)
grid_filename = f"any_to_any_{os.path.splitext(EXAMPLE_FILE)[0]}.png"
grid_path = os.path.join(OUTPUT_DIR, grid_filename)
fig.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(f"Saved any-to-any grid: {grid_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Individual figures for each (input → output) pair
# ─────────────────────────────────────────────────────────────────────────────

log.info("Saving individual pair figures …")
for input_mod, m_output in outputs.items():
    for out_mod, val in m_output.items():
        fig_pair, axes_pair = plt.subplots(1, 2, figsize=(8, 4))
        plot_modality(input_mod, data[input_mod], ax=axes_pair[0])
        axes_pair[0].set_title(f"Input: {input_mod}", fontsize=9)
        plot_modality(out_mod, val, ax=axes_pair[1])
        axes_pair[1].set_title(f"Generated: {out_mod}", fontsize=9)
        fig_pair.suptitle(
            f"{input_mod} → {out_mod} — {EXAMPLE_FILE}", fontsize=9
        )
        fig_pair.tight_layout()
        pair_path = os.path.join(OUTPUT_DIR, f"{input_mod}_to_{out_mod}.png")
        fig_pair.savefig(pair_path, dpi=150, bbox_inches="tight")
        plt.close(fig_pair)
        log.info(f"  Saved: {pair_path}")

# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 70)
log.info("Done.")
log.info(f"All outputs saved to: {OUTPUT_DIR}/")
log.info(
    "INTERPRETATION NOTES:\n"
    "  • Good generations indicate the model's pre-training covers the input\n"
    "    modality well — useful domain for cross-modal synthesis.\n"
    "  • Poor or blurry generations suggest out-of-domain inputs (expected\n"
    "    for Antarctic scenes at fine drone resolution).\n"
    "  • DEM generation in tiled mode can look patchy (elevation is global;\n"
    "    the model can't infer it from local texture alone)."
)
log.info("=" * 70)
