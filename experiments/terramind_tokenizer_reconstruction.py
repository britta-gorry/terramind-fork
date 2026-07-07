#!/usr/bin/env python
"""
terramind_tokenizer_reconstruction.py
Converted from: terramind_tokenizer_reconstruction.ipynb

PURPOSE:
  Tests TerraMind's visual tokenizer in isolation.  The tokenizer encodes
  an input image into discrete tokens, then decodes those tokens back to
  pixel space using the diffusion decoder.  The side-by-side comparison of
  input vs. reconstruction shows how much information the tokenizer retains.

  The script also demonstrates the encode/decode API separately so you can
  inspect intermediate token representations — useful for understanding the
  model's internal feature space.

RELEVANCE TO YOUR RESEARCH:
  The tokenizer is the bridge between raw satellite data and the model's
  internal world.  For your Antarctic domain adaptation work:
    • Tokenizer reconstruction quality tells you how well the pre-training
      distribution captures your data's statistics.
    • You can encode your drone data into tokens and inspect their structure
      without any labels — a form of unsupervised representation analysis.
    • Passing drone imagery through the tokenizer and measuring reconstruction
      error is a quantitative domain-gap metric.

  Available tokenizers (set TOKENIZER_NAME below):
    terramind_v1_tokenizer_s2l2a   — Sentinel-2 L2A  (12 bands)
    terramind_v1_tokenizer_s1rtc   — Sentinel-1 RTC   (2 bands)
    terramind_v1_tokenizer_dem     — Digital Elevation Model (1 band)
    terramind_v1_tokenizer_lulc    — Land-Use / Land-Cover (10-class)
    terramind_v1_tokenizer_ndvi    — NDVI (1 band)

HPC USAGE:
  Interactive:  python terramind_tokenizer_reconstruction.py
  Via PBS:      qsub submit_tokenizer.pbs

PREREQUISITES:
  - Example .tif files (see EXAMPLES_DIR below).

OUTPUT:
  output/tokenizer_recon_YYYYMMDD_HHMM/
    run.log                 — full terminal output
    reconstruction.png      — side-by-side: input RGB vs. reconstructed RGB
    input_rgb.png           — input alone
    reconstruction_rgb.png  — reconstruction alone
    tokens_info.txt         — shape and stats of the intermediate token tensor
"""

# ─────────────────────────────────────────────────────────────────────────────
# MUST come before any other matplotlib import
import matplotlib
matplotlib.use("Agg")
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Directory containing S2L2A example .tif files
EXAMPLES_DIR = "examples"

# Which example to use (0–4)
EXAMPLE_ID = 1

# Which tokenizer to use.  Options listed in the docstring above.
# To use a different modality, also update MODALITY_KEY and RGB_BANDS below.
TOKENIZER_NAME = "terramind_v1_tokenizer_s2l2a"

# The normalisation key for this modality.
# Run: print(list(v1_pretraining_mean.keys()))  to see all available keys.
NORM_KEY = "untok_sen2l2a@224"

# Modality name — used to find the right pre-training stats
MODALITY_KEY = "S2L2A"

# Band indices for an RGB preview (0-indexed).
# S2L2A: bands [3, 2, 1] = Red, Green, Blue
RGB_BANDS = [3, 2, 1]

# Scale factor for RGB display (raw S2L2A values are ~0–10000; /2000 → ~0–1)
RGB_SCALE = 2000.0

# Number of diffusion timesteps for the decoder
TIMESTEPS = 10

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
OUTPUT_DIR = os.path.join("output", f"tokenizer_recon_{timestamp}")
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
log.info("TerraMind Tokenizer Reconstruction")
log.info(f"  Script:          terramind_tokenizer_reconstruction.py")
log.info(f"  Tokenizer:       {TOKENIZER_NAME}")
log.info(f"  Norm key:        {NORM_KEY}")
log.info(f"  Example ID:      {EXAMPLE_ID}")
log.info(f"  Diffusion steps: {TIMESTEPS}")
log.info(f"  Output dir:      {OUTPUT_DIR}")
log.info("=" * 70)

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
# Build model  (notebook code — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

from terratorch import FULL_MODEL_REGISTRY
from terratorch.models.backbones.terramind.model.terramind_register import (
    v1_pretraining_mean, v1_pretraining_std,
)

log.info(f"Building tokenizer model: {TOKENIZER_NAME}")
log.info("  (Downloading weights if not cached — may take a minute)")

# Available tokenizer models (notebook comments — unchanged):
# model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_s2l2a', pretrained=True)
# model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_s1rtc', pretrained=True)
# model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_dem',   pretrained=True)
# model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_lulc',  pretrained=True)
# model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_ndvi',  pretrained=True)

model = FULL_MODEL_REGISTRY.build(TOKENIZER_NAME, pretrained=True)
model = model.to(device)
model.eval()
log.info("Tokenizer model built.")

# Print all available normalisation keys (informational)
log.info("All available normalisation keys (v1_pretraining_mean):")
for k in v1_pretraining_mean:
    log.info(f"  {k}")

# ─────────────────────────────────────────────────────────────────────────────
# Load data  (notebook code — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

import rioxarray as rxr

examples = [
    os.path.join(EXAMPLES_DIR, "S2L2A", "38D_378R_2_3.tif"),
    os.path.join(EXAMPLES_DIR, "S2L2A", "282D_485L_3_3.tif"),
    os.path.join(EXAMPLES_DIR, "S2L2A", "433D_629L_3_1.tif"),
    os.path.join(EXAMPLES_DIR, "S2L2A", "637U_59R_1_3.tif"),
    os.path.join(EXAMPLES_DIR, "S2L2A", "609U_541L_3_0.tif"),
]

example_path = examples[EXAMPLE_ID]

if not os.path.isfile(example_path):
    log.error(
        f"Example file not found: {example_path}\n"
        "  Download from the TerraMind repo or Hugging Face (on a login node)."
    )
    sys.exit(1)

log.info(f"Loading: {example_path}")

# Select example between 0 and 4  (notebook line — unchanged)
data = rxr.open_rasterio(example_path)
# Convert to shape [B, C, 224, 224]  (notebook line — unchanged)
data = torch.Tensor(data.values).unsqueeze(0)

log.info(f"Input tensor shape: {data.shape}  (batch, channels, H, W)")
log.info(f"Input value range:  min={data.min():.1f}  max={data.max():.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# Visualise input RGB  (notebook code — unchanged, plt.show() → savefig)
# ─────────────────────────────────────────────────────────────────────────────

def render_rgb(tensor_BCHW, rgb_bands=RGB_BANDS, scale=RGB_SCALE):
    """Convert a [B, C, H, W] or [C, H, W] tensor to a uint8 [H, W, 3] array."""
    t = tensor_BCHW[0] if tensor_BCHW.ndim == 4 else tensor_BCHW
    t = t.cpu() if hasattr(t, 'cpu') else t
    rgb = t[rgb_bands].permute(1, 2, 0).clone()
    rgb = (rgb / scale).clip(0, 1) * 255
    return rgb.numpy().round().astype(np.uint8)

# Visualize S-2 L2A input as RGB  (notebook section — unchanged)
rgb_input = render_rgb(data)

fig_in, ax_in = plt.subplots(1, 1, figsize=(6, 6))
ax_in.imshow(rgb_input)
ax_in.axis("off")
ax_in.set_title(f"{MODALITY_KEY} input — example {EXAMPLE_ID}", fontsize=10)
fig_in.tight_layout()
input_fig_path = os.path.join(OUTPUT_DIR, "input_rgb.png")
fig_in.savefig(input_fig_path, dpi=150, bbox_inches="tight")
plt.close(fig_in)
log.info(f"Saved input figure: {input_fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Normalise input  (notebook code — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

mean = torch.Tensor(v1_pretraining_mean[NORM_KEY])
std  = torch.Tensor(v1_pretraining_std[NORM_KEY])

log.info(f"Normalisation key: {NORM_KEY}")
log.info(f"  mean shape: {mean.shape}  std shape: {std.shape}")

# Normalise  (notebook line — unchanged)
input_norm = (data - mean[None, :, None, None]) / std[None, :, None, None]

log.info(f"Normalised value range: min={input_norm.min():.3f}  max={input_norm.max():.3f}")

# See keys for other modalities:  (notebook comment — retained)
log.info("  (v1_pretraining_mean.keys() = see above)")

# ─────────────────────────────────────────────────────────────────────────────
# Run encode + decode  (notebook code — unchanged, plus separate token step)
# ─────────────────────────────────────────────────────────────────────────────

input_gpu = input_norm.to(device)

log.info(f"Running tokenizer encode+decode ({TIMESTEPS} timesteps) …")

with torch.no_grad():
    # ── Option 1: encode & decode in one call  (notebook lines — unchanged) ──
    reconstruction = model(input_gpu, timesteps=TIMESTEPS)

    # ── Option 2: split encode and decode to inspect tokens  ─────────────────
    # This mirrors the commented-out section in the notebook:
    #   _, _, tokens = model.encode(input_gpu)
    #   reconstruction_split = model.decode_tokens(tokens, verbose=True, timesteps=TIMESTEPS)
    log.info("  Also running split encode/decode to inspect token tensor …")
    _, _, tokens = model.encode(input_gpu)
    log.info(f"  Token tensor shape: {tokens.shape}")
    log.info(f"  Token tensor dtype: {tokens.dtype}")
    log.info(f"  Token value range:  min={tokens.min():.4f}  max={tokens.max():.4f}")

    # Decode from tokens (should give the same result as direct reconstruction)
    reconstruction_from_tokens = model.decode_tokens(tokens, verbose=True, timesteps=TIMESTEPS)

# Denormalise  (notebook lines — unchanged)
reconstruction = reconstruction.cpu()
reconstruction = (reconstruction * std[None, :, None, None]) + mean[None, :, None, None]

reconstruction_from_tokens = reconstruction_from_tokens.cpu()
reconstruction_from_tokens = (
    reconstruction_from_tokens * std[None, :, None, None]
) + mean[None, :, None, None]

log.info(f"Reconstruction value range: min={reconstruction.min():.1f}  max={reconstruction.max():.1f}")

# ── Save token information to a text file ────────────────────────────────────
token_info_path = os.path.join(OUTPUT_DIR, "tokens_info.txt")
with open(token_info_path, "w") as f:
    f.write("TerraMind Tokenizer — Intermediate Token Statistics\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Tokenizer:       {TOKENIZER_NAME}\n")
    f.write(f"Input modality:  {MODALITY_KEY}\n")
    f.write(f"Input shape:     {data.shape}\n")
    f.write(f"Norm key:        {NORM_KEY}\n\n")
    f.write(f"Token shape:     {tokens.shape}\n")
    f.write(f"  Interpretation: (batch, num_tokens, token_dim) or similar\n\n")
    t_np = tokens.cpu().numpy()
    f.write(f"Token statistics:\n")
    f.write(f"  min:  {t_np.min():.6f}\n")
    f.write(f"  max:  {t_np.max():.6f}\n")
    f.write(f"  mean: {t_np.mean():.6f}\n")
    f.write(f"  std:  {t_np.std():.6f}\n")
    f.write(f"\nReconstruction shape:  {reconstruction.shape}\n")
    r_np = reconstruction.numpy()
    f.write(f"  min: {r_np.min():.1f}  max: {r_np.max():.1f}\n")
log.info(f"Saved token information: {token_info_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Compute reconstruction error metrics
# ─────────────────────────────────────────────────────────────────────────────

original_np = data.numpy().squeeze(0).astype(np.float32)      # (C, H, W)
recon_np    = reconstruction.numpy().squeeze(0).astype(np.float32)

mae  = np.abs(original_np - recon_np).mean()
rmse = np.sqrt(((original_np - recon_np) ** 2).mean())

log.info("Reconstruction error (in raw pixel units):")
log.info(f"  MAE:  {mae:.2f}")
log.info(f"  RMSE: {rmse:.2f}")
log.info(
    "  NOTE: Higher error in Antarctic/polar scenes (vs these global examples)\n"
    "        is a quantitative signal of domain gap."
)

# ─────────────────────────────────────────────────────────────────────────────
# Plot and save  (notebook code — unchanged, plt.show() → savefig)
# ─────────────────────────────────────────────────────────────────────────────

rgb_recon = render_rgb(reconstruction)

# Side-by-side comparison  (notebook figure layout — unchanged)
fig_cmp, ax_cmp = plt.subplots(1, 2, figsize=(10, 5))

ax_cmp[0].imshow(rgb_input)
ax_cmp[0].axis("off")
ax_cmp[0].set_title("Input", fontsize=10)

ax_cmp[1].imshow(rgb_recon)
ax_cmp[1].axis("off")
ax_cmp[1].set_title("Reconstruction", fontsize=10)

fig_cmp.suptitle(
    f"TerraMind tokenizer reconstruction — {TOKENIZER_NAME}\n"
    f"example {EXAMPLE_ID} — {TIMESTEPS} steps  |  MAE={mae:.1f}  RMSE={rmse:.1f}",
    fontsize=9,
)
fig_cmp.tight_layout()
cmp_path = os.path.join(OUTPUT_DIR, "reconstruction.png")
fig_cmp.savefig(cmp_path, dpi=150, bbox_inches="tight")
plt.close(fig_cmp)
log.info(f"Saved comparison figure: {cmp_path}")

# Reconstruction alone
fig_r, ax_r = plt.subplots(1, 1, figsize=(6, 6))
ax_r.imshow(rgb_recon)
ax_r.axis("off")
ax_r.set_title(f"Reconstruction — {TOKENIZER_NAME}", fontsize=10)
fig_r.tight_layout()
recon_path = os.path.join(OUTPUT_DIR, "reconstruction_rgb.png")
fig_r.savefig(recon_path, dpi=150, bbox_inches="tight")
plt.close(fig_r)
log.info(f"Saved reconstruction figure: {recon_path}")

# Difference map
diff = np.abs(original_np[[3, 2, 1]] - recon_np[[3, 2, 1]]).mean(axis=0)   # avg over RGB bands
fig_d, ax_d = plt.subplots(1, 1, figsize=(6, 6))
im = ax_d.imshow(diff, cmap="hot", vmin=0)
plt.colorbar(im, ax=ax_d, label="Absolute error (raw pixel units)")
ax_d.axis("off")
ax_d.set_title("Reconstruction error (mean over RGB bands)", fontsize=9)
fig_d.tight_layout()
diff_path = os.path.join(OUTPUT_DIR, "reconstruction_error_map.png")
fig_d.savefig(diff_path, dpi=150, bbox_inches="tight")
plt.close(fig_d)
log.info(f"Saved error map: {diff_path}")

# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 70)
log.info("Done.")
log.info(f"All outputs saved to: {OUTPUT_DIR}/")
log.info("=" * 70)
