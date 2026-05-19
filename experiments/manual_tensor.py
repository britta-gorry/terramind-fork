"""
manual_tensor.py
────────────────────────────────────────────────────────────────────────────
Build a TerraMind input tensor from scratch, one step at a time.
Based on Johan's example workflow (created by Perplexity AI).

Run from the folder that contains hls_burn_scars/:
    python manual_tensor.py
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import rasterio
import torch
from terratorch import BACKBONE_REGISTRY


# ────────────────────────────────────────────────────────────────────────────
# STEP 0 — Point at one image file
#
# Change this path to any BurnScars *_merged.tif file you have on disk.
# ────────────────────────────────────────────────────────────────────────────

IMAGE_PATH = "hls_burn_scars/data/subsetted_512x512_HLS.S30.T10SEH.2018280.v1.4.mask.tif"


# ────────────────────────────────────────────────────────────────────────────
# STEP 1 — Read the raw pixel values from the file
#
# rasterio.open() reads a GeoTIFF.
# src.read() returns a numpy array of shape [C, H, W]:
#   C = number of bands (channels) in the file
#   H = height in pixels
#   W = width  in pixels
#
# .astype(np.float32) converts from whatever the file stores (often uint16)
# to 32-bit float, which is what PyTorch expects.
# ────────────────────────────────────────────────────────────────────────────

with rasterio.open(IMAGE_PATH) as src:
    raw = src.read().astype(np.float32)   # shape: [C, H, W]

print("STEP 1 — raw pixel values loaded from file")
print(f"  shape : {raw.shape}   (C={raw.shape[0]} bands, H={raw.shape[1]}, W={raw.shape[2]})")
print(f"  dtype : {raw.dtype}")
print(f"  min   : {raw.min():.2f}")
print(f"  max   : {raw.max():.2f}")
print()


# ────────────────────────────────────────────────────────────────────────────
# STEP 2 — Normalise each band
#
# The model was pre-trained on data that was z-score normalised:
#   normalised = (pixel_value - mean) / std
#
# This centres each band around 0 with a spread of roughly 1.
# Using the WRONG means/stds would give garbage features — these must match
# whatever was used when the model was originally trained.
#
# These are the official TerraMind BurnScars statistics from the YAML config.
# Each number corresponds to one band in the order: BLUE, GREEN, RED,
# NIR_NARROW, SWIR_1, SWIR_2.
# ────────────────────────────────────────────────────────────────────────────

MEANS = np.array([0.033349706741586264,
                  0.05701185520536176,
                  0.05889748132001316,
                  0.2323245113436119,
                  0.1972854853760658,
                  0.11944914225186566], dtype=np.float32)

STDS  = np.array([0.02269135568823774,
                  0.026807560223070237,
                  0.04004109844362779,
                  0.07791732423672691,
                  0.08708738838140137,
                  0.07241979477437814], dtype=np.float32)

# Reshape to [C, 1, 1] so numpy broadcasts across H and W automatically.
# Without this reshape, the subtraction would fail because raw is [C, H, W]
# and means is [C] — the dimensions don't line up.
means_bcw = MEANS.reshape(-1, 1, 1)   # [6, 1, 1]
stds_bcw  = STDS.reshape(-1, 1, 1)    # [6, 1, 1]

normalised = (raw - means_bcw) / (stds_bcw + 1e-8)

# Replace any NaN or Inf that came from no-data pixels with 0.
# np.nan_to_num does this in one call.
normalised = np.nan_to_num(normalised, nan=0.0, posinf=0.0, neginf=0.0)

print("STEP 2 — normalised (z-score: subtract mean, divide by std)")
print(f"  shape : {normalised.shape}")
print(f"  min   : {normalised.min():.4f}   (should be roughly -3 to +3)")
print(f"  max   : {normalised.max():.4f}")
print(f"  mean  : {normalised.mean():.4f}  (should be close to 0)")
print()


# ────────────────────────────────────────────────────────────────────────────
# STEP 3 — Convert to a PyTorch tensor and add a batch dimension
#
# torch.from_numpy() wraps the numpy array without copying the data.
# .unsqueeze(0) inserts a new dimension at position 0.
#
# The model always expects input shape [B, C, H, W] where B is batch size.
# We only have one image here, so B=1.
#
# Before unsqueeze:  [C, H, W]  = [6, 512, 512]
# After  unsqueeze:  [B, C, H, W] = [1, 6, 512, 512]
# ────────────────────────────────────────────────────────────────────────────

tensor = torch.from_numpy(normalised).unsqueeze(0)   # [1, C, H, W]

print("STEP 3 — converted to PyTorch tensor with batch dimension added")
print(f"  shape : {tensor.shape}   [B, C, H, W]")
print(f"  dtype : {tensor.dtype}")
print()


# ────────────────────────────────────────────────────────────────────────────
# STEP 4 — Load the TerraMind backbone
#
# BACKBONE_REGISTRY.build() downloads pretrained weights and returns the
# encoder (backbone) part of the model.
#
# modalities tells the model which sensor type this data comes from.
# bands tells the model which specific spectral bands we are providing,
# so it can select the correct learned embeddings for each band.
#
# We are NOT loading the full segmentation head here — just the backbone
# that converts image patches into feature vectors.  This is enough to
# confirm that our tensor flows through the model correctly.
# ────────────────────────────────────────────────────────────────────────────

print("STEP 4 — loading TerraMind backbone (downloading weights if needed)")

backbone = BACKBONE_REGISTRY.build(
    "terramind_v1_base",
    pretrained=True,
    modalities=["S2L2A"],
    bands={
        "S2L2A": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
    },
)

backbone.eval()

n_params = sum(p.numel() for p in backbone.parameters())
print(f"  backbone loaded  ✓")
print(f"  parameters : {n_params:,}")
print()


# ────────────────────────────────────────────────────────────────────────────
# STEP 5 — Run a forward pass through the backbone
#
# The backbone expects a dict: {modality_name: tensor}
# This matches the format of batch["image"] from the DataLoader.
#
# torch.no_grad() tells PyTorch not to track gradients — we are just doing
# inference, not training, so this saves memory and speeds things up.
#
# The output is a list of feature tensors, one per transformer layer selected
# by SelectIndices in the full model.  At this stage (backbone only) we get
# the raw patch token outputs.
# ────────────────────────────────────────────────────────────────────────────

print("STEP 5 — running forward pass through the backbone")

with torch.no_grad():
    output = backbone({"S2L2A": tensor})

print(f"  input  shape : {tensor.shape}")
print(f"  output type  : {type(output)}")

# The backbone returns a list of tensors (one per selected layer).
if isinstance(output, (list, tuple)):
    print(f"  output is a list of {len(output)} feature tensors (one per neck layer):")
    for idx, feat in enumerate(output):
        print(f"    [{idx}] shape: {feat.shape}")
else:
    print(f"  output shape : {output.shape}")

print()
print("Pipeline complete — the tensor flowed through the backbone without errors.")
print("This same tensor format is what the full SemanticSegmentationTask receives.")
