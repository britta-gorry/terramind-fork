#!/usr/bin/env python3
"""
generate_pseudolabels.py
========================
Uses TerraMind's generative model to produce LULC pseudo-labels for a folder
of input images. Runs BEFORE fine-tuning — the output folder can then be used
as a label source in your TerraTorch config.

This script is called by run_experiment.py when the GENERATE step is enabled.
You do not need to edit this file. All settings come from the start config in
run_experiment.py (or from command-line arguments when called directly).

Usage (direct):
    pixi run python scripts/generate_pseudolabels.py \
        --input_dir  data/sen1floods11_v1.1/data/S2L1CHand \
        --output_dir output/myrun_20260601_1200/pseudolabels \
        --modality   S2L1C \
        --img_glob   "*_S2Hand.tif" \
        --split_file data/sen1floods11_v1.1/splits/flood_train_data.txt

Supported input modalities: S2L2A, S2L1C, S1GRD, S1RTC, DEM, LULC, NDVI
Output: one GeoTIFF per input image, integer class index (0-9), saved to
        <output_dir>/<stem>_lulc.tif

Note on S2L1C vs S2L2A:
    TerraMind's generation tokenizer was trained on S2L2A (12 bands, surface
    reflectance). Sen1Floods11 uses S2L1C (13 bands, top-of-atmosphere). We
    handle this by dropping S2L1C band 0 (the coastal/aerosol band) and then
    treating the remaining 12 bands as if they were S2L2A. This is a practical
    approximation — not ideal but sufficient for producing pseudo-labels.
    The --s2l1c_as_l2a flag (True by default) enables this behaviour.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import rasterio                          # for reading/writing GeoTIFFs
import torch
from rasterio.transform import from_bounds

# ──────────────────────────────────────────────────────────────────────────────
# SUPPORTED INPUT MODALITY → GENERATION CONFIG
# Maps the modality name you pass in to the correct TerraMind tokenizer key.
# ──────────────────────────────────────────────────────────────────────────────
_MODALITY_MAP = {
    "S2L2A":  "S2L2A",   # 12-band surface reflectance sentinel-2 (ideal input)
    "S2L1C":  "S2L2A",   # 13-band TOA → we drop band 0 and treat as S2L2A (approximation)
    "S1GRD":  "S1GRD",   # 2-band Sentinel-1 GRD
    "S1RTC":  "S1RTC",   # 2-band Sentinel-1 RTC
    "DEM":    "DEM",     # 1-band digital elevation model
    "NDVI":   "NDVI",    # 1-band NDVI
}

# Only these modalities have TerraMind generation tokenizers
_VALID_MODALITIES = set(_MODALITY_MAP.keys())

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

_window = 80 # window size for printing dividers

def _setup_logging(log_path: Path | None = None):
    """Set up logging to stdout (and optionally a file)."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

def _log(msg: str):
    logging.getLogger().info(msg)


# ──────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ──────────────────────────────────────────────────────────────────────────────

def find_images(input_dir: Path, img_glob: str, split_file: Path | None) -> list[Path]:
    """
    Return a sorted list of image paths to process.

    If a split_file is given, only images whose stem appears in that file are
    returned. This mirrors how TerraTorch uses split files so you generate
    pseudo-labels for exactly the scenes you will train/test on.

    The split file format is one stem per line, e.g.:
        Bolivia_103160_S2Hand     (Sen1Floods11 style — modality suffix stripped)
        hls_burn_scars_...        (BurnScars style)
    """
    all_images = sorted(input_dir.glob(img_glob))       # all TIFs matching the pattern
    if not all_images:
        _log(f"  WARNING: No images found in {input_dir} with glob '{img_glob}'")
        return []

    if split_file and split_file.exists():
        # Read the split file and build a set of stems to keep
        split_stems = set()
        for line in split_file.read_text().splitlines():
            stem = line.strip()
            if stem:
                split_stems.add(stem)
        # Keep only images whose stem appears in the split list.
        # We check both the full stem and a truncated version because split files
        # sometimes list scene IDs that are substrings of the actual filename stem.
        filtered = []
        for img in all_images:
            img_stem = img.stem                          # e.g. "Bolivia_103160_S2Hand"
            # Direct match
            if img_stem in split_stems:
                filtered.append(img)
                continue
            # Substring match: split stem is within the file stem
            for s in split_stems:
                if s in img_stem or img_stem in s:
                    filtered.append(img)
                    break
        _log(f"  Split file: {split_file.name}  ({len(split_stems)} entries)")
        _log(f"  Images after split filter: {len(filtered)} / {len(all_images)}")
        return filtered

    _log(f"  No split file — processing all {len(all_images)} images in {input_dir}")
    return all_images


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_model(MODEL: str, input_modality: str, output_modalities: list[str], device: str):
    """
    Build and return the TerraMind generation model.

    This is done once before the image loop to avoid re-downloading weights
    on every iteration. The model takes ~3-5 GB of GPU RAM.
    """
    _log(f"  Loading TerraMind generation model ...")
    _log(f"      model           : {MODEL}")
    _log(f"      input  modality : {input_modality}")
    _log(f"      output modalities: {output_modalities}")

    # Import here so that import errors surface with a clear message
    try:
        from terratorch.registry import FULL_MODEL_REGISTRY
    except ImportError as e:
        _log(f"  ERROR: Could not import TerraTorch: {e}")
        _log("  Make sure you are running inside your pixi environment.")
        sys.exit(1)

    model = FULL_MODEL_REGISTRY.build(
        MODEL,
        modalities=[input_modality],         # what we pass in
        output_modalities=output_modalities, # what we want out
        pretrained=True,
        standardize=True,                    # model handles normalisation internally
    )
    model = model.to(device)
    model.eval()                             # no dropout, no gradient tracking
    _log(f"  Model loaded on {device}.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_image(img_path: Path, modality: str, s2l1c_as_l2a: bool) -> tuple[torch.Tensor, dict]:
    """
    Load a single GeoTIFF and return (tensor [1, C, H, W], rasterio profile).

    The profile is saved alongside the output so the pseudo-label GeoTIFF has
    the same CRS and spatial extent as the input image.
    """
    with rasterio.open(img_path) as src:
        data = src.read().astype(np.float32)  # shape: [C, H, W]
        profile = src.profile.copy()          # CRS, transform, shape

    # S2L1C → S2L2A band-count fix
    # Sen1Floods11 S2L1C has 13 bands; the first band is the coastal/aerosol (B01).
    # TerraMind's S2L2A tokenizer expects 12 bands. We drop band 0 as an approximation.
    if modality == "S2L1C" and s2l1c_as_l2a:
        if data.shape[0] == 13:
            data = data[1:, :, :]             # drop band 0, keep bands 1–12
        elif data.shape[0] == 12:
            pass                              # already 12 bands, nothing to do
        else:
            _log(f"  WARNING: {img_path.name} has {data.shape[0]} bands, expected 13 for S2L1C.")

    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # [1, C, H, W]
    return tensor, profile


def save_lulc(lulc_tensor: torch.Tensor, out_path: Path, src_profile: dict):
    """
    Save the LULC prediction as a single-band uint8 GeoTIFF.

    lulc_tensor: shape [10, H, W] — class logits from the generation model.
    We take argmax to get integer class indices 0-9.
    """
    # [10, H, W] → argmax → [H, W], integer class index
    lulc_array = lulc_tensor.cpu().numpy()
    if lulc_array.ndim == 3 and lulc_array.shape[0] == 10:
        lulc_class = lulc_array.argmax(axis=0).astype(np.uint8)  # [H, W]
    elif lulc_array.ndim == 2:
        lulc_class = lulc_array.astype(np.uint8)                 # already argmaxed
    else:
        _log(f"  WARNING: unexpected LULC shape {lulc_array.shape}, skipping save.")
        return

    # Build the output profile: 1-band uint8, same CRS/transform as input
    out_profile = src_profile.copy()
    out_profile.update(
        count=1,          # single band
        dtype="uint8",    # class indices 0–9 fit in uint8
        nodata=255,       # 255 = no-data sentinel (same convention as many datasets)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(lulc_class[np.newaxis, :, :])  # rasterio expects [bands, H, W]


# ──────────────────────────────────────────────────────────────────────────────
# TILED INFERENCE (for images larger than 224×224)
# ──────────────────────────────────────────────────────────────────────────────

def run_generation_tiled(model, tensor: torch.Tensor, device: str,
                          output_modalities: list[str],
                          crop: int = 224, stride: int = 168,
                          timesteps: int = 10) -> dict[str, torch.Tensor]:
    """
    Run generation using TerraTorch's tiled_inference for images larger than 224px.

    Returns a dict {modality_name: tensor [C, H, W]} matching output_modalities.
    """
    try:
        from terratorch.tasks.tiled_inference import tiled_inference
    except ImportError as e:
        _log(f"  ERROR importing tiled_inference: {e}")
        sys.exit(1)

    # Channel counts for each modality (needed to unstack the concatenated output)
    _CHANNEL_COUNTS = {"S2L2A": 12, "S2L1C": 12, "S1GRD": 2, "S1RTC": 2,
                       "DEM": 1, "LULC": 10, "NDVI": 1}
    num_channels = {m: _CHANNEL_COUNTS[m] for m in output_modalities}

    def model_forward(x):
        """Wrapper so tiled_inference gets a plain tensor back."""
        generated = model(x, timesteps=timesteps)
        return torch.cat([generated[m] for m in output_modalities], dim=1)

    input_device = tensor.to(device)
    pred = tiled_inference(model_forward, input_device,
                           crop=crop, stride=stride,
                           batch_size=4, verbose=False)   # [1, C_total, H, W]
    pred = pred.squeeze(0).cpu()                           # [C_total, H, W]

    # Split back into per-modality tensors
    start_idx = list(np.cumsum([0] + list(num_channels.values())))
    results = {}
    for i, (mod, n) in enumerate(num_channels.items()):
        results[mod] = pred[start_idx[i]: start_idx[i] + n]
    return results


def run_generation_direct(model, tensor: torch.Tensor, device: str,
                           timesteps: int = 10) -> dict[str, torch.Tensor]:
    """
    Run generation directly (no tiling) for 224x224 images.
    Returns a dict {modality_name: tensor [C, H, W]}.
    """
    with torch.no_grad():
        generated = model(tensor.to(device), timesteps=timesteps)
    # Move all outputs to CPU and remove batch dimension
    return {mod: val.squeeze(0).cpu() for mod, val in generated.items()}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def generate_pseudolabels(
    input_dir:         Path,
    output_dir:        Path,
    modality:          str,
    MODEL:             str,
    img_glob:          str,
    split_file:        Path | None,
    timesteps:         int,
    use_tiling:        bool,
    tile_crop:         int,
    tile_stride:       int,
    s2l1c_as_l2a:      bool,
    output_modalities: list[str],
    device:            str,
):
    """Main loop: load each image, run generation, save LULC GeoTIFF."""

    t0 = time.time()
    _log(f"{'='*_window}")
    _log(f"  GENERATE PSEUDO-LABELS")
    _log(f"  model      : {MODEL}")
    _log(f"  input_dir  : {input_dir}")
    _log(f"  output_dir : {output_dir}")
    _log(f"  modality   : {modality}")
    _log(f"  img_glob   : {img_glob}")
    _log(f"  outputs    : {output_modalities}")
    _log(f"  timesteps  : {timesteps}")
    _log(f"  tiling     : {'yes (crop=%d, stride=%d)' % (tile_crop, tile_stride) if use_tiling else 'no (direct 224px)'}")
    _log(f"  device     : {device}")
    _log(f"{'='*_window}")

    # Validate modality
    if modality not in _VALID_MODALITIES:
        _log(f"  ERROR: modality '{modality}' not supported. Choose from: {sorted(_VALID_MODALITIES)}")
        sys.exit(1)

    # Map S2L1C → S2L2A for the generation tokenizer
    tokenizer_modality = _MODALITY_MAP[modality]
    if modality == "S2L1C" and s2l1c_as_l2a:
        _log("  NOTE: S2L1C input → dropping band 0 (coastal/aerosol) → treating as S2L2A.")
        _log("        This is an approximation. Results may differ from true S2L2A inputs.")

    # Discover images
    images = find_images(input_dir, img_glob, split_file)
    if not images:
        _log("  No images to process. Exiting.")
        return
    _log(f" Found {len(images)} images to process.")

    # Load model once (weights are ~1GB, expensive to reload per image)
    model = load_model(MODEL, tokenizer_modality, output_modalities, device)

    # Track results for summary
    n_ok, n_skip, n_fail = 0, 0, 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(images):
        stem = img_path.stem                                    # filename without extension
        out_path = output_dir / f"{stem}_lulc.tif"

        _log(f"[{i+1}/{len(images)}] {img_path.name}")

        # Skip if already generated (allows resuming interrupted runs)
        if out_path.exists():
            # _log(f"    → already exists, skipping.")
            n_skip += 1
            continue

        try:
            # Load image
            tensor, profile = load_image(img_path, modality, s2l1c_as_l2a)
            H, W = tensor.shape[2], tensor.shape[3]
            # _log(f"    shape: {tuple(tensor.shape)}  (H={H}, W={W})")

            # Choose tiled vs direct inference
            if use_tiling or H > 224 or W > 224:
                if not use_tiling:
                    _log(f"Image larger than 224px — switching to tiled inference automatically.")
                results = run_generation_tiled(
                    model, tensor, device, output_modalities,
                    crop=tile_crop, stride=tile_stride, timesteps=timesteps,
                )
            else:
                results = run_generation_direct(model, tensor, device, timesteps=timesteps)

            # Save each requested output modality (currently LULC by default)
            for out_modality, out_tensor in results.items():
                if out_modality == "LULC":
                    save_lulc(out_tensor, out_path, profile)
                    _log(f"    ✓ saved LULC → {out_path.name}")
                else:
                    # For non-LULC modalities, save the raw float32 GeoTIFF
                    raw_path = output_dir / f"{stem}_{out_modality.lower()}.tif"
                    raw_profile = profile.copy()
                    raw_profile.update(count=out_tensor.shape[0], dtype="float32", nodata=np.nan)
                    with rasterio.open(raw_path, "w", **raw_profile) as dst:
                        dst.write(out_tensor.cpu().numpy())
                    _log(f"    ✓ saved {out_modality} → {raw_path.name}")

            n_ok += 1

        except Exception as exc:
            _log(f"    ✗ FAILED: {exc}")
            import traceback
            _log(traceback.format_exc())
            n_fail += 1

    # Final summary
    elapsed = time.time() - t0
    _log(f"{'='*_window}")
    _log(f"  GENERATION COMPLETE")
    _log(f"  processed : {len(images)} images")
    _log(f"  ok        : {n_ok}")
    _log(f"  skipped   : {n_skip}  (already existed)")
    _log(f"  failed    : {n_fail}")
    _log(f"  duration  : {elapsed:.1f}s  ({elapsed/max(n_ok,1):.1f}s/image avg)")
    _log(f"  output    : {output_dir}")
    _log(f"{'='*_window}\n")

    if n_fail > 0:
        sys.exit(1)   # Signal failure to the parent run_experiment.py


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate LULC pseudo-labels using TerraMind generation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--input_dir",  type=Path, required=True,
        help="Directory containing input image TIFs.")
    parser.add_argument("--output_dir", type=Path, required=True,
        help="Directory to write pseudo-label GeoTIFFs.")
    parser.add_argument("--modality",   type=str,  required=True,
        help=f"Input modality. One of: {sorted(_VALID_MODALITIES)}")

    # Optional
    parser.add_argument("--model", type=str, default="terramind_v1_base_generate",
        help="TerraMind model from model registry.")
    parser.add_argument("--img_glob",   type=str,  default="*.tif",
        help="Glob pattern to match input images.")
    parser.add_argument("--split_file", type=Path, default=None,
        help="Path to a split .txt file. If given, only listed scenes are processed.")
    parser.add_argument("--timesteps",  type=int,  default=10,
        help="Diffusion timesteps. More = slower but higher quality. 10 is a good default.")
    parser.add_argument("--output_modalities", type=str, nargs="+", default=["LULC"],
        help="Which modalities to generate. Default is LULC for pseudo-labels.")
    parser.add_argument("--use_tiling", action="store_true", default=False,
        help="Force tiled inference even for small images.")
    parser.add_argument("--tile_crop",  type=int, default=224,
        help="Tile crop size for tiled inference.")
    parser.add_argument("--tile_stride", type=int, default=168,
        help="Tile stride for tiled inference. Smaller = more overlap = smoother boundaries.")
    parser.add_argument("--s2l1c_as_l2a", action="store_true", default=True,
        help="For S2L1C input: drop band 0 and treat as S2L2A (default: True).")
    parser.add_argument("--log_file",   type=Path, default=None,
        help="Optional path to write a log file.")

    args = parser.parse_args()

    # Setup logging
    _setup_logging(args.log_file)

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        _log("  WARNING: CUDA not available — running on CPU will be very slow.")
    _log(f"  Using device: {device}")

    generate_pseudolabels(
        input_dir         = args.input_dir,
        output_dir        = args.output_dir,
        modality          = args.modality,
        MODEL             = args.model,
        img_glob          = args.img_glob,
        split_file        = args.split_file,
        timesteps         = args.timesteps,
        use_tiling        = args.use_tiling,
        tile_crop         = args.tile_crop,
        tile_stride       = args.tile_stride,
        s2l1c_as_l2a      = args.s2l1c_as_l2a,
        output_modalities = args.output_modalities,
        device            = device,
    )


if __name__ == "__main__":
    main()
