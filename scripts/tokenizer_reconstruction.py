#!/usr/bin/env python3
"""
tokenizer_reconstruction.py
============================
Tile a stacked S2L2A GeoTIFF into 224x224 patches, run each patch through
TerraMind's S2L2A tokenizer (encode → decode), compute reconstruction error
(MAE and RMSE) per patch, and produce visualisations and a summary CSV.

This is the domain-gap diagnostic: scenes from TerraMind's training distribution
(temperate/global) should reconstruct well (low error). Antarctic scenes should
reconstruct poorly (high error) because the model has never seen them.
Running multiple scenes lets you compare quantitatively.

Usage:
    # Single scene:
    pixi run python scripts/tokenizer_reconstruction.py \
        --input   data/stacked/my_antarctic_scene.tif \
        --output  output/reconstruction/antarctic_scene \
        --label   "Antarctic (moss/lichen)"

    # Multiple scenes for comparison:
    pixi run python scripts/tokenizer_reconstruction.py \
        --input   data/stacked/scene1.tif data/stacked/scene2.tif \
        --label   "Antarctic" "Temperate" \
        --output  output/reconstruction/comparison

Output files (inside --output directory):
    reconstruction_metrics.csv   — per-patch MAE and RMSE
    summary.txt                  — mean/std MAE and RMSE per scene
    error_map.png                — spatial heatmap of MAE across the scene
    rgb_input_reconstruction.png — side-by-side RGB input vs reconstruction
    patch_examples.png           — best, median, and worst patches
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for HPC
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.transform import from_bounds

from terratorch import FULL_MODEL_REGISTRY
from terratorch.models.backbones.terramind.model.terramind_register import v1_pretraining_mean, v1_pretraining_std


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
def _setup_logging(log_path: Path | None = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

def _log(msg): logging.getLogger().info(msg)


# ──────────────────────────────────────────────────────────────────────────────
# TERRAMIND NORMALISATION STATS (S2L2A)
# Used to normalise before the tokenizer and denormalise after.
# ──────────────────────────────────────────────────────────────────────────────
# S2L2A_MEAN = torch.tensor([1390.458, 1503.317, 1718.197, 1853.910, 2199.100,
#                             2779.975, 2987.011, 3083.234, 3132.220, 3162.988,
#                             2424.884, 1857.648], dtype=torch.float32)

# S2L2A_STD  = torch.tensor([2106.761, 2141.107, 2038.973, 2134.138, 2085.321,
#                             1889.926, 1820.257, 1871.918, 1753.829, 1797.379,
#                             1434.261, 1334.311], dtype=torch.float32)

S2L2A_MEAN = mean = torch.tensor(v1_pretraining_mean['untok_sen2l2a@224'])
S2L2A_STD = torch.tensor(v1_pretraining_std['untok_sen2l2a@224'])

PATCH_SIZE = 224    # TerraMind native patch size
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_tokenizer(device: str):
    """Load the TerraMind S2L2A tokenizer. Weights download on first call."""
    _log("  Loading TerraMind S2L2A tokenizer...")
    # try:
    #     from terratorch.registry import FULL_MODEL_REGISTRY
    # except ImportError as e:
    #     _log(f"  ERROR: Cannot import TerraTorch: {e}")
    #     sys.exit(1)

    model = FULL_MODEL_REGISTRY.build("terramind_v1_tokenizer_s2l2a", pretrained=True)
    model = model.to(device)
    model.eval()
    _log(f"  Tokenizer loaded on {device}.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# TILING
# ──────────────────────────────────────────────────────────────────────────────
def tile_image(image: np.ndarray, patch_size: int = 224) -> tuple[list[np.ndarray], list[tuple]]:
    """
    Cut a [C, H, W] image into non-overlapping [C, 224, 224] patches.
    Returns:
        patches   : list of [C, patch_size, patch_size] arrays
        positions : list of (row, col) grid positions (0-indexed patch indices)

    Patches at the right/bottom edge are padded with zeros if the image
    dimensions are not exact multiples of patch_size. The padding is masked
    when computing error so edge patches are not penalised unfairly.
    """
    C, H, W = image.shape
    n_rows = (H + patch_size - 1) // patch_size   # ceiling division
    n_cols = (W + patch_size - 1) // patch_size

    patches   = []
    positions = []

    for r in range(n_rows):
        for c in range(n_cols):
            y0 = r * patch_size
            x0 = c * patch_size
            y1 = min(y0 + patch_size, H)
            x1 = min(x0 + patch_size, W)

            patch = np.zeros((C, patch_size, patch_size), dtype=np.float32)
            patch[:, :y1 - y0, :x1 - x0] = image[:, y0:y1, x0:x1]

            patches.append(patch)
            positions.append((r, c))

    _log(f"  Tiled into {len(patches)} patches ({n_rows} rows x {n_cols} cols)")
    return patches, positions, n_rows, n_cols


# ──────────────────────────────────────────────────────────────────────────────
# RECONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────
def reconstruct_patches(
    model,
    patches: list[np.ndarray],
    device: str,
    timesteps: int = 10,
    batch_size: int = 4,
) -> list[np.ndarray]:
    """
    Run each patch through the tokenizer (encode → decode) and return
    reconstructed patches in the original (non-normalised) value scale.

    Processes in mini-batches to avoid GPU OOM.
    """
    reconstructions = []

    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i: i + batch_size]
        _log(f"  Patch {i+1}-{min(i+len(batch_patches), len(patches))} / {len(patches)}")

        # Stack into [B, 12, 224, 224] tensor
        batch_np = np.stack(batch_patches, axis=0)
        batch_tensor = torch.tensor(batch_np, dtype=torch.float32)

        # Normalise using TerraMind stats  [B, 12, 224, 224]
        # shape broadcast: [12] → [1, 12, 1, 1]
        mean = S2L2A_MEAN[None, :, None, None]
        std  = S2L2A_STD[None, :, None, None]
        batch_norm = (batch_tensor - mean) / std

        batch_device = batch_norm.to(device)

        with torch.no_grad():
            recon_norm = model(batch_device, timesteps=timesteps)

        # Denormalise back to reflectance scale
        recon = recon_norm.cpu() * std + mean   # [B, 12, 224, 224]

        for j in range(recon.shape[0]):
            reconstructions.append(recon[j].numpy())

    return reconstructions


# ──────────────────────────────────────────────────────────────────────────────
# ERROR METRICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_patch_errors(
    patches: list[np.ndarray],
    reconstructions: list[np.ndarray],
) -> list[dict]:
    """
    Compute MAE and RMSE for each patch.
    Ignores zero-padded pixels at image edges (where original patch had zeros).
    Returns a list of dicts with keys: patch_idx, mae, rmse.
    """
    results = []
    for i, (orig, recon) in enumerate(zip(patches, reconstructions)):
        # Build a mask: valid pixels are those where at least one band in the
        # original patch is non-zero (zeros are padding or nodata)
        valid_mask = (orig.sum(axis=0) > 0)   # [H, W] boolean

        if valid_mask.sum() == 0:
            results.append({"patch_idx": i, "mae": float("nan"), "rmse": float("nan")})
            continue

        orig_valid  = orig[:, valid_mask].astype(np.float32)   # [12, N_valid]
        recon_valid = recon[:, valid_mask].astype(np.float32)

        diff = orig_valid - recon_valid
        mae  = float(np.abs(diff).mean())
        rmse = float(np.sqrt((diff ** 2).mean()))

        results.append({"patch_idx": i, "mae": mae, "rmse": rmse})

    return results


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────
def bands_to_rgb(data: np.ndarray) -> np.ndarray:
    """
    Convert [12, H, W] S2L2A array to uint8 RGB [H, W, 3].
    Uses bands B04 (index 2), B03 (index 1), B02 (index 0) with percentile stretch.
    """
    rgb = data[[2, 1, 0], :, :].transpose(1, 2, 0).astype(np.float32)  # [H, W, 3]
    valid = rgb[rgb > 0]
    if len(valid) == 0:
        return np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)
    lo = np.percentile(valid, 2)
    hi = np.percentile(valid, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def save_error_map(
    errors: list[dict],
    n_rows: int,
    n_cols: int,
    output_path: Path,
    scene_label: str,
):
    """Save a spatial heatmap of per-patch MAE values."""
    mae_grid = np.full((n_rows, n_cols), np.nan)
    for e in errors:
        idx = e["patch_idx"]
        r, c = divmod(idx, n_cols)
        mae_grid[r, c] = e["mae"]

    fig, ax = plt.subplots(figsize=(max(4, n_cols), max(3, n_rows)))
    im = ax.imshow(mae_grid, cmap="hot_r", interpolation="nearest",
                   vmin=0, vmax=np.nanpercentile(mae_grid, 95))
    plt.colorbar(im, ax=ax, label="MAE (reflectance units)")
    ax.set_title(f"Tokenizer Reconstruction Error — {scene_label}", fontsize=10)
    ax.set_xlabel("Patch column"); ax.set_ylabel("Patch row")
    plt.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _log(f"  Saved error map: {output_path}")


def save_rgb_comparison(
    image: np.ndarray,
    reconstructed_image: np.ndarray,
    output_path: Path,
    scene_label: str,
):
    """Save side-by-side RGB comparison of input and reconstruction."""
    rgb_in   = bands_to_rgb(image)
    rgb_recon = bands_to_rgb(reconstructed_image)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(rgb_in);    axes[0].set_title(f"Input — {scene_label}");      axes[0].axis("off")
    axes[1].imshow(rgb_recon); axes[1].set_title("Reconstruction");               axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _log(f"  Saved RGB comparison: {output_path}")


def save_patch_examples(
    patches: list[np.ndarray],
    reconstructions: list[np.ndarray],
    errors: list[dict],
    output_path: Path,
    scene_label: str,
):
    """Save a 3-column figure: best (lowest MAE), median, and worst (highest MAE) patch."""
    valid_errors = [(e["patch_idx"], e["mae"]) for e in errors
                    if not np.isnan(e["mae"])]
    if not valid_errors:
        return
    valid_errors.sort(key=lambda x: x[1])

    best_idx   = valid_errors[0][0]
    median_idx = valid_errors[len(valid_errors) // 2][0]
    worst_idx  = valid_errors[-1][0]

    selected = [
        ("Best (lowest MAE)",   best_idx,   valid_errors[0][1]),
        ("Median MAE",          median_idx, valid_errors[len(valid_errors) // 2][1]),
        ("Worst (highest MAE)", worst_idx,  valid_errors[-1][1]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Patch Examples — {scene_label}", fontsize=10)

    for col, (title, idx, mae) in enumerate(selected):
        rgb_in    = bands_to_rgb(patches[idx])
        rgb_recon = bands_to_rgb(reconstructions[idx])
        axes[0, col].imshow(rgb_in)
        axes[0, col].set_title(f"{title}\nMAE={mae:.1f}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(rgb_recon)
        axes[1, col].set_title("Reconstruction", fontsize=8)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Input", rotation=0, labelpad=40, fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", rotation=0, labelpad=60, fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _log(f"  Saved patch examples: {output_path}")


def reassemble_image(
    patches: list[np.ndarray],
    n_rows: int,
    n_cols: int,
    original_shape: tuple,
) -> np.ndarray:
    """Reassemble patches back into a full image for RGB comparison."""
    C, H, W = original_shape
    out = np.zeros((C, H, W), dtype=np.float32)
    patch_size = PATCH_SIZE
    for idx, patch in enumerate(patches):
        r, c = divmod(idx, n_cols)
        y0 = r * patch_size;  y1 = min(y0 + patch_size, H)
        x0 = c * patch_size;  x1 = min(x0 + patch_size, W)
        out[:, y0:y1, x0:x1] = patch[:, :y1-y0, :x1-x0]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# PROCESS ONE SCENE
# ──────────────────────────────────────────────────────────────────────────────
def process_scene(
    tif_path: Path,
    output_dir: Path,
    label: str,
    model,
    device: str,
    timesteps: int,
    batch_size: int,
    max_patches: int | None,
) -> dict:
    """
    Full pipeline for one scene: load → tile → reconstruct → metrics → plots.
    Returns a summary dict for cross-scene comparison.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"\n{'='*60}")
    _log(f"  Scene : {label}")
    _log(f"  File  : {tif_path}")
    _log(f"  Output: {output_dir}")

    # ── Load image ──────────────────────────────────────────────────
    with rasterio.open(tif_path) as src:
        image = src.read().astype(np.float32)    # [C, H, W]
        profile = src.profile
    C, H, W = image.shape
    _log(f"  Loaded: {C} bands, {H}x{W} pixels")

    if C != 12:
        _log(f"  ERROR: Expected 12 bands (S2L2A TerraMind order), got {C}.")
        _log("  Run stack_S2L2A.py first to produce a correctly ordered stack.")
        return {}

    # ── Tile ─────────────────────────────────────────────────────────
    patches, positions, n_rows, n_cols = tile_image(image, PATCH_SIZE)

    # Optionally limit patches (for quick tests)
    if max_patches and len(patches) > max_patches:
        _log(f"  Limiting to {max_patches} patches (--max_patches).")
        patches = patches[:max_patches]
        n_rows_eff = (max_patches + n_cols - 1) // n_cols
    else:
        n_rows_eff = n_rows

    # ── Reconstruct ──────────────────────────────────────────────────
    reconstructions = reconstruct_patches(model, patches, device,
                                          timesteps=timesteps, batch_size=batch_size)

    # ── Metrics ──────────────────────────────────────────────────────
    errors = compute_patch_errors(patches, reconstructions)

    valid_maes  = [e["mae"]  for e in errors if not np.isnan(e["mae"])]
    valid_rmses = [e["rmse"] for e in errors if not np.isnan(e["rmse"])]

    mean_mae  = float(np.mean(valid_maes))  if valid_maes  else float("nan")
    std_mae   = float(np.std(valid_maes))   if valid_maes  else float("nan")
    mean_rmse = float(np.mean(valid_rmses)) if valid_rmses else float("nan")
    std_rmse  = float(np.std(valid_rmses))  if valid_rmses else float("nan")

    _log(f"\n  Results for: {label}")
    _log(f"  MAE   mean={mean_mae:.1f}  std={std_mae:.1f}  (reflectance units, 0-10000 scale)")
    _log(f"  RMSE  mean={mean_rmse:.1f}  std={std_rmse:.1f}")
    _log(f"  n_patches={len(valid_maes)}")

    # ── Save metrics CSV ─────────────────────────────────────────────
    csv_path = output_dir / "reconstruction_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patch_idx", "mae", "rmse"])
        writer.writeheader()
        writer.writerows(errors)
    _log(f"  Saved: {csv_path}")

    # ── Visualisations ───────────────────────────────────────────────
    # 1. Error heatmap
    save_error_map(errors, n_rows_eff, n_cols, output_dir / "error_map.png", label)

    # 2. RGB comparison (full image)
    recon_full = reassemble_image(reconstructions, n_rows_eff, n_cols, image.shape)
    save_rgb_comparison(image, recon_full, output_dir / "rgb_comparison.png", label)

    # 3. Best / median / worst patch examples
    save_patch_examples(patches, reconstructions, errors,
                        output_dir / "patch_examples.png", label)

    # ── Summary ──────────────────────────────────────────────────────
    summary = {
        "label":      label,
        "file":       tif_path.name,
        "n_patches":  len(valid_maes),
        "mean_mae":   round(mean_mae, 2),
        "std_mae":    round(std_mae, 2),
        "mean_rmse":  round(mean_rmse, 2),
        "std_rmse":   round(std_rmse, 2),
    }
    # Write per-scene summary text
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Scene: {label}\n")
        f.write(f"File:  {tif_path}\n\n")
        f.write(f"Patches:   {len(valid_maes)}\n")
        f.write(f"MAE:   {mean_mae:.2f} ± {std_mae:.2f}  (reflectance units)\n")
        f.write(f"RMSE:  {mean_rmse:.2f} ± {std_rmse:.2f}  (reflectance units)\n")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CROSS-SCENE COMPARISON PLOT
# ──────────────────────────────────────────────────────────────────────────────
def save_comparison_plot(summaries: list[dict], output_path: Path):
    """Bar chart comparing MAE across scenes — your domain-gap figure."""
    labels    = [s["label"]    for s in summaries]
    mean_maes = [s["mean_mae"] for s in summaries]
    std_maes  = [s["std_mae"]  for s in summaries]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, mean_maes, yerr=std_maes, capsize=5,
                  color=["#3171AD" if "Antarctic" in l or "antarctic" in l else "#469C76"
                         for l in labels],
                  edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean MAE (reflectance units)")
    ax.set_title("TerraMind S2L2A Tokenizer Reconstruction Error\n(Higher = Greater Domain Gap)")
    ax.grid(axis="y", alpha=0.3)
    # Annotate bar values
    for bar, val in zip(bars, mean_maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(std_maes) * 0.05,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"  Saved comparison plot: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tokenizer reconstruction error for domain-gap analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True,
        help="One or more stacked 12-band S2L2A GeoTIFFs (from stack_S2L2A.py).")
    parser.add_argument("--output", type=Path, required=True,
        help="Output directory for metrics, plots, and summaries.")
    parser.add_argument("--label", type=str, nargs="*", default=None,
        help="Human-readable label for each input scene (same order as --input). "
             "Defaults to the filename stem.")
    parser.add_argument("--timesteps", type=int, default=10,
        help="Diffusion timesteps. 10 is fast; 50 is higher quality.")
    parser.add_argument("--batch_size", type=int, default=4,
        help="Patches per GPU batch. Reduce if you get OOM errors.")
    parser.add_argument("--max_patches", type=int, default=None,
        help="Limit patches per scene (useful for quick tests). None = process all.")
    args = parser.parse_args()

    inputs = args.input
    labels = args.label or [p.stem for p in inputs]
    if len(labels) != len(inputs):
        _log("ERROR: --label must have the same number of entries as --input.")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.output / "reconstruction.log")

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        _log("WARNING: CUDA not available. CPU will be very slow.")
    _log(f"Device: {device}")

    # Load tokenizer once
    model = load_tokenizer(device)

    # Process each scene
    summaries = []
    for tif_path, label in zip(inputs, labels):
        scene_out = args.output / label.replace(" ", "_").replace("/", "_")
        summary = process_scene(
            tif_path    = tif_path,
            output_dir  = scene_out,
            label       = label,
            model       = model,
            device      = device,
            timesteps   = args.timesteps,
            batch_size  = args.batch_size,
            max_patches = args.max_patches,
        )
        if summary:
            summaries.append(summary)

    # Cross-scene summary
    if summaries:
        # Write combined CSV
        combined_csv = args.output / "all_scenes_summary.csv"
        with open(combined_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)
        _log(f"\n  Combined summary: {combined_csv}")

        # Cross-scene comparison plot (only meaningful with >1 scene)
        if len(summaries) > 1:
            save_comparison_plot(summaries, args.output / "domain_gap_comparison.png")

        # Print table
        _log("\n  ╔══ DOMAIN GAP SUMMARY ══╗")
        _log(f"  {'Scene':<30}  {'MAE mean':>10}  {'MAE std':>9}  {'RMSE mean':>10}")
        _log("  " + "─" * 65)
        for s in summaries:
            _log(f"  {s['label']:<30}  {s['mean_mae']:>10.1f}  {s['std_mae']:>9.1f}  {s['mean_rmse']:>10.1f}")

    _log("\nDone.")


if __name__ == "__main__":
    main()
