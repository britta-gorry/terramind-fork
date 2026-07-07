"""
visualise_prediction_burnscars.py  —  Make burn scar map figures from a completed run

Usage (from the terramind root, in an interactive HPC session):
    pixi run python experiments/visualise_prediction_burnscars.py <run_dir> [--n 5]

Arguments:
    run_dir   The timestamped run folder, e.g. output/burnscars_terramind_base_20260427_1529
    --n       Number of scenes to plot (default: 5). Use --n 0 to plot all scenes.

What it does:
    Finds all prediction GeoTIFFs in run_dir/predictions/, matches each one
    to its input S2L1C image and ground-truth label, and saves a 3-panel
    PNG figure (RGB input | prediction | ground truth) to run_dir/figures/.

How to start an interactive session for this:
    qsub -I -l select=1:ncpus=2:mem=8gb -l walltime=01:00:00
    cd /path/to/terramind
    export PATH="$HOME/.pixi/bin:$PATH"
    pixi run python experiments/visualise_prediction_burnscars.py output/burnscars_terramind_base_20260427_1529

Relationship to the TerraMind notebook:
    The notebook's plotting cell calls dataset.plot(sample) which does the
    same thing interactively — it shows one batch of predictions side by side.
    This script does the equivalent for saved GeoTIFF predictions, which is
    what terratorch predict produces on the HPC.
"""

import argparse
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio


# ─────────────────────────────────────────────────────────────────────────────
# Colour map for the two classes
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLOURS = {
    0: (0.80, 0.80, 0.80),   # Others — grey
    1: (0.50, 0.00, 0.00),   # Burn   — burgundy
}


def label_to_rgb(arr):
    """Convert a 2D integer class array to a 3-channel RGB image for display."""
    h, w = arr.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)   # default: white (for no-data)
    for cls, colour in CLASS_COLOURS.items():
        rgb[arr == cls] = colour
    return rgb


def load_rgb(path, bands=(4, 3, 2)):
    """
    Load three bands from a multi-band GeoTIFF and return a display-ready RGB.
    bands: 1-indexed band numbers. Default (4,3,2) = Red, Green, Blue in S2L1C.
    Applies a 2-98 percentile stretch so the image looks good on screen.
    """
    with rasterio.open(path) as src:
        data = [src.read(b).astype(np.float32) for b in bands]

    def stretch(band):
        valid = band[band > 0]
        if len(valid) == 0:
            return band
        lo, hi = np.percentile(valid, [2, 98])
        return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

    return np.stack([stretch(b) for b in data], axis=-1)


def load_band1(path):
    """Load the first (and usually only) band from a single-band GeoTIFF."""
    with rasterio.open(path) as src:
        return src.read(1)


def find_source_files(pred_path):
    """
    Edited to suit burnscar dataset - data all in one folder, ground truth is .mask.tif, image is _merged.tif
    Given a prediction GeoTIFF path, try to find the matching S2L1C image
    and ground-truth label using the filename stem.

    TerraTorch predict names output files based on the input filename.
    For example:
      Input:      Bolivia_103757_S2Hand.tif
      Prediction: Bolivia_103757_S2Hand_pred.tif  (or similar)

    This function tries a few common patterns to find the match.
    Returns (image_path, label_path) — either may be None if not found.
    """
    pred_name = os.path.basename(pred_path)

    # Strip common suffixes that terratorch predict may add
    for suffix in ["_pred.tif", "_prediction.tif", ".tif"]:
        if pred_name.endswith(suffix):
            stem = pred_name[: -len(suffix)]
            break
    else:
        stem = pred_name

    # The stem may still contain _S2Hand — try to extract the scene prefix
    # e.g. "Bolivia_103757_S2Hand" → scene = "Bolivia_103757"
    scene = re.sub(r"subsetted_512x512_HLS.S30.*", "", stem)
    if not scene:
        scene = stem

    # Common dataset root locations (relative to terramind root)
    img_dir  = "data/hls_burn_scars/data"
    lbl_dir  = "data/hls_burn_scars/data"

    # Find image matching this scene
    image_path = None
    for pattern in [f"{scene}_merged.tif", f"{scene}*_merged.tif"]:
        matches = glob.glob(os.path.join(img_dir, pattern))
        if matches:
            image_path = matches[0]
            break

    # Find ground-truth label matching this scene
    label_path = None
    for pattern in [f"{scene}.mask.tif", f"{scene}*.mask.tif"]:
        matches = glob.glob(os.path.join(lbl_dir, pattern))
        if matches:
            label_path = matches[0]
            break

    return image_path, label_path


def make_figure(pred_path, image_path, label_path, out_path):
    """Save a 3-panel PNG: RGB input | prediction | ground truth."""

    n_panels = 3 if label_path else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6),
                             constrained_layout=True)

    scene_name = os.path.splitext(os.path.basename(pred_path))[0]
    fig.suptitle(scene_name, fontsize=13, fontweight="bold")

    # Panel 1 — RGB input
    if image_path and os.path.exists(image_path):
        axes[0].imshow(load_rgb(image_path))
        axes[0].set_title("S2L1C RGB input\n(bands 4-3-2)", fontsize=11)
    else:
        axes[0].text(0.5, 0.5, "Image not found", ha="center", va="center",
                     transform=axes[0].transAxes)
        axes[0].set_title("S2L2A input", fontsize=11)
    axes[0].axis("off")

    # Panel 2 — Prediction
    pred = load_band1(pred_path)
    axes[1].imshow(label_to_rgb(pred))
    axes[1].set_title("TerraMind prediction", fontsize=11)
    axes[1].axis("off")

    # Panel 3 — Ground truth (if available)
    if n_panels == 3:
        if label_path and os.path.exists(label_path):
            gt = load_band1(label_path)
            axes[2].imshow(label_to_rgb(gt))
            axes[2].set_title("Ground truth", fontsize=11)
        else:
            axes[2].text(0.5, 0.5, "Label not found", ha="center", va="center",
                         transform=axes[2].transAxes)
            axes[2].set_title("Ground truth", fontsize=11)
        axes[2].axis("off")

    # Legend
    patches = [mpatches.Patch(color=CLASS_COLOURS[0], label="Not burned"),
               mpatches.Patch(color=CLASS_COLOURS[1], label="Burn scar")]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.04))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Make burn scar map figures from a completed TerraMind run")
    parser.add_argument("run_dir",
        help="Path to the run folder, e.g. output/burnscars_terramind_base_20260427_1529")
    parser.add_argument("--n", type=int, default=0,
        help="Number of scenes to plot (default: 0). Use 0 for all scenes.")
    args = parser.parse_args()

    pred_dir  = os.path.join(args.run_dir, "predictions")
    fig_dir   = os.path.join(args.run_dir, "figures")

    # Find all prediction GeoTIFFs
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.tif")))

    if not pred_files:
        print(f"\nNo prediction GeoTIFFs found in: {pred_dir}")
        print("Did the predict step in run_experiment.aqua complete successfully?")
        print("Check the job log — predictions/ may be empty if terratorch predict failed.")
        return

    # Limit to --n scenes (unless --n 0 = plot all)
    if args.n > 0:
        pred_files = pred_files[: args.n]

    print(f"\nFound {len(pred_files)} prediction file(s). Saving figures to {fig_dir}/\n")

    for pred_path in pred_files:
        image_path, label_path = find_source_files(pred_path)
        scene = os.path.splitext(os.path.basename(pred_path))[0]
        out_path = os.path.join(fig_dir, f"{scene}.png")
        make_figure(pred_path, image_path, label_path, out_path)
        # status = "✓" if os.path.exists(out_path) else "✗"
        # print(f"  {status}  {out_path}")
    print(f"\nDone. {len(pred_files)} figure(s) saved to {fig_dir}/")


if __name__ == "__main__":
    main()
