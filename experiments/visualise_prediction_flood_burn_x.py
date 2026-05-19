"""
visualise_prediction.py  —  Make segmentation map figures for any supported dataset

Usage:
    pixi run python experiments/visualise_prediction.py <run_dir> --dataset <name>

Arguments:
    run_dir     Timestamped run folder, e.g. output/burnscars_base_20260501_1430
    --dataset   Which dataset profile to use (see DATASET_PROFILES below)
    --n         Number of scenes to plot (default: 5, use 0 for all)

Examples:
    pixi run python experiments/visualise_prediction.py output/burnscars_base_20260501_1430 --dataset burnscars
    pixi run python experiments/visualise_prediction.py output/sen1floods11_base_20260407_2038 --dataset sen1floods11

Adding a new dataset:
    Add one new entry to DATASET_PROFILES below. The keys are explained in the
    "ADDING A NEW DATASET" section. That is the only change needed.
"""

import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio


# =============================================================================
# DATASET PROFILES
# To add a new dataset, copy one of the entries below and fill in your values.
# =============================================================================

DATASET_PROFILES = {

    # ── Sen1Floods11 ──────────────────────────────────────────────────────────
    # Images and labels are in separate folders.
    # Image: sen1floods11_v1.1/data/S2L1CHand/Bolivia_103757_S2Hand.tif
    # Label: sen1floods11_v1.1/data/LabelHand/Bolivia_103757_LabelHand.tif
    "sen1floods11": {
        # Folder containing the input images (relative to terramind root)
        "image_dir":    "sen1floods11_v1.1/data/S2L1CHand",
        # Suffix that identifies an image file (used to find the image from the prediction name)
        "image_suffix": "_S2Hand.tif",
        # Folder containing ground-truth label files (can be the same as image_dir)
        "label_dir":    "sen1floods11_v1.1/data/LabelHand",
        # Suffix of the label file. Applied to the scene stem (filename minus image_suffix).
        # e.g. scene "Bolivia_103757" + "_LabelHand.tif" → "Bolivia_103757_LabelHand.tif"
        "label_suffix": "_LabelHand.tif",
        # Which 3 bands to use for the RGB preview (1-indexed, rasterio convention)
        # S2L1C band order: B1,B2,B3,B4(red),B5,B6,B7,B8,B8A,B9,B10,B11,B12
        # Bands 4,3,2 = Red, Green, Blue → natural colour preview
        "rgb_bands":    (4, 3, 2),
        # Class names (index = class value in the prediction/label raster)
        "class_names":  ["Others", "Flood"],
        # Colours for each class as (R, G, B) floats in [0, 1]
        "class_colours": [(0.80, 0.80, 0.80), (0.08, 0.35, 0.75)],
    },

    # ── BurnScars (HLS) ───────────────────────────────────────────────────────
    # Images and labels are in the SAME folder (hls_burn_scars/data/).
    # Image: hls_burn_scars/data/subsetted_512x512_HLS.S30.T10SEH.2018245.v1.4_merged.tif
    # Label: hls_burn_scars/data/subsetted_512x512_HLS.S30.T10SEH.2018245.v1.4.mask.tif
    # Note the label replaces "_merged.tif" with ".mask.tif" on the scene stem.
    "burnscars": {
        "image_dir":    "hls_burn_scars/data",
        "image_suffix": "_merged.tif",
        "label_dir":    "hls_burn_scars/data",   # same folder as images
        "label_suffix": ".mask.tif",
        # HLS S30 bands in merged.tif: Blue(1), Green(2), Red(3), NIR(4), SWIR1(5), SWIR2(6)
        # Bands 3,2,1 = Red, Green, Blue → natural colour
        "rgb_bands":    (3, 2, 1),
        "class_names":  ["Unburned", "Burned"],
        "class_colours": [(0.80, 0.80, 0.80), (0.75, 0.15, 0.05)],
    },

    # ── TEMPLATE — copy this block for a new dataset ──────────────────────────
    # "my_new_dataset": {
    #     "image_dir":    "path/to/images",
    #     "image_suffix": "_image.tif",
    #     "label_dir":    "path/to/labels",   # or same as image_dir
    #     "label_suffix": "_label.tif",
    #     "rgb_bands":    (3, 2, 1),           # 1-indexed band numbers for R, G, B
    #     "class_names":  ["Background", "Class1", "Class2"],
    #     "class_colours": [(0.8, 0.8, 0.8), (0.2, 0.6, 0.2), (0.8, 0.2, 0.2)],
    # },
}


# =============================================================================
# Core functions — you do not need to edit these for routine use
# =============================================================================

def label_to_rgb(arr, class_colours):
    """Convert a 2D integer class array to a 3-channel RGB image for display."""
    h, w = arr.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)   # white = no-data / unknown
    for cls_idx, colour in enumerate(class_colours):
        rgb[arr == cls_idx] = colour
    return rgb


def load_rgb(path, bands=(3, 2, 1)):
    """
    Load three bands from a multi-band GeoTIFF and return a display-ready image.
    bands: 1-indexed band numbers (rasterio convention).
    Applies a 2–98 percentile stretch so the image looks natural on screen.
    """
    with rasterio.open(path) as src:
        data = [src.read(b).astype(np.float32) for b in bands]

    def stretch(band):
        valid = band[band > 0]
        if len(valid) == 0:
            return np.zeros_like(band)
        lo, hi = np.percentile(valid, [2, 98])
        return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

    return np.stack([stretch(b) for b in data], axis=-1)


def load_band1(path):
    """Load just the first band from a single-band GeoTIFF (labels, predictions)."""
    with rasterio.open(path) as src:
        return src.read(1)


def find_source_files(pred_path, profile):
    """
    Given the path to a prediction GeoTIFF and the dataset profile,
    find the matching input image and ground-truth label file.

    Logic:
      1. Get the prediction stem (filename without extension).
         terratorch predict may or may not append "_pred" — we handle both.
      2. From the stem, reconstruct the image filename using image_suffix.
      3. From the image stem (minus image_suffix), build the label filename.

    Returns (image_path_or_None, label_path_or_None).
    """
    pred_name = os.path.basename(pred_path)
    pred_stem = os.path.splitext(pred_name)[0]  # strip .tif

    # terratorch predict may append "_pred" — strip it if present
    if pred_stem.endswith("_pred"):
        image_stem = pred_stem[:-5]   # remove trailing _pred
    else:
        image_stem = pred_stem

    # --- Find the input image ---
    image_filename = image_stem + profile["image_suffix"]
    image_path = os.path.join(profile["image_dir"], image_filename)
    if not os.path.exists(image_path):
        # Try a glob in case there's a slight mismatch
        matches = glob.glob(os.path.join(profile["image_dir"], image_stem + "*" + profile["image_suffix"]))
        image_path = matches[0] if matches else None

    # --- Find the label ---
    # The scene stem is the image stem minus the image suffix
    img_suffix = profile["image_suffix"]   # e.g. "_merged.tif" or "_S2Hand.tif"
    if image_stem.endswith(os.path.splitext(img_suffix)[0]):
        # Strip the suffix part without .tif  (e.g. strip "_merged" from "...T10SEH_merged")
        scene_stem = image_stem[: -len(os.path.splitext(img_suffix)[0])]
    else:
        scene_stem = image_stem

    label_filename = scene_stem + profile["label_suffix"]
    label_path = os.path.join(profile["label_dir"], label_filename)
    if not os.path.exists(label_path):
        matches = glob.glob(os.path.join(profile["label_dir"], scene_stem + "*" + profile["label_suffix"]))
        label_path = matches[0] if matches else None

    return image_path, label_path


def make_figure(pred_path, image_path, label_path, out_path, profile):
    """Save a 3-panel PNG: RGB input | prediction | ground truth."""

    has_label = label_path is not None and os.path.exists(label_path)
    n_panels = 3 if has_label else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6),
                             constrained_layout=True)

    scene_name = os.path.splitext(os.path.basename(pred_path))[0]
    fig.suptitle(scene_name, fontsize=11, fontweight="bold")

    # Panel 1 — RGB input
    if image_path and os.path.exists(image_path):
        axes[0].imshow(load_rgb(image_path, profile["rgb_bands"]))
        r, g, b = profile["rgb_bands"]
        axes[0].set_title(f"Input RGB (bands {r}-{g}-{b})", fontsize=10)
    else:
        axes[0].text(0.5, 0.5, "Image not found", ha="center", va="center",
                     transform=axes[0].transAxes, color="red")
        axes[0].set_title("Input", fontsize=10)
    axes[0].axis("off")

    # Panel 2 — Model prediction
    pred = load_band1(pred_path)
    axes[1].imshow(label_to_rgb(pred, profile["class_colours"]))
    axes[1].set_title("TerraMind prediction", fontsize=10)
    axes[1].axis("off")

    # Panel 3 — Ground truth (optional)
    if has_label:
        gt = load_band1(label_path)
        axes[2].imshow(label_to_rgb(gt, profile["class_colours"]))
        axes[2].set_title("Ground truth", fontsize=10)
        axes[2].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=col, label=name)
        for name, col in zip(profile["class_names"], profile["class_colours"])
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualise TerraMind segmentation predictions")
    parser.add_argument("run_dir",
        help="Path to the run folder, e.g. output/burnscars_terramind_base_20260501_1430")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_PROFILES.keys()),
        help=f"Dataset profile to use. Available: {list(DATASET_PROFILES.keys())}")
    parser.add_argument("--n", type=int, default=5,
        help="Number of scenes to plot (default: 5). Use 0 for all.")
    args = parser.parse_args()

    profile   = DATASET_PROFILES[args.dataset]
    pred_dir  = os.path.join(args.run_dir, "predictions")
    fig_dir   = os.path.join(args.run_dir, "figures")

    # Find all prediction GeoTIFFs
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.tif")))

    if not pred_files:
        print(f"\nNo prediction GeoTIFFs found in: {pred_dir}")
        print("Did the predict step complete successfully?")
        print("Check the job log — the predictions/ folder may be empty.")
        return

    if args.n > 0:
        pred_files = pred_files[: args.n]

    print(f"\nDataset profile : {args.dataset}")
    print(f"Predictions     : {pred_dir}  ({len(pred_files)} files)")
    print(f"Saving figures  : {fig_dir}/\n")

    for pred_path in pred_files:
        image_path, label_path = find_source_files(pred_path, profile)
        scene = os.path.splitext(os.path.basename(pred_path))[0]
        out_path = os.path.join(fig_dir, f"{scene}.png")

        if image_path is None:
            print(f"  ⚠  Image not found for: {os.path.basename(pred_path)} — skipping")
            continue

        make_figure(pred_path, image_path, label_path, out_path, profile)
        has_gt = "with GT" if label_path and os.path.exists(label_path) else "no GT"
        print(f"  ✓  {out_path}  ({has_gt})")

    print(f"\nDone. Figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
