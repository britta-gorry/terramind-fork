"""
visualise_prediction.py  —  Make segmentation map figures for any supported dataset

Usage (from the terramind root, in an interactive HPC session):
    pixi run python experiments/visualise_prediction.py <run_dir> --dataset <name>

Arguments:
    run_dir     The timestamped run folder, e.g. output/burnscars_base_20260501_1430
    --dataset   Which dataset profile to use (see DATASET_PROFILES below)
    --n         How many scenes to plot (default: 5, use 0 for all)

Examples:
    pixi run python experiments/visualise_prediction.py output/burnscars_base_20260501_1430 --dataset burnscars
    pixi run python experiments/visualise_prediction.py output/sen1floods11_base_20260407_2038 --dataset sen1floods11

Adding a new dataset:
    Add one new entry to DATASET_PROFILES below. That is the only change needed.
    The keys are explained in the TEMPLATE entry at the bottom of the profiles dict.
"""

import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # must come before pyplot import — works on HPC with no display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio


# =============================================================================
# DATASET PROFILES
# This is the only section you need to edit when adding a new dataset.
# =============================================================================

DATASET_PROFILES = {

    # ─── Sen1Floods11 ─────────────────────────────────────────────────────────
    # Layout: images and labels are in separate subdirectories.
    #   Image: sen1floods11_v1.1/data/S2L1CHand/Bolivia_103757_S2Hand.tif
    #   Label: sen1floods11_v1.1/data/LabelHand/Bolivia_103757_LabelHand.tif
    # Prediction file from terratorch predict will be named something like:
    #   Bolivia_103757_S2Hand_pred.tif   or   Bolivia_103757_S2Hand.tif
    "sen1floods11": {
        # Directory containing the input image .tif files
        "image_dir": "sen1floods11_v1.1/data/S2L1CHand",
        # The suffix that appears at the END of every image filename
        # Used to strip it off and find the scene stem
        "image_suffix": "_S2Hand.tif",
        # Directory containing ground-truth label files (may be different from image_dir)
        "label_dir": "sen1floods11_v1.1/data/LabelHand",
        # Suffix appended to the scene stem to find the label file
        # e.g. scene stem "Bolivia_103757" + "_LabelHand.tif" = "Bolivia_103757_LabelHand.tif"
        "label_suffix": "_LabelHand.tif",
        # Which bands to load as Red, Green, Blue for the preview image (1-indexed)
        # S2L1C band order: B1 B2(blue) B3(green) B4(red) B5 B6 B7 B8 B8A B9 B10 B11 B12
        "rgb_bands": (4, 3, 2),
        # Name for each class — index matches the integer value in the prediction raster
        "class_names": ["Others", "Flood"],
        # Colour for each class as (R, G, B) floats between 0 and 1
        "class_colours": [(0.80, 0.80, 0.80), (0.08, 0.35, 0.75)],
    },

    # ─── BurnScars (HLS Burn Scars) ───────────────────────────────────────────
    # Layout: images AND labels are in the SAME folder (hls_burn_scars/data/).
    #   Image: hls_burn_scars/data/subsetted_512x512_HLS.S30.T10SEH.2018245.v1.4_merged.tif
    #   Label: hls_burn_scars/data/subsetted_512x512_HLS.S30.T10SEH.2018245.v1.4.mask.tif
    # The label name is the image name with "_merged.tif" replaced by ".mask.tif".
    "burnscars": {
        "image_dir": "hls_burn_scars/data",
        "image_suffix": "_merged.tif",
        "label_dir": "hls_burn_scars/data",   # same folder as images
        "label_suffix": ".mask.tif",
        # HLS S30 merged band order: Blue(1) Green(2) Red(3) NIR_Narrow(4) SWIR1(5) SWIR2(6)
        "rgb_bands": (3, 2, 1),
        "class_names": ["Unburned", "Burned"],
        "class_colours": [(0.80, 0.80, 0.80), (0.75, 0.15, 0.05)],
    },

    # ─── TEMPLATE — copy this block to add your next dataset ──────────────────
    # "my_dataset": {
    #     "image_dir":     "path/to/images",
    #     "image_suffix":  "_image.tif",
    #     "label_dir":     "path/to/labels",     # can be same as image_dir
    #     "label_suffix":  "_label.tif",
    #     "rgb_bands":     (3, 2, 1),            # 1-indexed
    #     "class_names":   ["Background", "Class1"],
    #     "class_colours": [(0.8, 0.8, 0.8), (0.2, 0.6, 0.2)],
    # },
}


# =============================================================================
# Helper functions — no need to edit these for routine use
# =============================================================================

def label_to_rgb(arr, class_colours):
    """
    Convert a 2D integer array (each pixel = class index) into an RGB image.
    Pixels with values not in class_colours (e.g. -1 no-data) appear white.
    """
    h, w = arr.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)   # start white
    for cls_idx, colour in enumerate(class_colours):
        rgb[arr == cls_idx] = colour
    return rgb


def load_rgb_preview(path, bands):
    """
    Load three bands from a multi-band GeoTIFF as a display-ready RGB image.
    bands: tuple of three 1-indexed band numbers, e.g. (4, 3, 2).
    Applies a 2–98 percentile contrast stretch so images look natural on screen.
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


def load_single_band(path):
    """Load just band 1 from a GeoTIFF — used for prediction and label rasters."""
    with rasterio.open(path) as src:
        return src.read(1)


def find_image_and_label(pred_path, profile):
    """
    Given a prediction GeoTIFF path and a dataset profile, find the matching
    input image and ground-truth label on disk.

    How filename matching works:
      terratorch predict names output as: <input_stem>_pred.tif
      Stripping "_pred" gives the image stem. The image file is image_stem + ".tif".
      image_suffix is only used to strip the dataset-specific part from image_stem
      to get the scene_stem for finding the label.

      BurnScars example:
        pred:       "...T10SDH_merged_pred.tif"
        image_stem: "...T10SDH_merged"   (strip _pred)
        image_path: image_dir / "...T10SDH_merged.tif"
        scene_stem: "...T10SDH"          (strip "_merged" using image_suffix)
        label_path: label_dir / "...T10SDH.mask.tif"

      Sen1Floods11 example:
        pred:       "Bolivia_103757_S2Hand_pred.tif"
        image_stem: "Bolivia_103757_S2Hand"
        image_path: image_dir / "Bolivia_103757_S2Hand.tif"
        scene_stem: "Bolivia_103757"
        label_path: label_dir / "Bolivia_103757_LabelHand.tif"

    Returns (image_path, label_path) — either may be None if not found on disk.
    """
    pred_name = os.path.basename(pred_path)
    pred_stem = os.path.splitext(pred_name)[0]   # strip .tif

    # Strip "_pred" suffix that terratorch predict appends
    image_stem = pred_stem[:-5] if pred_stem.endswith("_pred") else pred_stem

    # ── Find the input image ──────────────────────────────────────────────────
    # The image file is image_stem + ".tif" — do NOT append image_suffix here.
    # Appending image_suffix would double it (e.g. "_merged_merged.tif").
    image_path = os.path.join(profile["image_dir"], image_stem + ".tif")
    if not os.path.exists(image_path):
        matches = glob.glob(os.path.join(profile["image_dir"], image_stem + "*.tif"))
        matches = [m for m in matches if m.endswith(".tif") and ".aux" not in m]
        image_path = matches[0] if matches else None

    # ── Find the ground-truth label ───────────────────────────────────────────
    # scene_stem is the image stem without the dataset-specific suffix part
    # e.g. image_suffix "_merged.tif" → strip "_merged" from image_stem
    # e.g. image_suffix "_S2Hand.tif" → strip "_S2Hand" from image_stem
    img_suffix_no_ext = os.path.splitext(profile["image_suffix"])[0]   # "_merged" or "_S2Hand"
    if image_stem.endswith(img_suffix_no_ext):
        scene_stem = image_stem[: len(image_stem) - len(img_suffix_no_ext)]
    else:
        scene_stem = image_stem

    label_path = os.path.join(profile["label_dir"], scene_stem + profile["label_suffix"])
    if not os.path.exists(label_path):
        matches = glob.glob(os.path.join(profile["label_dir"],
                                          scene_stem + "*" + profile["label_suffix"]))
        label_path = matches[0] if matches else None

    return image_path, label_path


def save_figure(pred_path, image_path, label_path, out_path, profile):
    """
    Save a 3-panel figure: RGB input | prediction | ground truth.
    If the label is not found, saves a 2-panel figure instead.
    """
    has_label = (label_path is not None and os.path.exists(label_path))
    n_panels = 3 if has_label else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6),
                              constrained_layout=True)

    scene_name = os.path.splitext(os.path.basename(pred_path))[0]
    fig.suptitle(scene_name, fontsize=11, fontweight="bold")

    # Panel 1 — RGB preview of the input image
    if image_path and os.path.exists(image_path):
        axes[0].imshow(load_rgb_preview(image_path, profile["rgb_bands"]))
        r, g, b = profile["rgb_bands"]
        axes[0].set_title(f"Input RGB (bands {r}-{g}-{b})", fontsize=10)
    else:
        axes[0].text(0.5, 0.5, "Input image not found\n(check image_dir in profile)",
                     ha="center", va="center", transform=axes[0].transAxes,
                     color="red", fontsize=9)
        axes[0].set_title("Input", fontsize=10)
    axes[0].axis("off")

    # Panel 2 — Model prediction
    pred = load_single_band(pred_path)
    axes[1].imshow(label_to_rgb(pred, profile["class_colours"]))
    axes[1].set_title("TerraMind prediction", fontsize=10)
    axes[1].axis("off")

    # Panel 3 — Ground truth (only if label was found)
    if has_label:
        gt = load_single_band(label_path)
        axes[2].imshow(label_to_rgb(gt, profile["class_colours"]))
        axes[2].set_title("Ground truth", fontsize=10)
        axes[2].axis("off")

    # Shared legend at the bottom
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
        help="Path to the run folder, e.g. output/burnscars_base_20260501_1430")
    parser.add_argument("--dataset", required=True,
        choices=list(DATASET_PROFILES.keys()),
        help="Dataset profile. Available: " + ", ".join(DATASET_PROFILES.keys()))
    parser.add_argument("--n", type=int, default=5,
        help="Number of scenes to plot (default 5, use 0 for all)")
    args = parser.parse_args()

    profile  = DATASET_PROFILES[args.dataset]
    pred_dir = os.path.join(args.run_dir, "predictions")
    fig_dir  = os.path.join(args.run_dir, "figures")

    # Find all prediction GeoTIFFs in the predictions/ subfolder
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.tif")))

    if not pred_files:
        print(f"\nNo prediction GeoTIFFs found in: {pred_dir}")
        print("Did the predict step in run_experiment.aqua complete successfully?")
        print("If the folder is empty, re-run predictions manually:")
        print("  pixi run terratorch predict \\")
        print(f"    --config {args.run_dir}/run_config.yaml \\")
        print(f"    --ckpt_path {args.run_dir}/checkpoints/best.ckpt \\")
        print(f"    --predict_output_dir {pred_dir}")
        return

    if args.n > 0:
        pred_files = pred_files[: args.n]

    print(f"\nDataset profile : {args.dataset}")
    print(f"Predictions dir : {pred_dir}  ({len(pred_files)} file(s) selected)")
    print(f"Figures dir     : {fig_dir}/")
    print()

    ok = 0
    for pred_path in pred_files:
        image_path, label_path = find_image_and_label(pred_path, profile)
        scene = os.path.splitext(os.path.basename(pred_path))[0]
        out_path = os.path.join(fig_dir, f"{scene}.png")

        if image_path is None:
            print(f"  ⚠  {scene}  — input image not found, skipping")
            print(f"     Expected: {profile['image_dir']}/{scene}{profile['image_suffix']}")
            continue

        gt_note = "with ground truth" if (label_path and os.path.exists(label_path)) else "no ground truth found"
        save_figure(pred_path, image_path, label_path, out_path, profile)
        print(f"  ✓  {out_path}  ({gt_note})")
        ok += 1

    print(f"\n{ok}/{len(pred_files)} figure(s) saved to {fig_dir}/")


if __name__ == "__main__":
    main()
