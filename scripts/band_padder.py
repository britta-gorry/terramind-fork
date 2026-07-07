#!/usr/bin/env python3
"""
band_padder.py
==============
Pads BurnScars' 6-band images to TerraMind's full 12-band S2L2A order,
implemented as an albumentations transform — the same mechanism your
config already uses for albumentations.D4 and ToTensorV2.

WHY A TRANSFORM, NOT A CUSTOM DATAMODULE
------------------------------------------
A previous version of this fix subclassed GenericNonGeoSegmentationDataModule
and pointed data.class_path at it. That failed with "No module named
'scripts'" — confirmed (by reading terratorch/cli_tools.py and
reproducing the exact error in isolation) to be a real ordering bug:
terratorch's custom_modules_path is only loaded inside
instantiate_classes(), which runs AFTER LightningCLI has already
type-validated data.class_path against the LightningDataModule
registry. The custom module is loaded too late to satisfy that check.

Transforms don't have this problem. Looking at the installed
terratorch source (terratorch/datamodules/generic_pixel_wise_data_module.py
and terratorch/datamodules/utils.py):

  - train_transform / val_transform / test_transform are typed as
    plain `list[Any] | None` in the DataModule's __init__ — there is
    no CLI-level subclass-type validation for list items.
  - Each dict entry (class_path / init_args) is resolved LAZILY, at
    runtime, via instantiate_transform_from_dict(), which does a
    plain importlib.import_module() — not jsonargparse subclass
    resolution.

That means a transform listed in train_transform/val_transform/
test_transform is loaded the same permissive way as
albumentations.D4 already is in your config — no custom_modules_path
needed, no package/__init__.py needed, no PYTHONPATH needed.

WHERE THIS RUNS IN THE DATA PIPELINE
--------------------------------------
Confirmed directly from terratorch/datasets/generic_pixel_wise_dataset.py,
GenericNonGeoPixelwiseRegressionDataset.__getitem__():

    image = image.to_numpy()
    image = np.moveaxis(image, 0, -1)      # -> channels-last [H, W, C]
    output = {"image": image.astype(np.float32) * self.constant_scale}
    output["mask"] = mask.to_numpy()[0]
    if self.transform:
        output = self.transform(**output)  # <- our padder runs here

So by the time any transform runs, the image is already a numpy array
shaped [H, W, C] with C=6 (the raw BurnScars bands) — this is the
standard albumentations image convention (channels last). Our
transform turns that into [H, W, 12] before D4 and ToTensorV2 run.

WHERE THIS GOES IN YOUR PROJECT
----------------------------------
Same place as before — scripts/band_padder.py. No __init__.py needed
this time (it's just a plain importable module via class_path's
dotted-path resolution at runtime, not a package import at CLI
startup). Confirmed by testing instantiate_transform_from_dict()
directly against this file (see test notes accompanying this patch).

THREE STRATEGIES — unchanged from before
-------------------------------------------
  zero      Missing bands = 0. Unphysical; smoke-test only.
  mean      Missing bands = TerraMind's S2L2A pretraining mean for
            that band position. Recommended default.
  spectral  Missing bands linearly interpolated by wavelength from
            the nearest available real bands. Most physically
            motivated; best for genuinely out-of-distribution scenes
            (this is the one to prefer once you move to Antarctic
            drone data).
"""

import numpy as np
import albumentations as A


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# Bands present in BurnScars' 6-band files, in file order.
# Must match dataset_bands in your BurnScars data config exactly.
BURNSCARS_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

STRATEGY_ZERO     = "zero"
STRATEGY_MEAN     = "mean"
STRATEGY_SPECTRAL = "spectral"
DEFAULT_STRATEGY  = STRATEGY_MEAN

# ── END CONFIGURATION ──────────────────────────────────────────────────────────


# Full 12-band S2L2A order TerraMind's tokenizer/TiM backbone expect.
S2L2A_BANDS = [
    "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2",
    "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "SWIR_1", "SWIR_2",
]

S2L2A_WAVELENGTHS = {
    "COASTAL_AEROSOL": 443, "BLUE": 490, "GREEN": 560, "RED": 665,
    "RED_EDGE_1": 705, "RED_EDGE_2": 740, "RED_EDGE_3": 783,
    "NIR_BROAD": 842, "NIR_NARROW": 865, "WATER_VAPOR": 945,
    "SWIR_1": 1610, "SWIR_2": 2190,
}

# TerraMind v1 pretraining means for S2L2A (raw scale), in S2L2A_BANDS order.
# Source: terratorch terramind_register.py -> v1_pretraining_mean["untok_sen2l2a@224"]
S2L2A_PRETRAIN_MEANS = {
    "COASTAL_AEROSOL": 1390.458, "BLUE": 1503.317, "GREEN": 1718.197,
    "RED": 1853.910, "RED_EDGE_1": 2199.100, "RED_EDGE_2": 2779.975,
    "RED_EDGE_3": 2987.011, "NIR_BROAD": 3083.234, "NIR_NARROW": 3132.220,
    "WATER_VAPOR": 3162.988, "SWIR_1": 2424.884, "SWIR_2": 1857.648,
}


def _spectral_interpolate_numpy(missing_band, input_bands, img):
    """
    img: numpy array [H, W, C] where C = len(input_bands).
    Returns: numpy array [H, W] — the interpolated fill for one band.
    """
    target_wl = S2L2A_WAVELENGTHS[missing_band]
    available = sorted(
        ((S2L2A_WAVELENGTHS[name], i) for i, name in enumerate(input_bands)
         if name in S2L2A_WAVELENGTHS),
        key=lambda x: x[0],
    )
    below = [(wl, i) for wl, i in available if wl <= target_wl]
    above = [(wl, i) for wl, i in available if wl > target_wl]

    if below and above:
        wl_lo, idx_lo = below[-1]
        wl_hi, idx_hi = above[0]
        t = (target_wl - wl_lo) / (wl_hi - wl_lo)
        return (1 - t) * img[:, :, idx_lo] + t * img[:, :, idx_hi]
    elif below:
        return img[:, :, below[-1][1]]
    else:
        return img[:, :, above[0][1]]


class PadToS2L2A12Band(A.ImageOnlyTransform):
    """
    Albumentations transform: pads a 6-band BurnScars image (numpy,
    channels-last [H, W, 6]) to the full 12-band S2L2A order TerraMind
    expects ([H, W, 12]).

    Only touches the image — masks pass through untouched (that's what
    ImageOnlyTransform guarantees), so your label/mask handling needs
    no changes at all.

    YAML usage — add this as the FIRST entry in train_transform /
    val_transform / test_transform, before D4 and ToTensorV2, e.g.:

        train_transform:
          - class_path: band_padder.PadToS2L2A12Band
            init_args:
              strategy: mean        # zero | mean | spectral
              verbose: false
          - class_path: albumentations.D4
          - class_path: albumentations.pytorch.transforms.ToTensorV2

    IMPORTANT: this must run on EVERY split that touches the model:
    train_transform, val_transform, AND test_transform (and, if you
    set one, predict — predict reuses test_transform by default in
    TerraTorch's GenericNonGeoSegmentationDataModule).
    """

    def __init__(
        self,
        strategy: str = DEFAULT_STRATEGY,
        input_bands: list = None,
        verbose: bool = False,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.strategy = strategy
        self.input_bands = input_bands or BURNSCARS_BANDS
        self.verbose = verbose
        self._logged_once = False

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # img: [H, W, C], C == len(self.input_bands)
        H, W, C = img.shape
        if C != len(self.input_bands):
            raise ValueError(
                f"PadToS2L2A12Band expected {len(self.input_bands)} input "
                f"bands {self.input_bands}, but got an image with {C} channels."
            )

        input_band_index = {name: i for i, name in enumerate(self.input_bands)}
        out = np.zeros((H, W, 12), dtype=img.dtype)

        kept, padded = [], []
        for out_idx, band_name in enumerate(S2L2A_BANDS):
            if band_name in input_band_index:
                in_idx = input_band_index[band_name]
                out[:, :, out_idx] = img[:, :, in_idx]
                kept.append(band_name)
            else:
                padded.append(band_name)
                if self.strategy == STRATEGY_ZERO:
                    pass  # already zero
                elif self.strategy == STRATEGY_MEAN:
                    out[:, :, out_idx] = S2L2A_PRETRAIN_MEANS[band_name]
                elif self.strategy == STRATEGY_SPECTRAL:
                    out[:, :, out_idx] = _spectral_interpolate_numpy(
                        band_name, self.input_bands, img
                    )
                else:
                    raise ValueError(
                        f"Unknown strategy '{self.strategy}'. "
                        f"Use 'zero', 'mean', or 'spectral'."
                    )

        if self.verbose and not self._logged_once:
            print(f"[band_padder] strategy='{self.strategy}'  "
                  f"kept={kept}  padded={padded}  out_shape={out.shape}")
            self._logged_once = True

        return out

    def get_transform_init_args_names(self):
        # Required by albumentations for serialization/repr — lists the
        # constructor args (beyond p) that define this transform.
        return ("strategy", "input_bands", "verbose")
