#!/usr/bin/env python3
"""
band_padder.py
==============
Place this file in scripts/, next to generate_pseudolabels.py and your
other helper scripts. It must be importable as `band_padder` when
`pixi run terratorch fit/test/predict` runs — see the PYTHONPATH note
in run_experiment.py for how that's wired up.

WHY THIS EXISTS
---------------
TerraMind's tokenizer and TiM backbone were pretrained on the full
12-band S2L2A layout:

    Position  Band name        Wavelength
    --------  ---------------  ----------
       0      COASTAL_AEROSOL  443 nm
       1      BLUE             490 nm
       2      GREEN            560 nm
       3      RED              665 nm
       4      RED_EDGE_1       705 nm
       5      RED_EDGE_2       740 nm
       6      RED_EDGE_3       783 nm
       7      NIR_BROAD        842 nm
       8      NIR_NARROW       865 nm
       9      WATER_VAPOR      945 nm
      10      SWIR_1          1610 nm
      11      SWIR_2          2190 nm

HLS BurnScars only has 6 of these:
    BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2

TiM cannot take a band subset (that's the `backbone_bands` restriction
that the normal fine-tuning mode allows but TiM does not — this is
also why `tim_compatibility()` in run_experiment.py strips
`backbone_bands` whenever the target backbone has `_tim` in its name).
So before the tensor reaches TerraMind, this module pads it from 6 to
12 channels.

THREE STRATEGIES
----------------
  zero      Missing bands = 0. Unphysical — only for quick smoke tests.
  mean      Missing bands = TerraMind's own S2L2A pretraining mean for
            that band position. Neutral, in-distribution. Recommended
            default for BurnScars and for an initial Antarctic pass.
  spectral  Missing bands linearly interpolated from the two nearest
            available bands by wavelength. Most physically motivated;
            best once you move to data (e.g. Antarctic) where the
            global S2L2A pretraining mean is a poor stand-in.

HOW THIS PLUGS INTO YOUR EXISTING WORKFLOW
-------------------------------------------
You do NOT need a new standalone YAML. Your existing pipeline already
works like this:

    datasets_config.yaml["burnscars"]["data"]
        -> copied verbatim by run_experiment.py into run_config["data"]
        -> terratorch fit/test/predict reads run_config["data"]["class_path"]

So the only change needed in datasets_config.yaml is:

    burnscars:
      data:
        class_path: band_padder.BurnScarsPaddedDataModule   # was: terratorch.datamodules.GenericNonGeoSegmentationDataModule
        init_args:
          padding_strategy: mean      # new
          padding_verbose: false      # new
          batch_size: 8               # unchanged -- everything else stays the same
          ...

Nothing else in datasets_config.yaml, model_config.yaml, start_config.yaml,
or run_experiment.py needs to change for the DATA side. Two small
additions are still required on the MODEL side and the IMPORT side --
see the patch notes sent alongside this script.
"""

import torch
from typing import Optional


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# Which bands does BurnScars actually contain, in file order?
# Must match `dataset_bands` in your BurnScars data config exactly.
BURNSCARS_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

STRATEGY_ZERO     = "zero"
STRATEGY_MEAN     = "mean"
STRATEGY_SPECTRAL = "spectral"
DEFAULT_STRATEGY  = STRATEGY_MEAN

# ── END CONFIGURATION ──────────────────────────────────────────────────────────


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


def pad_to_s2l2a_12band(
    tensor: torch.Tensor,
    input_bands: Optional[list] = None,
    strategy: str = DEFAULT_STRATEGY,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Expand a partial-band tensor [B, C, H, W] (or [C, H, W]) to the full
    12-band S2L2A order [B, 12, H, W] (or [12, H, W]).
    """
    if input_bands is None:
        input_bands = BURNSCARS_BANDS

    added_batch_dim = False
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        added_batch_dim = True

    B, C, H, W = tensor.shape
    if C != len(input_bands):
        raise ValueError(
            f"Tensor has {C} channels but input_bands lists {len(input_bands)} names."
        )

    device, dtype = tensor.device, tensor.dtype
    input_band_index = {name: i for i, name in enumerate(input_bands)}
    out = torch.zeros(B, 12, H, W, device=device, dtype=dtype)

    kept, padded = [], []
    for out_idx, band_name in enumerate(S2L2A_BANDS):
        if band_name in input_band_index:
            in_idx = input_band_index[band_name]
            out[:, out_idx, :, :] = tensor[:, in_idx, :, :]
            kept.append(band_name)
        else:
            padded.append(band_name)
            if strategy == STRATEGY_ZERO:
                pass  # already zero
            elif strategy == STRATEGY_MEAN:
                out[:, out_idx, :, :] = S2L2A_PRETRAIN_MEANS[band_name]
            elif strategy == STRATEGY_SPECTRAL:
                out[:, out_idx, :, :] = _spectral_interpolate(
                    band_name, input_bands, tensor
                )
            else:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. Use 'zero', 'mean', or 'spectral'."
                )

    if verbose:
        print(f"[band_padder] strategy='{strategy}'  kept={kept}  padded={padded}  "
              f"out_shape={tuple(out.shape)}")

    return out.squeeze(0) if added_batch_dim else out


def _spectral_interpolate(missing_band, input_bands, tensor):
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
        return (1 - t) * tensor[:, idx_lo, :, :] + t * tensor[:, idx_hi, :, :]
    elif below:
        return tensor[:, below[-1][1], :, :]
    else:
        return tensor[:, above[0][1], :, :]


# ── Lightning DataModule wrapper ────────────────────────────────────────────
# This is what your YAML's class_path actually points to.

from terratorch.registry import TERRATORCH_FULL_MODEL_REGISTRY
from terratorch.datamodules.generic_pixel_wise_data_module import GenericNonGeoSegmentationDataModule

@TERRATORCH_FULL_MODEL_REGISTRY.register
class BurnScarsPaddedDataModule(GenericNonGeoSegmentationDataModule):
    """
    Drop-in replacement for GenericNonGeoSegmentationDataModule.
    Reads BurnScars' normal 6-band files, then pads every batch's
    image tensor to 12-band S2L2A order before it reaches the model.

    All normal init_args (batch_size, train_data_root, means, stds,
    etc.) pass straight through to the parent class unchanged. Only
    two new init_args are added:

        padding_strategy: zero | mean | spectral   (default: mean)
        padding_verbose:  true | false              (default: false)
    """

    def __init__(
        self,
        padding_strategy: str = DEFAULT_STRATEGY,
        padding_verbose: bool = False,
        padding_input_bands: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.padding_strategy = padding_strategy
        self.padding_verbose = padding_verbose
        self.padding_input_bands = padding_input_bands or BURNSCARS_BANDS

    def _pad(self, batch):
        img = batch["image"]
        n_bands = img.shape[1] if img.dim() == 4 else img.shape[0]
        if n_bands < 12:
            batch["image"] = pad_to_s2l2a_12band(
                img,
                input_bands=self.padding_input_bands,
                strategy=self.padding_strategy,
                verbose=self.padding_verbose,
            )
        return batch

    def train_dataloader(self):
        return _PaddingLoader(super().train_dataloader(), self._pad)

    def val_dataloader(self):
        return _PaddingLoader(super().val_dataloader(), self._pad)

    def test_dataloader(self):
        return _PaddingLoader(super().test_dataloader(), self._pad)

    def predict_dataloader(self):
        return _PaddingLoader(super().predict_dataloader(), self._pad)


class _PaddingLoader:
    """Wraps a DataLoader, applying a function to each batch as it's yielded."""

    def __init__(self, dataloader, fn):
        self._dl = dataloader
        self._fn = fn

    def __iter__(self):
        for batch in self._dl:
            yield self._fn(batch)

    def __len__(self):
        return len(self._dl)

    @property
    def dataset(self):
        return self._dl.dataset

    @property
    def batch_size(self):
        return self._dl.batch_size
