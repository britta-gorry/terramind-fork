#!/usr/bin/env python3
"""
Adapt bands from burnscars dataset to pretrained bands expected by
TerraMind untok_sen2l2a@224
"""

# PRETRAINED_BANDS = {
#     "untok_sen2l2a@224": [
#         "COASTAL_AEROSOL",
#         "BLUE",
#         "GREEN",
#         "RED",
#         "RED_EDGE_1",
#         "RED_EDGE_2",
#         "RED_EDGE_3",
#         "NIR_BROAD",
#         "NIR_NARROW",
#         "WATER_VAPOR",
#         "SWIR_1",
#         "SWIR_2",
#     ]}
# v1_pretraining_mean = {
#     "untok_sen2l2a@224": [
#         1390.458,
#         1503.317,
#         1718.197,
#         1853.91,
#         2199.1,
#         2779.975,
#         2987.011,
#         3083.234,
#         3132.22,
#         3162.988,
#         2424.884,
#         1857.648,
#     ]}

import torch
import numpy as np

# Strict 12-band order expected by the TiM tokenizer
TIM_BAND_ORDER = [
    "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", 
    "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", 
    "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", 
    "SWIR_1", "SWIR_2"
]

# Tokenizer pretrained means to prevent out-of-distribution noise
TIM_PRETRAINED_MEANS = {
    "COASTAL_AEROSOL": 1390.458, "BLUE": 1503.317, "GREEN": 1718.197, "RED": 1853.91,
    "RED_EDGE_1": 2199.1, "RED_EDGE_2": 2779.975, "RED_EDGE_3": 2987.011,
    "NIR_BROAD": 3083.234, "NIR_NARROW": 3132.22, "WATER_VAPOR": 3162.988,
    "SWIR_1": 2424.884, "SWIR_2": 1857.648,
}

class AlignBandsToTiM:
    def __init__(self, dataset_bands: list[str], fill_method: str = "mean"):
        """
        Args:
            dataset_bands: List of band names present in your HLS burnscars TIFFs.
                           e.g., ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]
        """
        self.dataset_bands = dataset_bands
        self.fill_method = fill_method
        
    def __call__(self, **kwargs):
        # Terratorch/Albumentations passes data as a dict containing 'image'
        image = kwargs.get("image")
        if image is None:
            return kwargs
            
        # DEBUG TRACE: See shape and type entering the transform
        input_type = type(image)
        input_shape = image.shape if hasattr(image, 'shape') else "No Shape Attribute"
        print(f"\n🚀 [TRACE] AlignBandsToTiM input -> Type: {input_type} | Shape: {input_shape}")

        # Convert to torch tensor if it's a numpy array, preserving shape (C, H, W)
        is_numpy = isinstance(image, np.ndarray)
        if is_numpy:
            # Albumentations usually uses (H, W, C) for numpy
            if image.ndim == 3:
                tensor_img = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                tensor_img = torch.from_numpy(image).float()
        else:
            tensor_img = image.float()

        channels, height, width = tensor_img.shape
        output_tensors = []
        
        for band in TIM_BAND_ORDER:
            if band in self.dataset_bands:
                # Physics match: extract channel index from source HLS data
                idx = self.dataset_bands.index(band)
                output_tensors.append(tensor_img[idx])
            else:
                # ML patch: fill with tokenizer expected constant values
                fill_val = TIM_PRETRAINED_MEANS[band] if self.fill_method == "mean" else 0.0
                dummy_band = torch.full((height, width), fill_val, dtype=tensor_img.dtype)
                output_tensors.append(dummy_band)
                
        aligned_tensor = torch.stack(output_tensors, dim=0)
        
        # DEBUG TRACE: See shape and type entering the transform
        input_type = type(image)
        input_shape = image.shape if hasattr(image, 'shape') else "No Shape Attribute"
        print(f"\n🚀 [TRACE] AlignBandsToTiM input -> Type: {input_type} | Shape: {input_shape}")

        # Convert back to original format if required by terratorch pipeline
        if is_numpy:
            kwargs["image"] = aligned_tensor.permute(1, 2, 0).numpy()
        else:
            kwargs["image"] = aligned_tensor
            
        return kwargs
