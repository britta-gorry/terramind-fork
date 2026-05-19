# ========= Rasterize Shapefile Labels into a GeoTIFF Mask ==================
# This script reads a shapefile containing polygon labels, rasterizes it to match

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd

# ── Paths ──────────────────────────────────────────────────────────────────
image_tif   = "sample_data/aspa135-arthur-data/altum_aspa135_20230128_f1-f2_or_gr_corrected.tif"
shapefile   = "sample_data/aspa135-arthur-data/altum_aspa135_20230128_f1-f2_or_gr_corrected.shp"
output_mask = "sample_data/aspa135-arthur-data/altum_aspa135_20230128_f1-f2_or_gr_corrected_mask.tif"

# ── Step 1: Read the shapefile ─────────────────────────────────────────────
# geopandas reads all the .shp/.dbf/.prj components automatically
gdf = gpd.read_file(shapefile)
print(gdf.head())          # inspect: shows geometry + attribute columns
print(gdf.columns.tolist()) # find which column holds your class labels
print(gdf.crs)              # check the coordinate system

# ── Step 2: Open the image to get its grid metadata ────────────────────────
with rasterio.open(image_tif) as src:
    image_crs       = src.crs
    image_transform = src.transform   # maps pixel -> world coordinates
    image_height    = src.height
    image_width     = src.width
    image_profile   = src.profile     # copy of all metadata

print(f"Image CRS:  {image_crs}")
print(f"Image size: {image_height} x {image_width}")

# ── Step 3: Reproject shapefile to match the image CRS if needed ───────────
if gdf.crs != image_crs:
    print(f"Reprojecting shapefile from {gdf.crs} to {image_crs}")
    gdf = gdf.to_crs(image_crs)

# ── Step 4: Build (geometry, class_value) pairs ────────────────────────────
# Replace 'class_col' with the actual column name from gdf.columns above.
# If your polygons have string labels like "moss"/"lichen", map them to ints.
CLASS_MAP = {"moss": 1, "lichen": 2, "rock": 0}  # adjust to your labels
label_col = "class_col"   # ← CHANGE THIS to your actual column name

shapes = [
    (geom, CLASS_MAP.get(label, 0))
    for geom, label in zip(gdf.geometry, gdf[label_col])
]

# ── Step 5: Rasterize onto the image grid ──────────────────────────────────
mask = rasterize(
    shapes=shapes,
    out_shape=(image_height, image_width),
    transform=image_transform,
    fill=255,          # pixels not covered by any polygon → 255 (ignore index)
    dtype=np.uint8,
)

print(f"Mask shape: {mask.shape}")
print(f"Unique values: {np.unique(mask)}")

# ── Step 6: Save as a GeoTIFF ──────────────────────────────────────────────
mask_profile = image_profile.copy()
mask_profile.update({"count": 1, "dtype": "uint8", "nodata": 255})

with rasterio.open(output_mask, "w", **mask_profile) as dst:
    dst.write(mask[np.newaxis, :, :])   # rasterio expects (bands, H, W)

print(f"Saved mask: {output_mask}")


# ========= Visualisation (optional) =========================================
# import matplotlib
# matplotlib.use("Agg")   # no display needed on HPC
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].set_title("Mask values")
# axes[0].imshow(mask, cmap="tab10")
# axes[1].set_title("Mask histogram")
# axes[1].bar(*np.unique(mask, return_counts=True))
# plt.tight_layout()
# plt.savefig("mask_check.png", dpi=150)
# print("Saved mask_check.png")