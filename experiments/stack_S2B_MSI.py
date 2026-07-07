import glob
import os
import rioxarray
import xarray as xr

# 1. Define the correct, sequence-ordered source paths for a SAFE granule
# Always grab the highest native resolution available for that specific band
granule_path = "data/S2B_MSIL2A_20230324T013549_N0510_R088_T49DDG_20240803T170113.SAFE/GRANULE/L2A_T49DDG_A031574_20230324T013543/IMG_DATA/"

band_paths = [
    glob.glob(os.path.join(granule_path, "R60m/*_B01_60m.jp2"))[0],  # 60m Coastal Aerosol
    glob.glob(os.path.join(granule_path, "R10m/*_B02_10m.jp2"))[0],  # 10m Blue
    glob.glob(os.path.join(granule_path, "R10m/*_B03_10m.jp2"))[0],  # 10m Green
    glob.glob(os.path.join(granule_path, "R10m/*_B04_10m.jp2"))[0],  # 10m Red
    glob.glob(os.path.join(granule_path, "R20m/*_B05_20m.jp2"))[0],  # 20m Red Edge 1
    glob.glob(os.path.join(granule_path, "R20m/*_B06_20m.jp2"))[0],  # 20m Red Edge 2
    glob.glob(os.path.join(granule_path, "R20m/*_B07_20m.jp2"))[0],  # 20m Red Edge 3
    glob.glob(os.path.join(granule_path, "R10m/*_B08_10m.jp2"))[0],  # 10m NIR Broad
    glob.glob(os.path.join(granule_path, "R20m/*_B8A_20m.jp2"))[0],  # 20m NIR Narrow
    glob.glob(os.path.join(granule_path, "R60m/*_B09_60m.jp2"))[0],  # 60m Water Vapour
    glob.glob(os.path.join(granule_path, "R20m/*_B11_20m.jp2"))[0],  # 20m SWIR 1
    glob.glob(os.path.join(granule_path, "R20m/*_B12_20m.jp2"))[0],  # 20m SWIR 2
]

# 2. Use B02 (10m) as our master grid template for spatial alignment
target_grid = rioxarray.open_rasterio(band_paths[1]) 

loaded_bands = []
for path in band_paths:
    # Open the band band natively
    band = rioxarray.open_rasterio(path)
    
    # Match the master 10m spatial resolution and clip boundaries using nearest-neighbor
    if band.rio.shape != target_grid.rio.shape:
        band = band.rio.reproject_match(target_grid)
        
    loaded_bands.append(band)

# 3. Stack into a unified 12-band multi-spectral DataArray
stacked_tensor = xr.concat(loaded_bands, dim="band")

# 4. Save to a modern Cloud-Optimised GeoTIFF for model ingestion
output_filename = "data/S2B_MSIL2A_20230324T013549_N0510_R088_T49DDG_20240803T170113.SAFE/S2B_MSIL2A_20230324T013549_N0510_R088_T49DDG_20240803T170113.tif"
stacked_tensor.rio.to_raster(output_filename, driver="COG")
print(f"File successfully saved to: {output_filename}")
