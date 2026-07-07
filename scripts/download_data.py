#!/usr/bin/env python3
"""
download_data.py
================
Download data from for TerraMind experiments.
"""
import os
import ee
import torch
import argparse
from pathlib import Path
from datetime import datetime, timedelta

ee.Authenticate()
ee.Initialize()

default_name = "GEE_test"
# Target Benchmark coordinates (McMurdo Sound area, Antarctica)
bbox = [165.0, -78.0, 167.0, -77.0]
date_start = "2024-12-01"
date_end = "2025-01-31"
region = ee.Geometry.Rectangle(bbox)
scale = 20
tileSize = 244
crs = "EPSG:3031"

# Define bands in order
target_bands_L2A = ["B01",  # Coastal Aerosol
                    "B02",  # Blue
                    "B03",  # Green
                    "B04",  # Red
                    "B05",  # Red Edge 1
                    "B06",  # Red Edge 2
                    "B07",  # Red Edge 3
                    "B08",  # NIR Broad
                    "B8A",  # NIR Narrow
                    "B09",  # Water Vapour
                    "B11",  # SWIR 1
                    "B12"]  # SWIR 2

# --- Setup Output Directory and Logging ---
def _setup(args) -> Path: 
    if "output" in os.environ:
        out = Path(os.environ["output"])
    else:
        ts  = datetime.now().strftime("%Y%m%d_%H%M")
        out_dir = Path("output") / "data" / f"{args.name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download satellite data from Google Earth Engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", type=str, default=default_name,
                        help="Name of output dataset or output directory.")
    parser.add_argument("--start_date", type=str, default=date_start,
                        help="Start date of target download.")
    parser.add_argument("--end_date", type=str, default=date_end,
                        help="End date of target download.")
    parser.add_argument("--bbox", type=list, default=region,
                        help="List of four coordinates of target region, Format: [min_lon, min_lat, max_lon, max_lat]")
    args = parser.parse_args()

    out_dir = _setup(args)

    # Grab the collection and apply basic cloud sorting
    image = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(region)
                  .filterDate(args.start_date, args.end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                  .sort('CLOUDY_PIXEL_PERCENTAGE').first()
                  .select(target_bands_L2A))

    # GEE tiling and export
    task = ee.batch.Export.image.toDrive(
        image = image,
        description = "Google Earth Engine, Copernicus S2 SR Harmonized",
        folder = out_dir,
        region = region,
        scale = scale,
        crs = crs,
        shardSize = tileSize,
        fileFormat = "GeoTIFF"
    )

    task.start()
    print(f"Export task sent to Google servers. \nGEE is now splitting your data into {tileSize} tiles. \n Output to: {out_dir}")
    print(f"To copy downloaded data to another location: \ rclone sync google_drive:{out_dir} /path/to/hpc/scratch/")

if __name__ == "__main__":
    main()