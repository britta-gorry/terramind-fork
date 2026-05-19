"""
inspect_geo_data.py
===============
Diagnostic script for remote sensing data files.

Scans a folder and reports on every file it finds, grouped by format.
Tells you what each file contains and whether gdal_rasterize would work.

Usage:
    pixi run python inspect_geo_data.py                  # scans current directory
    pixi run python inspect_geo_data.py /path/to/folder  # scans a specific folder
    pixi run python inspect_geo_data.py /path/to/folder --recurse  # includes subfolders

Dependencies (all should already be in your TerraMind pixi env, or add with pixi add):
    rasterio, geopandas, h5py
"""

import os
import sys
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def subsection(title):
    print(f"\n  ── {title}")

def check_import(name):
    """Try importing a library; return the module or None with a clear message."""
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        print(f"  [MISSING] '{name}' not installed — run: pixi add {name}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

# Extensions we know how to handle, grouped by format family
KNOWN_EXTENSIONS = {
    # Raster formats
    "raster":  [".tif", ".tiff", ".img", ".vrt"],
    # Raster sidecar files (accompany a TIF but are not images themselves)
    "sidecar": [".tfw", ".ovr", ".xml", ".aux", ".aux.xml"],
    # Vector formats
    "vector":  [".shp", ".gpkg", ".geojson", ".kml", ".gml"],
    # Vector sidecar files (must travel with the .shp)
    "shp_sidecar": [".dbf", ".shx", ".prj", ".cpg", ".qmd", ".sbn", ".sbx"],
    # HDF5
    "hdf5":    [".h5", ".hdf5", ".he5"],
    # ENVI hyperspectral format
    "envi":    [".hdr", ".dat", ".bin", ".envi"],
    # Other hyperspectral / proprietary
    "other":   [".enp", ".raw", ".nc", ".nc4"],
    # Metadata / config
    "meta":    [".yaml", ".yml", ".json", ".txt", ".csv"],
}

# Flatten to a lookup: extension → family
EXT_TO_FAMILY = {}
for family, exts in KNOWN_EXTENSIONS.items():
    for ext in exts:
        EXT_TO_FAMILY[ext] = family


def discover_files(root: Path, recurse: bool):
    """Return all files under root, grouped by extension family."""
    found = {}  # family → list of Path

    iterator = root.rglob("*") if recurse else root.iterdir()
    for p in sorted(iterator):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        # Special case: .aux.xml
        if p.name.lower().endswith(".aux.xml"):
            ext = ".aux.xml"
        family = EXT_TO_FAMILY.get(ext, "unknown")
        found.setdefault(family, []).append(p)

    return found


# ─────────────────────────────────────────────────────────────────────────────
# INSPECTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def inspect_raster(path: Path):
    """Print rasterio summary for a TIF or other raster file."""
    rasterio = check_import("rasterio")
    if rasterio is None:
        return

    try:
        with rasterio.open(path) as src:
            print(f"\n  File  : {path.name}")
            print(f"  Size  : {src.width} x {src.height} pixels (width x height)")
            print(f"  Bands : {src.count}")
            print(f"  Dtype : {src.dtypes[0]}")
            print(f"  CRS   : {src.crs}")
            print(f"  Transform (pixel → world coords):")
            print(f"          {src.transform}")
            print(f"  Bounds: left={src.bounds.left:.4f}  right={src.bounds.right:.4f}")
            print(f"          bottom={src.bounds.bottom:.4f}  top={src.bounds.top:.4f}")

            # Pixel size (resolution)
            px = abs(src.transform.a)
            py = abs(src.transform.e)
            print(f"  Pixel size (resolution): {px:.6f} x {py:.6f} (in CRS units)")
            if src.crs and src.crs.is_geographic:
                print(f"  [NOTE] CRS is geographic (degrees). Pixel size is in degrees, not metres.")
            elif src.crs and src.crs.is_projected:
                print(f"  [NOTE] CRS is projected (likely metres). Pixel size is in metres.")
            else:
                print(f"  [NOTE] CRS unknown or None — no georeferencing embedded.")

            # Nodata
            print(f"  Nodata value: {src.nodata}")

            # Value range check (reads band 1 only — safe for large files)
            import numpy as np
            data = src.read(1)
            print(f"  Band 1 value range: min={data.min():.4f}  max={data.max():.4f}")
            print(f"  Band 1 unique values (first 20): {sorted(set(data.flat))[:20]}")

            # gdal_rasterize assessment
            print(f"\n  ── gdal_rasterize compatibility for this raster ──")
            print(f"  Use this file's CRS and transform as the TARGET GRID when rasterizing.")
            print(f"  Flags to copy:")
            print(f"    -tr {px:.6f} {py:.6f}")
            print(f"    -te {src.bounds.left:.4f} {src.bounds.bottom:.4f} "
                  f"{src.bounds.right:.4f} {src.bounds.top:.4f}")
            if src.crs:
                print(f"    -a_srs \"{src.crs.to_wkt()[:60]}...\"  (or use EPSG code if known)")

    except Exception as e:
        print(f"  [ERROR reading {path.name}]: {e}")


def inspect_shapefile(path: Path):
    """Print geopandas summary for a shapefile."""
    gpd = check_import("geopandas")
    if gpd is None:
        return

    try:
        gdf = gpd.read_file(path)
        print(f"\n  File       : {path.name}")
        print(f"  CRS        : {gdf.crs}")
        print(f"  Num rows   : {len(gdf)}")
        print(f"  Geometry type(s): {gdf.geom_type.unique().tolist()}")
        print(f"  Columns    : {gdf.columns.tolist()}")

        # Show first few rows of non-geometry columns
        non_geo_cols = [c for c in gdf.columns if c != "geometry"]
        if non_geo_cols:
            print(f"\n  First 5 rows of attribute table:")
            print(gdf[non_geo_cols].head().to_string(index=False))

        # Report column dtypes — important for gdal_rasterize
        print(f"\n  Column dtypes:")
        for col in non_geo_cols:
            dtype = gdf[col].dtype
            sample = gdf[col].iloc[0] if len(gdf) > 0 else "n/a"
            print(f"    {col:30s}  {str(dtype):15s}  (example: {sample})")

        # gdal_rasterize assessment
        print(f"\n  ── gdal_rasterize compatibility ──")
        numeric_cols = gdf[non_geo_cols].select_dtypes(include=["int", "float"]).columns.tolist()
        string_cols  = gdf[non_geo_cols].select_dtypes(include=["object"]).columns.tolist()

        if numeric_cols:
            print(f"  [OK] Numeric columns found: {numeric_cols}")
            print(f"  gdal_rasterize CAN use these directly with -a <column_name>")
            for col in numeric_cols:
                print(f"    Unique values in '{col}': {sorted(gdf[col].dropna().unique().tolist())}")
        else:
            print(f"  [WARNING] No numeric columns found.")

        if string_cols:
            print(f"  [NOTE] String columns: {string_cols}")
            print(f"  gdal_rasterize CANNOT use string columns directly.")
            print(f"  Options:")
            print(f"    1. Use the Python rasterize script (CLASS_MAP handles string→int)")
            print(f"    2. Add a numeric column in QGIS before running gdal_rasterize")
            for col in string_cols:
                print(f"    Unique string values in '{col}': {sorted(gdf[col].dropna().unique().tolist())}")

    except Exception as e:
        print(f"  [ERROR reading {path.name}]: {e}")


def inspect_hdf5(path: Path):
    """Print h5py structure for an HDF5 file."""
    h5py = check_import("h5py")
    if h5py is None:
        return

    def _print_tree(name, obj, indent=4):
        pad = " " * indent
        if hasattr(obj, "shape"):  # dataset
            print(f"{pad}{name}")
            print(f"{pad}  shape : {obj.shape}")
            print(f"{pad}  dtype : {obj.dtype}")
            if obj.attrs:
                print(f"{pad}  attrs : {dict(obj.attrs)}")
        else:  # group
            print(f"{pad}{name}/  (group)")
            if obj.attrs:
                print(f"{pad}  attrs : {dict(obj.attrs)}")

    try:
        print(f"\n  File: {path.name}")
        with h5py.File(path, "r") as f:
            print(f"  Top-level keys: {list(f.keys())}")
            print(f"\n  Full tree:")
            f.visititems(_print_tree)

            # HDF5 → gdal_rasterize note
            print(f"\n  ── gdal_rasterize compatibility ──")
            print(f"  gdal_rasterize does NOT work directly on HDF5 files.")
            print(f"  You need to extract the array and save as TIF first.")
            print(f"  Suggested steps:")
            print(f"    1. Identify the image dataset key from the tree above")
            print(f"    2. Run: pixi run python convert_h5_to_tif.py (see comment below)")
            print(f"  NOTE: If the HDF5 has a spatial reference stored in attrs,")
            print(f"        you can reconstruct georeferencing when saving the TIF.")

    except Exception as e:
        print(f"  [ERROR reading {path.name}]: {e}")


def inspect_sidecar(path: Path):
    """Explain what a sidecar file is and what it contains."""
    ext = path.suffix.lower()

    print(f"\n  File: {path.name}")

    if ext == ".tfw":
        print(f"  Type: TIFF World File — external georeferencing for a TIF.")
        print(f"  Contains 6 numbers (one per line):")
        print(f"    Line 1: pixel width (x resolution in CRS units)")
        print(f"    Line 2: row rotation (usually 0)")
        print(f"    Line 3: column rotation (usually 0)")
        print(f"    Line 4: pixel height (y resolution, usually negative)")
        print(f"    Line 5: X coordinate of upper-left pixel centre")
        print(f"    Line 6: Y coordinate of upper-left pixel centre")
        print(f"  [NOTE] This file only matters if the TIF has NO embedded CRS.")
        print(f"         If the TIF already has a CRS (check with rasterio above),")
        print(f"         the .tfw is redundant.")
        try:
            print(f"  Contents:")
            with open(path) as f:
                for i, line in enumerate(f, 1):
                    print(f"    Line {i}: {line.strip()}")
        except Exception as e:
            print(f"  [ERROR reading]: {e}")

    elif ext == ".ovr":
        print(f"  Type: GDAL Overview (pyramid) file.")
        print(f"  Contains downsampled versions of the TIF for faster display.")
        print(f"  You can IGNORE this file — it has no effect on analysis or training.")

    elif ext == ".xml":
        print(f"  Type: XML metadata file (likely ArcGIS or FGDC spatial metadata).")
        print(f"  Contains human-readable information about the dataset.")
        print(f"  You can IGNORE this file for TerraMind training.")
        try:
            with open(path) as f:
                content = f.read(800)
            print(f"  First 800 characters:")
            print(f"  {content}")
        except Exception as e:
            print(f"  [ERROR reading]: {e}")

    elif ext in (".aux", ".aux.xml"):
        print(f"  Type: GDAL auxiliary file — stores statistics (min, max, mean, std).")
        print(f"  You can IGNORE this file for TerraMind training.")

    else:
        print(f"  Sidecar type: {ext} — no specific handler. Showing raw content (first 400 chars).")
        try:
            with open(path) as f:
                print(f"  {f.read(400)}")
        except Exception:
            print(f"  [Binary file or unreadable]")


def inspect_enp(path: Path):
    """
    .enp files are not a single standard format.
    Try to detect what it is and report.
    """
    print(f"\n  File: {path.name}")
    print(f"  Type: .enp — not a universal standard format.")
    print(f"  Attempting to detect format...")

    # Check if it's readable as text (some .enp are XML or INI-style headers)
    try:
        with open(path, "r", errors="replace") as f:
            head = f.read(400)
        print(f"  First 400 characters (text attempt):")
        print(f"  {head}")
        print(f"  [HINT] If the above looks like XML or key=value pairs,")
        print(f"         it is a metadata/header file. Find the associated data file.")
        print(f"  [HINT] If it is garbled/binary, it is a binary proprietary format.")
    except Exception as e:
        print(f"  [Could not read as text]: {e}")

    # Check if rasterio can open it (some .enp are GDAL-readable)
    rasterio = check_import("rasterio")
    if rasterio:
        try:
            with rasterio.open(path) as src:
                print(f"  [OK] rasterio CAN open this file as a raster!")
                print(f"       Bands: {src.count}, Size: {src.width}x{src.height}, CRS: {src.crs}")
        except Exception:
            print(f"  rasterio cannot open this as a raster (expected for header-only files).")


def inspect_envi_header(path: Path):
    """Read an ENVI .hdr file and report band/size/interleave info."""
    print(f"\n  File: {path.name}")
    print(f"  Type: ENVI header file (.hdr) — describes a companion binary data file.")

    try:
        params = {}
        with open(path, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.split("=", 1)
                    params[k.strip().lower()] = v.strip()

        important = ["samples", "lines", "bands", "data type", "interleave",
                     "byte order", "wavelength", "wavelength units", "map info",
                     "coordinate system string"]
        for key in important:
            if key in params:
                val = params[key]
                # Truncate long values for display
                display = val if len(val) < 120 else val[:120] + "..."
                print(f"    {key:30s}: {display}")

        print(f"\n  ── What this means for TerraMind ──")
        bands = params.get("bands", "?")
        print(f"  This is a hyperspectral/multispectral image with {bands} bands.")
        print(f"  The companion data file (same name, no extension or .dat/.bin) contains the pixels.")
        print(f"  GDAL can read ENVI format directly — rasterio will open the .hdr file.")

        rasterio = check_import("rasterio")
        if rasterio:
            try:
                with rasterio.open(path) as src:
                    print(f"  [OK] rasterio opened the ENVI dataset via the .hdr file.")
                    print(f"       Bands: {src.count}, Size: {src.width}x{src.height}, CRS: {src.crs}")
            except Exception as e:
                print(f"  rasterio could not open via .hdr: {e}")

    except Exception as e:
        print(f"  [ERROR reading .hdr]: {e}")


def inspect_meta(path: Path):
    """Print first few lines of metadata/config files."""
    print(f"\n  File: {path.name}")
    try:
        with open(path, "r", errors="replace") as f:
            content = f.read(600)
        print(f"  First 600 characters:")
        print(f"  {content}")
    except Exception as e:
        print(f"  [ERROR]: {e}")


def inspect_shp_sidecar(path: Path):
    ext = path.suffix.lower()
    labels = {
        ".dbf": "Attribute table (class labels live here)",
        ".shx": "Spatial index — needed for fast access, don't delete",
        ".prj": "Coordinate reference system (CRS) definition",
        ".cpg": "Character encoding (e.g. UTF-8)",
        ".qmd": "QGIS metadata — safe to ignore for TerraMind",
        ".sbn": "Spatial index (Esri) — safe to ignore",
        ".sbx": "Spatial index (Esri) — safe to ignore",
    }
    print(f"\n  File: {path.name}")
    print(f"  Type: Shapefile component — {labels.get(ext, 'companion file')}")
    print(f"  You do not need to open this directly. geopandas handles it via the .shp file.")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY + RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(found: dict):
    section("SUMMARY AND NEXT STEPS")

    has_raster  = bool(found.get("raster"))
    has_vector  = bool(found.get("vector"))
    has_hdf5    = bool(found.get("hdf5"))
    has_sidecar = bool(found.get("sidecar"))
    has_envi    = bool(found.get("envi"))

    print("\n  Files found:")
    for family, files in sorted(found.items()):
        print(f"    {family:15s}: {len(files)} file(s)")

    print()

    if has_raster and has_vector:
        print("  [SCENARIO A] TIF image + Shapefile labels")
        print("  ─────────────────────────────────────────")
        print("  This is the cleanest scenario for TerraMind.")
        print("  Steps:")
        print("    1. Check above whether the shapefile label column is numeric or string.")
        print("    2. If numeric  → use gdal_rasterize (fast, no Python needed)")
        print("    3. If string   → use the Python rasterize script with CLASS_MAP")
        print("    4. Result: two aligned TIF files (image + mask) ready for TerraTorch.")
        print()

    if has_hdf5:
        print("  [SCENARIO B] HDF5 file(s) present")
        print("  ──────────────────────────────────")
        print("  gdal_rasterize does NOT work on HDF5.")
        print("  Steps:")
        print("    1. Look at the HDF5 tree printed above to find image + label arrays.")
        print("    2. Extract arrays with h5py and save as TIF using rasterio.")
        print("    3. Then proceed as Scenario A.")
        print()

    if has_envi:
        print("  [SCENARIO C] ENVI hyperspectral format (.hdr + data file)")
        print("  ──────────────────────────────────────────────────────────")
        print("  rasterio/GDAL can read ENVI format directly via the .hdr file.")
        print("  Steps:")
        print("    1. Point rasterio at the .hdr file — it reads the full cube.")
        print("    2. Save selected bands as a GeoTIFF for TerraTorch.")
        print()

    if has_sidecar and not has_raster:
        print("  [NOTE] Sidecar files (.tfw, .ovr, .xml) found without a matching TIF.")
        print("         Check if the TIF is in a different folder.")
        print()

    if not has_raster and not has_vector and not has_hdf5:
        print("  [NOTE] No raster, vector, or HDF5 files detected in this folder.")
        print("         Check you are pointing at the right directory.")
        print()

    print("  General TerraMind pipeline reminder:")
    print("    Raw data (any format)")
    print("      → aligned image TIF + mask TIF  (pre-processing, done once)")
    print("        → tile into patches (optional, for large drone images)")
    print("          → TerraTorch datamodule (treats them like Sen1Floods11)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect remote sensing data files.")
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Folder to scan (default: current directory)"
    )
    parser.add_argument(
        "--recurse", "-r",
        action="store_true",
        help="Recurse into subdirectories"
    )
    args = parser.parse_args()

    root = Path(args.folder).resolve()
    if not root.exists():
        print(f"[ERROR] Folder not found: {root}")
        sys.exit(1)

    print(f"\nScanning: {root}")
    print(f"Recurse:  {args.recurse}")

    found = discover_files(root, args.recurse)

    # ── Rasters ──────────────────────────────────────────────────────────────
    if found.get("raster"):
        section("RASTER FILES (.tif, .tiff, .img, .vrt)")
        for p in found["raster"]:
            inspect_raster(p)

    # ── Sidecar files ─────────────────────────────────────────────────────────
    if found.get("sidecar"):
        section("RASTER SIDECAR FILES (.tfw, .ovr, .xml, .aux)")
        for p in found["sidecar"]:
            inspect_sidecar(p)

    # ── Shapefiles ───────────────────────────────────────────────────────────
    if found.get("vector"):
        section("VECTOR / LABEL FILES (.shp, .gpkg, .geojson, ...)")
        shp_files = [p for p in found["vector"] if p.suffix.lower() == ".shp"]
        other_vec = [p for p in found["vector"] if p.suffix.lower() != ".shp"]
        for p in shp_files:
            inspect_shapefile(p)
        for p in other_vec:
            inspect_shapefile(p)

    # ── Shapefile sidecar components ──────────────────────────────────────────
    if found.get("shp_sidecar"):
        section("SHAPEFILE COMPONENT FILES (.dbf, .shx, .prj, etc.)")
        for p in found["shp_sidecar"]:
            inspect_shp_sidecar(p)

    # ── HDF5 ─────────────────────────────────────────────────────────────────
    if found.get("hdf5"):
        section("HDF5 FILES (.h5, .hdf5)")
        for p in found["hdf5"]:
            inspect_hdf5(p)

    # ── ENVI hyperspectral ────────────────────────────────────────────────────
    if found.get("envi"):
        section("ENVI HYPERSPECTRAL FORMAT (.hdr, .dat, .bin)")
        hdr_files = [p for p in found["envi"] if p.suffix.lower() == ".hdr"]
        other_envi = [p for p in found["envi"] if p.suffix.lower() != ".hdr"]
        for p in hdr_files:
            inspect_envi_header(p)
        for p in other_envi:
            print(f"\n  File: {p.name}")
            print(f"  Type: ENVI data file — binary pixel data. Open via the companion .hdr file.")

    # ── .enp and other unknowns ───────────────────────────────────────────────
    if found.get("other"):
        section("OTHER / PROPRIETARY FILES (.enp, .raw, .nc, ...)")
        for p in found["other"]:
            if p.suffix.lower() == ".enp":
                inspect_enp(p)
            else:
                print(f"\n  File: {p.name}  [{p.suffix}]")
                rasterio = check_import("rasterio")
                if rasterio:
                    try:
                        with rasterio.open(p) as src:
                            print(f"  [OK] rasterio opened this as a raster.")
                            print(f"       Bands: {src.count}, Size: {src.width}x{src.height}")
                    except Exception:
                        print(f"  rasterio could not open this file.")
                print(f"  Showing first 400 chars as text:")
                try:
                    with open(p, "r", errors="replace") as f:
                        print(f"  {f.read(400)}")
                except Exception:
                    print(f"  [Binary or unreadable]")

    # ── Metadata files ────────────────────────────────────────────────────────
    if found.get("meta"):
        section("METADATA / CONFIG FILES (.yaml, .txt, .csv, .json)")
        for p in found["meta"]:
            inspect_meta(p)

    # ── Unknown files ─────────────────────────────────────────────────────────
    if found.get("unknown"):
        section("UNRECOGNISED FILES")
        for p in found["unknown"]:
            print(f"\n  File: {p.name}  (extension: {p.suffix})")
            # Try rasterio anyway — GDAL supports a huge range of formats
            rasterio = check_import("rasterio")
            if rasterio:
                try:
                    with rasterio.open(p) as src:
                        print(f"  [OK] rasterio/GDAL can open this!")
                        print(f"       Bands: {src.count}, Size: {src.width}x{src.height}, CRS: {src.crs}")
                except Exception:
                    print(f"  rasterio cannot open this file.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(found)
    print()


if __name__ == "__main__":
    main()
