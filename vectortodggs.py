# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:58:42 2026

@author: Ashton.Eaves
"""

###### raster2dggs ###########################################################
### adapted from:  https://github.com/manaakiwhenua/raster2dggs/tree/master
##############################################################################

# In anaconda prompt open env: conda activate h3raster
# Then run spyder

import h3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# =============================================================================
# User settings
# =============================================================================

# H3 grid table
h3_csv = (
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability"
    r"\1_Data\H3Res11.csv"
)

# Vector polygon layer
vector_path = (
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability"
    r"\1_Data\LUDB_2026.shp"
)

# Name of the feature class when reading from a geodatabase.
# Set to None when reading a shapefile, GeoPackage layer with only one layer,
# or another single-layer vector file.
vector_layer = None

# Output table
output_csv = (
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability"
    r"\3_Outputs\LUDB_H3.csv"
)

# H3 ID field
h3_field = "GRID_ID"

# Polygon attributes to transfer to the H3 grid
vector_fields = [
    "LandUseNam",
    "LandUseGro",
]

# Number of H3 cells processed at once
chunk_size = 500_000

# Spatial relationship:
#
# "within"     = H3 centre must be within a polygon
# "intersects" = H3 centre intersects a polygon
#
# For points and polygons, these usually produce the same result.
spatial_predicate = "within"

# =============================================================================
# Read vector polygons
# =============================================================================

print("Reading vector layer...")

if vector_layer is None:
    polygons = gpd.read_file(vector_path)
else:
    polygons = gpd.read_file(
        vector_path,
        layer=vector_layer,
    )

print(f"Polygon features loaded: {len(polygons):,}")
print(f"Polygon CRS: {polygons.crs}")

if polygons.crs is None:
    raise ValueError(
        "The vector layer has no defined coordinate reference system."
    )

# Check that the requested fields exist
missing_fields = [
    field
    for field in vector_fields
    if field not in polygons.columns
]

if missing_fields:
    raise KeyError(
        "The following vector fields were not found:\n"
        + "\n".join(missing_fields)
    )

# Keep only the fields needed for the spatial join
polygons = polygons[
    vector_fields + ["geometry"]
].copy()

# Remove empty or missing geometries
polygons = polygons[
    polygons.geometry.notna()
    & ~polygons.geometry.is_empty
].copy()

# Optional geometry repair.
# Useful when the vector layer contains invalid polygons.
invalid_count = (~polygons.geometry.is_valid).sum()

if invalid_count > 0:
    print(f"Repairing {invalid_count:,} invalid polygon geometries...")
    polygons["geometry"] = polygons.geometry.make_valid()

# Create the spatial index before processing the H3 chunks
print("Creating polygon spatial index...")
_ = polygons.sindex

# =============================================================================
# Read H3 table in chunks and spatially join
# =============================================================================

print("Processing H3 cells...")

first_chunk = True
total_rows = 0
matched_rows = 0

for chunk_number, h3_df in enumerate(
    pd.read_csv(
        h3_csv,
        usecols=[h3_field],
        chunksize=chunk_size,
        dtype={h3_field: "string"},
    ),
    start=1,
):

    print(
        f"\nChunk {chunk_number}: "
        f"{len(h3_df):,} H3 cells"
    )

    # -------------------------------------------------------------------------
    # Convert H3 IDs to centre latitude and longitude
    # -------------------------------------------------------------------------

    coordinates = h3_df[h3_field].map(h3.cell_to_latlng)

    h3_df["lat"] = coordinates.str[0]
    h3_df["lon"] = coordinates.str[1]

    # -------------------------------------------------------------------------
    # Create H3 centre points in WGS 84
    # -------------------------------------------------------------------------

    h3_points = gpd.GeoDataFrame(
        h3_df,
        geometry=gpd.points_from_xy(
            h3_df["lon"],
            h3_df["lat"],
        ),
        crs="EPSG:4326",
    )

    # -------------------------------------------------------------------------
    # Reproject H3 points to match the polygon CRS
    # -------------------------------------------------------------------------

    h3_points = h3_points.to_crs(polygons.crs)

    # -------------------------------------------------------------------------
    # Spatial join: transfer polygon attributes to H3 centres
    # -------------------------------------------------------------------------

    joined = gpd.sjoin(
        h3_points,
        polygons,
        how="left",
        predicate=spatial_predicate,
    )

    # -------------------------------------------------------------------------
    # Handle points matching more than one polygon
    #
    # This can happen where polygons overlap or where "intersects" is used for
    # points lying exactly on polygon boundaries.
    # Keep the first polygon match for each H3 cell.
    # -------------------------------------------------------------------------

    duplicate_count = joined.duplicated(
        subset=[h3_field],
        keep=False,
    ).sum()

    if duplicate_count > 0:
        print(
            f"Warning: {duplicate_count:,} joined rows belong to H3 cells "
            "with multiple polygon matches."
        )

        joined = joined.drop_duplicates(
            subset=[h3_field],
            keep="first",
        )

    # -------------------------------------------------------------------------
    # Export only the join fields
    # -------------------------------------------------------------------------

    out = joined[
        [h3_field] + vector_fields
    ].copy()

    chunk_matches = out[vector_fields].notna().any(axis=1).sum()

    total_rows += len(out)
    matched_rows += chunk_matches

    # Append each chunk to the same CSV
    out.to_csv(
        output_csv,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False,
    )

    first_chunk = False

    print(f"Matched cells in chunk: {chunk_matches:,}")
    print(f"Total cells processed: {total_rows:,}")

# =============================================================================
# Final summary
# =============================================================================

print("\nFinished")
print(f"Output: {output_csv}")
print(f"Total H3 cells: {total_rows:,}")
print(f"Cells matched to a polygon: {matched_rows:,}")
print(f"Cells without a polygon match: {total_rows - matched_rows:,}")