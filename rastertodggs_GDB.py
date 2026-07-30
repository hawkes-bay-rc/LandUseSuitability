# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:04:43 2026

@author: Ashton.Eaves
"""
###### raster2dggs ###########################################################
### adapted from:  https://github.com/manaakiwhenua/raster2dggs/tree/master
##############################################################################

# In anaconda prompt open env: conda activate h3raster
# Then run spyder

import os
import h3
import pandas as pd
import rasterio
from pyproj import Transformer

# =============================================================================
# User settings
# =============================================================================

# H3 Resolution 11 grid
h3_csv = (
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability"
    r"\1_Data\H3Res11.csv"
)

# Folder containing the species raster geodatabases
species_gdb_folder = (
    r"D:\Land\GIS_DATA\Forestry\Forestry_RTRP_GIS\Data"
    r"\SCION_RTRP_original_delivery\AfforErodLandHB_deliverables"
    r"\TreeSpeciesSiteSuit"
)

# Combined output CSV
output_csv = (
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability"
    r"\3_Outputs\RTRP_TreeSpecies_H3Res11.csv"
)

# H3 identifier field
h3_field = "GRID_ID"

# Band to sample from each raster
raster_band = 1

# =============================================================================
# Species raster configuration
# =============================================================================

# Each dictionary identifies:
#   gdb_name     = file geodatabase containing the raster
#   raster_name  = raster dataset inside the geodatabase
#   output_field = field added to the combined CSV

species_rasters = [
    {
        "gdb_name": "Prad_Site_Suit.gdb",
        "raster_name": "SiteSuit_Prad",
        "output_field": "PRad",
    },
    {
        "gdb_name": "Clus_Site_Suit.gdb",
        "raster_name": "SiteSuit_C_lus",
        "output_field": "CLus",
    },
    {
        "gdb_name": "EucGen_Site_Suit.gdb",
        "raster_name": "SiteSuit_Egenx",
        "output_field": "EGenx",
    },
    {
        "gdb_name": "Manuka_Site_Suit.gdb",
        "raster_name": "SiteSuit_Manuka",
        "output_field": "Manuka",
    },
    {
        "gdb_name": "Redwood_Site_Suit.gdb",
        "raster_name": "SiteSuit_Redwoodx",
        "output_field": "Redwoodx",
    },
    {
        "gdb_name": "Totara_Site_Suit.gdb",
        "raster_name": "SiteSuit_Totarax",
        "output_field": "Totarax",
    },
]

# =============================================================================
# Initial checks
# =============================================================================

if not os.path.exists(h3_csv):
    raise FileNotFoundError(
        f"H3 CSV was not found:\n{h3_csv}"
    )

if not os.path.exists(species_gdb_folder):
    raise FileNotFoundError(
        f"Species geodatabase folder was not found:\n"
        f"{species_gdb_folder}"
    )

output_folder = os.path.dirname(output_csv)

if output_folder:
    os.makedirs(output_folder, exist_ok=True)

# =============================================================================
# Read the H3 grid
# =============================================================================

print("=" * 79)
print("READING H3 GRID")
print("=" * 79)

h3_df = pd.read_csv(
    h3_csv,
    usecols=[h3_field],
    dtype={h3_field: "string"},
)

print(f"H3 cells loaded: {len(h3_df):,}")

# Remove blank H3 IDs
blank_h3_mask = h3_df[h3_field].isna()

if blank_h3_mask.any():
    blank_count = blank_h3_mask.sum()

    print(
        f"Warning: removing {blank_count:,} rows "
        f"with missing {h3_field} values."
    )

    h3_df = h3_df.loc[~blank_h3_mask].copy()

# Check for duplicate H3 IDs
duplicate_count = h3_df[h3_field].duplicated().sum()

if duplicate_count > 0:
    print(
        f"Warning: {duplicate_count:,} duplicate H3 IDs found."
    )
    print("Keeping the first occurrence of each GRID_ID.")

    h3_df = h3_df.drop_duplicates(
        subset=[h3_field],
        keep="first",
    ).copy()

h3_df = h3_df.reset_index(drop=True)

# =============================================================================
# Validate H3 IDs
# =============================================================================

print("\nValidating H3 IDs...")

valid_h3_mask = h3_df[h3_field].map(h3.is_valid_cell)

invalid_count = (~valid_h3_mask).sum()

if invalid_count > 0:
    invalid_examples = (
        h3_df.loc[~valid_h3_mask, h3_field]
        .head(10)
        .tolist()
    )

    raise ValueError(
        f"{invalid_count:,} invalid H3 IDs were found.\n"
        f"Examples:\n{invalid_examples}"
    )

print("All H3 IDs are valid.")

# =============================================================================
# Calculate H3 centre coordinates once
# =============================================================================

print("\nCalculating H3 centre coordinates...")

centres = h3_df[h3_field].map(h3.cell_to_latlng)

h3_df["lat"] = centres.str[0].astype("float64")
h3_df["lon"] = centres.str[1].astype("float64")

longitude_values = h3_df["lon"].to_numpy()
latitude_values = h3_df["lat"].to_numpy()

print("H3 centre coordinates calculated.")

# =============================================================================
# Process each species raster
# =============================================================================

successful_fields = []
failed_fields = []

for raster_number, raster_config in enumerate(
    species_rasters,
    start=1,
):

    gdb_name = raster_config["gdb_name"]
    raster_name = raster_config["raster_name"]
    output_field = raster_config["output_field"]

    gdb_path = os.path.join(
        species_gdb_folder,
        gdb_name,
    )

    print("\n" + "=" * 79)
    print(
        f"PROCESSING RASTER {raster_number} "
        f"OF {len(species_rasters)}: {output_field}"
    )
    print("=" * 79)

    print(f"Geodatabase: {gdb_path}")
    print(f"Raster dataset: {raster_name}")
    print(f"Output field: {output_field}")

    try:

        # ---------------------------------------------------------------------
        # Check that the geodatabase exists
        # ---------------------------------------------------------------------

        if not os.path.exists(gdb_path):
            raise FileNotFoundError(
                f"Geodatabase was not found:\n{gdb_path}"
            )

        # ---------------------------------------------------------------------
        # List raster subdatasets
        # ---------------------------------------------------------------------

        print("\nInspecting geodatabase...")

        with rasterio.open(gdb_path) as gdb:
            subdatasets = gdb.subdatasets

        if not subdatasets:
            raise ValueError(
                "No raster subdatasets were returned from the "
                "file geodatabase. The current Rasterio/GDAL "
                "installation may not support raster datasets "
                "stored in this geodatabase."
            )

        print(
            f"Raster subdatasets found: "
            f"{len(subdatasets):,}"
        )

        # ---------------------------------------------------------------------
        # Find the requested raster dataset
        # ---------------------------------------------------------------------

        # First try an exact match against the final part of the
        # raster subdataset path.
        exact_matches = [
            subdataset
            for subdataset in subdatasets
            if subdataset.lower().endswith(
                raster_name.lower()
            )
        ]

        # Fall back to a partial text match if needed.
        if exact_matches:
            matching_subdatasets = exact_matches
        else:
            matching_subdatasets = [
                subdataset
                for subdataset in subdatasets
                if raster_name.lower()
                in subdataset.lower()
            ]

        if not matching_subdatasets:

            print("\nAvailable raster subdatasets:")

            for subdataset in subdatasets:
                print(f"  {subdataset}")

            raise ValueError(
                f"Raster '{raster_name}' was not found in:\n"
                f"{gdb_path}"
            )

        if len(matching_subdatasets) > 1:

            print(
                f"Warning: {len(matching_subdatasets)} "
                f"possible matches were found."
            )

            for match in matching_subdatasets:
                print(f"  {match}")

            print("Using the first match.")

        raster_path = matching_subdatasets[0]

        print(f"\nUsing raster:\n{raster_path}")

        # ---------------------------------------------------------------------
        # Open raster and sample H3 centres
        # ---------------------------------------------------------------------

        with rasterio.open(raster_path) as src:

            print("\nRaster information:")
            print(f"  Driver: {src.driver}")
            print(f"  CRS: {src.crs}")
            print(f"  Transform: {src.transform}")
            print(f"  Width: {src.width:,}")
            print(f"  Height: {src.height:,}")
            print(f"  Band count: {src.count}")
            print(f"  Data types: {src.dtypes}")
            print(f"  NoData: {src.nodata}")
            print(f"  Bounds: {src.bounds}")

            if src.crs is None:
                raise ValueError(
                    f"Raster '{raster_name}' has no readable CRS."
                )

            if src.count < raster_band:
                raise ValueError(
                    f"Raster '{raster_name}' does not contain "
                    f"band {raster_band}."
                )

            # Transform H3 longitude and latitude coordinates into
            # the raster coordinate reference system.
            transformer = Transformer.from_crs(
                "EPSG:4326",
                src.crs,
                always_xy=True,
            )

            print(
                "\nTransforming H3 centres into "
                "raster coordinates..."
            )

            xs, ys = transformer.transform(
                longitude_values,
                latitude_values,
            )

            print("Sampling raster values...")

            coordinates = zip(xs, ys)

            values = [
                sampled[raster_band - 1]
                for sampled in src.sample(
                    coordinates,
                    indexes=raster_band,
                    masked=False,
                )
            ]

            nodata = src.nodata

        # ---------------------------------------------------------------------
        # Add sampled values to the master H3 dataframe
        # ---------------------------------------------------------------------

        h3_df[output_field] = pd.to_numeric(
            values,
            errors="coerce",
        )

        # Replace the raster's defined NoData value.
        if nodata is not None:

            h3_df.loc[
                h3_df[output_field] == nodata,
                output_field,
            ] = pd.NA

        # Some rasters use extremely large floating-point values
        # as NoData.
        h3_df.loc[
            h3_df[output_field].abs() > 1e20,
            output_field,
        ] = pd.NA

        # Replace positive and negative infinity.
        h3_df.loc[
            h3_df[output_field].isin(
                [float("inf"), float("-inf")]
            ),
            output_field,
        ] = pd.NA

        valid_count = (
            h3_df[output_field]
            .notna()
            .sum()
        )

        missing_count = (
            h3_df[output_field]
            .isna()
            .sum()
        )

        print("\nRaster completed:")
        print(f"  Valid values: {valid_count:,}")
        print(f"  Missing values: {missing_count:,}")

        if valid_count > 0:

            print(
                f"  Minimum: "
                f"{h3_df[output_field].min()}"
            )

            print(
                f"  Maximum: "
                f"{h3_df[output_field].max()}"
            )

            print(
                f"  Mean: "
                f"{h3_df[output_field].mean()}"
            )

        successful_fields.append(output_field)

    except Exception as error:

        print("\nRASTER FAILED")
        print(f"  Output field: {output_field}")
        print(f"  Error: {error}")

        # Add the field anyway so the final output retains a
        # consistent schema.
        h3_df[output_field] = pd.NA

        failed_fields.append(
            {
                "output_field": output_field,
                "gdb_name": gdb_name,
                "raster_name": raster_name,
                "error": str(error),
            }
        )

# =============================================================================
# Prepare the combined output
# =============================================================================

print("\n" + "=" * 79)
print("PREPARING COMBINED OUTPUT")
print("=" * 79)

output_fields = [
    raster_config["output_field"]
    for raster_config in species_rasters
]

output_df = h3_df[
    [h3_field] + output_fields
].copy()

print("\nOutput preview:")
print(output_df.head())

print("\nOutput fields:")
print(output_df.columns.tolist())

# =============================================================================
# Write the combined CSV
# =============================================================================

print(f"\nWriting combined CSV:\n{output_csv}")

output_df.to_csv(
    output_csv,
    index=False,
)

# =============================================================================
# Final summary
# =============================================================================

print("\n" + "=" * 79)
print("TREE-SPECIES RASTER PROCESSING FINISHED")
print("=" * 79)

print(f"\nH3 cells exported: {len(output_df):,}")
print(f"Output fields: {len(output_fields):,}")
print(f"Successful rasters: {len(successful_fields):,}")
print(f"Failed rasters: {len(failed_fields):,}")

print("\nSuccessful output fields:")

if successful_fields:
    for field in successful_fields:
        print(f"  - {field}")
else:
    print("  None")

print("\nFailed output fields:")

if failed_fields:

    for failure in failed_fields:

        print(
            f"  - {failure['output_field']} "
            f"({failure['raster_name']}): "
            f"{failure['error']}"
        )

else:
    print("  None")

print("\nValid values by field:")

for field in output_fields:

    valid_count = output_df[field].notna().sum()

    print(
        f"  {field}: "
        f"{valid_count:,} of {len(output_df):,}"
    )

print(f"\nCombined output saved to:\n{output_csv}")