# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:21:11 2026

@author: Ashton.Eaves
"""

###### netcdf2dggs ###########################################################

##############################################################################

# In anaconda prompt open env: conda activate h3raster
# Then run spyder

###############################################################################
# Extract climate NetCDF values at H3 cell centres
#
# Inputs currently included:
#   - Frost days: March–May
#   - Frost days: September–November
#   - Growing degree days
#
# The H3 CSV is processed in chunks so approximately 49 million cells do not
# need to be held in memory at once.
#
# For every NetCDF input, the script prints:
#   - dataset dimensions and coordinates;
#   - data-variable names;
#   - variable dtype, shape, dimensions, units and long name;
#   - coordinate ranges;
#   - selected climate variable;
#   - valid climate-value range.
###############################################################################

import os
import time

import h3
import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# User settings
# =============================================================================

h3_csv = (
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability"
    r"\1_Data\H3Res11.csv"
)

output_csv = (
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability"
    r"\3_Outputs\ClimateVariables_H3.csv"
)

h3_field = "GRID_ID"

# Number of H3 rows processed at once.
# Reduce to 250_000 if RAM becomes constrained.
chunk_size = 500_000


# =============================================================================
# NetCDF input folders
# =============================================================================

frost_folder = (
    r"\\fileenviro\Esci\Climate"
    r"\NIWA_Climate_Change_Projections_2024"
    r"\Individual_Variables\Number_Frost_Days"
)

gdd_folder = (
    r"\\fileenviro\Esci\Climate"
    r"\NIWA_Climate_Change_Projections_2024"
    r"\Individual_Variables\Growing_Degree_Days"
)

hfl_folder = (
    r"\\fileenviro\Esci\Climate"
    r"\NIWA_Climate_Change_Projections_2024"
    r"\Individual_Variables\Number_Very_Hot_Days_30deg"
)

met_folder = (
    r"\\fileenviro\Esci\Climate"
    r"\NIWA_Climate_Change_Projections_2024"
    r"\Individual_Variables\Average_Daily_AirT_Tmean"
)

mnt_folder = (
    r"\\fileenviro\Esci\Climate"
    r"\NIWA_Climate_Change_Projections_2024"
    r"\Individual_Variables\Average_Daily_AirT_Tmin"
)

mxt_folder = (
    r"\\fileenviro\Esci\Climate"
    r"\NIWA_Climate_Change_Projections_2024"
    r"\Individual_Variables\Average_Daily_AirT_TMax"
)

rah_folder = (
    r"\\fileenviro\Esci\Climate"
    r"\NIWA_Climate_Change_Projections_2024"
    r"\Individual_Variables\Heavy_Rainfall"
)


# =============================================================================
# NetCDF input files
# =============================================================================

# Each dictionary value must be the complete path to one NetCDF file.
# The dictionary key becomes the output CSV field name.

netcdf_files = {
    "FrostDays_MAM": os.path.join(
        frost_folder,
        "FD_historical_MMM_CCAM_base_bp1995-2014_MAM_NZ5km.nc",
    ),

    "FrostDays_SON": os.path.join(
        frost_folder,
        "FD_historical_MMM_CCAM_base_bp1995-2014_SON_NZ5km.nc",
    ),

    "GrowingDegreeDays": os.path.join(
        gdd_folder,
        "GDD10_historical_MMM_CCAM_base_bp1995-2014_ANN_NZ5km.nc",
    ),

    "HeatStress_DJF": os.path.join(
        hfl_folder,
        "TX30_historical_MMM_CCAM_base_bp1995-2014_DJF_NZ5km.nc",
    ),

    "MeanTemp_SON": os.path.join(
        met_folder,
        "T_historical_MMM_CCAM_base_bp1995-2014_SON_NZ5km.nc",
    ),

    "MinTemp_SON": os.path.join(
        mnt_folder,
        "TN_historical_MMM_CCAM_base_bp1995-2014_SON_NZ5km.nc",
    ),

    "MaxTemp_SON": os.path.join(
        mxt_folder,
        "TX_historical_MMM_CCAM_base_bp1995-2014_SON_NZ5km.nc",
    ),

    "MaxTemp_DJF": os.path.join(
        mxt_folder,
        "TX_historical_MMM_CCAM_base_bp1995-2014_DJF_NZ5km.nc",
    ),

    "HarvestRainfall_DJF": os.path.join(
        rah_folder,
        "R99pVAL_historical_MMM_CCAM_base_bp1995-2014_DJF_NZ5km.nc",
    ),
}


# =============================================================================
# Helper functions
# =============================================================================

def find_coordinate_name(dataset, possible_names):
    """
    Find a coordinate or variable name without case sensitivity.
    """

    available_names = list(dataset.coords) + list(dataset.variables)

    name_lookup = {
        str(name).lower(): name
        for name in available_names
    }

    for possible_name in possible_names:
        matched_name = name_lookup.get(possible_name.lower())

        if matched_name is not None:
            return matched_name

    return None


def find_data_variable(dataset):
    """
    Identify the main two-dimensional climate data variable.

    Coordinate, projection and bounds variables are excluded.
    """

    excluded_names = {
        "lat",
        "latitude",
        "lon",
        "longitude",
        "x",
        "y",
        "time",
        "crs",
        "projection",
        "rotated_pole",
        "spatial_ref",
        "lat_bnds",
        "lon_bnds",
        "latitude_bnds",
        "longitude_bnds",
        "time_bnds",
        "bounds",
    }

    candidates = []

    for variable_name, variable in dataset.data_vars.items():

        if variable_name.lower() in excluded_names:
            continue

        # Climate layers usually have latitude and longitude dimensions,
        # sometimes plus a one-value time dimension.
        if variable.ndim >= 2:
            candidates.append(variable_name)

    if not candidates:
        raise ValueError(
            "No suitable climate data variable was found.\n"
            f"Available data variables: {list(dataset.data_vars)}"
        )

    print(f"\nCandidate climate variables: {candidates}")

    # Prefer familiar climate-variable names when there is more than one
    # candidate.
    preferred_terms = [
        "frost",
        "gdd",
        "degree",
        "tx30",
        "heat",
        "hot",
        "r99",
        "rain",
        "temperature",
        "temp",
        "tmean",
        "tmin",
        "tmax",
        "fd",
        "tn",
        "tx",
    ]

    for term in preferred_terms:
        for variable_name in candidates:
            if term in variable_name.lower():
                return variable_name

    # If no preferred name is found, use the first viable spatial variable.
    return candidates[0]


def print_netcdf_metadata(dataset, nc_path):
    """
    Print dimensions, coordinates, variables, metadata and coordinate ranges
    for one NetCDF dataset.
    """

    print("\n" + "=" * 79)
    print(f"NETCDF METADATA: {os.path.basename(nc_path)}")
    print("=" * 79)

    print("\nDataset summary")
    print(dataset)

    print("\nDataset dimensions")
    print(dict(dataset.sizes))

    print("\nCoordinate names")
    print(list(dataset.coords))

    print("\nData variables")
    print(list(dataset.data_vars))

    print("\nVariable metadata")

    for variable_name in dataset.data_vars:

        variable = dataset[variable_name]
        attributes = variable.attrs

        print(f"\n{variable_name}")
        print(f"  dtype         : {variable.dtype}")
        print(f"  shape         : {variable.shape}")
        print(f"  dimensions    : {variable.dims}")
        print(f"  units         : {attributes.get('units', '—')}")
        print(f"  long_name     : {attributes.get('long_name', '—')}")
        print(f"  standard_name : {attributes.get('standard_name', '—')}")
        print(f"  description   : {attributes.get('description', '—')}")

        # _FillValue may be held in encoding after xarray decodes the data.
        fill_value = attributes.get(
            "_FillValue",
            variable.encoding.get("_FillValue", "—"),
        )

        missing_value = attributes.get(
            "missing_value",
            variable.encoding.get("missing_value", "—"),
        )

        print(f"  fill_value    : {fill_value}")
        print(f"  missing_value : {missing_value}")

    print("\nCoordinate ranges")

    for coordinate_name in dataset.coords:

        coordinate = dataset[coordinate_name]
        values = coordinate.values

        if values.size == 0:
            print(f"\ncoord {coordinate_name}: empty")
            continue

        try:
            valid_values = values[
                pd.notna(values)
            ]

            if valid_values.size == 0:
                print(
                    f"\ncoord {coordinate_name}: "
                    f"no valid values (n={values.size})"
                )
                continue

            minimum = valid_values.min()
            maximum = valid_values.max()

            print(
                f"\ncoord {coordinate_name}: "
                f"{minimum} -> {maximum} "
                f"(n={values.size}, shape={values.shape})"
            )

            # Print an approximate coordinate interval for one-dimensional,
            # numeric coordinate arrays.
            if coordinate.ndim == 1 and valid_values.size > 1:

                try:
                    numeric_values = valid_values.astype(np.float64)

                    differences = np.diff(numeric_values)

                    if differences.size > 0:
                        print(
                            f"  approximate step: "
                            f"{np.nanmedian(differences)}"
                        )

                except (TypeError, ValueError):
                    # Time and object coordinates may not convert directly.
                    pass

        except (TypeError, ValueError):
            print(
                f"\ncoord {coordinate_name}: "
                f"range could not be calculated "
                f"(n={values.size}, shape={values.shape})"
            )


def nearest_indices(coordinate_values, query_values):
    """
    Return the nearest coordinate-array index for every query value.

    Works for ascending and descending one-dimensional coordinate arrays.
    """

    coordinate_values = np.asarray(
        coordinate_values,
        dtype=np.float64,
    )

    query_values = np.asarray(
        query_values,
        dtype=np.float64,
    )

    if coordinate_values.ndim != 1:
        raise ValueError(
            "nearest_indices requires a one-dimensional coordinate array."
        )

    if len(coordinate_values) < 2:
        raise ValueError(
            "The coordinate array must contain at least two values."
        )

    descending = coordinate_values[0] > coordinate_values[-1]

    if descending:
        working_values = coordinate_values[::-1]
    else:
        working_values = coordinate_values

    insertion_indices = np.searchsorted(
        working_values,
        query_values,
        side="left",
    )

    insertion_indices = np.clip(
        insertion_indices,
        1,
        len(working_values) - 1,
    )

    lower_indices = insertion_indices - 1
    upper_indices = insertion_indices

    lower_values = working_values[lower_indices]
    upper_values = working_values[upper_indices]

    choose_lower = (
        np.abs(query_values - lower_values)
        <= np.abs(upper_values - query_values)
    )

    result_indices = np.where(
        choose_lower,
        lower_indices,
        upper_indices,
    )

    if descending:
        result_indices = (
            len(coordinate_values)
            - 1
            - result_indices
        )

    return result_indices


def load_netcdf_grid(nc_path):
    """
    Load one regular latitude/longitude NetCDF into a compact lookup object.

    The metadata-reporting function is called here so each NetCDF is opened
    only once.
    """

    with xr.open_dataset(
        nc_path,
        decode_coords="all",
        mask_and_scale=True,
    ) as dataset:

        # Print complete metadata for this input.
        print_netcdf_metadata(
            dataset=dataset,
            nc_path=nc_path,
        )

        latitude_name = find_coordinate_name(
            dataset,
            [
                "latitude",
                "lat",
                "nav_lat",
                "grid_latitude",
                "rlat",
            ],
        )

        longitude_name = find_coordinate_name(
            dataset,
            [
                "longitude",
                "lon",
                "nav_lon",
                "grid_longitude",
                "rlon",
            ],
        )

        if latitude_name is None or longitude_name is None:
            raise ValueError(
                "Could not identify latitude and longitude coordinates.\n"
                f"Available coordinates: {list(dataset.coords)}\n"
                f"Available variables: {list(dataset.variables)}"
            )

        data_variable_name = find_data_variable(dataset)

        print("\nSelected NetCDF components")
        print(f"  latitude variable  : {latitude_name}")
        print(f"  longitude variable : {longitude_name}")
        print(f"  climate variable   : {data_variable_name}")

        data_array = (
            dataset[data_variable_name]
            .squeeze(drop=True)
        )

        latitude = (
            dataset[latitude_name]
            .squeeze(drop=True)
        )

        longitude = (
            dataset[longitude_name]
            .squeeze(drop=True)
        )

        print(f"  climate dimensions : {data_array.dims}")
        print(f"  climate shape      : {data_array.shape}")
        print(f"  latitude shape     : {latitude.shape}")
        print(f"  longitude shape    : {longitude.shape}")

        if latitude.ndim != 1 or longitude.ndim != 1:
            raise ValueError(
                "This script expects one-dimensional latitude and longitude "
                "coordinates.\n"
                f"Latitude dimensions: {latitude.dims}\n"
                f"Longitude dimensions: {longitude.dims}"
            )

        if data_array.ndim != 2:
            raise ValueError(
                "The selected climate variable was not two-dimensional after "
                "singleton dimensions were removed.\n"
                f"Variable: {data_variable_name}\n"
                f"Dimensions: {data_array.dims}\n"
                f"Shape: {data_array.shape}"
            )

        latitude_dimension = latitude.dims[0]
        longitude_dimension = longitude.dims[0]

        expected_dimensions = {
            latitude_dimension,
            longitude_dimension,
        }

        if set(data_array.dims) != expected_dimensions:
            raise ValueError(
                "The climate data dimensions do not match the identified "
                "latitude and longitude dimensions.\n"
                f"Climate dimensions: {data_array.dims}\n"
                f"Latitude dimension: {latitude_dimension}\n"
                f"Longitude dimension: {longitude_dimension}"
            )

        # Ensure the climate array is ordered [latitude, longitude].
        data_array = data_array.transpose(
            latitude_dimension,
            longitude_dimension,
        )

        latitude_values = latitude.to_numpy().astype(
            np.float64,
            copy=False,
        )

        longitude_values = longitude.to_numpy().astype(
            np.float64,
            copy=False,
        )

        climate_values = data_array.to_numpy().astype(
            np.float32,
            copy=False,
        )

        # Remove any undecoded fill or invalid values.
        invalid_values = (
            ~np.isfinite(climate_values)
            | (np.abs(climate_values) > 1.0e20)
        )

        climate_values[invalid_values] = np.nan

        print("\nPrepared lookup grid")
        print(f"  final grid shape : {climate_values.shape}")

        print(
            "  latitude range   : "
            f"{np.nanmin(latitude_values):.6f} to "
            f"{np.nanmax(latitude_values):.6f}"
        )

        print(
            "  longitude range  : "
            f"{np.nanmin(longitude_values):.6f} to "
            f"{np.nanmax(longitude_values):.6f}"
        )

        finite_values = climate_values[
            np.isfinite(climate_values)
        ]

        if len(finite_values) > 0:
            print(
                "  climate range    : "
                f"{finite_values.min():.4f} to "
                f"{finite_values.max():.4f}"
            )

            print(
                "  climate mean     : "
                f"{finite_values.mean():.4f}"
            )

            print(
                "  valid cells      : "
                f"{len(finite_values):,} of "
                f"{climate_values.size:,}"
            )

        else:
            print(
                "  warning          : "
                "no valid climate values were found"
            )

        return {
            "latitude": latitude_values,
            "longitude": longitude_values,
            "values": climate_values,
            "variable_name": data_variable_name,
            "units": dataset[data_variable_name].attrs.get(
                "units",
                "",
            ),
            "long_name": dataset[data_variable_name].attrs.get(
                "long_name",
                "",
            ),
        }


def sample_regular_grid(
    grid,
    query_latitudes,
    query_longitudes,
):
    """
    Sample a regular latitude/longitude grid using nearest-neighbour lookup.
    """

    query_latitudes = np.asarray(
        query_latitudes,
        dtype=np.float64,
    )

    query_longitudes = np.asarray(
        query_longitudes,
        dtype=np.float64,
    )

    latitude_values = grid["latitude"]
    longitude_values = grid["longitude"]
    climate_values = grid["values"]

    # Adjust H3 longitudes where a NetCDF uses a 0–360 longitude convention.
    if (
        np.nanmax(longitude_values) > 180
        and np.nanmin(query_longitudes) < 0
    ):
        adjusted_longitudes = query_longitudes % 360
    else:
        adjusted_longitudes = query_longitudes

    latitude_min = np.nanmin(latitude_values)
    latitude_max = np.nanmax(latitude_values)

    longitude_min = np.nanmin(longitude_values)
    longitude_max = np.nanmax(longitude_values)

    # Do not assign the nearest edge cell to H3 centres outside the NetCDF.
    inside_extent = (
        np.isfinite(query_latitudes)
        & np.isfinite(adjusted_longitudes)
        & (query_latitudes >= latitude_min)
        & (query_latitudes <= latitude_max)
        & (adjusted_longitudes >= longitude_min)
        & (adjusted_longitudes <= longitude_max)
    )

    sampled_values = np.full(
        len(query_latitudes),
        np.nan,
        dtype=np.float32,
    )

    if not inside_extent.any():
        return sampled_values

    latitude_indices = nearest_indices(
        latitude_values,
        query_latitudes[inside_extent],
    )

    longitude_indices = nearest_indices(
        longitude_values,
        adjusted_longitudes[inside_extent],
    )

    sampled_values[inside_extent] = climate_values[
        latitude_indices,
        longitude_indices,
    ]

    return sampled_values


def h3_centres(h3_cells):
    """
    Convert H3 identifiers to WGS84 latitude and longitude arrays.
    """

    number_of_cells = len(h3_cells)

    latitudes = np.empty(
        number_of_cells,
        dtype=np.float64,
    )

    longitudes = np.empty(
        number_of_cells,
        dtype=np.float64,
    )

    for index, cell in enumerate(h3_cells):

        try:
            latitude, longitude = h3.cell_to_latlng(cell)

        except Exception as error:
            raise ValueError(
                f"Invalid H3 identifier at chunk position {index}: "
                f"{cell}"
            ) from error

        latitudes[index] = latitude
        longitudes[index] = longitude

    return latitudes, longitudes


# =============================================================================
# Validate inputs
# =============================================================================

if not os.path.isfile(h3_csv):
    raise FileNotFoundError(
        f"H3 input CSV was not found:\n{h3_csv}"
    )

if not netcdf_files:
    raise ValueError(
        "No NetCDF inputs have been configured."
    )

print("\nConfigured climate inputs")

for output_field, nc_path in netcdf_files.items():

    print(f"  {output_field}: {nc_path}")

    if not os.path.isfile(nc_path):
        raise FileNotFoundError(
            f"NetCDF input for '{output_field}' was not found:\n"
            f"{nc_path}"
        )

output_directory = os.path.dirname(output_csv)

if output_directory:
    os.makedirs(
        output_directory,
        exist_ok=True,
    )


# =============================================================================
# Load the small NetCDF grids once
# =============================================================================

netcdf_grids = {}

for output_field, nc_path in netcdf_files.items():

    print("\n" + "#" * 79)
    print(f"PREPARING OUTPUT FIELD: {output_field}")
    print(f"Path: {nc_path}")
    print("#" * 79)

    netcdf_grids[output_field] = load_netcdf_grid(
        nc_path
    )


# =============================================================================
# Prepare output
# =============================================================================

if os.path.exists(output_csv):
    os.remove(output_csv)
    print(f"\nRemoved existing output: {output_csv}")

start_time = time.perf_counter()

total_rows = 0

total_valid = {
    output_field: 0
    for output_field in netcdf_files
}

header_written = False


# =============================================================================
# Process the H3 file in chunks
# =============================================================================

try:
    chunk_reader = pd.read_csv(
        h3_csv,
        usecols=[h3_field],
        dtype={h3_field: "string"},
        chunksize=chunk_size,
    )

except ValueError as error:
    available_columns = pd.read_csv(
        h3_csv,
        nrows=0,
    ).columns.tolist()

    raise KeyError(
        f"'{h3_field}' was not found in the H3 CSV.\n"
        f"Available columns: {available_columns}"
    ) from error


for chunk_number, chunk_df in enumerate(
    chunk_reader,
    start=1,
):

    chunk_start_time = time.perf_counter()

    chunk_df.columns = chunk_df.columns.str.strip()

    chunk_df = chunk_df.loc[
        chunk_df[h3_field].notna()
    ].copy()

    chunk_df[h3_field] = (
        chunk_df[h3_field]
        .astype("string")
        .str.strip()
    )

    chunk_df = chunk_df.loc[
        chunk_df[h3_field] != ""
    ].copy()

    number_of_rows = len(chunk_df)

    if number_of_rows == 0:
        print(
            f"Chunk {chunk_number:,} contained no valid H3 identifiers."
        )
        continue

    print("\n" + "=" * 79)

    print(
        f"Processing chunk {chunk_number:,}: "
        f"{number_of_rows:,} H3 cells"
    )

    latitudes, longitudes = h3_centres(
        chunk_df[h3_field].tolist()
    )

    for output_field, grid in netcdf_grids.items():

        sampled_values = sample_regular_grid(
            grid=grid,
            query_latitudes=latitudes,
            query_longitudes=longitudes,
        )

        if len(sampled_values) != number_of_rows:
            raise ValueError(
                f"Sampling '{output_field}' returned "
                f"{len(sampled_values):,} values for "
                f"{number_of_rows:,} H3 cells."
            )

        chunk_df[output_field] = sampled_values

        valid_mask = np.isfinite(sampled_values)
        valid_count = int(valid_mask.sum())

        total_valid[output_field] += valid_count

        if valid_count > 0:
            valid_values = sampled_values[valid_mask]

            print(
                f"{output_field}: "
                f"{valid_count:,} valid; "
                f"min={valid_values.min():.4f}; "
                f"max={valid_values.max():.4f}; "
                f"mean={valid_values.mean():.4f}"
            )

        else:
            print(
                f"{output_field}: no valid values"
            )

    output_fields = [
        h3_field,
        *netcdf_files.keys(),
    ]

    output_chunk = chunk_df[
        output_fields
    ].copy()

    output_chunk.to_csv(
        output_csv,
        mode="a",
        header=not header_written,
        index=False,
    )

    header_written = True
    total_rows += number_of_rows

    chunk_elapsed = (
        time.perf_counter()
        - chunk_start_time
    )

    total_elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Chunk completed in "
        f"{chunk_elapsed / 60:.2f} minutes"
    )

    print(
        f"Total rows written: {total_rows:,}; "
        f"elapsed time: {total_elapsed / 60:.2f} minutes"
    )


# =============================================================================
# Completion summary
# =============================================================================

elapsed_time = (
    time.perf_counter()
    - start_time
)

print("\n" + "=" * 79)

if total_rows == 0:
    print("No rows were written.")

else:
    print("Done")
    print(f"Output saved to: {output_csv}")
    print(f"Rows written: {total_rows:,}")

    print("\nValid output values")

    for output_field, valid_count in total_valid.items():
        print(
            f"  {output_field}: "
            f"{valid_count:,} of {total_rows:,}"
        )

print(
    f"\nTotal processing time: "
    f"{elapsed_time / 60:.2f} minutes"
)