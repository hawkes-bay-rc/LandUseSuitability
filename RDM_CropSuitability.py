# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:19:36 2026

@author: Ashton.Eaves
"""

# -*- coding: utf-8 -*-

###############################################################################
# Crop Suitability Classification
#
# Reads crop suitability criteria from Suitability_Ruleset_MACHINE.csv
# and applies them automatically to every crop in the ruleset.
#
# Created 2026
###############################################################################
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import numpy as np

# =============================================================================
# User settings
# =============================================================================

input_csv = (
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability"
    r"\3_Outputs\CropSuitability.csv"
)

rules_csv = (
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability"
    r"\1_Data\Suitability_Ruleset_MACHINE.csv"
)

output_folder = (
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability"
    r"\3_Outputs"
)

output_csv = os.path.join(
    output_folder,
    "CropSuitabilityClassified_ALL.csv"
)


# =============================================================================
# Read crop suitability input
# =============================================================================

print("Reading crop suitability input...")

df = pd.read_csv(
    input_csv,
    dtype={
        "GRID_ID": "string",
    },
)

df.columns = df.columns.str.strip()

print(f"Input rows: {len(df):,}")
print(f"Input columns: {len(df.columns):,}")


# =============================================================================
# Read machine-readable suitability rules
# =============================================================================

print("\nReading suitability ruleset...")

rules = pd.read_csv(
    rules_csv
)

rules.columns = rules.columns.str.strip()

print(f"Rule rows: {len(rules):,}")
print(f"Crops: {rules['Crop'].nunique():,}")

print("\nCrops found:")

for crop in rules["Crop"].dropna().unique():
    print(f"  - {crop}")


# =============================================================================
# Clean ruleset field types
# =============================================================================

# Numeric fields
for column in [
    "ClassScore",
    "Lower",
    "Upper",
]:

    if column in rules.columns:

        rules[column] = pd.to_numeric(
            rules[column],
            errors="coerce",
        )


# =============================================================================
# Convert CSV Boolean fields safely
# =============================================================================

def to_bool(value):
    """
    Convert common CSV Boolean representations to True/False.
    """

    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    return value in [
        "true",
        "1",
        "yes",
        "y",
    ]


for column in [
    "LowerInc",
    "UpperInc",
    "HardLimit",
]:

    if column in rules.columns:

        rules[column] = (
            rules[column]
            .apply(to_bool)
        )


# =============================================================================
# Helper function: normalise categorical text
# =============================================================================

def normalise_text(series):
    """
    Standardise categorical text before mapping.

    Handles:
        - leading/trailing whitespace
        - repeated spaces
        - non-breaking spaces
        - en/em dashes
        - upper/lower-case differences
    """

    return (
        series
        .astype("string")
        .str.replace("\xa0", " ", regex=False)
        .str.replace("–", "-", regex=False)
        .str.replace("—", "-", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


# =============================================================================
# Pre-processing
# =============================================================================


# =============================================================================
# DrainageClass -> DrainageScore
#
# 1 = Excessively / Well drained
# 2 = Moderately well drained
# 3 = Imperfectly drained
# 4 = Poorly drained
# 5 = Very poorly drained
# =============================================================================

drainage_map = {

    "excessively drained": 1,
    "well drained": 1,

    "moderately well drained": 2,

    "imperfectly drained": 3,
    "imperfect drained": 3,

    "poorly drained": 4,

    "very poorly drained": 5,
    "very-poorly drained": 5,
}


if "DrainageClass" in df.columns:

    print(
        "\nConverting DrainageClass "
        "to DrainageScore..."
    )

    drainage_normalised = normalise_text(
        df["DrainageClass"]
    )

    df["DrainageScore"] = (
        drainage_normalised
        .map(drainage_map)
        .astype("Float64")
    )

    print("\nDrainage conversion:")

    print(
        df[
            [
                "DrainageClass",
                "DrainageScore",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "DrainageScore"
        )
        .to_string(index=False)
    )

    unmapped_drainage = (
        df["DrainageClass"].notna()
        & df["DrainageScore"].isna()
    )

    print(
        "\nUnmapped drainage records:",
        f"{unmapped_drainage.sum():,}"
    )

    if unmapped_drainage.any():

        print(
            "\nUnmapped DrainageClass values:"
        )

        print(
            df.loc[
                unmapped_drainage,
                "DrainageClass",
            ]
            .value_counts()
        )

else:

    print(
        "\nWARNING: DrainageClass "
        "was not found."
    )

    df["DrainageScore"] = pd.NA


# =============================================================================
# RootDepthRange -> RootDepth_cm
#
# RootDepthRange contains categories such as:
#
#   > 1 m
#   60 - 100 cm
#   45 - 50 cm
#   10 - 15 cm
#
# For bounded ranges, the midpoint is used as the representative depth.
#
# Examples:
#
#   60 - 100 cm -> 80 cm
#   45 - 50 cm  -> 47.5 cm
#
# > 1 m is represented as 101 cm.
#
# NOTE:
# This introduces a midpoint assumption because the original source
# provides a depth range rather than an exact root depth.
# =============================================================================

def root_depth_to_cm(value):
    """
    Convert RootDepthRange text to a representative depth in cm.
    """

    if pd.isna(value):
        return np.nan

    value = str(value)

    # Normalise text
    value = (
        value
        .replace("\xa0", " ")
        .replace("–", "-")
        .replace("—", "-")
        .strip()
        .lower()
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    # -------------------------------------------------------------------------
    # Greater-than values in metres
    #
    # > 1 m
    # >1 m
    # -------------------------------------------------------------------------

    match = re.match(
        r"^>\s*([0-9.]+)\s*m$",
        value,
    )

    if match:

        metres = float(
            match.group(1)
        )

        return (
            metres * 100
            + 1
        )


    # -------------------------------------------------------------------------
    # Greater-than values in cm
    #
    # >100 cm
    # > 100 cm
    # -------------------------------------------------------------------------

    match = re.match(
        r"^>\s*([0-9.]+)\s*cm$",
        value,
    )

    if match:

        return (
            float(match.group(1))
            + 1
        )


    # -------------------------------------------------------------------------
    # Less-than values
    #
    # <30 cm
    # < 30 cm
    #
    # Use a value just below the threshold.
    # -------------------------------------------------------------------------

    match = re.match(
        r"^<\s*([0-9.]+)\s*cm$",
        value,
    )

    if match:

        upper = float(
            match.group(1)
        )

        return max(
            upper - 1,
            0,
        )


    # -------------------------------------------------------------------------
    # Range values
    #
    # 60 - 100 cm
    # 45 - 50 cm
    # 10 - 15 cm
    #
    # Use midpoint.
    # -------------------------------------------------------------------------

    match = re.match(
        r"^([0-9.]+)\s*-\s*([0-9.]+)\s*cm$",
        value,
    )

    if match:

        lower = float(
            match.group(1)
        )

        upper = float(
            match.group(2)
        )

        return (
            lower + upper
        ) / 2


    # -------------------------------------------------------------------------
    # Alternative wording:
    #
    # 60 to 100 cm
    # -------------------------------------------------------------------------

    match = re.match(
        r"^([0-9.]+)\s+to\s+([0-9.]+)\s*cm$",
        value,
    )

    if match:

        lower = float(
            match.group(1)
        )

        upper = float(
            match.group(2)
        )

        return (
            lower + upper
        ) / 2


    # Could not interpret value
    return np.nan


if "RootDepthRange" in df.columns:

    print(
        "\nConverting RootDepthRange "
        "to RootDepth_cm..."
    )

    df["RootDepth_cm"] = (
        df["RootDepthRange"]
        .apply(root_depth_to_cm)
        .astype("Float64")
    )

    print("\nRoot depth conversion:")

    print(
        df[
            [
                "RootDepthRange",
                "RootDepth_cm",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "RootDepth_cm"
        )
        .to_string(index=False)
    )

    unmapped_root_depth = (
        df["RootDepthRange"].notna()
        & df["RootDepth_cm"].isna()
    )

    print(
        "\nUnmapped root-depth records:",
        f"{unmapped_root_depth.sum():,}"
    )

    if unmapped_root_depth.any():

        print(
            "\nUnmapped RootDepthRange values:"
        )

        print(
            df.loc[
                unmapped_root_depth,
                "RootDepthRange",
            ]
            .value_counts()
        )

else:

    print(
        "\nWARNING: RootDepthRange "
        "was not found."
    )

    df["RootDepth_cm"] = pd.NA


# =============================================================================
# Rule names for output
# =============================================================================

rule_names = {

    "ANR": "Rainfall requirement",
    "ART": "Rainfall excess",

    "SLP": "Slope",

    "PRD": "Root depth",

    "DRC": "Drainage",

    "PWC": "Plant available water",

    "STN": "Topsoil stones",

    "FFB": "SON frost days",
    "FFH": "MAM frost days",

    "GDD": "Growing Degree Days",

    "ECS": "Salinity",

    "HFL": "Heat stress",

    "HFO": "Harvest heat",

    "MET": "Mean temperature",

    "MNT": "Minimum temperature",

    "MXT": "Maximum temperature",

    "PHH": "Soil pH",

    "RAH": "Harvest rainfall",
}


# =============================================================================
# Convert mean score to final suitability class
# =============================================================================

def mean_score_to_class(score):
    """
    Convert mean criterion score into overall suitability class.
    """

    if pd.isna(score):

        return pd.NA

    elif score < 1.5:

        return "Well Suited"

    elif score < 2.0:

        return "Suited"

    elif score < 2.5:

        return "Moderately Suited"

    else:

        return "Unsuitable"


# =============================================================================
# Class score lookup
# =============================================================================

score_class_map = {

    1: "Well Suited",

    2: "Suited",

    3: "Moderately Suited",

    4: "Unsuitable",
}


# =============================================================================
# Process every crop in machine ruleset
# =============================================================================

crops = (
    rules["Crop"]
    .dropna()
    .drop_duplicates()
    .tolist()
)


for crop in crops:

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Processing crop: {crop}"
    )

    print(
        "=" * 70
    )


    # =========================================================================
    # Extract rules for current crop
    # =========================================================================

    crop_rules = (
        rules.loc[
            rules["Crop"] == crop
        ]
        .copy()
    )


    rule_ids = (
        crop_rules["Rule_id"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )


    score_cols = []


    # =========================================================================
    # Apply each suitability criterion
    # =========================================================================

    for rule_id in rule_ids:

        criterion_rules = (
            crop_rules.loc[
                crop_rules["Rule_id"]
                == rule_id
            ]
            .copy()
        )


        # ---------------------------------------------------------------------
        # Get input field
        # ---------------------------------------------------------------------

        input_fields = (
            criterion_rules[
                "InputField"
            ]
            .dropna()
            .unique()
        )


        if len(input_fields) == 0:

            print(
                f"WARNING: No InputField "
                f"for {crop} {rule_id}"
            )

            continue


        input_field = (
            input_fields[0]
        )


        output_score_col = (
            f"{crop}_{rule_id}_score"
        )


        # ---------------------------------------------------------------------
        # Missing input field
        # ---------------------------------------------------------------------

        if input_field not in df.columns:

            print(
                f"WARNING: Missing input field: "
                f"{input_field} "
                f"for {crop} {rule_id}"
            )

            df[output_score_col] = (
                pd.Series(
                    pd.NA,
                    index=df.index,
                    dtype="Float64",
                )
            )

            score_cols.append(
                output_score_col
            )

            continue


        # ---------------------------------------------------------------------
        # Convert source variable to numeric
        # ---------------------------------------------------------------------

        numeric_series = (
            pd.to_numeric(
                df[input_field],
                errors="coerce",
            )
        )


        # ---------------------------------------------------------------------
        # Empty criterion score
        # ---------------------------------------------------------------------

        criterion_score = (
            pd.Series(
                pd.NA,
                index=df.index,
                dtype="Float64",
            )
        )


        # =====================================================================
        # Apply every rule row belonging to criterion
        # =====================================================================

        for _, rule_row in (
            criterion_rules.iterrows()
        ):

            class_score = (
                rule_row["ClassScore"]
            )

            lower = (
                rule_row["Lower"]
            )

            upper = (
                rule_row["Upper"]
            )

            lower_inc = (
                rule_row["LowerInc"]
            )

            upper_inc = (
                rule_row["UpperInc"]
            )


            # -----------------------------------------------------------------
            # Exact-value rule
            #
            # Examples:
            #
            # Salinity == 0
            # Salinity == 1
            #
            # DrainageScore == 1
            # DrainageScore == 5
            # -----------------------------------------------------------------

            if (
                pd.notna(lower)
                and pd.notna(upper)
                and float(lower)
                == float(upper)
            ):

                mask = (
                    numeric_series
                    == float(lower)
                )


            # -----------------------------------------------------------------
            # Range / threshold rule
            # -----------------------------------------------------------------

            else:

                mask = (
                    numeric_series
                    .notna()
                    .copy()
                )


                # -------------------------------------------------------------
                # Lower bound
                # -------------------------------------------------------------

                if pd.notna(lower):

                    lower_value = float(
                        lower
                    )

                    if lower_inc:

                        mask &= (
                            numeric_series
                            >= lower_value
                        )

                    else:

                        mask &= (
                            numeric_series
                            > lower_value
                        )


                # -------------------------------------------------------------
                # Upper bound
                # -------------------------------------------------------------

                if pd.notna(upper):

                    upper_value = float(
                        upper
                    )

                    if upper_inc:

                        mask &= (
                            numeric_series
                            <= upper_value
                        )

                    else:

                        mask &= (
                            numeric_series
                            < upper_value
                        )


            # -----------------------------------------------------------------
            # Remove NA from Boolean mask
            # -----------------------------------------------------------------

            mask = (
                mask
                .fillna(False)
            )


            # -----------------------------------------------------------------
            # Assign class score
            # -----------------------------------------------------------------

            criterion_score.loc[
                mask
            ] = class_score


        # =====================================================================
        # IMPORTANT:
        #
        # Write completed criterion back to main dataframe.
        # =====================================================================

        df[output_score_col] = (
            criterion_score
        )


        score_cols.append(
            output_score_col
        )


        print(
            f"  {rule_id:<5} "
            f"{input_field:<30} "
            f"{df[output_score_col].notna().sum():,} "
            f"classified"
        )


    # =========================================================================
    # Skip crop if there are no usable criteria
    # =========================================================================

    if not score_cols:

        print(
            f"WARNING: No usable "
            f"criteria for {crop}"
        )

        continue


    # =========================================================================
    # Mean suitability score
    # =========================================================================

    df[
        f"{crop}_MeanScore"
    ] = (
        df[score_cols]
        .mean(
            axis=1,
            skipna=True,
        )
        .round(2)
    )


    # =========================================================================
    # Initial final class
    # =========================================================================

    df[
        f"{crop}_FinalClass"
    ] = (
        df[
            f"{crop}_MeanScore"
        ]
        .apply(
            mean_score_to_class
        )
    )


    # =========================================================================
    # Worst individual criterion
    # =========================================================================

    df[
        f"{crop}_WorstScore"
    ] = (
        df[score_cols]
        .max(
            axis=1,
            skipna=True,
        )
    )


    # =========================================================================
    # Normal limiting factor
    #
    # Retain all criteria tied for the worst score.
    # =========================================================================

    limiting_factor = (
        pd.Series(
            pd.NA,
            index=df.index,
            dtype="object",
        )
    )


    for rule_id in rule_ids:

        score_col = (
            f"{crop}_{rule_id}_score"
        )

        if score_col not in df.columns:
            continue


        mask = (

            df[score_col].notna()

            & (

                df[score_col]

                == df[
                    f"{crop}_WorstScore"
                ]

            )

        )


        factor_name = (
            rule_names.get(
                rule_id,
                rule_id,
            )
        )


        existing = (
            limiting_factor.loc[
                mask
            ]
        )


        limiting_factor.loc[
            mask
        ] = (

            existing
            .fillna("")
            .apply(

                lambda value:

                (
                    f"{value}; {factor_name}"
                    if value
                    else factor_name
                )

            )

        )


    df[
        f"{crop}_LimitingFactor"
    ] = limiting_factor


    # =========================================================================
    # Limiting class
    # =========================================================================

    df[
        f"{crop}_LimitingClass"
    ] = (

        df[
            f"{crop}_WorstScore"
        ]

        .map(
            score_class_map
        )

    )


    # =========================================================================
    # Hard limiting factors
    #
    # HardLimit is read directly from machine ruleset.
    # Only ClassScore 4 triggers hard exclusion.
    # =========================================================================

    hard_rule_ids = (

        crop_rules.loc[

            crop_rules[
                "HardLimit"
            ] == True,

            "Rule_id",

        ]

        .dropna()

        .drop_duplicates()

        .tolist()

    )


    hard_excluded = (
        pd.Series(
            False,
            index=df.index,
        )
    )


    hard_limiting_factor = (
        pd.Series(
            pd.NA,
            index=df.index,
            dtype="object",
        )
    )


    for rule_id in hard_rule_ids:

        score_col = (
            f"{crop}_{rule_id}_score"
        )

        if score_col not in df.columns:
            continue


        # Score 4 = Unsuitable
        mask = (

            df[score_col]
            .eq(4)
            .fillna(False)

        )


        hard_excluded |= mask


        factor_name = (
            rule_names.get(
                rule_id,
                rule_id,
            )
        )


        existing = (

            hard_limiting_factor.loc[
                mask
            ]

        )


        hard_limiting_factor.loc[
            mask
        ] = (

            existing
            .fillna("")
            .apply(

                lambda value:

                (
                    f"{value}; {factor_name}"
                    if value
                    else factor_name
                )

            )

        )


    df[
        f"{crop}_HardExcluded"
    ] = hard_excluded


    df[
        f"{crop}_HardLimitingFactor"
    ] = hard_limiting_factor


    # -------------------------------------------------------------------------
    # Hard exclusion overrides mean suitability
    # -------------------------------------------------------------------------

    df.loc[

        hard_excluded,

        f"{crop}_FinalClass",

    ] = "Unsuitable"


    # =========================================================================
    # Required source fields for this crop
    # =========================================================================

    required_columns = (

        crop_rules[
            "InputField"
        ]

        .dropna()

        .drop_duplicates()

        .tolist()

    )


    existing_required_columns = [

        column

        for column
        in required_columns

        if column
        in df.columns

    ]


    absent_columns = [

        column

        for column
        in required_columns

        if column
        not in df.columns

    ]


    # =========================================================================
    # No SMU = Indeterminate
    # =========================================================================

    if "SMU" in df.columns:

        no_soil_mask = (
            df["SMU"].isna()
        )

    else:

        no_soil_mask = (
            pd.Series(
                False,
                index=df.index,
            )
        )


    # =========================================================================
    # Missing required data
    # =========================================================================

    if existing_required_columns:

        missing_data_mask = (

            df[
                existing_required_columns
            ]
            .isna()
            .any(axis=1)

            & ~no_soil_mask

        )

    else:

        missing_data_mask = (
            pd.Series(
                False,
                index=df.index,
            )
        )


    # -------------------------------------------------------------------------
    # If the entire required column is absent,
    # all mapped records are Missing data.
    # -------------------------------------------------------------------------

    if absent_columns:

        missing_data_mask |= (
            ~no_soil_mask
        )

        print(
            f"\nWARNING: {crop} "
            f"has missing source columns:"
        )

        for column in absent_columns:

            print(
                f"  - {column}"
            )


    # =========================================================================
    # Record which required fields are missing
    # =========================================================================

    missing_field_output = (
        f"{crop}_MissingData"
    )


    df[
        missing_field_output
    ] = pd.NA


    # -------------------------------------------------------------------------
    # Existing columns containing NA
    # -------------------------------------------------------------------------

    for column in existing_required_columns:

        column_missing_mask = (

            df[column].isna()

            & ~no_soil_mask

        )


        existing = (

            df.loc[
                column_missing_mask,
                missing_field_output,
            ]

        )


        df.loc[
            column_missing_mask,
            missing_field_output,
        ] = (

            existing
            .fillna("")
            .apply(

                lambda value:

                (
                    f"{value}; {column}"
                    if value
                    else column
                )

            )

        )


    # -------------------------------------------------------------------------
    # Entirely absent source columns
    # -------------------------------------------------------------------------

    for column in absent_columns:

        mask = (
            ~no_soil_mask
        )

        existing = (

            df.loc[
                mask,
                missing_field_output,
            ]

        )

        df.loc[
            mask,
            missing_field_output,
        ] = (

            existing
            .fillna("")
            .apply(

                lambda value:

                (
                    f"{value}; {column}"
                    if value
                    else column
                )

            )

        )


    # =========================================================================
    # Apply Missing data class
    # =========================================================================

    df.loc[

        missing_data_mask,

        f"{crop}_FinalClass",

    ] = "Missing data"


    df.loc[

        missing_data_mask,

        f"{crop}_HardExcluded",

    ] = False


    df.loc[

        missing_data_mask,

        f"{crop}_HardLimitingFactor",

    ] = pd.NA


    # =========================================================================
    # Apply Indeterminate LAST
    #
    # No SMU means outside the soil assessment domain.
    # =========================================================================

    df.loc[

        no_soil_mask,

        f"{crop}_FinalClass",

    ] = "Indeterminate"


    df.loc[

        no_soil_mask,

        f"{crop}_HardExcluded",

    ] = False


    df.loc[

        no_soil_mask,

        f"{crop}_HardLimitingFactor",

    ] = "Indeterminate"


    # =========================================================================
    # Crop summary
    # =========================================================================

    print(
        f"\n{crop} final classes:"
    )

    print(

        df[
            f"{crop}_FinalClass"
        ]

        .value_counts(
            dropna=False
        )

    )


# =============================================================================
# Export
# =============================================================================

print(
    "\n"
    + "=" * 70
)

print(
    "Exporting"
)

print(
    "=" * 70
)


df.to_csv(
    output_csv,
    index=False,
)


print(
    "\nDone"
)

print(
    f"Output saved to:\n"
    f"{output_csv}"
)


###############################################################################
# End
###############################################################################

###############################################################################
# End
###############################################################################