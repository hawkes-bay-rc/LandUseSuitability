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

import os
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
# Read input data
# =============================================================================

print("Reading crop suitability input...")

df = pd.read_csv(
    input_csv,
    dtype={"GRID_ID": "string"},
)

df.columns = df.columns.str.strip()

print(f"Input rows: {len(df):,}")
print(f"Input columns: {len(df.columns):,}")


# =============================================================================
# Read machine-readable ruleset
# =============================================================================

print("\nReading suitability ruleset...")

rules = pd.read_csv(rules_csv)

rules.columns = rules.columns.str.strip()

print(f"Rule rows: {len(rules):,}")
print(f"Crops: {rules['Crop'].nunique()}")

print("\nCrops found:")
for crop in rules["Crop"].dropna().unique():
    print(f"  - {crop}")


# =============================================================================
# Clean ruleset field types
# =============================================================================

# Numeric fields
numeric_rule_fields = [
    "ClassScore",
    "Lower",
    "Upper",
]

for col in numeric_rule_fields:
    if col in rules.columns:
        rules[col] = pd.to_numeric(
            rules[col],
            errors="coerce",
        )


# ---------------------------------------------------------------------------
# Convert True/False fields safely
# ---------------------------------------------------------------------------

def to_bool(value):
    """
    Convert common CSV representations of True/False
    to Python Boolean values.
    """

    if pd.isna(value):
        return False

    return str(value).strip().lower() in [
        "true",
        "1",
        "yes",
        "y",
    ]


for col in [
    "LowerInc",
    "UpperInc",
    "HardLimit",
]:
    if col in rules.columns:
        rules[col] = rules[col].apply(to_bool)


# =============================================================================
# Convert categorical source variables to numeric variables
# =============================================================================

# ---------------------------------------------------------------------------
# Drainage class
#
# Lower scores represent better drainage conditions.
#
# This converts the categorical S-map drainage description into a simple
# ordered numeric variable that can be used by the machine ruleset.
# ---------------------------------------------------------------------------

drainage_map = {

    "excessively drained": 1,
    "well drained": 2,
    "moderately well drained": 3,

    # Handle both spellings found previously
    "imperfectly drained": 4,
    "imperfect drained": 4,

    "poorly drained": 5,

    # Handle both spellings
    "very poorly drained": 6,
    "very-poorly drained": 6,
}


def normalise_text(series):
    """
    Normalise categorical text before lookup.
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


if "DrainageClass" in df.columns:

    print("\nConverting DrainageClass to DrainageScore...")

    drainage_normalised = normalise_text(
        df["DrainageClass"]
    )

    df["DrainageScore"] = (
        drainage_normalised
        .map(drainage_map)
        .astype("Float64")
    )

    print(
        df[
            ["DrainageClass", "DrainageScore"]
        ]
        .drop_duplicates()
        .sort_values("DrainageScore")
        .to_string(index=False)
    )

else:

    print(
        "\nWARNING: DrainageClass does not exist "
        "in the input dataset."
    )

    df["DrainageScore"] = pd.NA


# =============================================================================
# Rule names
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
    "MNT": "Minimum temperature",
    "MXT": "Maximum temperature",
    "PHH": "Soil pH",
}


# =============================================================================
# Mean score to final class
# =============================================================================

def mean_score_to_class(score):
    """
    Convert mean suitability score into final suitability class.

    Uses the thresholds developed in the previous suitability model.
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
# Apply one rule range
# =============================================================================

def build_range_mask(
    series,
    lower,
    upper,
    lower_inc,
    upper_inc,
):
    """
    Build a Boolean mask for one row of the machine ruleset.

    Supports:
        lower + upper
        lower only
        upper only

    and honours inclusive/exclusive boundaries.
    """

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    mask = numeric_series.notna()

    # -------------------------------------------------------------------------
    # Lower boundary
    # -------------------------------------------------------------------------

    if pd.notna(lower):

        if lower_inc:
            mask &= numeric_series >= lower

        else:
            mask &= numeric_series > lower

    # -------------------------------------------------------------------------
    # Upper boundary
    # -------------------------------------------------------------------------

    if pd.notna(upper):

        if upper_inc:
            mask &= numeric_series <= upper

        else:
            mask &= numeric_series < upper

    return mask


# =============================================================================
# Process each crop
# =============================================================================

crops = (
    rules["Crop"]
    .dropna()
    .drop_duplicates()
    .tolist()
)


for crop in crops:

    print("\n" + "=" * 70)
    print(f"Processing crop: {crop}")
    print("=" * 70)

    crop_rules = rules[
        rules["Crop"] == crop
    ].copy()

    rule_ids = (
        crop_rules["Rule_id"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    score_cols = []

    # =========================================================================
    # Apply each criterion
    # =========================================================================

    for rule_id in rule_ids:

        criterion_rules = crop_rules[
            crop_rules["Rule_id"] == rule_id
        ].copy()

        # All rows belonging to a Rule_id should use the same input field
        input_fields = (
            criterion_rules["InputField"]
            .dropna()
            .unique()
        )

        if len(input_fields) == 0:

            print(
                f"WARNING: No InputField for "
                f"{crop} {rule_id}"
            )

            continue

        input_field = input_fields[0]

        output_score_col = (
            f"{crop}_{rule_id}_score"
        )

        # ---------------------------------------------------------------------
        # Check source field
        # ---------------------------------------------------------------------

        if input_field not in df.columns:

            print(
                f"WARNING: Missing input field: "
                f"{input_field} "
                f"for {crop} {rule_id}"
            )

            df[output_score_col] = pd.NA

            score_cols.append(
                output_score_col
            )

            continue

        # ---------------------------------------------------------------------
        # Create empty score field
        # ---------------------------------------------------------------------

        criterion_score = pd.Series(
            pd.NA,
            index=df.index,
            dtype="Float64",
        )

        # ---------------------------------------------------------------------
        # Apply every range belonging to this criterion
        #
        # This naturally handles split ranges such as citrus pH:
        #
        # Score 2 = 5.0–5.5 OR 6.5–6.8
        # ---------------------------------------------------------------------

        for _, rule_row in criterion_rules.iterrows():

            class_score = rule_row["ClassScore"]

            lower = rule_row["Lower"]
            upper = rule_row["Upper"]

            lower_inc = rule_row["LowerInc"]
            upper_inc = rule_row["UpperInc"]

            mask = build_range_mask(
                df[input_field],
                lower,
                upper,
                lower_inc,
                upper_inc,
            )

            criterion_score.loc[mask] = (
                class_score
            )

        df[output_score_col] = (
            criterion_score
        )

        score_cols.append(
            output_score_col
        )

        print(
            f"  {rule_id:<5} "
            f"{input_field:<30} "
            f"{df[output_score_col].notna().sum():,} classified"
        )


    # =========================================================================
    # Mean suitability score
    # =========================================================================

    if not score_cols:

        print(
            f"WARNING: No usable criteria for {crop}"
        )

        continue

    df[f"{crop}_MeanScore"] = (
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

    df[f"{crop}_FinalClass"] = (
        df[f"{crop}_MeanScore"]
        .apply(mean_score_to_class)
    )


    # =========================================================================
    # Normal limiting factor
    #
    # Highest individual criterion score.
    # Multiple tied criteria are retained.
    # =========================================================================

    df[f"{crop}_WorstScore"] = (
        df[score_cols]
        .max(
            axis=1,
            skipna=True,
        )
    )

    limiting_factor = pd.Series(
        pd.NA,
        index=df.index,
        dtype="object",
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
                == df[f"{crop}_WorstScore"]
            )
        )

        factor_name = rule_names.get(
            rule_id,
            rule_id,
        )

        existing = limiting_factor.loc[mask]

        limiting_factor.loc[mask] = (
            existing
            .fillna("")
            .apply(
                lambda x:
                f"{x}; {factor_name}"
                if x
                else factor_name
            )
        )

    df[f"{crop}_LimitingFactor"] = (
        limiting_factor
    )


    # =========================================================================
    # Limiting class
    # =========================================================================

    score_class_map = {
        1: "Well Suited",
        2: "Suited",
        3: "Moderately Suited",
        4: "Unsuitable",
    }

    df[f"{crop}_LimitingClass"] = (
        df[f"{crop}_WorstScore"]
        .map(score_class_map)
    )


    # =========================================================================
    # Hard limiting factors
    #
    # Read directly from HardLimit in the rules CSV.
    #
    # A hard-limit criterion only overrides the final result when its
    # criterion score = 4 (Unsuitable).
    # =========================================================================

    hard_rule_ids = (
        crop_rules.loc[
            crop_rules["HardLimit"] == True,
            "Rule_id",
        ]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    hard_excluded = pd.Series(
        False,
        index=df.index,
    )

    hard_limiting_factor = pd.Series(
        pd.NA,
        index=df.index,
        dtype="object",
    )

    for rule_id in hard_rule_ids:

        score_col = (
            f"{crop}_{rule_id}_score"
        )

        if score_col not in df.columns:
            continue

        # Score 4 = Unsuitable
        mask = (
            df[score_col] == 4
        ).fillna(False)

        hard_excluded |= mask

        factor_name = rule_names.get(
            rule_id,
            rule_id,
        )

        existing = (
            hard_limiting_factor.loc[mask]
        )

        hard_limiting_factor.loc[mask] = (
            existing
            .fillna("")
            .apply(
                lambda x:
                f"{x}; {factor_name}"
                if x
                else factor_name
            )
        )

    df[f"{crop}_HardExcluded"] = (
        hard_excluded
    )

    df[f"{crop}_HardLimitingFactor"] = (
        hard_limiting_factor
    )

    # Hard limit overrides mean score
    df.loc[
        hard_excluded,
        f"{crop}_FinalClass",
    ] = "Unsuitable"


    # =========================================================================
    # Missing data
    #
    # Determine required source fields directly from ruleset.
    # =========================================================================

    required_columns = (
        crop_rules["InputField"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    # Only fields that actually exist can be checked row-by-row
    existing_required_columns = [
        col
        for col in required_columns
        if col in df.columns
    ]


    # =========================================================================
    # No SMU = Indeterminate
    #
    # Outside soil mapping / assessment domain.
    # =========================================================================

    if "SMU" in df.columns:

        no_soil_mask = (
            df["SMU"].isna()
        )

    else:

        no_soil_mask = pd.Series(
            False,
            index=df.index,
        )


    # =========================================================================
    # Missing crop input data
    # =========================================================================

    if existing_required_columns:

        missing_data_mask = (
            df[existing_required_columns]
            .isna()
            .any(axis=1)
            & ~no_soil_mask
        )

    else:

        missing_data_mask = pd.Series(
            False,
            index=df.index,
        )


    # -------------------------------------------------------------------------
    # Also detect required columns completely absent from input table
    # -------------------------------------------------------------------------

    absent_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if absent_columns:

        print(
            f"\nWARNING: {crop} has missing source columns:"
        )

        for col in absent_columns:
            print(f"  - {col}")


    # =========================================================================
    # Record missing fields
    # =========================================================================

    missing_field_output = (
        f"{crop}_MissingData"
    )

    df[missing_field_output] = pd.NA

    for column in existing_required_columns:

        mask = (
            df[column].isna()
            & ~no_soil_mask
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
                lambda x:
                f"{x}; {column}"
                if x
                else column
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
    # Apply Indeterminate class LAST
    #
    # This guarantees no-SMU areas always remain Indeterminate.
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
    # Summary
    # =========================================================================

    print(f"\n{crop} final classes:")

    print(
        df[f"{crop}_FinalClass"]
        .value_counts(
            dropna=False
        )
    )


# =============================================================================
# Export
# =============================================================================

print("\n" + "=" * 70)
print("Exporting")
print("=" * 70)

df.to_csv(
    output_csv,
    index=False,
)

print("\nDone")
print(f"Output saved to:\n{output_csv}")

###############################################################################
# End
###############################################################################