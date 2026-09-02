# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:54:37 2026

@author: Ashton.Eaves
"""
# In anaconda prompt open env: conda activate h3raster2
# Then run spyder

### Conditional Statements for Crop Suitability ###############################

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

csv_path = (
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability"
    r"\3_Outputs\CropSuitability.csv"
)

out_path = (
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability"
    r"\3_Outputs\CropSuitabilityClassified_Onions.csv"
)

df = pd.read_csv(
    csv_path,
    dtype={"GRID_ID": "string"},
)

df.columns = df.columns.str.strip()

# ---------------------------------------------------------------------------
# Generic classifiers
# ---------------------------------------------------------------------------

def classify_numeric(series, rules):
    """
    Classify a numeric pandas Series using operator-based rules.

    Supported formats:

        ("<", 100)
        ("<=", 100)
        (">", 100)
        (">=", 100)
        ("==", 0)
        ("!=", 0)

        ("between", 100, 200)

    or:

        ("between", (100, 200))

    Multiple alternative conditions:

        (
            "or",
            [
                ("between", (5.0, 5.5)),
                ("between", (6.5, 6.8)),
            ],
        )
    """

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="object",
    )

    # -----------------------------------------------------------------------
    # Helper function to evaluate one numeric condition
    # -----------------------------------------------------------------------

    def evaluate_condition(condition):

        if not isinstance(condition, (tuple, list)) or len(condition) < 2:
            raise ValueError(
                f"Invalid numeric condition: {condition}"
            )

        operator = condition[0]

        if operator == ">":

            threshold = condition[1]
            return numeric_series > threshold

        elif operator == ">=":

            threshold = condition[1]
            return numeric_series >= threshold

        elif operator == "<":

            threshold = condition[1]
            return numeric_series < threshold

        elif operator == "<=":

            threshold = condition[1]
            return numeric_series <= threshold

        elif operator == "==":

            threshold = condition[1]
            return numeric_series == threshold

        elif operator == "!=":

            threshold = condition[1]
            return numeric_series != threshold

        elif operator == "between":

            # Supports:
            # ("between", lower, upper)
            if len(condition) == 3:

                lower = condition[1]
                upper = condition[2]

            # Supports:
            # ("between", (lower, upper))
            elif (
                len(condition) == 2
                and isinstance(condition[1], (tuple, list))
                and len(condition[1]) == 2
            ):

                lower, upper = condition[1]

            else:

                raise ValueError(
                    f"Invalid 'between' condition: {condition}"
                )

            return numeric_series.between(
                lower,
                upper,
                inclusive="both",
            )

        else:

            raise ValueError(
                f"Unsupported numeric operator "
                f"'{operator}'. "
                f"Full condition: {condition}"
            )

    # -----------------------------------------------------------------------
    # Apply each suitability rule
    # -----------------------------------------------------------------------

    for suitability_class, rule in rules.items():

        if not isinstance(rule, (tuple, list)) or len(rule) < 2:

            raise ValueError(
                f"Invalid numeric rule for "
                f"'{suitability_class}': {rule}"
            )

        operator = rule[0]

        # -------------------------------------------------------------------
        # OR conditions
        # -------------------------------------------------------------------

        if operator == "or":

            conditions = rule[1]

            if not isinstance(conditions, (tuple, list)):
                raise ValueError(
                    f"'or' rule for '{suitability_class}' "
                    f"must contain a list of conditions."
                )

            # Start with all False
            mask = pd.Series(
                False,
                index=series.index,
            )

            # A row matches if ANY condition matches
            for condition in conditions:

                mask = mask | evaluate_condition(condition)

        # -------------------------------------------------------------------
        # Normal single condition
        # -------------------------------------------------------------------

        else:

            mask = evaluate_condition(rule)

        # Do not classify missing source values
        mask = (
            mask
            & numeric_series.notna()
        )

        result.loc[mask] = suitability_class

    return result


def classify_categorical(series, rules):
    """
    Classify a categorical pandas Series.

    Matching is case-insensitive and ignores:
    - leading/trailing whitespace
    - repeated spaces
    - non-breaking spaces
    - en dashes and em dashes
    """

    normalised_series = (
        series
        .astype("string")
        .str.replace("\xa0", " ", regex=False)
        .str.replace("–", "-", regex=False)
        .str.replace("—", "-", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="object",
    )

    for suitability_class, accepted_values in rules.items():

        normalised_values = []

        for value in accepted_values:

            if pd.isna(value):
                continue

            normalised_value = (
                str(value)
                .replace("\xa0", " ")
                .replace("–", "-")
                .replace("—", "-")
                .strip()
                .lower()
            )

            normalised_value = " ".join(
                normalised_value.split()
            )

            normalised_values.append(normalised_value)

        mask = normalised_series.isin(normalised_values)

        result.loc[mask] = suitability_class

    return result
# ---------------------------------------------------------------------------
# Rule names for output
# ---------------------------------------------------------------------------

rule_names = {
    "ANR": "Rainfall requirement",
    "ART": "Rainfall excess",
    "DRC": "Drainage",
    "ECS": "Salinity",
    "FFB": "SON frost days",
    "HFO": "Harvest heat",
    "PHH": "pH",
    "PRD": "Root depth",        
    "PWC": "Plant available water",
    "RAH": "Harvest rainfall",
    "SLP": "Slope",    
    "STN": "Topsoil stones"
}
# ---------------------------------------------------------------------------
# Crop rule dictionary
# ---------------------------------------------------------------------------

crop_rules = {
    "Onions": {
        "ANR": {
            "column": "MeanAnnualRainfall",
            "type": "numeric",
            "rules": {
                "Well Suited": (">", 800),
                "Suited": ("between", (700, 800)),
                "Moderately Suited": ("between", (600, 700)),
                "Unsuitable": ("<", 600),
            },
        },
        "ART": {
            "column": "MeanAnnualRainfall",
            "type": "numeric",
            "rules": {
                "Well Suited": ("<", 1500),
                "Suited": ("between", (1500, 1800)),
                "Moderately Suited": ("between", (1800, 2000)),
                "Unsuitable": (">", 2000),
            },
        },
        "DRC": {
            "column": "DrainageClass",
            "type": "categorical",
            "rules": {
                "Well Suited": [
                    "Excessively drained",
                    "Well drained",
                    "Moderately well drained",
                ],
                "Suited": [
                    "Imperfectly drained",
                    "Imperfect drained",
                ],
                "Moderately Suited": [
                    "Poorly drained"
                ],
                "Unsuitable": [
                    "Very poorly drained",
                    "Very-poorly drained",
                ],
            },
        },
        "ECS": {
            "column": "Salinity",
            "type": "numeric",
            "rules": {
                "Well Suited": ("==", 0),
                "Unsuitable": ("==", 1),
            },
        },
        "FFB": {
             "column": "FrostDays_SON",
             "type": "numeric",
             "rules": {
                 "Well Suited": ("<", 2),
                 "Suited": ("between", (2, 3)),
                 "Moderately Suited": ("between", (3, 4)),
                 "Unsuitable": (">", 4),
             },
         },
        "HFO": {
            "column": "MaxTemp_DJF",
            "type": "numeric",
            "rules": {
                "Well Suited": ("<", 3),
                "Suited": ("between", (3, 6)),
                "Moderately Suited": ("between", (6, 9)),
                "Unsuitable": (">", 9),
            },
        },
        "PHH": {
            "column": "PH_MID",
            "type": "numeric",
            "rules": {
                "Well Suited": (">", 6.0),
                "Moderately Suited": ("between", (5.8, 6.0)),
                "Unsuitable": ("<", 5.8),
            },
        },
        "PRD": {
            "column": "RootDepthRange",
            "type": "categorical",
            "rules": {
                "Well Suited": [">60 cm"],
                "Suited": ["45 - 60 cm", "45 to 60 cm"],
                "Moderately Suited": ["20 - 45 cm", "20 to 45 cm"],
                "Unsuitable": ["< 20 cm", "<20 cm"],
            },
        },        
        "PWC": {
            "column": "PAWmm",
            "type": "numeric",
            "rules": {
                "Well Suited": (">", 150),
                "Suited": ("between", (90, 150)),
                "Moderately Suited": ("between", (60, 90)),
                "Unsuitable": ("<", 60),
            },
        },
        "RAH": {
            "column": "HarvestRainfall_DJF",
            "type": "numeric",
            "rules": {
                "Well Suited": ("<", 1),
                "Suited": ("between", (1, 2)),
                "Moderately Suited": ("between", (2, 3)),
                "Unsuitable": (">", 3),
            },
        },          
        "SLP": {
            "column": "MeanSlope",
            "type": "numeric",
            "rules": {
                "Well Suited": ("<", 3),
                "Suited": ("between", (3, 7)),
                "Moderately Suited": ("between", (7, 15)),
                "Unsuitable": (">", 15),
            },
        },
        "STN": {
            "column": "SiblingTopsoilStonesCode",
            "type": "numeric",
            "rules": {
                "Well Suited": ("<", 1),
                "Suited": ("between", (1, 5)),
                "Moderately Suited": ("between", (5, 15)),
                "Unsuitable": (">", 15),
            },
        },             
    }
}

# ---------------------------------------------------------------------------
# Apply crop suitability rules
# ---------------------------------------------------------------------------

for crop, ruleset in crop_rules.items():

    for rule_id, rule in ruleset.items():

        column = rule["column"]
        output_col = f"{crop}_{rule_id}"

        if column not in df.columns:
            print(
                f"Missing column for {crop} {rule_id}: "
                f"{column}"
            )

            df[output_col] = pd.NA
            continue

        if rule["type"] == "numeric":

            df[output_col] = classify_numeric(
                df[column],
                rule["rules"],
            )

        elif rule["type"] == "categorical":

            df[output_col] = classify_categorical(
                df[column],
                rule["rules"],
            )

        else:
            raise ValueError(
                f"Unsupported rule type: {rule['type']} "
                f"for {crop} {rule_id}"
            )

# ---------------------------------------------------------------------------
# Suitability scoring:
# mean score + hard limiting constraints + normal limiting factors
# ---------------------------------------------------------------------------

score_map = {
    "Well Suited": 1,
    "Suited": 2,
    "Moderately Suited": 3,
    "Unsuitable": 4,
}

#Split by histogram of draft output:
def mean_score_to_class(score):
    """Convert the mean criterion score into an overall suitability class."""

    if pd.isna(score):
        return pd.NA
    elif score < 1.5:
        return "Well Suited"
    elif score < 2.5:
        return "Suited"
    elif score < 3.5:
        return "Moderately Suited"
    else:
        return "Unsuitable"


# Rules that cannot be averaged away when they are Unsuitable.
# These can be set separately for each crop.
hard_exclusion_rules = {
    "MaizeGrain": ["SLP", "PRD", "DRC"],
}


for crop, ruleset in crop_rules.items():

    score_cols = []

    # -----------------------------------------------------------------------
    # Convert each criterion class to a numeric score
    # -----------------------------------------------------------------------

    for rule_id in ruleset.keys():

        class_col = f"{crop}_{rule_id}"
        score_col = f"{class_col}_score"

        df[score_col] = df[class_col].map(score_map)
        score_cols.append(score_col)

    # -----------------------------------------------------------------------
    # Calculate the mean suitability score
    # -----------------------------------------------------------------------

    df[f"{crop}_MeanScore"] = (
        df[score_cols]
        .mean(axis=1, skipna=True)
        .round(2)
    )

    # Initial overall class based on the mean score
    df[f"{crop}_FinalClass"] = (
        df[f"{crop}_MeanScore"]
        .apply(mean_score_to_class)
    )

    # -----------------------------------------------------------------------
    # Normal limiting factor
    #
    # This is the worst-scoring criterion, as used previously.
    # All criteria tied for the worst score are retained.
    # -----------------------------------------------------------------------

    df[f"{crop}_WorstScore"] = (
        df[score_cols]
        .max(axis=1, skipna=True)
    )

    def get_limiting_factors(row):
        """Return all criteria tied for the worst individual score."""

        worst_score = row[f"{crop}_WorstScore"]

        if pd.isna(worst_score):
            return pd.NA

        limiting_rules = []

        for rule_id in ruleset.keys():

            score_col = f"{crop}_{rule_id}_score"

            if (
                score_col in row.index
                and pd.notna(row[score_col])
                and row[score_col] == worst_score
            ):
                limiting_rules.append(
                    rule_names.get(rule_id, rule_id)
                )

        return (
            "; ".join(limiting_rules)
            if limiting_rules
            else pd.NA
        )

    df[f"{crop}_LimitingFactor"] = df.apply(
        get_limiting_factors,
        axis=1,
    )

    # Optional: record the class associated with the normal limiting factor
    reverse_score_map = {
        value: key
        for key, value in score_map.items()
    }

    df[f"{crop}_LimitingClass"] = (
        df[f"{crop}_WorstScore"]
        .map(reverse_score_map)
    )

# -----------------------------------------------------------------------
# Hard limiting factors
#
# Any hard-exclusion criterion classified as Unsuitable overrides
# the mean suitability class.
#
# Additional handling:
#   - No SMU              -> Indeterminate
#   - Missing crop input  -> Missing data
# -----------------------------------------------------------------------

exclusion_rule_ids = hard_exclusion_rules.get(crop, [])

exclusion_cols = [
    f"{crop}_{rule_id}"
    for rule_id in exclusion_rule_ids
    if f"{crop}_{rule_id}" in df.columns
]

# -----------------------------------------------------------------------
# 1. Apply hard exclusions
# -----------------------------------------------------------------------

if exclusion_cols:

    hard_exclusion_mask = (
        df[exclusion_cols]
        .eq("Unsuitable")
        .any(axis=1)
    )

    # Hard exclusions override the mean-based class
    df.loc[
        hard_exclusion_mask,
        f"{crop}_FinalClass",
    ] = "Unsuitable"

    def get_hard_limiting_factors(row):
        """Return all hard-exclusion criteria classified as Unsuitable."""

        limiting_rules = []

        for rule_id in exclusion_rule_ids:

            class_col = f"{crop}_{rule_id}"

            if class_col not in row.index:
                continue

            value = row[class_col]

            if pd.notna(value) and value == "Unsuitable":
                limiting_rules.append(
                    rule_names.get(rule_id, rule_id)
                )

        return (
            "; ".join(limiting_rules)
            if limiting_rules
            else pd.NA
        )

    df[f"{crop}_HardLimitingFactor"] = df.apply(
        get_hard_limiting_factors,
        axis=1,
    )

    df[f"{crop}_HardExcluded"] = hard_exclusion_mask

else:

    df[f"{crop}_HardLimitingFactor"] = pd.NA
    df[f"{crop}_HardExcluded"] = False


# -----------------------------------------------------------------------
# 2. No SMU = Indeterminate
#
# These areas are outside the soil mapping / assessment domain.
# They are not considered genuine hard exclusions.
# -----------------------------------------------------------------------

no_soil_mask = df["SMU"].isna()

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


# -----------------------------------------------------------------------
# 3. Missing data in one or more crop criteria
#
# Only check source columns actually required by the current crop.
# Do not overwrite Indeterminate areas.
# -----------------------------------------------------------------------

required_columns = [
    rule["column"]
    for rule in ruleset.values()
    if rule["column"] in df.columns
]

missing_data_mask = (
    df[required_columns]
    .isna()
    .any(axis=1)
    & ~no_soil_mask
)

df.loc[
    missing_data_mask,
    f"{crop}_FinalClass",
] = "Missing data"

# Missing data is not a hard exclusion
df.loc[
    missing_data_mask,
    f"{crop}_HardExcluded",
] = False

# Clear any hard limiting factor where the result is due to missing data
df.loc[
    missing_data_mask,
    f"{crop}_HardLimitingFactor",
] = pd.NA


# -----------------------------------------------------------------------
# 4. Record which required fields are missing
# -----------------------------------------------------------------------

df[f"{crop}_MissingData"] = pd.NA

for column in required_columns:

    column_missing_mask = (
        df[column].isna()
        & ~no_soil_mask
    )

    existing = df.loc[
        column_missing_mask,
        f"{crop}_MissingData"
    ]

    df.loc[
        column_missing_mask,
        f"{crop}_MissingData"
    ] = existing.fillna("").apply(
        lambda x: f"{x}; {column}" if x else column
    )
# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

df.to_csv(out_path, index=False)

print("Done")
print(f"Output saved to: {out_path}")

##############################################################################
