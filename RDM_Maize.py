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
    r"\3_Outputs\CropSuitabilityClassified_Maize.csv"
)

df = pd.read_csv(
    csv_path,
    dtype={"GRID_ID": "string"},
)

df.columns = df.columns.str.strip()

# ---------------------------------------------------------------------------
# Generic classifiers
# ---------------------------------------------------------------------------

def classify_numeric(value, rules):
    """Classify numeric values using threshold rules."""

    if pd.isna(value):
        return np.nan

    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    for suitability, condition in rules.items():
        op, threshold = condition

        if op == "==" and value == threshold:
            return suitability
        elif op == "!=" and value != threshold:
            return suitability
        elif op == ">" and value > threshold:
            return suitability
        elif op == ">=" and value >= threshold:
            return suitability
        elif op == "<" and value < threshold:
            return suitability
        elif op == "<=" and value <= threshold:
            return suitability
        elif op == "between":
            low, high = threshold
            if low <= value <= high:
                return suitability
        else:
            if op not in {"==", "!=", ">", ">=", "<", "<=", "between"}:
                raise ValueError(f"Unsupported operator: {op}")

    return np.nan


def classify_categorical(value, rules):
    """Classify text or integer categories."""

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    for suitability, valid_values in rules.items():
        valid_values = [str(v).strip().lower() for v in valid_values]

        if value in valid_values:
            return suitability

    return np.nan


# ---------------------------------------------------------------------------
# Rule names for output
# ---------------------------------------------------------------------------

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
    "ECS": "Salinity"
}

# ---------------------------------------------------------------------------
# Crop rule dictionary
# ---------------------------------------------------------------------------

crop_rules = {
    "MaizeGrain": {
        "ANR": {
            "column": "MeanAnnualRainfall",
            "type": "numeric",
            "rules": {
                "Well Suited": (">", 1200),
                "Suited": ("between", (1000, 1200)),
                "Moderately Suited": ("between", (850, 1000)),
                "Unsuitable": ("<", 850),
            },
        },
        "ART": {
            "column": "MeanAnnualRainfall",
            "type": "numeric",
            "rules": {
                "Well Suited": ("<", 1300),
                "Suited": ("between", (1300, 1400)),
                "Moderately Suited": ("between", (1400, 1500)),
                "Unsuitable": (">", 1500),
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
        "PRD": {
            "column": "RootDepthRange",
            "type": "categorical",
            "rules": {
                "Well Suited": ["> 1 m", ">1 m", ">100 cm"],
                "Suited": ["60 - 100 cm", "60 to 100 cm"],
                "Moderately Suited": ["30 - 60 cm", "30 to 60 cm"],
                "Unsuitable": ["< 30 cm", "<30 cm"],
            },
        },
        "DRC": {
            "column": "DrainageClass",
            "type": "categorical",
            "rules": {
                "Well Suited": [
                    "Excessively drained",
                    "Well drained",
                ],
                "Suited": [
                    "Moderately well drained",
                    "Imperfectly drained",
                    "Imperfect drained",
                ],
                "Unsuitable": [
                    "Poorly drained",
                    "Very poorly drained",
                    "Very-poorly drained",
                ],
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
        "STN": {
            "column": "SiblingTopsoilStonesCode",
            "type": "numeric",
            "rules": {
                "Well Suited": ("<", 1),
                "Suited": ("between", (1, 5)),
                "Moderately Suited": ("between", (5, 35)),
                "Unsuitable": (">", 35),
            },
        },     
        "FFH": {
            "column": "FrostDays_MAM",
            "type": "numeric",
            "rules": {
                "Well Suited": ("<", 1),
                "Suited": ("between", (1, 2)),
                "Moderately Suited": ("between", (2, 3)),
                "Unsuitable": (">", 3),
            },
        },
       "FFB": {
            "column": "FrostDays_SON",
            "type": "numeric",
            "rules": {
                "Well Suited": ("<", 1),
                "Suited": ("between", (1, 2)),
                "Moderately Suited": ("between", (2, 3)),
                "Unsuitable": (">", 3),
            },
        },      
        "GDD": {
            "column": "GrowingDegreeDays",
            "type": "numeric",
            "rules": {
                "Well Suited": (">", 1400),
                "Suited": ("between", (1300, 1400)),
                "Moderately Suited": ("between", (1100, 1300)),
                "Unsuitable": ("<", 1100),
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
    }
}

# ---------------------------------------------------------------------------
# Apply rules
# ---------------------------------------------------------------------------

for crop, ruleset in crop_rules.items():

    for rule_id, rule in ruleset.items():

        column = rule["column"]
        output_col = f"{crop}_{rule_id}"

        if column not in df.columns:
            df[output_col] = np.nan
            print(f"Missing column for {crop} {rule_id}: {column}")
            continue

        if rule["type"] == "numeric":
            df[output_col] = df[column].apply(
                lambda x: classify_numeric(x, rule["rules"])
            )

        elif rule["type"] == "categorical":
            df[output_col] = df[column].apply(
                lambda x: classify_categorical(x, rule["rules"])
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
    # -----------------------------------------------------------------------

    exclusion_rule_ids = hard_exclusion_rules.get(crop, [])

    exclusion_cols = [
        f"{crop}_{rule_id}"
        for rule_id in exclusion_rule_ids
        if f"{crop}_{rule_id}" in df.columns
    ]

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
            """Return all hard-exclusion criteria that are Unsuitable."""

            limiting_rules = []

            for rule_id in exclusion_rule_ids:

                class_col = f"{crop}_{rule_id}"

                if (
                    class_col in row.index
                    and row[class_col] == "Unsuitable"
                ):
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

        # Useful Boolean field for filtering and GIS symbology
        df[f"{crop}_HardExcluded"] = hard_exclusion_mask

    else:

        df[f"{crop}_HardLimitingFactor"] = pd.NA
        df[f"{crop}_HardExcluded"] = False
        
# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

df.to_csv(out_path, index=False)

print("Done")
print(f"Output saved to: {out_path}")