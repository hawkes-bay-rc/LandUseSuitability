# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 13:16:54 2026

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

def classify_numeric(value, rules):
    """
    Classify numeric values using one or more conditions.

    Supported operators:
        ==, !=, >, >=, <, <=, between, or
    """

    if pd.isna(value):
        return np.nan

    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    def condition_matches(value, condition):
        """Return True when a value matches one condition."""

        op, threshold = condition

        if op == "==":
            return value == threshold

        elif op == "!=":
            return value != threshold

        elif op == ">":
            return value > threshold

        elif op == ">=":
            return value >= threshold

        elif op == "<":
            return value < threshold

        elif op == "<=":
            return value <= threshold

        elif op == "between":
            low, high = threshold
            return low <= value <= high

        elif op == "or":
            return any(
                condition_matches(value, subcondition)
                for subcondition in threshold
            )

        else:
            raise ValueError(
                f"Unsupported operator: {op}"
            )

    for suitability, condition in rules.items():
        if condition_matches(value, condition):
            return suitability

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
# Suitability scoring and final limiting class
# ---------------------------------------------------------------------------

score_map = {
    "Well Suited": 1,
    "Suited": 2,
    "Moderately Suited": 3,
    "Unsuitable": 4,
}

reverse_score_map = {v: k for k, v in score_map.items()}

for crop, ruleset in crop_rules.items():

    score_cols = []

    for rule_id in ruleset.keys():
        class_col = f"{crop}_{rule_id}"
        score_col = f"{class_col}_score"

        df[score_col] = df[class_col].map(score_map)
        score_cols.append(score_col)

    df[f"{crop}_FinalScore"] = df[score_cols].max(axis=1)
    df[f"{crop}_FinalClass"] = df[f"{crop}_FinalScore"].map(reverse_score_map)

    df[f"{crop}_LimitingFactor"] = df[score_cols].idxmax(axis=1)

    df[f"{crop}_LimitingFactor"] = (
        df[f"{crop}_LimitingFactor"]
        .str.replace(f"{crop}_", "", regex=False)
        .str.replace("_score", "", regex=False)
        .map(rule_names)
    )

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

df.to_csv(out_path, index=False)

print("Done")
print(f"Output saved to: {out_path}")