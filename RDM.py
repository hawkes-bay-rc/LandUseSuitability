# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:54:37 2026

@author: Ashton.Eaves
"""

### Conditional Statements for crops #########################################

# In anaconda prompt open env: conda activate h3raster
# Then run spyder

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

csv_path = r"D:\Land\GIS_DATA\Landuse\LanduseSuitability\3_Outputs\CropSuitabilityTest.csv"

out_path = r"D:\Land\GIS_DATA\Landuse\LanduseSuitability\3_Outputs\CropSuitabilityTest_Classified.csv"

df = pd.read_csv(csv_path)

# ---------------------------------------------------------------------------
# Generic classifiers
# ---------------------------------------------------------------------------

def classify_numeric(value, rules):
    if pd.isna(value):
        return np.nan

    value = float(value)

    for suitability, condition in rules.items():
        op, threshold = condition

        if op == ">" and value > threshold:
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

    return np.nan


def classify_categorical(value, rules):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    for suitability, valid_values in rules.items():
        valid_values = [str(v).strip().lower() for v in valid_values]

        if value in valid_values:
            return suitability

    return np.nan


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
            "column": "rootDepthRange",
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

        "SPM": {
            "column": "SoilProfileMaterial",
            "type": "categorical",
            "rules": {
                "Well Suited": ["Md"],
                "Suited": [],
                "Moderately Suited": [],
                "Unsuitable": [],
            },
        },

        "STN": {
            "column": "OVSR_TopSoilStony",
            "type": "categorical",
            "rules": {
                "Well Suited": ["<1", "< 1", "0", "None", "Null"],
                "Suited": ["1 to 5", "1 - 5"],
                "Moderately Suited": ["5 to 35", "5 - 35"],
                "Unsuitable": [">35", "> 35"],
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
            print(f"Missing column: {column}")
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
# Optional: suitability scoring and final limiting class
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

    # Limiting factor: the rule with the worst score
    df[f"{crop}_LimitingFactor"] = df[score_cols].idxmax(axis=1)
    df[f"{crop}_LimitingFactor"] = (
        df[f"{crop}_LimitingFactor"]
        .str.replace(f"{crop}_", "", regex=False)
        .str.replace("_score", "", regex=False)
    )

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

df.to_csv(out_path, index=False)

print("Done")
print(f"Output saved to: {out_path}")