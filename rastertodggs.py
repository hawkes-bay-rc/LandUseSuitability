# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:39:35 2026

@author: Ashton.Eaves
"""
###### raster2dggs ###########################################################
### adapted from:  https://github.com/manaakiwhenua/raster2dggs/tree/master
##############################################################################

# In anaconda prompt open env: conda activate h3raster
# Then run spyder

import h3
import pandas as pd
import rasterio
from pyproj import Transformer

# ==========================
# User settings
# ==========================

h3_csv = r"D:\Land\GIS_DATA\Landuse\LanduseSuitability\1_Data\H3Res11.csv"

### Select one raster at a time: 
    
#raster_path = r"D:\Land\GIS_DATA\Landuse\LanduseSuitability\1_Data\MeanAnnualRainfall.tif"
raster_path = r"\\gisdalton\gishub3\Land\HawkesBayRegion_LiDAR_2020\LUCWorkStream\D1\hbrc_dem_5m_slope_degrees_r2.tif"

#output_csv = r"D:\Land\GIS_DATA\Landuse\LanduseSuitability\3_Outputs\MeanAnnualRainfall_H3.csv"
output_csv = r"D:\Land\GIS_DATA\Landuse\LanduseSuitability\3_Outputs\MeanSlope_H3.csv"


### Select only one output field at a time:

h3_field = "GRID_ID"
#output_field = "MeanAnnualRainfall"
output_field = "MeanSlope"

# ==========================
# Read H3 table
# ==========================

h3_df = pd.read_csv(h3_csv)

# ==========================
# Get H3 centre coordinates
# ==========================

centres = [h3.cell_to_latlng(cell) for cell in h3_df[h3_field]]

h3_df["lat"] = [c[0] for c in centres]
h3_df["lon"] = [c[1] for c in centres]

# ==========================
# Sample raster at H3 centres
# ==========================

with rasterio.open(raster_path) as src:
    transformer = Transformer.from_crs(
        "EPSG:4326",
        src.crs,
        always_xy=True
    )

    xs, ys = transformer.transform(
        h3_df["lon"].to_numpy(),
        h3_df["lat"].to_numpy()
    )

    coords = list(zip(xs, ys))

    values = [
        v[0] for v in src.sample(coords)
    ]

    nodata = src.nodata

h3_df[output_field] = values

# Remove NoData if wanted
if nodata is not None:
    h3_df.loc[h3_df[output_field] == nodata, output_field] = pd.NA

# ==========================
# Export only join fields
# ==========================

out = h3_df[[h3_field, output_field]]

out.to_csv(output_csv, index=False)

print(out.head())
print(len(out))
print(out[output_field].notna().sum())