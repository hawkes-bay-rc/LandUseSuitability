# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:39:35 2026

@author: Ashton.Eaves
"""
###### raster2dggs ###########################################################
### adapted from:  https://github.com/manaakiwhenua/raster2dggs/tree/master

import rasterio
import pandas as pd
import numpy as np
import h3
from pyproj import Transformer

# ==========================
# User settings
# ==========================

#raster_path = r"D:\Land\GIS_DATA\Landuse\LanduseSuitability\1_Data\MeanAnnualRainfall.tif"
raster_path = r"\\gisdalton\gishub3\Land\HawkesBayRegion_LiDAR_2020\LUCWorkStream\D1\hbrc_dem_5m_slope_degrees_r2.tif"

#output_csv = r"D:\Land\GIS_DATA\Landuse\LanduseSuitability\3_Outputs\MeanAnnualRainfall_H3.csv"
output_csv = r"D:\Land\GIS_DATA\Landuse\LanduseSuitability\3_Outputs\MeanSlope_H3.csv"

#field_name = "MeanAnnualRainfall"
field_name = "MeanSlope"

h3_resolution = 12

##############################################################################

with rasterio.open(raster_path) as src:
    band = src.read(1)
    nodata = src.nodata
    transform = src.transform
    crs = src.crs

    rows, cols = np.where(band != nodata)

    values = band[rows, cols]

    xs, ys = rasterio.transform.xy(
        transform,
        rows,
        cols,
        offset="center"
    )

# Convert from raster CRS to WGS84 lat/lon for H3
transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

lons, lats = transformer.transform(xs, ys)

df = pd.DataFrame({
    "lon": lons,
    "lat": lats,
    "value": values
})

df["h3"] = [
    h3.latlng_to_cell(lat, lon, h3_resolution)
    for lat, lon in zip(df["lat"], df["lon"])
]

# Aggregate raster values by H3 cell
h3_df = (
    df.groupby("h3", as_index=False)["value"]
      .mean()
)

print(h3_df.head())
print(len(h3_df))

h3_df = (
    df.groupby("h3", as_index=False)["value"]
      .mean()
      .rename(columns={"value": field_name})
)

# Save to output
h3_df.to_csv(output_csv, index=False)