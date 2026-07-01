# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:03:07 2026

@author: Ashton.Eaves
"""
###### raster2dggs ###########################################################
### adapted from:  https://github.com/manaakiwhenua/raster2dggs/tree/master

import rasterio
import pandas as pd
import numpy as np
import h3
from pyproj import Transformer


def raster_to_h3_table(
    raster_path,
    output_csv,
    field_name,
    h3_resolution=12,
    statistic="mean"
):
    raster_to_h3_table(
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability\1_Data\MeanAnnualRainfall.tif",
    r"D:\Land\GIS_DATA\Landuse\LanduseSuitability\MeanAnnualRainfall_H3.csv",
    "MeanAnnualRainfall"
)

