import numpy as np
import pandas as pd
from scipy.interpolate import Rbf

from data_processing.utils.utils import generate_mesh


def interpolate(df, data_type, debug=False):
    lons = df['Longitude'].values
    lats = df['Latitude'].values
    values = df[data_type].values

    meshes = generate_mesh()

    rbf_interp = Rbf(lons, lats, values, function='inverse', smooth=0)

    interp_lons = []
    interp_lats = []
    interp_values = []

    for lon_lat_grid_north, lon_lat_grid_south in meshes:
        lon_grid_north, lat_grid_north = lon_lat_grid_north[:, 0], lon_lat_grid_north[:, 1]
        interpolated_north = rbf_interp(lon_grid_north, lat_grid_north)

        lon_grid_south, lat_grid_south = lon_lat_grid_south[:, 0], lon_lat_grid_south[:, 1]
        interpolated_south = rbf_interp(lon_grid_south, lat_grid_south)

        interp_lons.extend(lon_grid_north)
        interp_lats.extend(lat_grid_north)
        interp_values.extend(interpolated_north)

        interp_lons.extend(lon_grid_south)
        interp_lats.extend(lat_grid_south)
        interp_values.extend(interpolated_south)

    interpolated_df = pd.DataFrame({
        'Longitude': interp_lons,
        'Latitude': interp_lats,
        data_type: interp_values
    })

    if debug:
        print(f"\nIntperpolated {data_type} df:")
        print(df.describe())
        print(df.head())

    return interpolated_df
