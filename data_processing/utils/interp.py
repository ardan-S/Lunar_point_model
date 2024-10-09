import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
import os

from data_processing.utils.utils import generate_mesh, save_by_lon_range


def interpolate(data_dict, data_type, batch_size = 1_000, debug=False):
    print("hello from interpolate")
    return

    csvs = sorted(os.listdir(data_dict['save_path']))
    meshes = generate_mesh()
    save_path = data_dict['interp_dir']

    for (csv, (lon_lat_grid_north, lon_lat_grid_south)) in zip(csvs, meshes):
        df = pd.read_csv(f"{data_dict['save_path']}/{csv}")

        lons = df['Longitude'].values
        lats = df['Latitude'].values
        values = df[data_type].values

        assert len(lons) == len(lats) == len(values)
        assert np.all(np.isfinite(lons)), "Longitude contains NaN or inf"
        assert np.all(np.isfinite(lats)), "Latitude contains NaN or inf"
        assert np.all(np.isfinite(values)), f"{data_type} contains NaN or inf"

        interp_lons = []
        interp_lats = []
        interp_values = []

        num_batches = len(lons) // batch_size + 1

        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(lons))
        
            rbf_interp = Rbf(lons[batch_start:batch_end], lats[batch_start:batch_end], values[batch_start:batch_end], function='inverse', smooth=0)

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

        save_by_lon_range(interpolated_df, save_path)

        if debug:
            print(f"\nIntperpolated {data_type} df:")
            print(df.describe())
            print(df.head())



