import numpy as np
import pandas as pd # type: ignore
from scipy.interpolate import Rbf, griddata
import os

from data_processing.utils.utils import generate_mesh, save_by_lon_range, plot_polar_data

def interpolate(data_dict, data_type, plot_save_path=None, method='linear', debug=False):
    if len([f for f in os.listdir(data_dict['interp_dir']) if f.endswith('.csv') and 'lon' in f]) == 12:
        print(f"Interpolated CSVs appear to exist for {data_type} data. Skipping interpolation.")
        return

    csvs = sorted(os.listdir(data_dict['save_path']))
    meshes = generate_mesh()
    save_path = data_dict['interp_dir']

    interp_lons = []
    interp_lats = []
    interp_values = []
    interp_elev = []

    for (csv, (lon_lat_grid_north, lon_lat_grid_south)) in zip(csvs, meshes):
        df = pd.read_csv(f"{data_dict['save_path']}/{csv}")

        lons = df['Longitude'].values
        lats = df['Latitude'].values
        values = df[data_type].values

        if data_type == 'M3':
            elev = df['Elevation'].values
            assert len(elev) == len(values)
        else:
            elev = None

        if len(values) == 0:
            print(f"No data for range: {csv}")
            continue
        if np.isnan(values).any():
            print(f"WARNING: Nans present in {data_type}: {csv}")
        if np.isinf(values).any():
            print(f"WARNING: Infs present in {data_type}: {csv}")

        assert len(lons) == len(lats) == len(values)
        assert np.all(np.isfinite(lons)), "Longitude contains NaN or inf"
        assert np.all(np.isfinite(lats)), "Latitude contains NaN or inf"
        assert np.all(np.isfinite(values)), f"{data_type} contains NaN or inf"

        points = np.column_stack((lons, lats))

        # Interpolation on northern mesh grid
        lon_grid_north, lat_grid_north = lon_lat_grid_north[:, 0], lon_lat_grid_north[:, 1]
        grid_north = np.column_stack((lon_grid_north, lat_grid_north))
        interpolated_north = griddata(points, values, grid_north, method=method)

        # Interpolation on southern mesh grid
        lon_grid_south, lat_grid_south = lon_lat_grid_south[:, 0], lon_lat_grid_south[:, 1]
        grid_south = np.column_stack((lon_grid_south, lat_grid_south))
        interpolated_south = griddata(points, values, grid_south, method=method)

        if data_type == 'M3':
            interpolated_elev_north = griddata(points, elev, grid_north, method=method)
            interpolated_elev_south = griddata(points, elev, grid_south, method=method)

            # Handle NaNs for Elevation
            nan_indices_elev_north = np.isnan(interpolated_elev_north)
            nan_indices_elev_south = np.isnan(interpolated_elev_south)

            interpolated_elev_north[nan_indices_elev_north] = griddata(points, elev, grid_north[nan_indices_elev_north], method='nearest')
            interpolated_elev_south[nan_indices_elev_south] = griddata(points, elev, grid_south[nan_indices_elev_south], method='nearest')

            interp_elev.extend(np.concatenate([interpolated_elev_north, interpolated_elev_south]))            

        # Find indices of NaNs and conduct a second pass with 'nearest' method
        nan_indices_north = np.isnan(interpolated_north)
        nan_indices_south = np.isnan(interpolated_south)

        interpolated_north[nan_indices_north] = griddata(points, values, grid_north[nan_indices_north], method='nearest')
        interpolated_south[nan_indices_south] = griddata(points, values, grid_south[nan_indices_south], method='nearest')

        interp_lons.extend(np.concatenate([lon_grid_north, lon_grid_south]))
        interp_lats.extend(np.concatenate([lat_grid_north, lat_grid_south]))
        interp_values.extend(np.concatenate([interpolated_north, interpolated_south]))

    interpolated_df = pd.DataFrame({
        'Longitude': interp_lons,
        'Latitude': interp_lats,
        data_type: interp_values
    })

    if data_type == 'M3':
        interpolated_df['Elevation'] = interp_elev

    save_by_lon_range(interpolated_df, save_path)

    if plot_save_path:
        plot_polar_data(interpolated_df, data_type, graph_cat='interp', frac=0.25, save_path=plot_save_path)

    if debug:
        print(f"\nInterpolated {data_type} df:")
        print(interpolated_df.describe())
        print(interpolated_df.head())
