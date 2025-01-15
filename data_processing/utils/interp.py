import numpy as np
import pandas as pd # type: ignore
from scipy.interpolate import Rbf, griddata
import os
import sys
from scipy.stats import binned_statistic_2d

from data_processing.utils.utils import generate_mesh, save_by_lon_range, plot_polar_data, load_every_nth_line

# def interpolate(data_dict, data_type, plot_save_path=None, method='linear', debug=False):
#     if not os.path.exists(data_dict['interp_dir']):
#         print(f"Creating interp dir for {data_type}")
#         os.mkdir(data_dict['interp_dir'])

#     if len([f for f in os.listdir(data_dict['interp_dir']) if f.endswith('.csv') and 'lon' in f]) == 12:
#         print(f"Interpolated CSVs appear to exist for {data_type} data. Skipping interpolation.")
#         return

#     csvs = sorted(os.listdir(data_dict['save_path']))
#     meshes = generate_mesh()
#     save_path = data_dict['interp_dir']

#     div_frac = 0.25
#     print(f"NOTE: Diviner resampled for {div_frac*100}% of data across all csvs due to abundance of data. Weighted to higher values"); sys.stdout.flush()

#     for (csv, (lon_lat_grid_north, lon_lat_grid_south)) in zip(csvs, meshes):
#         df = pd.read_csv(f"{data_dict['save_path']}/{csv}")

#         if data_type == 'Diviner':
#             weights = df[data_type].values / df[data_type].sum()
#             df = df.sample(frac=div_frac, weights=weights, random_state=42)    # Resample Diviner data, weighted to higher values

#         lons = df['Longitude'].values
#         lats = df['Latitude'].values
#         values = df[data_type].values

#         if data_type == 'M3':   
#             elev = df['Elevation'].values
#             assert len(elev) == len(values)
#         else:
#             elev = None

#         if len(values) == 0:
#             print(f"No data for range: {csv}")
#             continue
#         if np.isnan(values).any():
#             print(f"WARNING: Nans present in {data_type}: {csv}")
#         if np.isinf(values).any():
#             print(f"WARNING: Infs present in {data_type}: {csv}")

#         assert len(lons) == len(lats) == len(values), f"Length mismatch for {data_type}: {csv}"
#         assert np.all(np.isfinite(np.asarray(lons))), "Longitude contains NaN or inf"
#         assert np.all(np.isfinite(np.asarray(lats))), "Latitude contains NaN or inf"
#         assert np.all(np.isfinite(np.asarray(values))), f"{data_type} contains NaN or inf"

#         points = np.column_stack((np.asanyarray(lons), np.asanyarray(lats)))

#         # Interpolation on northern mesh grid
#         lon_grid_north, lat_grid_north = lon_lat_grid_north[:, 0], lon_lat_grid_north[:, 1]
#         grid_north = np.column_stack((lon_grid_north, lat_grid_north))
#         interpolated_north = griddata(points, values, grid_north, method=method)

#         # Interpolation on southern mesh grid
#         lon_grid_south, lat_grid_south = lon_lat_grid_south[:, 0], lon_lat_grid_south[:, 1]
#         grid_south = np.column_stack((lon_grid_south, lat_grid_south))
#         interpolated_south = griddata(points, values, grid_south, method=method)

#         if data_type == 'M3':
#             interpolated_elev_north = griddata(points, elev, grid_north, method=method)
#             interpolated_elev_south = griddata(points, elev, grid_south, method=method)

#             # Handle NaNs for Elevation
#             nan_indices_elev_north = np.isnan(interpolated_elev_north)
#             nan_indices_elev_south = np.isnan(interpolated_elev_south)

#             interpolated_elev_north[nan_indices_elev_north] = griddata(points, elev, grid_north[nan_indices_elev_north], method='nearest')
#             interpolated_elev_south[nan_indices_elev_south] = griddata(points, elev, grid_south[nan_indices_elev_south], method='nearest')

#             # interp_elev.extend(np.concatenate([interpolated_elev_north, interpolated_elev_south]))            

#         # Find indices of NaNs and conduct a second pass with 'nearest' method
#         nan_indices_north = np.isnan(interpolated_north)
#         nan_indices_south = np.isnan(interpolated_south)

#         interpolated_north[nan_indices_north] = griddata(points, values, grid_north[nan_indices_north], method='nearest')
#         interpolated_south[nan_indices_south] = griddata(points, values, grid_south[nan_indices_south], method='nearest')

#         # interp_lons.extend(np.concatenate([lon_grid_north, lon_grid_south]))
#         # interp_lats.extend(np.concatenate([lat_grid_north, lat_grid_south]))
#         # interp_values.extend(np.concatenate([interpolated_north, interpolated_south]))

#         interpolated_df = pd.DataFrame({
#             'Longitude': np.concatenate([lon_grid_north, lon_grid_south]),
#             'Latitude': np.concatenate([lat_grid_north, lat_grid_south]),
#             data_type: np.concatenate([interpolated_north, interpolated_south])
#         })

#         if data_type == 'M3':
#             interpolated_df['Elevation'] = np.concatenate([interpolated_elev_north, interpolated_elev_south]) # type: ignore (possible unbound warning)

#         save_by_lon_range(interpolated_df, save_path)

#         del interpolated_df, df, points, lons, lats, values

#     df_list = []

#     for file in os.listdir(save_path):
#         if file.endswith('.csv'):
#             df = load_every_nth_line(f"{save_path}/{file}", n=1)
#             df_list.append(df)

#     interpolated_df = pd.concat(df_list, ignore_index=True)

#     if plot_save_path:
#         plot_polar_data(interpolated_df, data_type, graph_cat='interp', frac=0.25, save_path=plot_save_path)

#     if debug:
#         print(f"\nInterpolated {data_type} df:")
#         print(interpolated_df.describe())
#         print(interpolated_df.head())

#     del interpolated_df, df_list



def interpolate_using_griddata(df, lon_lat_grid_north, lon_lat_grid_south, method='linear', elev=None, data_type=None):
    lons = df['Longitude'].values
    lats = df['Latitude'].values
    values = df[data_type].values
    
    points = np.column_stack((lons, lats))

    # Interpolation on northern mesh grid
    lon_grid_north, lat_grid_north = lon_lat_grid_north[:, 0], lon_lat_grid_north[:, 1]
    grid_north = np.column_stack((lon_grid_north, lat_grid_north))
    interpolated_north = griddata(points, values, grid_north, method=method)

    # Interpolation on southern mesh grid
    lon_grid_south, lat_grid_south = lon_lat_grid_south[:, 0], lon_lat_grid_south[:, 1]
    grid_south = np.column_stack((lon_grid_south, lat_grid_south))
    interpolated_south = griddata(points, values, grid_south, method=method)

    # Handle NaNs with 'nearest' method
    nan_indices_north = np.isnan(interpolated_north)
    nan_indices_south = np.isnan(interpolated_south)
    interpolated_north[nan_indices_north] = griddata(points, values, grid_north[nan_indices_north], method='nearest')
    interpolated_south[nan_indices_south] = griddata(points, values, grid_south[nan_indices_south], method='nearest')

    interpolated_df = pd.DataFrame({
        'Longitude': np.concatenate([lon_grid_north, lon_grid_south]),
        'Latitude': np.concatenate([lat_grid_north, lat_grid_south]),
        data_type: np.concatenate([interpolated_north, interpolated_south])
    })

    if elev is not None:
        interpolated_elev_north = griddata(points, elev, grid_north, method=method)
        interpolated_elev_south = griddata(points, elev, grid_south, method=method)

        # Handle NaNs for elevation
        nan_indices_elev_north = np.isnan(interpolated_elev_north)
        nan_indices_elev_south = np.isnan(interpolated_elev_south)
        interpolated_elev_north[nan_indices_elev_north] = griddata(points, elev, grid_north[nan_indices_elev_north], method='nearest')
        interpolated_elev_south[nan_indices_elev_south] = griddata(points, elev, grid_south[nan_indices_elev_south], method='nearest')

        interpolated_df['Elevation'] = np.concatenate([interpolated_elev_north, interpolated_elev_south])

    return interpolated_df


def interpolate_diviner(df, lon_lat_grid_north, lon_lat_grid_south, method='linear', n_bins=100):
    # Extract data
    lons = df['Longitude'].values
    lats = df['Latitude'].values
    values = df['Diviner'].values

    # Spatial binning and compute maxima in each bin
    lon_bins = np.linspace(lons.min(), lons.max(), n_bins + 1)
    lat_bins = np.linspace(lats.min(), lats.max(), n_bins + 1)

    max_temp, _, _, _ = binned_statistic_2d(
        x=lons, y=lats, values=values, statistic='max', bins=(lon_bins, lat_bins)   # type: ignore
    )

    # Compute centers of bins
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    # Flatten grids and filter valid points
    valid_mask = ~np.isnan(max_temp.ravel())
    points = np.column_stack((lon_grid.ravel()[valid_mask], lat_grid.ravel()[valid_mask]))
    max_values = max_temp.ravel()[valid_mask]

    # Interpolate onto northern and southern grids
    lon_grid_n = lon_lat_grid_north[:, 0]
    lat_grid_n = lon_lat_grid_north[:, 1]
    coords_n = np.column_stack((lon_grid_n, lat_grid_n))

    interp_n = griddata(points, max_values, coords_n, method=method)

    lon_grid_s = lon_lat_grid_south[:, 0]
    lat_grid_s = lon_lat_grid_south[:, 1]
    coords_s = np.column_stack((lon_grid_s, lat_grid_s))

    interp_s = griddata(points, max_values, coords_s, method=method)

    # Fill gaps with nearest interpolation
    if np.any(np.isnan(interp_n)):
        nan_idx_north = np.isnan(interp_n)
        interp_n[nan_idx_north] = griddata(
            points, max_values, coords_n[nan_idx_north], method='nearest'
        )

    if np.any(np.isnan(interp_s)):
        nan_idx_south = np.isnan(interp_s)
        interp_s[nan_idx_south] = griddata(
            points, max_values, coords_s[nan_idx_south], method='nearest'
        )
    interpolated_df = pd.DataFrame({
        'Longitude': np.concatenate([lon_grid_n, lon_grid_s]),
        'Latitude': np.concatenate([lat_grid_n, lat_grid_s]),
        'Diviner': np.concatenate([interp_n, interp_s])
    })

    return interpolated_df


def interpolate(data_dict, data_type, plot_save_path=None, method='linear', debug=False):
    if not os.path.exists(data_dict['interp_dir']):
        print(f"Creating interp dir for {data_type}")
        os.mkdir(data_dict['interp_dir'])

    if len([f for f in os.listdir(data_dict['interp_dir']) if f.endswith('.csv') and 'lon' in f]) == 12:
        print(f"Interpolated CSVs appear to exist for {data_type} data. Skipping interpolation.")
        return

    csvs = sorted(os.listdir(data_dict['save_path']))
    meshes = generate_mesh()
    save_path = data_dict['interp_dir']

    div_frac = 0.25
    if data_type == 'Diviner':
        print(f"NOTE: Diviner resampled for {div_frac*100}% of data across all csvs due to abundance of data. Weighted to higher values"); sys.stdout.flush()

    for (csv, (lon_lat_grid_north, lon_lat_grid_south)) in zip(csvs, meshes):
        df = pd.read_csv(f"{data_dict['save_path']}/{csv}")
        print(f"{len(df)} rows in raw csv: {csv}")

        if data_type == 'Diviner':
            print(f"Interpolating Diviner data for {csv}")
            weights = df[data_type].values / df[data_type].sum()
            df = df.sample(frac=div_frac, weights=weights, random_state=42)    # Resample Diviner data, weighted to higher values
            interpolated_df = interpolate_diviner(df, lon_lat_grid_north, lon_lat_grid_south, n_bins=400)
        else:
            print(f"Interpolating {data_type} data for {csv}")
            elev = df['Elevation'].values if data_type == 'M3' else None
            interpolated_df = interpolate_using_griddata(df, lon_lat_grid_north, lon_lat_grid_south, method, elev, data_type)
    
        print(f"Saving interpolated data length: {len(interpolated_df)}")
        save_by_lon_range(interpolated_df, save_path)

        del interpolated_df, df

    df_list = []

    for file in os.listdir(save_path):
        if file.endswith('.csv'):
            df = load_every_nth_line(f"{save_path}/{file}", n=1)
            df_list.append(df)

    interpolated_df = pd.concat(df_list, ignore_index=True)

    if plot_save_path:
        plot_polar_data(interpolated_df, data_type, graph_cat='interp', frac=0.25, save_path=plot_save_path)

    if debug:
        print(f"\nInterpolated {data_type} df:")
        print(interpolated_df.describe())
        print(interpolated_df.head())

    del interpolated_df, df_list
