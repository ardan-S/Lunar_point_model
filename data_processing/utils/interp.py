import numpy as np
import pandas as pd # type: ignore
from scipy.interpolate import Rbf, griddata
import os
import sys
from scipy.stats import binned_statistic_2d
from scipy.spatial import cKDTree

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


def interpolate_diviner(df, lon_lat_grid_north, lon_lat_grid_south, method='linear'):
    lons = df['Longitude'].values
    lats = df['Latitude'].values
    values = df['Diviner'].values

    assert np.all(np.abs(lats) >= 75), "Diviner data contains points between lat 75 and -75"

    lon_grid_north, lat_grid_north = lon_lat_grid_north[:, 0], lon_lat_grid_north[:, 1]
    lon_grid_south, lat_grid_south = lon_lat_grid_south[:, 0], lon_lat_grid_south[:, 1]

    north_mask = lats >= 75
    south_mask = lats <= -75

    points_n = np.column_stack((lons[north_mask], lats[north_mask]))
    values_n = values[north_mask]

    points_s = np.column_stack((lons[south_mask], lats[south_mask]))
    values_s = values[south_mask]

    grid_n = np.column_stack((lon_grid_north, lat_grid_north))
    grid_s = np.column_stack((lon_grid_south, lat_grid_south))

    print("Points stacked"); sys.stdout.flush()

    interp_n = max_rad_interp(points_n, values_n, grid_n)
    interp_s = max_rad_interp(points_s, values_s, grid_s)

    print("Interpolated"); sys.stdout.flush()

    # Fill gaps with nearest interpolation
    if np.any(np.isnan(interp_n)):
        nan_idx_north = np.isnan(interp_n)
        interp_n[nan_idx_north] = griddata(points_n, values, grid_n[nan_idx_north], method='nearest')

    if np.any(np.isnan(interp_s)):
        nan_idx_south = np.isnan(interp_s)
        interp_s[nan_idx_south] = griddata(points_n, values, grid_s[nan_idx_south], method='nearest')

    interpolated_df = pd.DataFrame({
        'Longitude': np.concatenate([lon_grid_north, lon_grid_south]),
        'Latitude': np.concatenate([lat_grid_north, lat_grid_south]),
        'Diviner': np.concatenate([interp_n, interp_s])
    })

    return interpolated_df


def max_rad_interp(points, values, targets, radius_degs=0.5):
    tree = cKDTree(points)
    print("Built tree"); sys.stdout.flush()
    idx_lists = tree.query_ball_point(targets, r=radius_degs)   # Vectorised query for all targets
    print("Queried"); sys.stdout.flush()

    interpolated = np.empty(len(targets), dtype=float)
    points_per_target = np.array([len(idxs) for idxs in idx_lists])
    print(f"Average num points per target: {np.mean(points_per_target)}")
    print(f"Min num points per target: {np.min(points_per_target)}")
    print(f"Max num points per target: {np.max(points_per_target)}")
    sys.stdout.flush()

    for i, idxs in enumerate(idx_lists):
        if idxs:
            interpolated[i] = np.nanmax(values[idxs])   # Take max if possible
        else:
            interpolated[i] = np.nan    # Set to nan if no points
    return interpolated


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
        print(f"NOTE: Diviner resampled for {div_frac:.2%} of data across all csvs due to abundance of data. Weighted to higher values"); sys.stdout.flush()

    for (csv, (lon_lat_grid_north, lon_lat_grid_south)) in zip(csvs, meshes):
        df = pd.read_csv(f"{data_dict['save_path']}/{csv}")

        if data_type == 'Diviner':
            weights = df[data_type].values / df[data_type].sum()
            df = df.sample(frac=div_frac, weights=weights, random_state=42)    # Resample Diviner data, weighted to higher values
            print(f"Interpolating Diviner for {csv}"); sys.stdout.flush()
            interpolated_df = interpolate_diviner(df, lon_lat_grid_north, lon_lat_grid_south)
            print()
        else:
            elev = df['Elevation'].values if data_type == 'M3' else None
            interpolated_df = interpolate_using_griddata(df, lon_lat_grid_north, lon_lat_grid_south, method, elev, data_type)
    
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
