import numpy as np
import pandas as pd # type: ignore
from scipy.interpolate import Rbf, griddata
import os
import sys
from scipy.stats import binned_statistic_2d
from scipy.spatial import cKDTree
from pyproj import Proj

from data_processing.utils.utils import generate_mesh, save_by_lon_range, plot_polar_data, load_every_nth_line, generate_xy_mesh
from data_processing.download_data import clear_dir


# def interpolate_using_griddata(df, lon_lat_grid_north, lon_lat_grid_south, method='linear', elev=None, data_type=None):
#     lons = df['Longitude'].values
#     lats = df['Latitude'].values
#     values = df[data_type].values
#     points = np.column_stack((lons, lats))

#     # Interpolation on northern mesh grid
#     lon_grid_north, lat_grid_north = lon_lat_grid_north[:, 0], lon_lat_grid_north[:, 1]
#     grid_north = np.column_stack((lon_grid_north, lat_grid_north))
#     interpolated_north = griddata(points, values, grid_north, method=method)

#     # Interpolation on southern mesh grid
#     lon_grid_south, lat_grid_south = lon_lat_grid_south[:, 0], lon_lat_grid_south[:, 1]
#     grid_south = np.column_stack((lon_grid_south, lat_grid_south))
#     interpolated_south = griddata(points, values, grid_south, method=method)

#     # Handle NaNs with 'nearest' method
#     nan_indices_north = np.isnan(interpolated_north)
#     nan_indices_south = np.isnan(interpolated_south)
#     interpolated_north[nan_indices_north] = griddata(points, values, grid_north[nan_indices_north], method='nearest')
#     interpolated_south[nan_indices_south] = griddata(points, values, grid_south[nan_indices_south], method='nearest')

#     interpolated_df = pd.DataFrame({
#         'Longitude': np.concatenate([lon_grid_north, lon_grid_south]),
#         'Latitude': np.concatenate([lat_grid_north, lat_grid_south]),
#         data_type: np.concatenate([interpolated_north, interpolated_south])
#     })

#     if elev is not None:
#         interpolated_elev_north = griddata(points, elev, grid_north, method=method)
#         interpolated_elev_south = griddata(points, elev, grid_south, method=method)

#         # Handle NaNs for elevation
#         nan_indices_elev_north = np.isnan(interpolated_elev_north)
#         nan_indices_elev_south = np.isnan(interpolated_elev_south)
#         interpolated_elev_north[nan_indices_elev_north] = griddata(points, elev, grid_north[nan_indices_elev_north], method='nearest')
#         interpolated_elev_south[nan_indices_elev_south] = griddata(points, elev, grid_south[nan_indices_elev_south], method='nearest')

#         interpolated_df['Elevation'] = np.concatenate([interpolated_elev_north, interpolated_elev_south])

#     return interpolated_df


def interpolate_csv(df, mesh_north, mesh_south, data_type,  elev=None, MOON_RADIUS_M=1737.4e3):
    # Define AEQD CRS for polar projections
    transformer_north = Proj(proj='aeqd', lat_0=90, lon_0=0, R=MOON_RADIUS_M, always_xy=True)
    transformer_south = Proj(proj='aeqd', lat_0=-90, lon_0=0, R=MOON_RADIUS_M, always_xy=True)

    lons = df['Longitude'].values
    lats = df['Latitude'].values

    north_mask = lats >= 75
    south_mask = lats <= -75

    if len(north_mask) == 0 or len(south_mask) == 0:
        raise ValueError("No data points found in the specified latitude ranges.")

    xs_n, ys_n = transformer_north(lons[north_mask], lats[north_mask])
    xs_s, ys_s = transformer_south(lons[south_mask], lats[south_mask])

    vals_n = df.loc[north_mask, data_type].values
    vals_s = df.loc[south_mask, data_type].values

    tree_n = cKDTree(np.column_stack((xs_n, ys_n)))
    tree_s = cKDTree(np.column_stack((xs_s, ys_s)))

    def idw_interpolate(tree, values, targets, k=1, power=2):
        dists, idxs = tree.query(targets, k=k, workers=-1)
        if k == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        dists = np.where(dists == 0, 1e-12, dists)
        weights = 1. / (dists ** power)
        weighted_vals = weights * values[idxs]
        return weighted_vals.sum(axis=1) / weights.sum(axis=1)
    
    # Interp onto each mesh
    vals_mesh_n = idw_interpolate(tree_n, vals_n, mesh_north)
    vals_mesh_s = idw_interpolate(tree_s, vals_s, mesh_south)

    # inverse project mesh points back to lon/lat
    lon_n, lat_n = transformer_north(mesh_north[:, 0], mesh_north[:, 1], inverse=True)
    lon_s, lat_s = transformer_south(mesh_south[:, 0], mesh_south[:, 1], inverse=True)

    # Convert from range [-180, 180] to range [0, 360]
    lon_n = (lon_n + 360) % 360
    lon_s = (lon_s + 360) % 360

    print(f"Interpolation of {data_type} returning df of length {len(lon_n) + len(lon_s)} from meshgrid of {len(mesh_north) + len(mesh_south)}")

    df_interp = pd.DataFrame({
        'Longitude': np.concatenate([lon_n, lon_s]),
        'Latitude': np.concatenate([lat_n, lat_s]),
        data_type: np.concatenate([vals_mesh_n, vals_mesh_s])
    })

    if elev is not None:
        elev_n = idw_interpolate(tree_n, elev[north_mask], mesh_north)
        elev_s = idw_interpolate(tree_s, elev[south_mask], mesh_south)
        df_interp['Elevation'] = np.concatenate([elev_n, elev_s])

    return df_interp


def interpolate_diviner(df, mesh_north, mesh_south, MOON_RADIUS_M=1737.4e3):
    # Define AEQD CRS for polar projections
    transformer_north = Proj(proj='aeqd', lat_0=90, lon_0=0, R=MOON_RADIUS_M, always_xy=True)
    transformer_south = Proj(proj='aeqd', lat_0=-90, lon_0=0, R=MOON_RADIUS_M, always_xy=True)

    lons = df['Longitude'].values
    lats = df['Latitude'].values

    north_mask = lats >= 75
    south_mask = lats <= -75

    if north_mask.sum() == 0 or south_mask.sum() == 0:
        raise ValueError("No data points found in the specified latitude ranges.")

    xs_n, ys_n = transformer_north(lons[north_mask], lats[north_mask])
    xs_s, ys_s = transformer_south(lons[south_mask], lats[south_mask])

    values_n = df.loc[north_mask, 'Diviner'].values
    values_s = df.loc[south_mask, 'Diviner'].values

    x_grid_north, y_grid_north = mesh_north[:, 0], mesh_north[:, 1]
    x_grid_south, y_grid_south = mesh_south[:, 0], mesh_south[:, 1]

    points_n = np.column_stack((xs_n, ys_n))
    points_s = np.column_stack((xs_s, ys_s))

    grid_n = np.column_stack((x_grid_north, y_grid_north))
    grid_s = np.column_stack((x_grid_south, y_grid_south))

    interp_n = max_rad_interp(points_n, values_n, grid_n)
    interp_s = max_rad_interp(points_s, values_s, grid_s)

    lon_grid_north, lat_grid_north = transformer_north(x_grid_north, y_grid_north, inverse=True)
    lon_grid_south, lat_grid_south = transformer_south(x_grid_south, y_grid_south, inverse=True)

    # Convert from range [-180, 180] to range [0, 360]
    lon_grid_north = (lon_grid_north + 360) % 360
    lon_grid_south = (lon_grid_south + 360) % 360

    print(f"Interpolation of Diviner returning df of length {len(lon_grid_north) + len(lon_grid_south)} from meshgrid of {len(mesh_north) + len(mesh_south)}")

    interpolated_df = pd.DataFrame({
        'Longitude': np.concatenate([lon_grid_north, lon_grid_south]),
        'Latitude': np.concatenate([lat_grid_north, lat_grid_south]),
        'Diviner': np.concatenate([interp_n, interp_s])
    })

    return interpolated_df


def max_rad_interp(points, values, targets, radius_m=1000, fallback='nearest'):
    tree = cKDTree(points)
    idx_lists = tree.query_ball_point(targets, r=radius_m)   # Vectorised query for all targets

    interpolated = np.empty(len(targets), dtype=float)
    points_per_target = np.array([len(idxs) for idxs in idx_lists])

    for i, idxs in enumerate(idx_lists):
        if idxs:
            interpolated[i] = np.nanmax(values[idxs])   # Take max if possible
        elif fallback == 'nearest':
            _, idx = tree.query(targets[i], k=1)
            interpolated[i] = values[idx]
        else:
            interpolated[i] = np.nan    # Set to nan if no points
    return interpolated


def interpolate(data_dict, data_type, plot_save_path=None, debug=False):
    if not os.path.exists(data_dict['interp_dir']):
        print(f"Creating interp dir for {data_type}")
        os.mkdir(data_dict['interp_dir'])

    # if len([f for f in os.listdir(data_dict['interp_dir']) if f.endswith('.csv') and 'lon' in f]) == 12:
    #     print(f"Interpolated CSVs appear to exist for {data_type} data. Skipping interpolation.")
    #     return
    
    # If CSVs dont already exist, clear the directory
    clear_dir(data_dict['interp_dir'], dirs_only=False)

    csvs = sorted(os.listdir(data_dict['save_path']))
    meshes = generate_xy_mesh()
    save_path = data_dict['interp_dir']


    for (csv, (mesh_north, mesh_south)) in zip(csvs, meshes):
        df = pd.read_csv(f"{data_dict['save_path']}/{csv}")
        print(f"{data_type} mesh sizes - n: {len(mesh_north)}, s: {len(mesh_south)}")
        if data_type == 'Diviner':
            # div_frac = 0.25
            # weights = df[data_type].values / df[data_type].sum()
            # df = df.sample(frac=div_frac, weights=weights, random_state=42)    # Resample Diviner data, weighted to higher values
            # print(f"NOTE: Diviner resampled for {div_frac:.2%} of data across all csvs due to abundance of data. Weighted to higher values"); sys.stdout.flush()

            print(f"Interpolating Diviner for {csv}"); sys.stdout.flush()
            interpolated_df = interpolate_diviner(df, mesh_north, mesh_south)
            print()
        else:
            elev = df['Elevation'].values if data_type == 'M3' else None
            interpolated_df = interpolate_csv(df, mesh_north, mesh_south, data_type, elev)
    
        print(f"Interpolated {data_type} df contains {len(interpolated_df)} points")
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
        print(f"Plot (hopefully) saved to {plot_save_path}")

    if debug:
        print(f"\nInterpolated {data_type} df:")
        print(interpolated_df.describe())
        print(interpolated_df.head())

    del interpolated_df, df_list
