import numpy as np
import pandas as pd # type: ignore
from scipy.interpolate import Rbf, griddata
import os
import sys
from scipy.stats import binned_statistic_2d
from scipy.spatial import cKDTree
from pyproj import Proj

from data_processing.utils.utils import save_by_lon_range, load_every_nth_line, generate_xy_mesh, plot_polar
from data_processing.download_data import clear_dir


def interpolate_csv(df, mesh_north, mesh_south, data_type,  elev=None, MOON_RADIUS_M=1737.4e3, block=500_000):
    # Define AEQD CRS for polar projections
    transformer_north = Proj(proj='aeqd', lat_0=90, lon_0=0, R=MOON_RADIUS_M, always_xy=True)
    transformer_south = Proj(proj='aeqd', lat_0=-90, lon_0=0, R=MOON_RADIUS_M, always_xy=True)

    lons = df['Longitude'].values.astype(np.float32, copy=False)
    lats = df['Latitude'].values.astype(np.float32, copy=False)

    north_mask = lats >= 75
    south_mask = lats <= -75

    if not north_mask.any() or not south_mask.any():
        raise ValueError("No data points found in the specified latitude ranges.")

    xs_n, ys_n = transformer_north(lons[north_mask], lats[north_mask])
    xs_s, ys_s = transformer_south(lons[south_mask], lats[south_mask])

    vals_n = df.loc[north_mask, data_type].values.astype(np.float32, copy=False)
    vals_s = df.loc[south_mask, data_type].values.astype(np.float32, copy=False)

    tree_n = cKDTree(np.column_stack((xs_n, ys_n)), balanced_tree=True, compact_nodes=True)
    tree_s = cKDTree(np.column_stack((xs_s, ys_s)), balanced_tree=True, compact_nodes=True)

    del xs_n, ys_n, xs_s, ys_s, lons, lats

    def _idw_one(tree, values, targets, k, power=2):
        """1-nearest-neighbour IDW – minimal allocations."""
        dists, idxs = tree.query(targets, k=k, workers=-1)

        if k == 1:
            return values[idxs]                            # in-place
    
        dists[dists == 0] = 1e-12
        weights = 1.0 / np.power(dists, power)
        weighted_vals = weights * values[idxs]           # broadcast to (N, k)
        return weighted_vals.sum(axis=1) / weights.sum(axis=1)

    def _idw_chunked(tree, values, targets, block=200_000, power=2, k=1):
        """Run _idw_one() in slices to cap peak RAM."""
        out = np.empty(targets.shape[0], dtype=values.dtype)
        for start in range(0, targets.shape[0], block):
            end = start + block
            out[start:end] = _idw_one(tree, values, targets[start:end], power=power, k=k)
        return out
    
    # Interp onto each mesh
    vals_mesh_n = _idw_chunked(tree_n, vals_n, mesh_north, block=block)
    vals_mesh_s = _idw_chunked(tree_s, vals_s, mesh_south, block=block)


    # inverse project mesh points back to lon/lat
    lon_n, lat_n = transformer_north(mesh_north[:, 0], mesh_north[:, 1], inverse=True)
    lon_s, lat_s = transformer_south(mesh_south[:, 0], mesh_south[:, 1], inverse=True)

    lon = np.concatenate((lon_n, lon_s)).astype(np.float32, copy=False)
    lat = np.concatenate((lat_n, lat_s)).astype(np.float32, copy=False)
    val = np.concatenate((vals_mesh_n, vals_mesh_s))        # already float32

    # Convert from range [-180, 180] to range [0, 360]
    lon = (lon + 360) % 360

    data = {'Longitude': lon, 'Latitude': lat, data_type: val}

    if elev is not None:
        elev_n = _idw_chunked(tree_n, elev[north_mask], mesh_north, block=block)
        elev_s = _idw_chunked(tree_s, elev[south_mask], mesh_south, block=block)
        data['Elevation'] = np.concatenate([elev_n, elev_s])

    return pd.DataFrame(data, copy=False)


def interpolate_diviner(df, mesh_north, mesh_south, MOON_RADIUS_M=1737.4e3, block=500_000):
    # Define AEQD CRS for polar projections
    transformer_north = Proj(proj='aeqd', lat_0=90, lon_0=0, R=MOON_RADIUS_M, always_xy=True)
    transformer_south = Proj(proj='aeqd', lat_0=-90, lon_0=0, R=MOON_RADIUS_M, always_xy=True)

    lons = df['Longitude'].values.astype(np.float32, copy=False)
    lats = df['Latitude'].values.astype(np.float32, copy=False)

    north_mask = lats >= 75
    south_mask = lats <= -75

    if not north_mask.any() or not south_mask.any():
        raise ValueError("No data points found in the specified latitude ranges.")

    xs_n, ys_n = transformer_north(lons[north_mask], lats[north_mask])
    xs_s, ys_s = transformer_south(lons[south_mask], lats[south_mask])

    values_n = df.loc[north_mask, 'Diviner'].values.astype(np.float32, copy=False)
    values_s = df.loc[south_mask, 'Diviner'].values.astype(np.float32, copy=False)

    points_n = np.column_stack((xs_n, ys_n))
    points_s = np.column_stack((xs_s, ys_s))

    interp_n = max_rad_interp(points_n, values_n, mesh_north, block=block)
    interp_s = max_rad_interp(points_s, values_s, mesh_south, block=block)

    lon_grid_north, lat_grid_north = transformer_north(mesh_north[:, 0], mesh_north[:, 1], inverse=True)
    lon_grid_south, lat_grid_south = transformer_south(mesh_south[:, 0], mesh_south[:, 1], inverse=True)

    lon = np.concatenate((lon_grid_north, lon_grid_south)).astype(np.float32, copy=False)
    lat = np.concatenate((lat_grid_north, lat_grid_south)).astype(np.float32, copy=False)
    val = np.concatenate((interp_n, interp_s))        # already float32

    lon = (lon + 360) % 360

    return pd.DataFrame({
        'Longitude': lon,
        'Latitude': lat,
        'Diviner': val
    }, copy=False)


def max_rad_interp(points, values, targets, block, radius_m=1000, fallback='nearest'):
    tree = cKDTree(points, balanced_tree=True, compact_nodes=True)
    del points

    out = np.empty(targets.shape[0], dtype=values.dtype)

    n_tot    = 0            # sum of neighbour counts
    n_pts    = 0            # number of target points processed
    n_min    = np.inf       # current minimum
    n_max    = -np.inf      # current maximum

    for s in range(0, targets.shape[0], block):
        e = s + block
        slab = targets[s:e]

        idx_lists = tree.query_ball_point(slab, r=radius_m)   # Vectorised query for all targets

        for i, idxs in enumerate(idx_lists):
            n = len(idxs)
            n_tot += n
            n_pts += 1
            if n < n_min:
                n_min = n
            if n > n_max:
                n_max = n
            if idxs:
                out[s + i] = np.nanmax(values[idxs])  # Use max value within radius
            elif fallback == 'nearest':
                _, idx = tree.query(slab[i], k=1)
                out[s + i] = values[idx]
            else:
                out[s + i] = np.nan

    print(f"radius_m: {radius_m}, n_min: {n_min}, n_max: {n_max}, mean: {n_tot/n_pts}")
    return out


def interpolate(data_dict, data_type, plot_save_path=None):
    if not os.path.exists(data_dict['interp_dir']):
        print(f"Creating interp dir for {data_type}")
        os.mkdir(data_dict['interp_dir'])

    if len([f for f in os.listdir(data_dict['interp_dir']) if f.endswith('.csv') and 'lon' in f]) == 12:
        print(f"Interpolated CSVs appear to exist for {data_type} data. Skipping interpolation.")
        return
    
    # If CSVs dont already exist, clear the directory
    clear_dir(data_dict['interp_dir'], dirs_only=False)

    csvs = sorted(os.listdir(data_dict['save_path']))
    meshes = generate_xy_mesh()
    save_path = data_dict['interp_dir']


    for (csv, (mesh_north, mesh_south)) in zip(csvs, meshes):
        # Cast lons/lats to float32 for memory efficiency
        mesh_north = mesh_north.astype(np.float32)
        mesh_south = mesh_south.astype(np.float32)

        df = pd.read_csv(f"{data_dict['save_path']}/{csv}")
        if data_type == 'Diviner':
            interpolated_df = interpolate_diviner(df, mesh_north, mesh_south)

        elif data_type == 'M3':
            elev = df['Elevation'].values
            interpolated_df = interpolate_csv(df, mesh_north, mesh_south, data_type, elev)

        else:
            interpolated_df = interpolate_csv(df, mesh_north, mesh_south, data_type)
    
        save_by_lon_range(interpolated_df, save_path)

    df_list = []

    for file in os.listdir(save_path):
        if file.endswith('.csv'):
            df = load_every_nth_line(f"{save_path}/{file}", n=1)
            df_list.append(df)

    interpolated_df = pd.concat(df_list, ignore_index=True)

    if plot_save_path:
        plot_polar(interpolated_df, data_type, frac=0.1, save_path=plot_save_path, name_add='interp')


    print(f"\nInterpolated {data_type} df:")
    print(interpolated_df.describe())
    print(interpolated_df.head())
