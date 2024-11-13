from astropy.coordinates import get_sun, AltAz, ITRS, CartesianRepresentation
from astropy.time import Time
import astropy.units as u
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from multiprocessing import Pool
import h5py
import os
from pyproj import Proj
import sys
import dask.dataframe as dd
import dask.delayed as delayed

from utils.utils import load_csvs_parallel

MOON_RADIUS = 1737.4 * 1000  # Convert to meters
GRID_RES = 240  # Resolution of the grid in meters
RES_DEG = (GRID_RES / MOON_RADIUS) * (180 / np.pi)  # Resolution in degrees


def load_and_prepare_data(args):
    # Load data, keep desired cols
    df = load_csvs_parallel(args.upload_dir, args.n_workers)
    print(f"Loaded {len(df)} points")
    df = df[['Longitude', 'Latitude', 'Elevation']]
    df['PSR'] = 1  # Assume all points are permanently shadowed regions (PSRs)
    df.reset_index(drop=True, inplace=True)

    assert np.isfinite(df['Longitude']).all() and np.isfinite(df['Latitude']).all() and np.isfinite(df['Elevation']).all(), "Data contains nans"
    assert len(df) > 0, "No data loaded"

    # Group lat, lon, and compute mean elevation
    df_grouped = df.groupby(['Longitude', 'Latitude'], as_index=False).mean()
    df_grouped = df_grouped.sort_values(by=['Latitude', 'Longitude']).reset_index(drop=True)
    df_grouped['Longitude'] = df_grouped['Longitude'].round(6)
    df_grouped['Latitude'] = df_grouped['Latitude'].round(6)

    df_north = df_grouped[df_grouped['Latitude'] > 0].reset_index(drop=True).copy()
    df_south = df_grouped[df_grouped['Latitude'] < 0].reset_index(drop=True).copy()


    def prepare_polar_data(df, pole):
        df = df.copy()
        df['r'] = 90 - df['Latitude'] if (pole == 'north') else 90 + df['Latitude']
        df['Longitude_rounded'] = (df['Longitude']*2).round() / 2
        df['Longitude_rounded'] = df['Longitude_rounded'].replace(360.0, 0.0)

        assert all((df['Longitude_rounded'] % 0.5) == 0), "Longitude values are not multiples of 0.5"
        df['theta'] = np.deg2rad(df['Longitude_rounded'])
        # Value count for rounded longitudes
        print(f"Unique longitudes for {pole} Pole: {df['Longitude_rounded'].nunique()}")
        print(f"{df['Longitude_rounded'].value_counts()}")
        return df
    
    lon_round_range = np.arange(0, 179.5, 0.5)

    df_north = prepare_polar_data(df_north, 'north')
    df_south = prepare_polar_data(df_south, 'south')

    for lon in lon_round_range:
        assert lon in df_north['Longitude_rounded'].values, f"Longitude {lon} not found in North"
        assert lon in df_south['Longitude_rounded'].values, f"Longitude {lon} not found in South"

        df_temp 



    # # Create a grid from the data
    # unique_lats_n = np.sort(df_north['Latitude'].unique())
    # unique_lons_n = np.sort(df_north['Longitude'].unique())
    # unique_lats_s = np.sort(df_south['Latitude'].unique())
    # unique_lons_s = np.sort(df_south['Longitude'].unique())

    # assert len(unique_lats_n) == len(unique_lats_s), "Number of unique latitudes differ between North and South"
    # assert len(unique_lons_n) == len(unique_lons_s), "Number of unique longitudes differ between North and South"
    
    # n_lats = len(unique_lats_n)
    # n_lons = len(unique_lons_n)

    # assert len(df_north) == n_lats * n_lons, "Number of points in North does not match grid size"
    # assert len(df_south) == n_lats * n_lons, "Number of points in South does not match grid size"

    # elevation_grid_n = df_north['Elevation'].to_numpy().reshape((n_lats, n_lons))
    # elevation_grid_s = df_south['Elevation'].to_numpy().reshape((n_lats, n_lons))

    # elevation_grid = np.stack([elevation_grid_n, elevation_grid_s], axis=-1)

    # print(f"Number of unique latitudes: {n_lats}, Number of unique longitudes: {n_lons}")
    # print(f"Grid shape: {elevation_grid.shape}")
    # print(f"Grid size: {elevation_grid.size}")
    # print(f"NP df length: {len(df_north)}, SP df length: {len(df_south)}")
    # sys.stdout.flush()
    # assert np.isnan(elevation_grid).sum() == 0, "Elevation grid contains nans"

    # return elevation_grid, df_north, df_south


def compute_horizon_for_point(args):
    i, j, elevation_grid = args
    horizon_angles = np.zeros(720)
    current_elevation = elevation_grid[i, j]
    height, width = elevation_grid.shape
    max_distance = max(height, width)
    
    # Precompute azimuth angles
    azimuths = np.deg2rad(np.arange(0, 360, 0.5))
    sin_azimuths = np.sin(azimuths)
    cos_azimuths = np.cos(azimuths)
    
    for idx, (sin_theta, cos_theta) in enumerate(zip(sin_azimuths, cos_azimuths)):
        max_angle = -np.inf
        # Traverse along the ray
        for distance in range(1, max_distance):
            x = int(j + distance * cos_theta)
            y = int(i + distance * sin_theta)
            if 0 <= x < width and 0 <= y < height:
                target_elevation = elevation_grid[y, x] # Elevation of the target point - ENSURE THIS IS ALWAYS FOUND.
                elevation_angle = np.arctan2(target_elevation - current_elevation, distance * GRID_RES)
                if elevation_angle > max_angle:
                    max_angle = elevation_angle
                    horizon_angles[idx] = np.rad2deg(max_angle)
            else:
                break  # Out of bounds
    return (i, j, horizon_angles)


def compute_horizons_chunks(df, pole, args, n_splits):
    essential_cols = ['Longitude', 'Latitude', 'Elevation','Longitude_rounded', 'PSR']
    horizon_cols = [f'Horizon_{i/2}' for i in range(0, 720)]
    ddf = dd.from_pandas(df[essential_cols], npartitions=n_splits)
    print(f"Dask df with {ddf.npartitions} partitions created"); sys.stdout.flush()

    os.makedirs(args.save_dir, exist_ok=True)

    def process_partition(partition_df, partition_info=None):
        for col in horizon_cols:
            partition_df[col] = np.nan

        partition_df = compute_horizon_elevation_chunk(partition_df, args.n_workers, args.save_dir, pole)
        partition_idx = partition_info['number'] if partition_info else 0

        output_file = os.path.join(args.save_dir, f'partition_{partition_idx}.parquet')
        partition_df.to_parquet(output_file, index=False, engine='pyarrow')

        print(f"Processed partition {partition_idx+1}/{n_splits}"); sys.stdout.flush()

        return pd.DataFrame(columns=empty_meta.columns)
    
    empty_meta = pd.DataFrame(columns=essential_cols + horizon_cols)

    ddf.map_partitions(
        process_partition,
        meta=empty_meta,
        include_partition_info=True,
    ).compute()

    print(f"Horizon data computed for {pole} Pole"); sys.stdout.flush()

def compute_horizon_elevation_chunk(elevation_grid_chunk, n_workers, save_dir, pole):
    # os.makedirs(save_dir, exist_ok=True)

    # height, width = elevation_grid_chunk.shape
    # points = [(i, j, elevation_grid_chunk) for i in range(height) for j in range(width)]
    
    # with Pool(n_workers) as pool:
    #     results = pool.map(compute_horizon_for_point, points)
    
    # h5_file_path = os.path.join(save_dir, f'horizon_data_{pole}.h5')

    # # Write results to an HDF5 file
    # with h5py.File(h5_file_path, 'a') as h5f:
    #     for i, j, horizon_angles in results:
    #         dataset_name = f"{chunk_indices[0]+i}_{chunk_indices[1]+j}"
    #         h5f.create_dataset(dataset_name, data=horizon_angles, compression="gzip")
    pass


def divide_into_chunks(df, essential_cols, horizon_cols, n_splits):
    cols_per_chunk = len(horizon_cols) // n_splits
    for i in range(0, len(horizon_cols), cols_per_chunk):
        cols = essential_cols + horizon_cols[i:i+cols_per_chunk]
        yield df[cols]


def compute_sun_position(lats, lons, time_step, start_time):
    """
    Take a given point P (lat, lon) and time t. Find the Sun's position (azimuth, elevation) at P.
    """
    time = start_time + timedelta(hours=time_step)
    t = Time(time.isoformat())

    sun_azimuths = np.zeros(len(lats))
    sun_elevs = np.zeros(len(lats))

    for i, (lat, lon) in enumerate(zip(lats, lons)):
        x = MOON_RADIUS * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
        y = MOON_RADIUS * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
        z = MOON_RADIUS * np.sin(np.radians(lat))

        moon_loc = ITRS(CartesianRepresentation(x=x, y=y, z=z), obstime=t)
        sun_loc = get_sun(t).transform_to(AltAz(obstime=t, location=moon_loc))

        # sun_azimuths[i] = sun_loc.az.deg
        # sun_elevs[i] = sun_loc.alt.deg

    return sun_azimuths, sun_elevs
