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

from utils.utils import load_csvs_parallel

MOON_RADIUS = 1737.4 * 1000  # Convert to meters
GRID_RES = 240  # Resolution of the grid in meters


def load_and_prepare_data(args, sample=1.0):
    df = load_csvs_parallel(args.upload_dir, args.n_workers)
    df.filter(items=['Longitude', 'Latitude', 'Elevation'])
    df = df.iloc[:int(sample * len(df))]
    print(f"Computing horizon elevation for {len(df)} points ({sample*100}% of the data)")
    df['PSR'] = 1  # Assume all points are permanently shadowed regions (PSRs)

    # Define polar stereographic projections for north and south poles
    proj_north = Proj(proj='stere', lat_0=90, lon_0=0, lat_ts=90, a=MOON_RADIUS, b=MOON_RADIUS, units='m')
    proj_south = Proj(proj='stere', lat_0=-90, lon_0=0, lat_ts=-90, a=MOON_RADIUS, b=MOON_RADIUS, units='m')

    # Convert latitude and longitude to x and y coordinates
    df_north = df[df['Latitude'] >= 0].copy()
    df_south = df[df['Latitude'] < 0].copy()

    print(f"Number of points: {len(df_north)} (North), {len(df_south)} (South), {len(df_north) + len(df_south)} (Total)")

    df_north['x'], df_north['y'] = proj_north(df_north['Longitude'].values, df_north['Latitude'].values)
    df_south['x'], df_south['y'] = proj_south(df_south['Longitude'].values, df_south['Latitude'].values)

    print(f"Number of x, y points: {len(df_north['x'])} (North), {len(df_south['x'])} (South), {len(df_north['x']) + len(df_south['x'])} (Total)")

    df = pd.concat([df_north, df_south], ignore_index=True)

    print(f"Number of points after projection: {len(df)}")

    initial_row_count = df.shape[0]
    df = df[np.isfinite(df['x']) & np.isfinite(df['y'])]
    final_row_count = df.shape[0]
    rows_removed = initial_row_count - final_row_count
    print(f"Removed {rows_removed} rows with NaN or Inf in 'x' or 'y' ({rows_removed/initial_row_count:.2%})")

    # Determine min_x and min_y to offset indices to start from zero
    min_x = df['x'].min()
    min_y = df['y'].min()
    grid_res = args.grid_resolution

    df['x_index'] = ((df['x'] - min_x) / grid_res).astype(int)
    df['y_index'] = ((df['y'] - min_y) / grid_res).astype(int)

    print(f"Number of idx points: {len(df['x_index'])} (x), {len(df['y_index'])} (y)")
    print(f"Number of unique x indices: {df['x_index'].nunique()}, Number of unique y indices: {df['y_index'].nunique()}")

    max_x_idx, max_y_idx = df['x_index'].max(), df['y_index'].max()

    print(f"Max x index: {max_x_idx}, Max y index: {max_y_idx}")

    elevation_grid = np.full((max_y_idx + 1, max_x_idx + 1), np.nan)
    print(f"Initialised grid shape: {elevation_grid.shape}")

    for _, row in df.iterrows():
        elevation_grid[row['y_index'].astype(int), row['x_index'].astype(int)] = row['Elevation']

    print(f"Grid shape: {elevation_grid.shape}")
    print(f"Grid size: {elevation_grid.size}")
    print(f"Number of non-nans in elev_grid: {np.count_nonzero(~np.isnan(elevation_grid))} ({np.count_nonzero(~np.isnan(elevation_grid)) / elevation_grid.size:.2%})")
    print(f"Number of nans in elev_grid: {np.isnan(elevation_grid).sum()} ({np.isnan(elevation_grid).sum() / elevation_grid.size:.2%})")

    return elevation_grid, min_x, min_y


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
                target_elevation = elevation_grid[y, x]
                elevation_angle = np.arctan2(target_elevation - current_elevation, distance * GRID_RES)
                if elevation_angle > max_angle:
                    max_angle = elevation_angle
                    horizon_angles[idx] = np.rad2deg(max_angle)
            else:
                break  # Out of bounds
    return (i, j, horizon_angles)


# def compute_horizon_elevation(elevation_grid, n_workers):
#     height, width = elevation_grid.shape
#     points = [(i, j, elevation_grid) for i in range(height) for j in range(width)]
    
#     with Pool(n_workers) as pool:
#         results = pool.map(compute_horizon_for_point, points)
    
#     # Organize results into a structured format
#     horizon_data = {}
#     for i, j, horizon_angles in results:
#         horizon_data[(i, j)] = horizon_angles
    
#     return horizon_data


def process_in_chunks(elevation_grid, args):
    chunks = divide_into_chunks(elevation_grid, chunk_size=100_000)
    print(f"\nNumber of chunks: {len(list(chunks))}")

    for chunk_indices, elevation_grid_chunk in chunks:
        compute_horizon_elevation_chunk(elevation_grid_chunk, chunk_indices, args.n_workers, args.save_dir)
        print(f"Processed chunk {chunk_indices} out of {elevation_grid.shape}")


def compute_horizon_elevation_chunk(elevation_grid_chunk, chunk_indices, n_workers, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    height, width = elevation_grid_chunk.shape
    points = [(i, j, elevation_grid_chunk) for i in range(height) for j in range(width)]
    
    with Pool(n_workers) as pool:
        results = pool.map(compute_horizon_for_point, points)
    
    h5_file_path = os.path.join(save_dir, 'horizon_data.h5')

    # Write results to an HDF5 file
    with h5py.File(h5_file_path, 'a') as h5f:
        for i, j, horizon_angles in results:
            dataset_name = f"{chunk_indices[0]+i}_{chunk_indices[1]+j}"
            h5f.create_dataset(dataset_name, data=horizon_angles, compression="gzip")


def divide_into_chunks(elevation_grid, chunk_size=100_000):
    # Divide the grid into manageable chunks
    height, width = elevation_grid.shape
    for i in range(0, height, chunk_size):
        for j in range(0, width, chunk_size):
            elevation_grid_chunk = elevation_grid[i:i+chunk_size, j:j+chunk_size]
            yield ((i, j), elevation_grid_chunk)


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
