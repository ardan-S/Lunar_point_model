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
RES_DEG = (GRID_RES / MOON_RADIUS) * (180 / np.pi)  # Resolution in degrees


def load_and_prepare_data(args, sample=1.0):
    df = load_csvs_parallel(args.upload_dir, args.n_workers)
    print(f"Loaded {len(df)} points")
    df = df[['Longitude', 'Latitude', 'Elevation']]
    df['PSR'] = 1  # Assume all points are permanently shadowed regions (PSRs)

    df = df.dropna(subset=['Longitude', 'Latitude', 'Elevation'])
    df = df[np.isfinite(df['Longitude']) & np.isfinite(df['Latitude']) & np.isfinite(df['Elevation'])]

    pts_to_take = int(sample * len(df))
    df = df.iloc[:pts_to_take] if len(df) > pts_to_take else df
    print(f"Computing horizon elevation for {len(df)} points ({sample*100}% of the data)")
    print(df.head())

    # min_lon, min_lat = df['Longitude'].min(), df['Latitude'].min()

    # df['lon_idx'] = ((df['Longitude'] - min_lon) / RES_DEG).round().astype(int)
    # df['lat_idx'] = ((df['Latitude'] - min_lat) / RES_DEG).round().astype(int)
    # print(f"Number of idx points: {len(df['lon_idx'])} (lon), {len(df['lat_idx'])} (lat)")
    # print(f"Number of unique lon idcs: {df['lon_idx'].nunique()}, Number of unique lat idcs: {df['lat_idx'].nunique()}")

    # max_lon_idx, max_lat_idx = df['lon_idx'].max(), df['lat_idx'].max()
    # print(f"Max lon index: {max_lon_idx}, Max lat index: {max_lat_idx}")

    # df_grouped = df.groupby(['lon_idx', 'lat_idx'], as_index=False).mean()
    df_grouped = df.groupby(['Longitude', 'Latitude'], as_index=False).mean()
    df_grouped = df_grouped.sort_values(by=['Latitude', 'Longitude']).reset_index(drop=True)
    df_grouped['Longitude'] = df_grouped['Longitude'].round(6)
    df_grouped['Latitude'] = df_grouped['Latitude'].round(6)

    unique_lats = np.sort(df_grouped['Latitude'].unique())
    unique_lons = np.sort(df_grouped['Longitude'].unique())
    n_lats = len(unique_lats)
    n_lons = len(unique_lons)

    elevation_grid = df_grouped['Elevation'].to_numpy()
    elevation_grid = elevation_grid.reshape((n_lats, n_lons))

    print(f"Number of unique latitudes: {n_lats}, Number of unique longitudes: {n_lons}")
    print(f"Grid shape: {elevation_grid.shape}")
    print(f"Grid size: {elevation_grid.size}")
    print(f"Number of non-nans in elev_grid: {np.count_nonzero(~np.isnan(elevation_grid))} ({np.count_nonzero(~np.isnan(elevation_grid)) / elevation_grid.size:.2%})")
    print(f"Number of nans in elev_grid: {np.isnan(elevation_grid).sum()} ({np.isnan(elevation_grid).sum() / elevation_grid.size:.2%})")

    return elevation_grid


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
