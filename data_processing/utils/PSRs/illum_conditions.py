from astropy.coordinates import get_sun, AltAz, ITRS, CartesianRepresentation
from astropy.time import Time
from datetime import timedelta
import numpy as np
import pandas as pd
from multiprocessing import Pool
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import BallTree
import traceback
import tracemalloc

from utils.utils import load_csvs_parallel

MOON_RADIUS = 1737.4  # km
GRID_RES = 240  # Resolution of the grid in meters
RES_DEG = (GRID_RES / MOON_RADIUS) * (180 / np.pi)  # Resolution in degrees
DEG_TO_RAD = np.pi / 180
RAD_TO_DEG = 180 / np.pi
DELTA_LON = 1.0  # Tolerance for longitude difference in degrees
MAX_DISTANCE = 200  # Maximum distance to consider in kilometers (euclidian radius of circle at 75 deg lat is approx 900km)


def save_table_as_parquet(df, table_id, save_dir, chunk_size, partition_cols=None):
    os.makedirs(save_dir, exist_ok=True)

    for i, chunk in enumerate(range(0, len(df), chunk_size)):
        chunk_df = df.iloc[chunk:chunk + chunk_size]
        table = pa.Table.from_pandas(chunk_df)
        file_path = os.path.join(save_dir, f"table_{table_id}_part_{i}.parquet")
        
        pq.write_table(table, file_path, compression='snappy', row_group_size=chunk_size)


def load_and_prepare_data(args):
    df = load_csvs_parallel(args.upload_dir, args.n_workers)
    print(f"Loaded {len(df)} points")
    df = df[['Longitude', 'Latitude', 'Elevation']]
    df.reset_index(drop=True, inplace=True)

    assert np.isfinite(df['Longitude']).all() and np.isfinite(df['Latitude']).all() and np.isfinite(df['Elevation']).all(), "Data contains nans"
    assert len(df) > 0, "No data loaded"

    # Group lat, lon, and compute mean elevation
    df_grouped = df.groupby(['Longitude', 'Latitude'], as_index=False).mean()
    df_grouped = pd.DataFrame(df_grouped) 
    df_grouped = df_grouped.sort_values(by=['Latitude', 'Longitude']).reset_index(drop=True)
    df_grouped['Longitude'] = df_grouped['Longitude'].round(6)
    df_grouped['Latitude'] = df_grouped['Latitude'].round(6)

    df_north = df_grouped[df_grouped['Latitude'] > 0].reset_index(drop=True).copy()
    df_south = df_grouped[df_grouped['Latitude'] < 0].reset_index(drop=True).copy()

    nfiles = 8

    def prepare_polar_data(df, pole):
        df = df.copy()
        df['Longitude_rounded'] = (df['Longitude']*2).round() / 2
        df['Longitude_rounded'] = df['Longitude_rounded'].replace(360.0, 0.0)
        df['Adjusted_Longitude'] = df['Longitude_rounded'].apply(lambda x: x - 180 if x >= 180 else x)
    
        df = df.sort_values(by='Adjusted_Longitude').reset_index(drop=True)

        unique_lons = df['Adjusted_Longitude'].unique()
        unique_lons.sort()
        lon_chunks = np.array_split(unique_lons, nfiles)    # Each chunk contains congtiguous range of longitudes

        chunks = []
        for lon_chunk in lon_chunks:
            chunks.append(df[df['Adjusted_Longitude'].isin(lon_chunk)])

        assert all((df['Longitude_rounded'] % 0.5) == 0), "Longitude values are not multiples of 0.5"
        assert df['Longitude_rounded'].nunique() == 720, f"Unique longitudes for {pole} Pole is not 720, it is {df['Longitude_rounded'].nunique()}"
        
        return chunks
    
    north_chunks = prepare_polar_data(df_north, 'north')
    south_chunks = prepare_polar_data(df_south, 'south')

    save_dir = args.parquet_save_dir
    lon_round_range = np.arange(0, 179.5, 0.5)

    expected_files = len(lon_round_range) * 2 * nfiles
    print(f"Expected number of files: {expected_files}")
    if len(os.listdir(save_dir)) == expected_files:
        print("Data already saved, skipping preparation")
        return df_north, df_south
    
    lst = []
    cnt = 0

    for region, chunks in [('north', north_chunks), ('south', south_chunks)]:   # Loop through north and south poles
        for i, chunk in enumerate(chunks):                                      # Loop through each chunk at the pole
            for lon in lon_round_range:                                         # Loop through each longitude in each chunk
                if lon in chunk['Longitude_rounded'].values:
                    head_1 = str(lon)
                    head_2 = str(lon + 180)

                    cols = ['Latitude', 'Elevation', 'Longitude_rounded', head_1, head_2]
                    df_temp = pd.DataFrame(columns=cols)
                    df_temp['Latitude'] = chunk['Latitude']
                    df_temp['Elevation'] = chunk['Elevation']
                    df_temp['Longitude_rounded'] = chunk['Longitude_rounded']
                    df_temp[head_1] = np.nan
                    df_temp[head_2] = np.nan

                    reg = 'n' if region == 'north' else 's'
                    id = f"{lon:03.1f}_{reg}_{i}".replace('.', '_')
                    chunk_save_path = os.path.join(save_dir, f"{id}.parquet")
                    df_temp.to_parquet(chunk_save_path, index=False)

                    del df_temp

                lst.append(lon)
                cnt += 1

    print(f"Completed {cnt} iterations")
    print("Unique longitudes:")
    for lon in sorted(lst):
        print(lon)
    print("Data prepared and saved")
    sys.stdout.flush()

    return df_north, df_south


def compute_horizons_chunks(args):
    pq_files = list(Path(args.parquet_save_dir).glob('*.parquet'))
    assert pq_files, "No data found in the parquet directory"
    pq_files = pq_files[:1]  # For testing
    print("PROCESSING ONLY ONE FILE FOR TESTING")

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [
            executor.submit(compute_pq_file, file_path, args)
            for file_path in pq_files
        ]

        # Wait for all futures to complete and handle exceptions
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {future}: {e}")
                print("Traceback:")
                traceback.print_exc()


def compute_pq_file(file_path, args):
    # Read the data
    df = pq.read_table(file_path).to_pandas()
    print(f"Read in {file_path}")
    print(df.head())
    print(f"Longitude counts:\n{df['Longitude_rounded'].value_counts()}")

    MAX_DISTANCE_RAD = MAX_DISTANCE / MOON_RADIUS  # Convert distance to angular distance in radians
    DELTA_LON_RAD = DELTA_LON * DEG_TO_RAD

    df['lon_rad'] = df['Longitude_rounded'] * DEG_TO_RAD
    df['lat_rad'] = df['Latitude'] * DEG_TO_RAD
    col_1 = df.columns[3]
    col_2 = df.columns[4]
    assert float(col_2) - float(col_1) == 180, "Columns are not 180 degrees apart"

    print(f"Computing directions {col_1} and {col_2}"); sys.stdout.flush()
    df['lon_dir1_rad'] = float(col_1) * DEG_TO_RAD
    df['lon_dir2_rad'] = float(col_2) * DEG_TO_RAD

    # Precompute ECEF coordinates
    df['x'] = (MOON_RADIUS + df['Elevation']) * np.cos(df['lat_rad']) * np.cos(df['lon_rad'])
    df['y'] = (MOON_RADIUS + df['Elevation']) * np.cos(df['lat_rad']) * np.sin(df['lon_rad'])
    df['z'] = (MOON_RADIUS + df['Elevation']) * np.sin(df['lat_rad'])

    # Initialize results
    df[col_1] = np.nan
    df[col_2] = np.nan

    # Prepare data for BallTree
    data = np.vstack((df['lat_rad'].values, df['lon_rad'].values)).T
    ball_tree = BallTree(data, metric='haversine')

    # Coordinates
    coords = df[['x', 'y', 'z']].values
    U_norms = np.linalg.norm(coords, axis=1)
    U_units = coords / U_norms[:, np.newaxis]  # Shape (N, 3)

    directions = [df['lon_dir1_rad'].values, df['lon_dir2_rad'].values]

    print("Starting computation"); sys.stdout.flush()

    for i in range(len(df)):    # Process each point A[i] in the df
        if i == 5:
            break
        print()
        A = coords[i]
        U_unit = U_units[i]
        indices_array = ball_tree.query_radius(data[i:i+1], r=MAX_DISTANCE_RAD) # Find all points within MAX_DISTANCE_RAD
        B_indices = indices_array[0]    # Indices of points within MAX_DISTANCE_RAD (just taking it out of an array here)
        print(f"\nNumber of neighbours found: {len(B_indices)}")
        print(f"Number of unique lons in neighbours: {len(np.unique(df['lon_rad'].values[B_indices]))}\n")

        print(f"[1]"); sys.stdout.flush()

        # Exclude self
        B_indices = B_indices[B_indices != i]
        if B_indices.size == 0:
            print(f"No points found for {i}")
            continue

        lon_B = df['lon_rad'].values[B_indices]

        # Compute the difference in longitude between current point (A) and its neighbors (B)
        lon_diff1 = np.abs(lon_B - df['lon_dir1_rad'].values[i]) # abs(lon_B - lon_A)
        lon_diff1 = np.minimum(lon_diff1, 2 * np.pi - lon_diff1)  # Wrap around 360°
        print(f"Range of lon diff 1: {np.min(lon_diff1) * RAD_TO_DEG} to {np.max(lon_diff1) * RAD_TO_DEG}")

        lon_diff2 = np.abs(lon_B - df['lon_dir2_rad'].values[i]) # abs(lon_B - lon_A)
        lon_diff2 = np.minimum(lon_diff2, 2 * np.pi - lon_diff2)  # Wrap around 360°
        print(f"Range of lon diff 2: {np.min(lon_diff2) * RAD_TO_DEG} to {np.max(lon_diff2) * RAD_TO_DEG}")

        # print(f"Lon rad values (B indices): {df['lon_rad'].values[B_indices]}")
        # print(f"Lon rad values (A): {lon_dir_rad[i]}")
        # print(f"Range of lon diffs: {np.min(lon_diff) * RAD_TO_DEG} - {np.max(lon_diff) * RAD_TO_DEG}")
        # lon_diff_deg = lon_diff * RAD_TO_DEG
        # unique_values, counts = np.unique(lon_diff_deg, return_counts=True)
        # print(f"Unique values: {unique_values}")
        # print(f"Delta lon rad: {DELTA_LON_RAD}")

        # Only consider points within a certain longitude difference
        lon_mask1 = lon_diff1 <= DELTA_LON_RAD
        lon_mask2 = lon_diff2 <= DELTA_LON_RAD

        # # Apply mask
        # B_indices = B_indices[lon_mask]
        # if B_indices.size == 0:
        #     print(f"No points found for {i} after lon mask")
        #     continue

        # # Compute elevation angles
        # B_candidates = coords[B_indices]
        # V = B_candidates - A
        # V_norms = np.linalg.norm(V, axis=1)
        # cos_theta = np.dot(V, U_unit) / V_norms
        # theta = np.arcsin(cos_theta) * RAD_TO_DEG  # Convert to degrees

        # max_theta = np.max(theta)

        # # Store result
        # if dir_idx == 0:
        #     df.at[i, col_1] = max_theta
        # else:
        #     df.at[i, col_2] = max_theta

       # Process direction 1
        B_indices1 = B_indices[lon_mask1]
        if B_indices1.size > 0:
            B_candidates1 = coords[B_indices1]
            V1 = B_candidates1 - A
            V_norms1 = np.linalg.norm(V1, axis=1)
            cos_theta1 = np.dot(V1, U_unit) / V_norms1
            theta1 = np.arcsin(cos_theta1) * RAD_TO_DEG  # Convert to degrees
            max_theta1 = np.max(theta1)
            df.at[i, col_1] = max_theta1
            print(f"Processed point {A} for lons {col_1} with horizon angle {max_theta1}")
        else:
            print(f"No points found for {i} after lon mask 1")

        # Process direction 2
        B_indices2 = B_indices[lon_mask2]
        if B_indices2.size > 0:
            B_candidates2 = coords[B_indices2]
            V2 = B_candidates2 - A
            V_norms2 = np.linalg.norm(V2, axis=1)
            cos_theta2 = np.dot(V2, U_unit) / V_norms2
            theta2 = np.arcsin(cos_theta2) * RAD_TO_DEG  # Convert to degrees
            max_theta2 = np.max(theta2)
            df.at[i, col_2] = max_theta2
            print(f"Processed point {A} for lons {col_2} with horizon angle {max_theta2}")
        else:
            print(f"No points found for {i} after lon mask 2")

        print(f"Considered A (converted to lat/lon): {df['Latitude'].values[i]}, {df['Longitude_rounded'].values[i]}")
        print(f"[2]"); sys.stdout.flush()

        if i == 1:
            print(f"Cols in df: {df.columns}")

    # return df
    # table = pa.Table.from_pandas(df)    
    # pq.write_table(table, file_path, compression='snappy')
    # del df
