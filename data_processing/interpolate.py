import numpy as np
import pandas as pd
import os
import sys
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial import KDTree
import signal
import gc
from multiprocessing import Value, Lock, Manager

sys.path.append(os.path.abspath('.'))
from utils_dask import generate_mesh

# Gracefully handle exits if walltime limit is reached
def handle_signal(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def calculate_normalised_weights(data, target_point, power=2, threshold_distance=None):
    target_point = np.array(target_point).flatten()
    points = data[['Longitude', 'Latitude']].values
    distances = np.linalg.norm(points - target_point, axis=1)

    if threshold_distance is not None:
        mask = distances <= threshold_distance
        filtered_distances = distances[mask]
    else:
        filtered_distances = distances

    if np.any(filtered_distances < 1e-3):   # Exact match
        weights = np.zeros_like(filtered_distances)
        weights[np.argmin(filtered_distances)] = 1   # Assign weight of 1 to the exact match
        sys.stdout.flush()
    else:
        weights = 1 / filtered_distances**power
    return weights / np.sum(weights)


def interpolate_point(data, tree, point, data_type, power=2):
    offset = 0.1
    max_offset = 10
    filtered_data = pd.DataFrame()

    while offset < max_offset:
        indices = tree.query_ball_point(point, offset)
        if indices:
            filtered_data = data.iloc[indices]
            if not filtered_data.empty:
                break

        offset += 0.1

    if filtered_data.empty:
        raise ValueError(f'No data found for point {point} within a max offset of {max_offset}')
    
    normalised_weights = calculate_normalised_weights(filtered_data, point, power)
    data_values = filtered_data[data_type].values
    interpolated_value = np.sum(data_values * normalised_weights)
    
    # Clean up
    del filtered_data
    del normalised_weights
    del data_values

    return [point[0], point[1], interpolated_value]


def interpolate(data_type, data, mesh, power=2):
    mesh_points = np.vstack((mesh[0], mesh[1]))
    tree = KDTree(data[['Longitude', 'Latitude']].values)
    interpolated_values = []

    iter = 0
    time_start = time.time()
    for point in mesh_points:
        interpolated_values.append(interpolate_point(data, tree, point, data_type, power))
        if iter % 1_000_000 == 0:
            print(f'Reached point ({point[0]:.4f}, {point[1]:.4f}) (iter: {iter:,} of {len(mesh_points):,}) after {(time.time() - time_start)/60:.2f} mins')
            sys.stdout.flush()
        iter += 1

    return pd.DataFrame(interpolated_values, columns=['Longitude', 'Latitude', data_type])


def process_file(file_index, data_type, data_path, output_csv_path, meshes):
    start_time = time.time()

    # CSV, mesh and directories
    all_csvs = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])
    filepaths = [os.path.join(data_path, f) for f in all_csvs]

    data = pd.read_csv(filepaths[file_index])
    mesh = meshes[file_index]
    gc.collect()

    print(f'Start interpolating CSV {all_csvs[file_index]} after {(time.time() - start_time)/60:.2f} mins')
    sys.stdout.flush()
    interpolated_data = interpolate(data_type, data, mesh)
    if data_type == 'M3':
        interp_elev_data = interpolate('Elevation', data, mesh)
        interpolated_data['Elevation'] = interp_elev_data['Elevation']
    print(f'Finished interpolating CSV {all_csvs[file_index]} after {(time.time() - start_time)/60:.2f} mins. Saving...')
    output_file = os.path.join(output_csv_path, all_csvs[file_index])
    interpolated_data.to_csv(output_file, index=False)
    sys.stdout.flush()
    gc.collect()


def main(data_type, data_path, output_csv_path, n_workers, filenum=0):
    start_time = time.time()

    # Validate input
    num_files = len([f for f in os.listdir(data_path) if f.endswith('.csv')])
    accepted_data_types = ['Diviner', 'LOLA', 'M3', 'MiniRF']
    if data_type not in accepted_data_types:
        raise ValueError(f"Invalid data type '{data_type}'. Accepted values are: {accepted_data_types}")
 
    if data_path == output_csv_path:
        raise ValueError('Input and output directories cannot be the same')

    # Generate mesh
    meshes = generate_mesh()
    os.makedirs(output_csv_path, exist_ok=True)

    # Parallel processing of files
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_file, i, data_type, data_path, output_csv_path, meshes) for i in range(num_files)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Interpolation failed: {e}")

    print(f'Finished all interpolations after {(time.time() - start_time)/60:.2f} mins')
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpolation script arguments')
    parser.add_argument('--data_type', type=str, default='Diviner', help='Label type (Diviner, LOLA, M3, Mini-RF)')
    parser.add_argument('--data_path', type=str, default='../data/M3/M3_CSVs', help='Path to the directory containing the CSV files')
    parser.add_argument('--output_csv_path', type=str, default='../data/M3/M3_interp', help='Path to the output CSV file')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--filenum', type=int, default=0, help='File number to interpolate')
    args = parser.parse_args()
    main(args.data_type, args.data_path, args.output_csv_path, args.n_workers, args.filenum)
