# import numpy as np
# import pandas as pd
# import os
# import sys
# import argparse
# import time
# import dask.dataframe as dd
# from scipy.spatial import KDTree

# sys.path.append(os.path.abspath('.'))
# from utils_dask import generate_mesh

# def calculate_distance(point1, point2):
#     return np.sqrt(np.sum((point1 - point2)**2, axis=-1))


# def calculate_weights(data, target_point, power=2, threshold_distance=None):
#     target_point = np.array(target_point).flatten()
#     points = data[['Longitude', 'Latitude']].values
#     distances = calculate_distance(points, target_point)

#     if threshold_distance is not None:
#         mask = distances <= threshold_distance
#         filtered_distances = distances[mask]
#     else:
#         filtered_distances = distances

#     if np.any(filtered_distances < 1e-6):
#         weights = np.zeros_like(filtered_distances)
#         weights[np.argmin(filtered_distances)] = 1   # Assign weight of 1 to the exact match
#         print(f'Exact match found at {target_point}')
#         sys.stdout.flush()
#     else:
#         weights = 1 / filtered_distances**power
#     return weights

# def normalise_weights(weights):
#     return weights / np.sum(weights)

# def interpolate(data_type, data, mesh, power=2):
#     interpolated_values = []
#     mesh_points = np.vstack((mesh[0], mesh[1]))
#     tree = KDTree(data[['Longitude', 'Latitude']].values)
    
#     iter = 0
#     offset_sum = 0.0
#     new_time = time.time()

#     for point in mesh_points:
#         offset = 0.1

#         while True:
#             indices = tree.query_ball_point(point, offset)
#             filtered_data = data.iloc[indices]

#             if len(filtered_data) > 0:
#                 break
#             offset += 0.1

#         weights = calculate_weights(filtered_data, point, power)
#         normalised_weights = normalise_weights(weights)

#         data_values = filtered_data[data_type].values
#         interpolated_value = np.sum(data_values * normalised_weights)   # Determine the value at the new points
#         interpolated_values.append([point[0], point[1], interpolated_value])
    
#         iter += 1
#         offset_sum += offset
#         if iter % 100_000 == 0:
#             diff = time.time() - new_time
#             ave_offset = offset_sum / iter
#             print(f'Interpolated point ({point[0]:.4f}, {point[1]:.4f}) (iter: {iter:,} of {len(mesh_points)}) after {diff:.2f} secs with new average offset: {ave_offset:.2f}')
#             sys.stdout.flush()
#             new_time = time.time()

#     return pd.DataFrame(interpolated_values, columns=['Longitude', 'Latitude', data_type])


# def main(data_type, data_path, output_csv_path):
#     start_time = time.time()

#     # Validate input
#     accepted_data_types = ['Diviner', 'LOLA', 'M3', 'MiniRF']
#     if data_type not in accepted_data_types:
#         raise ValueError(f"Invalid data type '{data_type}'. Accepted values are: {accepted_data_types}")

#     if data_path == output_csv_path:
#         raise ValueError('Input and output directories cannot be the same')

#     # CSV, mesh and directories
#     all_csvs = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])
#     filepaths = [os.path.join(data_path, f) for f in all_csvs]
#     meshes = generate_mesh()
#     os.makedirs(output_csv_path, exist_ok=True)

#     # Interpolate
#     for i in range(len(filepaths)):
#         data = pd.read_csv(filepaths[i])
#         mesh = meshes[i]
#         print(f'Start interpolating CSV {all_csvs[i]} after {(time.time() - start_time)/60:.2f} mins')
#         sys.stdout.flush()
#         interpolated_data = interpolate(data_type, data, mesh)
#         print(f'Finished interpolating CSV {all_csvs[i]} after {(time.time() - start_time)/60:.2f} mins. Saving...')
#         output_file = os.path.join(output_csv_path, all_csvs[i])
#         interpolated_data.to_csv(output_file, index=False)
#         sys.stdout.flush()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Interpolation script arguments')
#     parser.add_argument('--label_type', type=str, default='Diviner', help='Label type (Diviner, LOLA, M3, Mini-RF)')
#     parser.add_argument('--data_path', type=str, default='../data/M3/M3_CSVs', help='Path to the directory containing the CSV files')
#     parser.add_argument('--output_csv_path', type=str, default='../data/M3/M3_interp', help='Path to the output CSV file')
#     args = parser.parse_args()
#     main(args.label_type, args.data_path, args.output_csv_path)



