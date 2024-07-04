import numpy as np
import pandas as pd
import os
import sys
import argparse

sys.path.append(os.path.abspath('.'))
from utils_dask import generate_mesh


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_weights(data, target_point, power=2):
    distances = np.array([calculate_distance((row['lat'], row['lon']), target_point) for _, row in data.iterrows()])
    if np.any(distances == 0):
        weights = np.zeros_like(distances)
        weights[np.argmin(distances)] = 1   # Assign weight of 1 to the exact match
    else:
        weights = 1 / distances**power
    return weights


def normalise_weights(weights):
    return weights / np.sum(weights)


def interpolate(label_type, data, grid_points, power=2):
    interpolated_values = []
    for point in grid_points:
        weights = calculate_weights(data, point, power)
        normalised_weights = normalise_weights(weights)
        interpolated_value = np.sum(data['value'] * normalised_weights)
        interpolated_values.append([point[0], point[1], interpolated_value])
    return pd.DataFrame(interpolated_values, columns=['Latitude', 'Longitude', label_type])


def main(label_type, data_path, output_csv_path):
    all_csvs = os.listdir(data_path)
    all_csvs = [f for f in all_csvs if f.endswith('.csv')]

    filepaths = [os.path.join(data_path, f) for f in all_csvs]

    # Interpolate
    for i in range(filepaths.size):
        data = pd.read_csv(filepaths[i])
        mesh = generate_mesh()[i]

        interpolated_data = interpolate(data, mesh)
        interpolated_data.to_csv(os.path.join(output_csv_path, all_csvs[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpolation script arguments')
    parser.add_argument('--data_path', type=str, default='./data/M3/M3_CSVs', help='Path to the directory containing the CSV files')
    parser.add_argument('--output_csv_path', type=str, default='data/M3/M3_interp', help='Path to the output CSV file')
    args = parser.parse_args()
    main(args.label_type, args.data_path, args.output_csv_path)
