import os
import pandas as pd
import numpy as np
from scipy.ndimage import label
import random

def main(data_paths):
    all_data = {i: [] for i in range(12)}

    # Read all the data and store it in a dictionary based on the longitude range
    for data_path in data_paths:
        all_csvs = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        filepaths = [os.path.join(data_path, f) for f in all_csvs]
        print(f"Reading {len(filepaths)} files from {data_path}")

        for file in filepaths:
            data = pd.read_csv(file)
            lon_range = get_lon_range(file)
            print(f"Appending file {file} to lon_range {lon_range}")
            all_data[lon_range].append(data)

    # Combine files and create the labeled file for each longitude range
    for lon_range, data_list in all_data.items():
        if len(data_list) == 4:
            print(f"Combining data for lon_range {lon_range}")
            combined_data = combine_data(data_list)
            combined_data['Label'] = 0
            labeled_data = apply_labels(combined_data)
            print(f"Labeled data for lon_range {lon_range}")
            output_file = f'combined_{lon_range * 30:03d}-{(lon_range + 1) * 30:03d}.csv'
            labeled_data.to_csv(output_file, index=False)
            print(f'Saved {output_file}')


def get_lon_range(filepath):
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    lon_start = int(parts[-2])
    return lon_start // 30

def combine_data(data_list):
    combined_data = data_list[0]
    combined_data = combined_data.rename(columns={'Value': 'Value1'})
    for i, data in enumerate(data_list[1:], start=2):
        data = data.rename(columns={'Value': f'Value{i}'})
        combined_data = combined_data.merge(data[['Longitude', 'Latitude', f'Value{i}']], on=['Longitude', 'Latitude'])
    return combined_data

def apply_labels(data):
    # Diviner
    """ Add 2 to the label if Diviner is below 110"""
    data.loc[data['Diviner'] <= 110, 'Label'] += 2

    # LOLA
    """ Add 1 to the label if LOLA is above 0.5.
    Add another 1 to the label if the area of the region is larger than 3 km^2"""
    lola_val_thresh = 0.5
    lola_area_thresh = 3 * (1000**2)
    data = label_based_on_area(data, 'LOLA', lola_val_thresh, lola_area_thresh)

    # M3
    """ Add 1 to the label if M3 is above 0.5.
    Add another 1 to the label if the area of the region is larger than 2.4 km^2"""
    M3_val_thresh = 0.5
    M3_area_thresh = 2.4 * (1000**2)
    data = label_based_on_area(data, 'M3', M3_val_thresh, M3_area_thresh)

    # MiniRF
    """ Add 1 to the label if MiniRF is above 2 standard deviations from the mean"""
    std_dev = data['MiniRF'].std()
    data.loc[data['MiniRF'] > 2*std_dev, 'Label'] += 1
    # quantile = data['MiniRF'].quantile(0.90)
    # data.loc[data['MiniRF'] < quantile, 'Label'] += 1
    return data


def label_based_on_area(data, column, threshold, area_threshold):
    grid_size = 240  # grid size in meters

    # Create a binary array where 1 indicates the value is above the threshold
    # This is effectively a 2D map where each node is either a 1 or 0 depending on if its above threshold
    binary_array = (data[column] > threshold).astype(int).values.reshape(len(set(data['Latitude'])), -1)

    # Label connected regions
    # Identify connected regions within the binary array
    labeled_array, num_features = label(binary_array)

    # Create a dataframe from the labeled array
    labeled_df = pd.DataFrame(labeled_array, columns=data['Longitude'].unique(), index=data['Latitude'].unique())

    # Iterate through each feature
    for feature in range(1, num_features + 1):
        # Find the coordinates, area and label for the current feature
        coords = np.column_stack(np.where(labeled_array == feature))
        area = len(coords) * (grid_size ** 2)
        label_value = 1 if area < area_threshold else 2

        # Apply the label to the corresponding points in the original dataframe
        for coord in coords:
            lat = labeled_df.index[coord[0]]
            lon = labeled_df.columns[coord[1]]
            data.loc[(data['Latitude'] == lat) & (data['Longitude'] == lon), 'Label'] += label_value

    return data