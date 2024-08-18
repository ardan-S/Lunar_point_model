import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from scipy.ndimage import label, binary_dilation
from skimage.morphology import dilation, square
# from skimage.measure import label

import argparse
from sklearn.cluster import MiniBatchKMeans as KMeans
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce
import time
import dask.dataframe as dd
import math


import os
import sys

sys.path.append(os.path.abspath('.'))
start_time = None

def get_lon_range(filepath):
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    lon_start = int(parts[-2])
    return lon_start // 30

def combine_data(data_list, column_names):
    if len(data_list) != len(column_names):
        raise ValueError("Number of column names must match the number of dataframes")

    # Initialize the combined_data with the first DataFrame
    combined_data = data_list[0][['Longitude', 'Latitude', column_names[0]]].copy()
    combined_data.columns = ['Longitude', 'Latitude', column_names[0]]
    
    def merge_dfs(left, right):
        return pd.merge(left, right, on=['Longitude', 'Latitude'])
    
    dfs_to_merge = [data[['Longitude', 'Latitude', value_column]].copy() 
                    for data, value_column in zip(data_list[1:], column_names[1:])]
    
    combined_data = reduce(merge_dfs, [combined_data] + dfs_to_merge)

    return combined_data


def apply_labels(data):
    global start_time
    # batch_size = 10_000   # Batch size used for KMeans clustering

    def calculate_psr_perc(lats1=[-90, -75], lats2=[75,90], lunar_rad=1737.4, NP_PSR_area=12866, SP_PSR_area=16055):
        # Source for NP and SP areas - https://doi.org/10.1016/j.icarus.2010.10.030
        lat1, lat2 = lats1
        area1 = 2 * math.pi * (lunar_rad ** 2) * (math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)))
        
        lat1, lat2 = lats2
        area2 = 2 * math.pi * (lunar_rad ** 2) * (math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)))
        
        total_polar_area = abs(area1) + abs(area2)
        PSR_perc = (NP_PSR_area + SP_PSR_area) / total_polar_area
        
        return PSR_perc

    PSR_perc = calculate_psr_perc()
    print(f"Percentage of the poles covered by PSRs: {PSR_perc * 100:.2f}%")

    data[['Label', 'Diviner label', 'LOLA label', 'M3 label', 'MiniRF label']] = 0


    """ Add 2 to the label if Diviner is below 110"""
    data.loc[data['Diviner'] <= 110, 'Label'] += 2
    data.loc[data['Diviner'] <= 110, 'Diviner label'] += 2
    if start_time is not None:
        print(f"Diviner labelled after {(time.time() - start_time) / 60 :.2f} mins")
        sys.stdout.flush()

    """ Add 1 to the label if LOLA is above threshold.
    Add another 1 to the label if the area of the region is larger than 3 km^2"""
    # lola_val_thresh = data['LOLA'].std()
    # data_reshaped = data['LOLA'].values.reshape(-1, 1)
    # kmeans = KMeans(n_clusters=2, random_state=0, batch_size=batch_size).fit(data_reshaped)
    # lola_val_thresh = np.mean(kmeans.cluster_centers_.flatten())
    lola_val_thresh = data['LOLA'].quantile(1-PSR_perc)
    print(f"LOLA threshold: {lola_val_thresh}")
    sys.stdout.flush()
    lola_area_thresh = 3 * (1000**2)
    data = label_based_on_area(data, 'LOLA', lola_val_thresh, lola_area_thresh)
    if start_time is not None:
        print(f"LOLA labelled after {(time.time() - start_time) / 60 :.2f} mins")
        sys.stdout.flush()

    """ Add 1 to the label if M3 is above threshold.
    Add another 1 to the label if the area of the region is larger than 2.4 km^2"""
    # M3_val_thresh = data['M3'].std()
    # data_reshaped = data['M3'].values.reshape(-1, 1)
    # kmeans = KMeans(n_clusters=2, random_state=0, batch_size=batch_size).fit(data_reshaped)
    # M3_val_thresh = np.mean(kmeans.cluster_centers_.flatten())
    M3_val_thresh = data['M3'].quantile(1-PSR_perc)
    print(f"M3 threshold: {M3_val_thresh}")
    M3_area_thresh = 2.4 * (1000**2)
    data = label_based_on_area(data, 'M3', M3_val_thresh, M3_area_thresh)
    if start_time is not None:
        print(f"M3 labelled after {(time.time() - start_time) / 60 :.2f} mins")
        sys.stdout.flush()

    """ Add 1 to the label if MiniRF is above threshold"""
    # data_reshaped = data['MiniRF'].values.reshape(-1, 1)
    # kmeans = KMeans(n_clusters=2, random_state=0, batch_size=batch_size).fit(data_reshaped)
    # MiniRF_val_thresh = np.mean(kmeans.cluster_centers_.flatten())
    MiniRF_val_thresh = data['MiniRF'].quantile(1-PSR_perc)
    print(f"MiniRF threshold: {MiniRF_val_thresh}")
    data.loc[data['MiniRF'] > MiniRF_val_thresh, 'Label'] += 1
    data.loc[data['MiniRF'] > MiniRF_val_thresh, 'MiniRF label'] += 1
    if start_time is not None:
        print(f"MiniRF labelled after {(time.time() - start_time) / 60 :.2f} mins")
        sys.stdout.flush()

    return data


def label_based_on_area(data, column, threshold, area_threshold, img_save_path=None):

    grid_size = 240  # grid size in meters
    latitudes = data['Latitude'].unique()
    longitudes = data['Longitude'].unique()

    # 2D array
    binary_array = np.zeros((len(latitudes), len(longitudes)), dtype=int)

    # Create a mapping of coordinates to indices
    lat_to_index = {lat: i for i, lat in enumerate(latitudes)}
    lon_to_index = {lon: j for j, lon in enumerate(longitudes)}

    mask = data[column] > threshold # A mask for the points that satisfy the threshold
    # Get the indices of the points that satisfy the threshold and set the corresponding indices in the binary array to 1
    indices = data.loc[mask, ['Latitude', 'Longitude']].apply(lambda row: (lat_to_index[row['Latitude']], lon_to_index[row['Longitude']]), axis=1)
    for idx in indices:
        binary_array[idx] = 1


    dilated_array = binary_dilation(binary_array, square(5))
    dilated_labeled_array, num_features = label(dilated_array, structure=square(3))

    print(f"Number of features for {column} above threshold: {num_features} out of {len(indices)} points above threshold")

    latitude_indices, longitude_indices = np.where(dilated_labeled_array > 0)
    latitudes_feature = np.array(latitudes)[latitude_indices]
    longitudes_feature = np.array(longitudes)[longitude_indices]
    feature_labels = dilated_labeled_array[dilated_labeled_array > 0]

    feature_areas = np.bincount(feature_labels)[1:] * (grid_size ** 2)
    sorted_features = np.argsort(feature_areas)[::-1] + 1  # +1 to adjust for feature label indexing

    processed_points = set()

    for feature in sorted_features:
        label_value = 1 if feature_areas[feature - 1] < area_threshold else 2   # Determine score of all points in feature
        mask = feature_labels == feature    # Mask for the current feature

        current_pairs = set(zip(latitudes_feature[mask], longitudes_feature[mask])) # Define lat lon pairs for the current feature
        unprocessed_pairs = current_pairs - processed_points    # Remove the already processed points from the current feature
    
        for current_point in unprocessed_pairs:
            lat, lon = current_point
        
            if (lat, lon) in processed_points:
                continue

            data.loc[
                (data['Latitude'] == lat) & (data['Longitude'] == lon),
                ['Label', f'{column} label']
            ] += label_value

            processed_points.add((lat, lon))


    # Plotting the binary array in polar projection
    def plot_polar_binary_array(binary_array, latitudes, longitudes, pole, save_path=f'{column}_polar.png'):
        if pole == 'north':
            latitudes_polar = 90 - latitudes
        else:
            latitudes_polar = 90 + latitudes
    
        theta, r = np.meshgrid(np.deg2rad(longitudes), latitudes_polar)
        
        sample_size = int(len(theta.flatten()) * 0.01)
        sample_idx = random.sample(range(len(theta.flatten())), sample_size)

        theta_sample = theta.flatten()[sample_idx]
        r_sample = r.flatten()[sample_idx]
        binary_array_sample = binary_array.flatten()[sample_idx]
        
    
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
        ax.scatter(theta_sample, r_sample, c=binary_array_sample, cmap='binary_r', s=1)

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f'{column} Distribution - {pole.capitalize()} Pole')
        plt.savefig(save_path)
        plt.close(fig)

    # Plot for North Pole
    if img_save_path:
        np_save_path = os.path.join(img_save_path, column, 'north_polar_binary.png')
        north_latitudes = latitudes[latitudes >= 0]
        north_binary_array = binary_array[np.isin(latitudes, north_latitudes)]
        plot_polar_binary_array(north_binary_array, north_latitudes, longitudes, 'north', np_save_path)

        # Plot for South Pole
        sp_save_path = os.path.join(img_save_path, column, 'south_polar_binary.png')
        south_latitudes = latitudes[latitudes < 0]
        south_binary_array = binary_array[np.isin(latitudes, south_latitudes)]
        plot_polar_binary_array(south_binary_array, south_latitudes, longitudes, 'south', sp_save_path)

    return data


def process_data(lon_range, data_list, column_names, output_path):
    global start_time
    global workers_per_range
    if len(data_list) == 4:

        """Create new features that capture additional information or interactions between existing features.
        DISTANCE FROM POLE, RATIO OF 2 SETS, SOME SETS SQUARED, ELEVATION?"""

        global start_time
        print(f"Combining data for lon_range {lon_range} after {(time.time() - start_time) / 60 :.2f} mins")
        sys.stdout.flush()
        combined_data = combine_data(data_list, column_names)

        print(f"Labeling data for lon_range {lon_range} after {(time.time() - start_time) / 60 :.2f} mins")
        sys.stdout.flush()

        # labeled_data = apply_labels(combined_data)

        chunks = np.array_split(combined_data, workers_per_range)
        labeled_data_chunks = []
        with ProcessPoolExecutor(max_workers=workers_per_range) as executor:
            futures = [executor.submit(apply_labels, chunk) for chunk in chunks]
            for future in as_completed(futures):
                labeled_data_chunks.append(future.result())

        # Concatenate all labeled data chunks
        labeled_data = pd.concat(labeled_data_chunks, ignore_index=True)

        print(f"Labeled data for lon_range {lon_range}:")
        print(labeled_data.head())
        print(f"Highest labels for lon_range {lon_range}:")
        print("Total label counts")
        print(labeled_data['Label'].value_counts())
        print("\nDiviner label counts")
        print(labeled_data['Diviner label'].value_counts())
        print("\nLOLA label counts")
        print(labeled_data['LOLA label'].value_counts())
        print("\nM3 label counts")
        print(labeled_data['M3 label'].value_counts())
        print("\nMiniRF label counts")
        print(labeled_data['MiniRF label'].value_counts())
        sys.stdout.flush()

        # Remove the 4 individual label columns after details were printed
        labeled_data = labeled_data.drop(columns=['Diviner label', 'LOLA label', 'M3 label', 'MiniRF label'])

        output_file = f'combined_{lon_range * 30:03d}-{(lon_range + 1) * 30:03d}.csv'
        os.makedirs(output_path, exist_ok=True)

        labeled_data.to_csv(os.path.join(output_path, output_file), index=False)
        
        print(f'Saved {output_file} to {output_path} after {(time.time() - start_time) / 60 :.2f} mins\n')
        sys.stdout.flush()


def main(n_workers, dataset1, dataset2, dataset3, dataset4, output_path):
    print(f"Labelling data with {n_workers} workers")

    global start_time
    global workers_per_range
    start_time = time.time()
    workers_per_range = n_workers // 12
    all_data = {i: [] for i in range(12)}
    data_paths = [dataset1, dataset2, dataset3, dataset4]

    # Read all the data and store it in a dictionary based on the longitude range
    for data_path in data_paths:
        data_path = data_path[0]    # Unpack the list
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"Directory {data_path} not found")

        all_csvs = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        filepaths = [os.path.join(data_path, f) for f in all_csvs]

        for file in filepaths:
            data = pd.read_csv(file)
            # data = data.sample(frac=0.05, random_state=42)      # !!!!! SAMPLE 5% OF DATA, for debugging
            # print('!! Only 5% of data is being used !!')        # Gentle reminder while debugging
            lon_range = get_lon_range(file)
            all_data[lon_range].append(data)

    print(f"Finished reading data from {len(data_paths)} directories after {(time.time() - start_time) / 60 :.2f} mins")
    sys.stdout.flush()

    # Combine files and create the labeled file for each longitude range
    column_names = ['Diviner', 'LOLA', 'M3', 'MiniRF']
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_data, lon_range, data_list, column_names, output_path)
            for lon_range, data_list in all_data.items()
        ]
        for future in futures:
            future.result()  # wait for all futures to complete

    print(f"Finished processing data after {(time.time() - start_time) / 60 :.2f} mins")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label script arguments')
    parser.add_argument('--n_workers', type=int, help='Number of workers')
    parser.add_argument('--dataset1', nargs='+', help='First dataset path')
    parser.add_argument('--dataset2', nargs='+', help='Second dataset path')
    parser.add_argument('--dataset3', nargs='+', help='Third dataset path')
    parser.add_argument('--dataset4', nargs='+', help='Fourth dataset path')
    parser.add_argument('--output_path', type=str, help='Path to the output directory')
    args = parser.parse_args()

    main(args.n_workers, args.dataset1, args.dataset2, args.dataset3, args.dataset4, args.output_path)
