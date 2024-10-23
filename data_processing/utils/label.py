import pandas as pd # type: ignore
from functools import reduce
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import DBSCAN  # type: ignore
import numpy as np
from shapely.geometry import Polygon, MultiPoint    # type: ignore
from shapely.ops import transform   # type: ignore
from functools import partial
from pyproj import CRS, Transformer
import sys

from data_processing.utils.utils import save_by_lon_range, plot_labeled_polar_data, load_csvs_parallel


def combine(*dirs, n_workers=None):
    all_dfs = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_dir = {executor.submit(load_csvs_parallel, dir, n_workers): dir for dir in dirs}
        
        for future in as_completed(future_to_dir):
            df = future.result()
            all_dfs.append(df)

    combined_df = reduce(lambda left, right: pd.merge(left, right, on=['Longitude', 'Latitude'], how='outer'), all_dfs)

    print(f"Dataframes combined. Shape: {combined_df.shape}\n")
    if combined_df.shape[1] != 7:
        print(f"ERROR - Combined dataframe columns: {combined_df.columns}")
        raise ValueError("Combined dataframe does not have the expected number of columns")

    return combined_df


def label(df, dataset_dict, plot_dir, lola_area_thresh=(3), m3_area_thresh=(2.4)):
    combined_save_path = dataset_dict['Combined']['combined_save_path']
    plot_save_path = plot_dir
    df[['Label', 'Diviner label', 'LOLA label', 'M3 label', 'MiniRF label']] = 0

    # DIVINER
    df.loc[df['Diviner'] <=110, 'Label'] += 2
    df.loc[df['Diviner'] <= 110, 'Diviner label'] += 2

    # LOLA
    lola_thresh = df['LOLA'].mean() - 2 * df['LOLA'].std()
    df = apply_area_label(df, 'LOLA', lola_thresh, lola_area_thresh, 'below')

    plot_labeled_polar_data(
        df=df,
        variable='LOLA',
        label_column='LOLA label',
        save_path=os.path.join(plot_save_path, 'LOLA_label_plot.png')
    )
    print()

    # M3
    m3_thresh = df['M3'].mean() + 2 * df['M3'].std()
    df = apply_area_label(df, 'M3', m3_thresh, m3_area_thresh, 'above')

    plot_labeled_polar_data(
        df=df,
        variable='M3',
        label_column='M3 label',
        save_path=os.path.join(plot_save_path, 'M3_label_plot.png')
    )
    print()

    # MiniRF
    MRF_thresh = df['MiniRF'].mean() + 2 * df['MiniRF'].std()
    df.loc[df['MiniRF'] > MRF_thresh, 'Label'] += 1
    df.loc[df['MiniRF'] > MRF_thresh, 'MiniRF label'] += 1

    def print_label_counts(df, label_name):
        counts = df[label_name].value_counts(normalize=True) * 100
        for label in [0, 1, 2]:  # Ensure that we always include 0, 1, and 2 even if they are missing
            percentage = counts.get(label, 0.0)  # Use .get() to handle cases where a label is missing
            print(f"{label_name} - {label}: {percentage:.2f}%")

    print("Label counts:")
    print_label_counts(df, 'Diviner label')
    print_label_counts(df, 'LOLA label')
    print_label_counts(df, 'M3 label')
    print_label_counts(df, 'MiniRF label')

    df.drop(columns=['Diviner label', 'LOLA label', 'M3 label', 'MiniRF label'], inplace=True)

    save_by_lon_range(df, combined_save_path)


def apply_area_label(df, data_type, threshold, area_thresh, direction):
    df = df.copy()
    
    assert 'Label' in df.columns, "Label column not found in dataframe"
    assert f'{data_type} label' in df.columns, f"{data_type} label column not found in dataframe"
    
    # Step 1: Filter the DataFrame based on the threshold and direction
    if direction == 'below':
        condition = df[data_type] < threshold
    elif direction == 'above':
        condition = df[data_type] > threshold
    else:
        raise ValueError("Invalid direction. Use 'above' or 'below'.")

    df.loc[condition, 'Label'] += 1
    df.loc[condition, f'{data_type} label'] += 1

    df_filtered = df[condition].copy()

    if df_filtered.empty:
        return df
    
    # Step 2: Convert coordinates to radians
    df_filtered['longitude_rad'] = np.deg2rad(df_filtered['Longitude'])
    df_filtered['latitude_rad'] = np.deg2rad(df_filtered['Latitude'])
    coords_rad = df_filtered[['latitude_rad', 'longitude_rad']].values

    # Step 3: Perform clustering using DBSCAN with Haversine metric
    # The Moon's mean radius in kilometers
    moon_radius_km = 1737.4

    # The 'eps' parameter in radians (e.g., 0.5 km radius)
    eps_km = 0.75    # ADJUST
    eps_rad = eps_km / moon_radius_km

    # Perform clustering
    clustering = DBSCAN(eps=eps_rad, min_samples=5, metric='haversine').fit(coords_rad)
    df_filtered['cluster_label'] = clustering.labels_

    # Step 4: Calculate area for each cluster and update labels
    cluster_labels = set(clustering.labels_)
    cluster_labels.discard(-1)  # Exclude noise points labeled as -1

    moon_radius_m = moon_radius_km * 1000  # Convert to meters

    crs_lunar_geo = CRS.from_proj4(
        f"+proj=longlat +a={moon_radius_m} +b={moon_radius_m} +no_defs +type=crs +celestial_body=Moon"
    )

    clusters_at_1 = 0
    clusters_at_2 = 0

    for cluster_label in cluster_labels:
        sys.stdout.flush()
        cluster_points = df_filtered[df_filtered['cluster_label'] == cluster_label]
        if len(cluster_points) < 3:
            print(f"Skipping cluster {cluster_label} with less than 3 points")
            continue

        # Extract longitude and latitude values
        cluster_coords_lon = cluster_points['Longitude'].values
        cluster_coords_lat = cluster_points['Latitude'].values

        # Project the cluster coordinates to planar coordinates
        # Use an Azimuthal Equidistant projection centered on the cluster centroid
        centroid_lon = cluster_coords_lon.mean()
        centroid_lat = cluster_coords_lat.mean()

        proj_str = (
            f"+proj=aeqd +lat_0={centroid_lat} +lon_0={centroid_lon} "
            f"+a={moon_radius_m} +b={moon_radius_m} +units=m +no_defs +type=crs +celestial_body=Moon"
        )
        crs_lunar_proj = CRS.from_proj4(proj_str)
        transformer = Transformer.from_crs(crs_lunar_geo, crs_lunar_proj, always_xy=True)

        # Project the coordinates
        x_proj, y_proj = transformer.transform(cluster_coords_lon, cluster_coords_lat)
        cluster_coords_proj = np.column_stack((x_proj, y_proj))
        if not np.isfinite(cluster_coords_proj).all():
            print(f"Invalid projected coordinates for cluster {cluster_label}. Skipping...")
            continue

        unique_coords_proj = np.unique(cluster_coords_proj, axis=0)
        if len(unique_coords_proj) < 3:
            print(f"Skipping cluster {cluster_label} with less than 3 unique points")
            continue

        # Create a Polygon from the projected cluster points
        multipoint = MultiPoint(unique_coords_proj)
        polygon = multipoint.convex_hull
        # polygon = Polygon(cluster_coords_proj)

        if not polygon.is_valid:
            print(f"Invalid polygon for cluster {cluster_label}. Skipping...")
            continue
        if polygon.area == 0 or polygon.is_empty:
            print(f"Polygon area is zero for cluster {cluster_label}. Skipping...")
            continue
        if polygon.geom_type != 'Polygon':
            print(f"Invalid geometry type for cluster {cluster_label}. Skipping...")
            continue

        # Calculate the area
        area_m2 = polygon.area  # Area in square meters

        # Determine label increment based on area threshold
        area_km2 = area_m2 / 1e6  # Convert to square kilometers
        if area_km2 < area_thresh:
            label_increment = 1
            clusters_at_1 += 1
        else:
            label_increment = 2
            clusters_at_2 += 1

        # Update labels for points in this cluster
        df_filtered.loc[cluster_points.index, 'Label'] += label_increment - 1
        df_filtered.loc[cluster_points.index, f'{data_type} label'] += label_increment - 1
        # print(f"Cluster {cluster_label:03d} - Area: {area_km2:.2f} km^2, Label increment: {label_increment}, Points: {len(cluster_points):03d}")

    # Update the original DataFrame with the new labels
    df.update(df_filtered[['Label', f'{data_type} label']])
    print(f"{data_type} - Clusters at 1: {clusters_at_1}, Clusters at 2: {clusters_at_2}, Total clusters: {len(cluster_labels)}")

    return df
