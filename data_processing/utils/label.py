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

from data_processing.utils.utils import save_by_lon_range


def load_csv(directory, csv_file):
    df = pd.read_csv(os.path.join(directory, csv_file))
    return df if not df.isna().all().all() else None


def load_csvs_parallel(directory, n_workers):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    dfs = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_csv = {executor.submit(load_csv, directory, csv_file): csv_file for csv_file in csv_files}
        
        for future in as_completed(future_to_csv):
            df = future.result()
            if df is not None:
                dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def combine(*dirs, n_workers=None):
    all_dfs = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_dir = {executor.submit(load_csvs_parallel, dir, n_workers): dir for dir in dirs}
        
        for future in as_completed(future_to_dir):
            df = future.result()
            all_dfs.append(df)

    combined_df = reduce(lambda left, right: pd.merge(left, right, on=['Longitude', 'Latitude'], how='outer'), all_dfs)

    print(f"Combined dataframe shape: {combined_df.shape}")
    if combined_df.shape[1] != 7:
        print(f"Combined dataframe columns: {combined_df.columns}")
        raise ValueError("Combined dataframe does not have the expected number of columns")

    return combined_df


def label(df, combined_save_path, lola_area_thresh=(3*(1000**2)), m3_area_thresh=(2.4*(1000**2))):
    df[['Label', 'Diviner label', 'LOLA label', 'M3 label', 'MiniRF label']] = 0

    # DIVINER
    df.loc[df['Diviner'] <=110, 'Label'] += 2
    df.loc[df['Diviner'] <= 110, 'Diviner label'] += 2

    # LOLA
    lola_thresh = df['LOLA'].mean() - 2 * df['LOLA'].std()
    df = apply_area_label(df, 'LOLA', lola_thresh, lola_area_thresh, 'below')

    # M3
    m3_thresh = df['M3'].mean() + 2 * df['M3'].std()
    df = apply_area_label(df, 'M3', m3_thresh, m3_area_thresh, 'above')

    # MiniRF
    MRF_thresh = df['MiniRF'].mean() + 2 * df['MiniRF'].std()
    df.loc[df['MiniRF'] > MRF_thresh, 'Label'] += 1
    df.loc[df['MiniRF'] > MRF_thresh, 'MiniRF label'] += 1

    save_by_lon_range(df, combined_save_path)


def apply_area_label(df, data_type, threshold, area_thresh, direction):
    df = df.copy()

    if 'Label' not in df.columns:
        raise ValueError("Label column not found in dataframe")
    if f'{data_type} label' not in df.columns:
        raise ValueError(f"{data_type} label column not found in dataframe")
    
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
    eps_km = 0.5    # ADJUST
    eps_rad = eps_km / moon_radius_km

    # Perform clustering
    clustering = DBSCAN(eps=eps_rad, min_samples=5, metric='haversine').fit(coords_rad)
    df_filtered['cluster_label'] = clustering.labels_

    # Step 4: Calculate area for each cluster and update labels
    cluster_labels = set(clustering.labels_)
    cluster_labels.discard(-1)  # Exclude noise points labeled as -1

    moon_radius_m = moon_radius_km * 1000  # Convert to meters

    for cluster_label in cluster_labels:
        cluster_points = df_filtered[df_filtered['cluster_label'] == cluster_label]
        if len(cluster_points) < 3:
            print(f"Skipping cluster {cluster_label} with less than 3 points")
            continue

        # Create a Polygon from the cluster points
        cluster_coords = cluster_points[['Longitude', 'Latitude']].values
        polygon = Polygon(cluster_coords)

        if not polygon.is_valid:
            print(f"Invalid polygon for cluster {cluster_label}. Skipping...")
            continue
        if polygon.area == 0:
            print(f"Polygon area is zero for cluster {cluster_label}. Skipping...")
            continue

        # Project the polygon to a planar coordinate system for area calculation
        # We can use an Azimuthal Equidistant projection centered on the cluster centroid
        centroid_lon = cluster_points['Longitude'].mean()
        centroid_lat = cluster_points['Latitude'].mean()

        crs_lunar_geo = CRS.from_proj4(
            f"+proj=longlat +a={moon_radius_m} +b={moon_radius_m} +no_defs +type=crs +celestial_body=Moon"
        )
        proj_str = (
            f"+proj=aeqd +lat_0={centroid_lat} +lon_0={centroid_lon} "
            f"+R={moon_radius_m} +units=m +no_defs +type=crs +celestial_body=Moon"
        )
        crs_lunar_proj = CRS.from_proj4(proj_str)
        transformer = Transformer.from_crs(crs_lunar_geo, crs_lunar_proj, always_xy=True)

        # Transform the polygon to the projected coordinate system
        try:
            polygon_proj = transform(transformer.transform, polygon)
            area_m2 = polygon_proj.area
        except Exception as e:
            print(f"Error projecting polygon for cluster {cluster_label}: {e}")
            continue

        # Determine label increment based on area threshold
        area_km2 = area_m2 / 1e6  # Convert to square kilometers
        if area_km2 < area_thresh:
            label_increment = 1
        else:
            label_increment = 2

        # Update labels for points in this cluster
        df_filtered.loc[cluster_points.index, 'Label'] += label_increment - 1
        df_filtered.loc[cluster_points.index, f'{data_type} label'] += label_increment - 1
        print(f"Cluster {cluster_label:03d} - Area: {area_km2:.2f} km^2, Label increment: {label_increment}, Points: {len(cluster_points):03d}")

    # Update the original DataFrame with the new labels
    df.update(df_filtered[['Label', f'{data_type} label']])

    return df