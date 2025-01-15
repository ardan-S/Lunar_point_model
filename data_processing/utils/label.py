import pandas as pd # type: ignore
from functools import reduce
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import DBSCAN, HDBSCAN  # type: ignore
import numpy as np
from shapely.geometry import Polygon, MultiPoint    # type: ignore
from shapely.ops import transform   # type: ignore
from functools import partial
from pyproj import CRS, Transformer
import sys

from data_processing.utils.utils import save_by_lon_range, plot_labeled_polar_data, load_csvs_parallel, plot_polar_data


def combine(*dirs, n_workers=None):
    all_dfs = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_dir = {executor.submit(load_csvs_parallel, dir, n_workers): dir for dir in dirs}
        
        for future in as_completed(future_to_dir):
            df = future.result()
            all_dfs.append(df)

    combined_df = reduce(lambda left, right: pd.merge(left, right, on=['Longitude', 'Latitude'], how='outer'), all_dfs)

    if combined_df.shape[1] != 7:
        print(f"ERROR - Combined dataframe columns: {combined_df.columns}")
        raise ValueError("Combined dataframe does not have the expected number of columns")

    print("Dataframes combined.\n")
    return combined_df


def label(df, dataset_dict, plot_dir, lola_area_thresh=3, m3_area_thresh=2.4, eps=0.75):

    combined_save_path = dataset_dict['Combined']['combined_save_path']

    if len([f for f in os.listdir(combined_save_path) if f.endswith('.csv') and 'lon' in f]) == 12:
        print(f"Combined CSVs appear to exist. Skipping label stage.")
        return

    plot_save_path = plot_dir
    df[['Label', 'Diviner label', 'LOLA label', 'M3 label', 'MiniRF label']] = 0
    df[['Label', 'Diviner label', 'LOLA label', 'M3 label', 'MiniRF label']] = df[['Label', 'Diviner label', 'LOLA label', 'M3 label', 'MiniRF label']].astype('int8')
    df[['Latitude', 'Longitude']] = df[['Latitude', 'Longitude']].astype('float32')

    # -------------------- DIVINER --------------------
    df.loc[df['Diviner'] <= 110, 'Label'] += 2
    df.loc[df['Diviner'] <= 110, 'Diviner label'] += 2

    plot_labeled_polar_data(df=df,variable='Diviner', label_column='Diviner label', save_path=os.path.join(plot_save_path, 'Diviner_label_plot.png'))

    # -------------------- LOLA --------------------
    lola_thresh_n = df.loc[df['Latitude'] >= 0, 'LOLA'].mean() + 2 * df.loc[df['Latitude'] >= 0, 'LOLA'].std()
    print(f"North pole LOLA threshold: {lola_thresh_n:.2f} ({df[df['LOLA'] >= lola_thresh_n].shape[0] / df.shape[0] :.2%} of data)")
    apply_area_label_2(df, 'LOLA', lola_thresh_n, lola_area_thresh, 'above', 'north', eps=eps)

    lola_thresh_s = df.loc[df['Latitude'] < 0, 'LOLA'].mean() + 2 * df.loc[df['Latitude'] < 0, 'LOLA'].std()
    print(f"South pole LOLA threshold: {lola_thresh_s:.2f} ({df[df['LOLA'] >= lola_thresh_s].shape[0] / df.shape[0] :.2%} of data)")
    apply_area_label_2(df, 'LOLA', lola_thresh_s, lola_area_thresh, 'above', 'south', eps=eps)

    plot_labeled_polar_data(df=df,variable='LOLA', label_column='LOLA label', save_path=os.path.join(plot_save_path, 'LOLA_label_plot.png'))

    # -------------------- M3 --------------------
    m3_thresh_n = df.loc[df['Latitude'] >= 0, 'M3'].mean() - 2 * df.loc[df['Latitude'] >= 0, 'M3'].std()
    print(f"North pole M3 threshold: {m3_thresh_n:.2f} ({df[df['M3'] <= m3_thresh_n].shape[0] / df.shape[0] :.2%} of data)")
    apply_area_label_2(df, 'M3', m3_thresh_n, m3_area_thresh, 'below', 'north', eps=eps)  # eps to 2.5

    m3_thresh_s = df.loc[df['Latitude'] < 0, 'M3'].mean() - 2 * df.loc[df['Latitude'] < 0, 'M3'].std()
    print(f"South pole M3 threshold: {m3_thresh_s:.2f} ({df[df['M3'] <= m3_thresh_s].shape[0] / df.shape[0] :.2%} of data)")
    apply_area_label_2(df, 'M3', m3_thresh_s, m3_area_thresh, 'below', 'south', eps=eps)  # eps to 2.5

    plot_labeled_polar_data(df=df,variable='M3', label_column='M3 label', save_path=os.path.join(plot_save_path, 'M3_label_plot.png'))

    # -------------------- MiniRF --------------------
    MRF_thresh = df['MiniRF'].mean() + 5.5 * df['MiniRF'].std()
    print(f"\nTaking the top {df[df['MiniRF'] >= MRF_thresh].shape[0] / df.shape[0] :.2%} of points from MiniRF. Threshold = {MRF_thresh:.2f}"); sys.stdout.flush()
    
    df.loc[df['MiniRF'] > MRF_thresh, 'Label'] += 1
    df.loc[df['MiniRF'] > MRF_thresh, 'MiniRF label'] += 1

    plot_labeled_polar_data(df=df,variable='MiniRF', label_column='MiniRF label', save_path=os.path.join(plot_save_path, 'MiniRF_label_plot.png'))

    # -------------------- Combine labels --------------------
    def print_label_counts(df, label_name):
        counts = df[label_name].value_counts(normalize=True) * 100
        for label in [0, 1, 2]:  # Ensure that we always include 0, 1, and 2 even if they are missing
            percentage = counts.get(label, 0.0)
            print(f"{label_name} - {label}: {percentage:.2f}%")

    # def print_label_counts(df, label_name):
    #     # Global percentage of 0
    #     global_counts = df[label_name].value_counts(normalize=True) * 100
    #     zero_percentage = global_counts.get(0, 0.0)
    #     print(f"{label_name} - 0 (Global): {zero_percentage:.2f}%")

    #     # Split the data into north and south poles
    #     north_pole_df = df[df['Latitude'] >= 0]
    #     south_pole_df = df[df['Latitude'] < 0]

    #     # Calculate label counts for north and south poles
    #     for label in [1, 2]:
    #         north_counts = north_pole_df[label_name].value_counts(normalize=True) * 100
    #         south_counts = south_pole_df[label_name].value_counts(normalize=True) * 100

    #         north_percentage = north_counts.get(label, 0.0)
    #         south_percentage = south_counts.get(label, 0.0)

    #         print(f"{label_name} - {label} (North Pole): {north_percentage:.2f}%")
    #         print(f"{label_name} - {label} (South Pole): {south_percentage:.2f}%")

    print()
    print("Label counts:")
    print_label_counts(df, 'Diviner label')
    print_label_counts(df, 'LOLA label')
    print_label_counts(df, 'M3 label')
    print_label_counts(df, 'MiniRF label')
    print()
    print("Label proportions after combining:")
    print(df.value_counts('Label', normalize=True) * 100)
    print()
    print(f"Total number of points: {df.shape[0]}")
    print()

    df.drop(columns=['Diviner label', 'LOLA label', 'M3 label', 'MiniRF label'], inplace=True)

    save_by_lon_range(df, combined_save_path)
    plot_polar_data(df, 'Label', graph_cat='labeled', save_path=plot_save_path)

    lbl = 3
    df_bin = df.copy()
    df_bin['Label'] = (df_bin['Label'] >= lbl).astype(int)

    print(f"Len of df: {len(df)}, dtype of label: {df['Label'].dtype}")
    print(f"Len of df_bin: {len(df_bin)}, dtype of label: {df_bin['Label'].dtype}")

    print(f"Percentage of labels >= {lbl}: {(df_bin['Label'] == 1).sum() / df_bin.shape[0]:.2%}")

    print()
    print(f"Binary label proportions (label={lbl}):")
    print(df_bin['Label'].value_counts(normalize=True) * 100)
    print()
    print(f"Total number of points: {df_bin.shape[0]}")

    plot_polar_data(df_bin, 'Label', frac=0.01, graph_cat='binary_orig', save_path=plot_save_path, dpi=400)


def apply_area_label_2(df, data_type, threshold, area_thresh, direction, pole, eps):
    assert 'Label' in df.columns, "Label column not found in dataframe"
    assert f'{data_type} label' in df.columns, f"{data_type} label column not found in dataframe"
    assert pole.lower() in ['north', 'south'], "Invalid pole. Use 'north' or 'south'."
    
    # Select data around threshold
    if direction.lower() == 'below':
        condition = df[data_type] < threshold
    elif direction.lower() == 'above':
        condition = df[data_type] > threshold
    else:
        raise ValueError("Invalid direction. Use 'above' or 'below'.")
    
    # Select data around pole
    if pole.lower() == 'north':
        condition &= (df['Latitude'] >= 0)
    elif pole.lower() == 'south':
        condition &= (df['Latitude'] < 0)
    else:
        raise ValueError("Invalid pole. Use 'north' or 'south'.")
    
    df.astype({'Longitude': 'float32', 'Latitude': 'float32', 'Label': 'int8', f'{data_type} label': 'int8'}, copy=False)

    cluster_pole(df, condition, data_type, area_thresh, eps)


def cluster_pole(df, condition, data_type, area_thresh, eps):
    if df.loc[condition].empty:
        return df
    
    # Step 1: Update labels for the rows which meet the threshold condition
    df.loc[condition, 'Label'] += 1
    df.loc[condition, f'{data_type} label'] += 1


    # Step 2: Convert filtered coordinates to radians
    coords_rad = np.radians(df.loc[condition, ['Latitude', 'Longitude']].to_numpy(dtype='float32'))

    # Step 3: Perform clustering using DBSCAN with Haversine metric
    moon_radius_km = 1737.4
    eps_rad = eps / moon_radius_km

    min_cluster = 85
    print(f"Clustering {data_type} pole with eps={eps:.2f} and min_cluster={min_cluster}"); sys.stdout.flush()

    clustering = DBSCAN(eps=eps_rad, min_samples=min_cluster, metric='haversine')   # DBSCAN
    clustering_labels = clustering.fit_predict(coords_rad)
    clustering_labels.astype('int32', copy=False)

    df.loc[condition, 'cluster_label'] = clustering_labels
    del coords_rad, clustering, clustering_labels

    all_labels = df.loc[condition, 'cluster_label'].unique()
    unique_labels = all_labels[all_labels != -1]

    moon_radius_m = moon_radius_km * 1000  # Convert to meters
    crs_lunar_geo = CRS.from_proj4(
        f"+proj=longlat +a={moon_radius_m} +b={moon_radius_m} +no_defs +type=crs +celestial_body=Moon"
    )

    clusters_at_1 = 0
    clusters_at_2 = 0

    # Step 4: Calculate area for each cluster and update labels
    for cluster_label in unique_labels:
        sys.stdout.flush()
        
        mask = (df['cluster_label'] == cluster_label) & condition
        cluster_points = df.loc[mask, ['Longitude', 'Latitude']].to_numpy(dtype='float32')
        
        if len(cluster_points) < 3: # Need at least 3 points to form a polygon
            continue

        # Project the cluster coordinates to planar coordinates
        # Use an Azimuthal Equidistant projection centered on the cluster centroid
        centroid_lon = float(cluster_points[:, 0].mean())
        centroid_lat = float(cluster_points[:, 1].mean())

        proj_str = (
            f"+proj=aeqd +lat_0={centroid_lat} +lon_0={centroid_lon} "
            f"+a={moon_radius_m} +b={moon_radius_m} +units=m +no_defs +type=crs +celestial_body=Moon"
        )
        crs_lunar_proj = CRS.from_proj4(proj_str)
        
        transformer = Transformer.from_crs(crs_lunar_geo, crs_lunar_proj, always_xy=True)
        x_proj, y_proj = transformer.transform(cluster_points[:, 0], cluster_points[:, 1])
        cluster_coords_proj = np.column_stack((x_proj, y_proj))

        if not np.isfinite(cluster_coords_proj).all():
            continue

        # Create a Polygon from the projected cluster points
        unique_coords_proj = np.unique(cluster_coords_proj, axis=0)
        multipoint = MultiPoint(unique_coords_proj)
        polygon = multipoint.convex_hull

        if len(unique_coords_proj) < 3: # Need at least 3 unique points to form a polygon
            continue
        if not polygon.is_valid:    # Check if the polygon is valid
            continue
        if polygon.area == 0 or polygon.is_empty:   # Check if the polygon has zero area
            continue
        if polygon.geom_type != 'Polygon':  # Check if the geometry type is Polygon
            continue

        area_m2 = polygon.area  # Area in square meters
        area_km2 = area_m2 / 1e6  # Convert to square kilometers

        # Determine label increment based on area threshold
        if area_km2 > area_thresh:  # Clusters with size greater than threshold
            df.loc[mask, 'Label'] += 1
            df.loc[mask, f'{data_type} label'] += 1
            clusters_at_2 += 1
        else:
            clusters_at_1 += 1

    print(f"{data_type} - Clusters at 1: {clusters_at_1}, Clusters at 2: {clusters_at_2}, Total clusters: {len(unique_labels)}")
    df.drop(columns=['cluster_label'], inplace=True)

