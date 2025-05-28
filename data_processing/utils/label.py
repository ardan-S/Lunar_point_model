import pandas as pd # type: ignore
from functools import reduce
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import DBSCAN, HDBSCAN  # type: ignore
import numpy as np
from shapely.geometry import Polygon, MultiPoint    # type: ignore
from shapely.ops import transform   # type: ignore
from functools import partial
from pyproj import CRS, Transformer, Proj, Geod
import sys

from data_processing.utils.utils import save_by_lon_range, load_csvs_parallel, plot_polar, plot_polar_overlay
from data_processing.download_data import clear_dir


def combine(*dirs, n_workers=None, combined_save_path=None):
    all_dfs = []

    if len([f for f in os.listdir(combined_save_path) if f.endswith('.csv') and 'lon' in f]) == 12:
        print(f"Combined CSVs appear to exist. Skipping combine stage.")
        return pd.DataFrame()  # Return empty DataFrame for label() to exit with

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_dir = {executor.submit(load_csvs_parallel, dir, n_workers): dir for dir in dirs}
        
        for future in as_completed(future_to_dir):
            df = future.result()
            all_dfs.append(df)

    combined_df = reduce(lambda left, right: pd.merge(left, right, on=['Longitude', 'Latitude'], how='outer'), all_dfs)

    if combined_df.shape[1] != 7:
        print(f"ERROR - Combined dataframe columns: {combined_df.columns}")
        raise ValueError("Combined dataframe does not have the expected number of columns")

    print(f"Dataframes combined successfully. Shape: {combined_df.shape}\n")
    return combined_df


def label(df, dataset_dict, plot_dir, lola_area_thresh=3, m3_area_thresh=2.4, eps=450, min_cluster=5):
    # -------------------- Setup --------------------
    combined_save_path = dataset_dict['Combined']['combined_save_path']

    if len([f for f in os.listdir(combined_save_path) if f.endswith('.csv') and 'lon' in f]) == 12:
        print(f"Labelled CSVs appear to exist. Skipping label stage.")
        return
    # print(f"REMINDER: Skip if combined CSVs already exist commented out in label.py")

    clear_dir(combined_save_path, dirs_only=False)

    plot_save_path = plot_dir
    df[['Label', 'Diviner label', 'LOLA label', 'M3 label', 'MiniRF label']] = 0
    df[['Label', 'Diviner label', 'LOLA label', 'M3 label', 'MiniRF label']] = df[['Label', 'Diviner label', 'LOLA label', 'M3 label', 'MiniRF label']].astype('int8')
    df[['Latitude', 'Longitude']] = df[['Latitude', 'Longitude']].astype('float32')

    # -------------------- DIVINER --------------------
    df.loc[df['Diviner'] <= 110, 'Label'] += 2
    df.loc[df['Diviner'] <= 110, 'Diviner label'] += 2

    print(f"Taking the top {df[df['Diviner'] <= 110].shape[0] / df.shape[0] :.2%} of points from Diviner. Threshold = 110")
    plot_polar(df, 'Diviner', save_path=plot_save_path, mode='labeled', label_col='Diviner label', frac=0.01, dpi=400)
    plot_polar_overlay(base_df=df, overlay_df=df, variable='Diviner', label_col='Diviner label', save_path=plot_save_path, dpi=400, poster=True)
    print()

    # -------------------- LOLA --------------------
    lola_thresh_n = df.loc[df['Latitude'] >= 0, 'LOLA'].mean() + 2 * df.loc[df['Latitude'] >= 0, 'LOLA'].std()
    print(f"North pole LOLA threshold: {lola_thresh_n:.2f} ({df[df['LOLA'] >= lola_thresh_n].shape[0] / df.shape[0] :.2%} of data)")
    apply_area_label_2(df, 'LOLA', lola_thresh_n, lola_area_thresh, 'above', 'north', eps=eps, min_cluster=min_cluster)

    lola_thresh_s = df.loc[df['Latitude'] < 0, 'LOLA'].mean() + 2 * df.loc[df['Latitude'] < 0, 'LOLA'].std()
    print(f"South pole LOLA threshold: {lola_thresh_s:.2f} ({df[df['LOLA'] >= lola_thresh_s].shape[0] / df.shape[0] :.2%} of data)")
    apply_area_label_2(df, 'LOLA', lola_thresh_s, lola_area_thresh, 'above', 'south', eps=eps, min_cluster=min_cluster)

    plot_polar(df, 'LOLA', save_path=plot_save_path, mode='labeled', label_col='LOLA label', frac=0.01, dpi=400)
    plot_polar_overlay(base_df=df, overlay_df=df, variable='LOLA', label_col='LOLA label', save_path=plot_save_path, dpi=400, poster=True)
    print()

    # -------------------- M3 --------------------
    m3_thresh_n = df.loc[df['Latitude'] >= 0, 'M3'].mean() - 2 * df.loc[df['Latitude'] >= 0, 'M3'].std()
    print(f"North pole M3 threshold: {m3_thresh_n:.2f} ({df[df['M3'] <= m3_thresh_n].shape[0] / df.shape[0] :.2%} of data)")
    apply_area_label_2(df, 'M3', m3_thresh_n, m3_area_thresh, 'below', 'north', eps=eps, min_cluster=min_cluster)  # eps to 2.5

    m3_thresh_s = df.loc[df['Latitude'] < 0, 'M3'].mean() - 2 * df.loc[df['Latitude'] < 0, 'M3'].std()
    print(f"South pole M3 threshold: {m3_thresh_s:.2f} ({df[df['M3'] <= m3_thresh_s].shape[0] / df.shape[0] :.2%} of data)")
    apply_area_label_2(df, 'M3', m3_thresh_s, m3_area_thresh, 'below', 'south', eps=eps, min_cluster=min_cluster)  # eps to 2.5

    plot_polar(df, 'M3', save_path=plot_save_path, mode='labeled', label_col='M3 label', frac=0.01, dpi=400)
    plot_polar_overlay(base_df=df, overlay_df=df, variable='M3', label_col='M3 label', save_path=plot_save_path, dpi=400, poster=True)
    print()

    # -------------------- MiniRF --------------------
    MRF_thresh = df['MiniRF'].mean() + 2 * df['MiniRF'].std()
    print(f"Taking the top {df[df['MiniRF'] >= MRF_thresh].shape[0] / df.shape[0] :.2%} of points from MiniRF. Threshold = {MRF_thresh:.2f}")
    
    df.loc[df['MiniRF'] > MRF_thresh, 'Label'] += 1
    df.loc[df['MiniRF'] > MRF_thresh, 'MiniRF label'] += 1

    plot_polar(df, 'MiniRF', save_path=plot_save_path, mode='labeled', label_col='MiniRF label', frac=0.01, dpi=400)
    plot_polar_overlay(base_df=df, overlay_df=df, variable='MiniRF', label_col='MiniRF label', save_path=plot_save_path, dpi=400, poster=True)

    # -------------------- Save labeled data --------------------
    save_by_lon_range(df, combined_save_path)

    # -------------------- Print labels --------------------
    def print_label_counts(df, label_name):
        counts = df[label_name].value_counts(normalize=True) * 100
        for label in [0, 1, 2]:  # Ensure that we always include 0, 1, and 2 even if they are missing
            percentage = counts.get(label, 0.0)
            print(f"{label_name:<14} - {label}: {percentage:6.2f}%")
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

    plot_polar(df, 'Label', save_path=plot_save_path, mode='continuous', label_col='Label', frac=0.01, dpi=400, poster=True)

    # -------------------- Ice threshold --------------------
    lbl = 3
    df_bin = df.copy()
    df_bin['Label'] = (df_bin['Label'] >= lbl).astype(int)

    assert len(df_bin) == len(df), f"Length of binary dataframe does not match original dataframe: {len(df_bin):,} != {len(df):,}"

    print(f"Percentage of labels >= {lbl}: {(df_bin['Label'] == 1).sum() / df_bin.shape[0]:.2%}")

    print()
    print(f"Binary label proportions (label={lbl}):")
    print(df_bin['Label'].value_counts(normalize=True) * 100)
    print()
    print(f"Total number of points: {df_bin.shape[0]}")

    # plot_polar_data(df_bin, 'Label', frac=0.01, graph_cat='binary_orig', save_path=plot_save_path, dpi=400)
    plot_polar(df_bin, 'Label', save_path=plot_save_path, mode='binary', label_col='Label', frac=0.01, dpi=400)


def apply_area_label_2(df, data_type, threshold, area_thresh, direction, pole, eps, min_cluster, R_MOON_M=1737.4e3):
    # -------------------- Setup --------------------
    pole = pole.lower()
    assert 'Label' in df.columns, "Label column not found in dataframe"
    assert f'{data_type} label' in df.columns, f"{data_type} label column not found in dataframe"
    assert pole in ['north', 'south'], "Invalid pole. Use 'north' or 'south'."
    df.astype({'Longitude': 'float32', 'Latitude': 'float32', 'Label': 'int8', f'{data_type} label': 'int8'}, copy=False)

    # ----------------- Define condition -----------------
    # Select data around threshold
    if direction.lower() == 'below':
        condition = df[data_type] < threshold
    elif direction.lower() == 'above':
        condition = df[data_type] > threshold
    else:
        raise ValueError("Invalid direction. Use 'above' or 'below'.")
    
    # Select data around pole
    if pole == 'north':
        condition &= (df['Latitude'] >= 0)
    elif pole == 'south':
        condition &= (df['Latitude'] < 0)
    else:
        raise ValueError("Invalid pole. Use 'north' or 'south'.")
    
#     cluster_pole(df, condition, data_type, area_thresh, eps, pole)


# def cluster_pole(df, condition, data_type, area_thresh, eps, pole, R_MOON=1737.4e3):
    if df.loc[condition].empty:
        print(f"WARNING: Clustering failed for {data_type} pole. No data points meet the pole/threshold conditions.")
        return df
    
    # ----------------- Clustering -----------------
    # Step 1: Update labels for the rows which meet the threshold condition
    df.loc[condition, ['Label', f'{data_type} label']] += 1

    # Step 2: Convert filtered coordinates to xy with polar aeqd projection 
    proj_n = Proj(proj='aeqd', lat_0=90, lon_0=0, R=R_MOON_M, always_xy=True)
    proj_s = Proj(proj='aeqd', lat_0=-90, lon_0=0, R=R_MOON_M, always_xy=True)

    lon = df.loc[condition, 'Longitude'].to_numpy(dtype='float32')
    lat = df.loc[condition, 'Latitude'].to_numpy(dtype='float32')

    x, y = proj_n(lon, lat) if pole == 'north' else proj_s(lon, lat)    # Case for invalid pole handled earlier

    coords_xy = np.column_stack((x, y))

    # Step 3: Perform clustering using DBSCAN
    # Note DBSCAN is a hard clustering algorithm, points are assigned to exactly one cluster so no double counted points
    print(f"Clustering {data_type} pole - eps={eps:.2f} m, min_cluster={min_cluster}")
    clustering = DBSCAN(eps=eps, min_samples=min_cluster, metric="euclidean")
    clustering_labels = clustering.fit_predict(coords_xy).astype('int32')

    df.loc[condition, 'cluster_label'] = clustering_labels

    # Step 4: area calculation and second increment
    unique_clusters = [l for l in np.unique(clustering_labels) if l != -1]
    clusters_at_1 = clusters_at_2 = 0

    geod = Geod(a=R_MOON_M, b=R_MOON_M)

    for clabel in unique_clusters:        
        mask = (df['cluster_label'] == clabel) & condition
        lon_cl = df.loc[mask, 'Longitude'].to_numpy(dtype='float32')
        lat_cl = df.loc[mask, 'Latitude'].to_numpy(dtype='float32')

        if len(np.unique(np.column_stack((lon_cl, lat_cl)), axis=0)) < 3:
            continue

        area_m2, _ = geod.polygon_area_perimeter(lon_cl, lat_cl)
        area_m2 = abs(area_m2)  # Orientation independent
    
        area_km2 = area_m2 / 1e6  # Convert to square kilometers

        # Determine label increment based on area threshold
        if area_km2 > area_thresh:  # Clusters with size greater than threshold
            df.loc[mask, ['Label', f'{data_type} label']] += 1
            clusters_at_2 += 1
        else:
            clusters_at_1 += 1

    print(f"{data_type} - Clusters at 1: {clusters_at_1:,}, Clusters at 2: {clusters_at_2:,}, Total clusters: {clusters_at_1 + clusters_at_2:,}")
    df.drop(columns=['cluster_label'], inplace=True)

