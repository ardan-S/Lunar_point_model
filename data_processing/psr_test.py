import requests
import glymur
import tempfile
import numpy as np
import pandas as pd # type: ignore
import argparse
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import gc
from concurrent.futures import ThreadPoolExecutor
import sys
import matplotlib.pyplot as plt 

from utils.utils import plot_polar_data, load_csvs_parallel, save_by_lon_range
from utils.utils import plot_psr_data, psr_eda


def compute_psrs(jp2_url, lbl_url, pole):
    pole = pole.lower()
    assert pole in ['north', 'south'], f"Invalid pole: {pole}"

    jp2_response = requests.get(jp2_url)
    jp2_response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=True, suffix='.jp2') as temp_file:
        temp_file.write(jp2_response.content)
        temp_file.flush()

        jp2 = glymur.Jp2k(temp_file.name)
        img_data = jp2[:]
        shape = jp2.shape

    lbl_response = requests.get(lbl_url)
    lbl_response.raise_for_status()

    lines = lbl_response.text.splitlines()

    # Define the desired keys with their expected data types
    desired_keys = {
        "A_AXIS_RADIUS": float,
        "B_AXIS_RADIUS": float,
        "C_AXIS_RADIUS": float,
        "CENTER_LATITUDE": float,
        "CENTER_LONGITUDE": float,
        "MAP_SCALE": float,         # UNITS: m/pix
        "LINE_PROJECTION_OFFSET": float,
        "SAMPLE_PROJECTION_OFFSET": float,
        "POSITIVE_LONGITUDE_DIRECTION": str,
        "MAP_PROJECTION_TYPE": str
    }

    meta = {k:[] for k in desired_keys}

    for line in lines:
        for key, dtype in desired_keys.items():
            if key in line:
                val = line.split('=')[1].strip()    # Extract the value
                val = val.split()[0] if '<' in val else val # Remove any trailing comments
                try: 
                    val = dtype(val.strip('"'))
                except ValueError: 
                    pass
                meta[key].append(val)

    assert meta['A_AXIS_RADIUS'] == meta['B_AXIS_RADIUS'] == meta['C_AXIS_RADIUS'], "Non-spherical body detected"

    radius = meta['A_AXIS_RADIUS'][0] * 1000    # m
    map_scale = meta['MAP_SCALE'][0]
    sample_offset = meta['SAMPLE_PROJECTION_OFFSET'][0]
    line_offset = meta['LINE_PROJECTION_OFFSET'][0]
    center_lon = meta['CENTER_LONGITUDE'][0]

    rows, cols = np.indices(shape)
    x = (cols - sample_offset) * map_scale
    y = (line_offset - rows) * map_scale  # Note inverted y-axis for image coordinates
    
    r = np.sqrt(x**2 + y**2)
    c = 2 * np.arctan(r/(2 * radius))

    if pole == 'south':
        lat = -90 + np.degrees(c)
        # lon = center_lon + np.degrees(np.arctan2(x, y)) 
        lon = center_lon + np.degrees(np.arctan2(x, y)) + 180

    else:
        lat = 90 - np.degrees(c)
        lon = center_lon + np.degrees(np.arctan2(x, -y))
    
    lon = lon % 360

    binary_psr = np.array(img_data == 20000, dtype=int)  # 1 for PSR, 0 otherwise

    # valid_mask = ((lat <= -80) & (lat >= -90)) | ((lat <= 90) & (lat >= 80))
    # valid_mask &= np.isfinite(binary_psr)

    df = pd.DataFrame({
        'Latitude': lat.ravel(),
        'Longitude': lon.ravel(),
        'psr': binary_psr.ravel()
    })

    return df


def gen_psr_df(args):
    jp2_url_s = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/extras/illumination/jp2/lpsr_65s_240m_201608.jp2"
    lbl_url_s = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/extras/illumination/jp2/lpsr_65s_240m_201608_jp2.lbl"

    psr_df_s = compute_psrs(jp2_url_s, lbl_url_s, 'south')

    jp2_url_n = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/extras/illumination/jp2/lpsr_65n_240m_201608.jp2"
    lbl_url_n = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/extras/illumination/jp2/lpsr_65n_240m_201608_jp2.lbl"

    psr_df_n = compute_psrs(jp2_url_n, lbl_url_n, 'north')

    psr_df = pd.concat([psr_df_s, psr_df_n], ignore_index=True)

    # Remove values not between latitudes [-75, -90] and [75, 90]
    psr_df = psr_df[(psr_df['Latitude'] <= -75) | (psr_df['Latitude'] >= 75)]
    psr_df_n = psr_df[(psr_df['Latitude'] >= 75)]
    psr_df_s = psr_df[(psr_df['Latitude'] <= -75)]

    assert np.all(psr_df['Latitude'] <= 90) and np.all(psr_df['Latitude'] >= -90), "Latitude values out of the allowed range"

    def compute_stats(name, df):
        return {
            'Region': name,
            'Number of points': len(df),
            'Non psr (%)': f"{np.sum(df['psr'] == 0) / len(df):.2%}",
            'psr (%)': f"{np.sum(df['psr'] == 1) / len(df):.2%}",
            'Lon min': f"{df['Longitude'].min():.3f}",
            'Lon max': f"{df['Longitude'].max():.3f}",
            'Lat min': f"{df['Latitude'].min():.3f}",
            'Lat max': f"{df['Latitude'].max():.3f}"
        }

    stats_s = compute_stats('South', psr_df_s)
    stats_n = compute_stats('North', psr_df_n)
    stats_tot = compute_stats('Total', psr_df)

    stats_df = pd.DataFrame([stats_s, stats_n, stats_tot])
    print(stats_df.to_string(index=False))
    print()

    plot_polar_data(psr_df, 'psr', frac=0.01, save_path=args.plot_dir, dpi=400)

    n_workers = args.n_workers

    combined_df = load_csvs_parallel('../../data/CSVs/combined', n_workers=n_workers)

    print()
    print("Label proportions after combining 1:")
    print(combined_df.value_counts('Label', normalize=True) * 100) # type: ignore
    print()
    print(f"Total number of points: {combined_df.shape[0]}")


    # ----------------------------------------------------
    combined_df_n = combined_df[(combined_df['Latitude'] >= 75)]
    combined_df_s = combined_df[(combined_df['Latitude'] <= -75)]

    coords_n = np.column_stack((combined_df_n['Latitude'].to_numpy(float), combined_df_n['Longitude'].to_numpy(float)))
    coords_s = np.column_stack((combined_df_s['Latitude'].to_numpy(float), combined_df_s['Longitude'].to_numpy(float)))

    psr_df_n = psr_df[(psr_df['Latitude'] >= 75)]
    psr_df_s = psr_df[(psr_df['Latitude'] <= -75)]

    coords_psr_n = np.column_stack((psr_df_n['Latitude'].to_numpy(float), psr_df_n['Longitude'].to_numpy(float)))
    coords_psr_s = np.column_stack((psr_df_s['Latitude'].to_numpy(float), psr_df_s['Longitude'].to_numpy(float)))

    tree_n = cKDTree(coords_n)
    tree_s = cKDTree(coords_s)

    def query_tree(tree, coords, k=1):
        dist, idxs = tree.query(coords, k=k)
        return dist, idxs
    
    def query_tree_using_rad(tree, coords, df, r=0.01):

        # Step A: Query the tree for all points within a radius r
        neighbour_idx_lst = tree.query_ball_point(coords, r)

        # Step B: Analyze the neighbors
        neighbor_counts = [len(indices) for indices in neighbour_idx_lst]
        mean_neighbors = np.mean(neighbor_counts)
        max_neighbors  = np.max(neighbor_counts)
        min_neighbors  = np.min(neighbor_counts)
        no_neighbours = np.sum(np.array(neighbor_counts) == 0)

        print()
        print(f"Mean number of neighbors: {mean_neighbors}")
        print(f"Max number of neighbors: {max_neighbors}")
        print(f"Min number of neighbors: {min_neighbors}")
        print(f"Number of points with no neighbors: {no_neighbours}")


        # Step C: pick exactly one best match (highest label, then closest)
        chosen_idx_list = np.full(len(coords), -1, dtype=int)  # default -1 for no neighbors
        
        for i, neighbor_indices in enumerate(neighbour_idx_lst):
            if len(neighbor_indices) == 0:
                continue  # remain -1 if no neighbors

            # 1) Extract labels for these neighbors
            local_labels = df.iloc[neighbor_indices]['Label'].values
            
            # 2) Find the maximum label among them
            max_label = local_labels.max()
            
            # 3) Identify the subset of neighbors that share this max label
            positions_of_max = np.where(local_labels == max_label)[0]
            max_label_indices = [neighbor_indices[j] for j in positions_of_max]
            
            # 4) Among those max-label neighbors, pick the one closest to coords[i]
            #    We'll do naive Euclidean distance in (Lat, Lon) space
            #    If you want geodesic/haversine, you need a different formula.
            query_latlon = coords[i]  # e.g. [Lat, Lon]
            neighbors_coord = df.iloc[max_label_indices][['Latitude','Longitude']].values
            dist = np.linalg.norm(neighbors_coord - query_latlon, axis=1)

            # index of the closest neighbor among the max-label set
            chosen_local_index = np.argmin(dist)
            
            # The actual row index in df for that chosen neighbor
            chosen_df_idx = max_label_indices[chosen_local_index]
            
            chosen_idx_list[i] = chosen_df_idx

        return chosen_idx_list

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_n = executor.submit(query_tree, tree_n, coords_psr_n)
        future_s = executor.submit(query_tree, tree_s, coords_psr_s)
        dist_n, idxs_n = future_n.result()
        dist_s, idxs_s = future_s.result()

    # with ThreadPoolExecutor(max_workers=n_workers) as executor:
    #     future_n = executor.submit(query_tree_using_rad, tree_n, coords_psr_n, combined_df_n)
    #     future_s = executor.submit(query_tree_using_rad, tree_s, coords_psr_s, combined_df_s)
    #     idxs_n = future_n.result()
    #     idxs_s = future_s.result()

    print()
    print(f"Max distance (northern): {dist_n.max()}")
    print(f"Mean distance (northern): {dist_n.mean()}")
    print(f"Max distance (southern): {dist_s.max()}")
    print(f"Mean distance (southern): {dist_s.mean()}")

    df_merged_n = psr_df_n.copy()
    df_merged_s = psr_df_s.copy()

    desired_cols = [c for c in combined_df_n.columns if c not in ['Latitude', 'Longitude']]
    df_merged_n[desired_cols] = combined_df_n.iloc[idxs_n][desired_cols].values

    desired_cols = [c for c in combined_df_s.columns if c not in ['Latitude', 'Longitude']]
    df_merged_s[desired_cols] = combined_df_s.iloc[idxs_s][desired_cols].values


    # df_merged_n['Label'] = labels_n
    # df_merged_s['Label'] = labels_s

    df_merged = pd.concat([df_merged_n, df_merged_s], ignore_index=True)

    print()
    print("Label proportions after combining 2:")
    print(df_merged.value_counts('Label', normalize=True) * 100) # type: ignore
    print()
    print(f"Total number of points: {df_merged.shape[0]}")

    print()
    print("Combined_df_n label proportions:")
    print(combined_df_n.value_counts('Label', normalize=True) * 100)    # type: ignore

    print("Psr_df_n label proportions after mapping:")
    print(df_merged_n.value_counts('Label', normalize=True) * 100)  # type: ignore

    print("Combined_df_s label proportions:")
    print(combined_df_s.value_counts('Label', normalize=True) * 100)    # type: ignore

    print("Psr_df_s label proportions after mapping:")
    print(df_merged_s.value_counts('Label', normalize=True) * 100)  # type: ignore

    print()
    print(f"Density of combined_df_n: {len(combined_df_n) / (psr_df_n['Latitude'].max() - psr_df_n['Latitude'].min())}")
    print(f"Density of psr_df_n: {len(psr_df_n) / (psr_df_n['Latitude'].max() - psr_df_n['Latitude'].min())}")

    print(f"Density of combined_df_s: {len(combined_df_s) / (psr_df_s['Latitude'].max() - psr_df_s['Latitude'].min())}")
    print(f"Density of psr_df_s: {len(psr_df_s) / (psr_df_s['Latitude'].max() - psr_df_s['Latitude'].min())}")



    # Assert values are only between latitudes [-75, -90] and [75, 90]
    valid_mask = ((df_merged['Latitude'] >= -90) & (df_merged['Latitude'] <= -75)) | \
                ((df_merged['Latitude'] >= 75) & (df_merged['Latitude'] <= 90))

    assert np.all(valid_mask), "Latitude values out of the allowed range"

    save_by_lon_range(df_merged, args.psr_save_dir)

    # del tree, idxs, psr_df, psr_df_s, psr_df_n, combined_df; gc.collect()
    del tree_n, tree_s, psr_df, psr_df_s, psr_df_n, combined_df, combined_df_n, combined_df_s; gc.collect()

    return df_merged

def main(args):
    save_dir = args.psr_save_dir

    # if len([file for file in os.listdir(save_dir) if file.endswith('.csv')]) == 12:
    #     print("Files found, loading psr data from CSVs")
    #     df_merged = load_csvs_parallel(save_dir, n_workers=args.n_workers)
    #     valid_mask = ((df_merged['Latitude'] <= -80) & (df_merged['Latitude'] >= -90)) | ((df_merged['Latitude'] <= 90) & (df_merged['Latitude'] >= 80))
    #     valid_mask &= np.isfinite(df_merged['psr'])
    # else:
    #     print("No files found, generating psr data")
    #     df_merged = gen_psr_df(args)
    df_merged = gen_psr_df(args)
    
    # Check cols
    expected_columns = ['Latitude', 'Longitude', 'psr', 'Diviner', 'MiniRF', 'LOLA', 'M3', 'Elevation', 'Label']
    assert all(col in df_merged.columns for col in expected_columns), \
        f"Missing columns! Expected columns: {expected_columns}, but found: {df_merged.columns.tolist()}"
    assert set(df_merged.columns) == set(expected_columns), \
        f"Unexpected columns! Expected exactly: {expected_columns}, but found: {df_merged.columns.tolist()}"

    # Check values
    assert round(df_merged['Latitude'].min()) == -90, f"Expected Latitude min to be -90, but got {df_merged['Latitude'].min()}"
    assert round(df_merged['Latitude'].max()) == 90, f"Expected Latitude max to be 90, but got {df_merged['Latitude'].max()}"
    assert round(df_merged['Longitude'].min()) == 0, f"Expected Longitude min to be 0, but got {df_merged['Longitude'].min()}"
    assert round(df_merged['Longitude'].max()) == 360, f"Expected Longitude max to be 360, but got {df_merged['Longitude'].max()}"
    assert round(df_merged['psr'].min()) == 0, f"Expected PSR min to be 0, but got {df_merged['psr'].min()}"
    assert round(df_merged['psr'].max()) == 1, f"Expected PSR max to be 1, but got {df_merged['psr'].max()}"
    assert round(df_merged['Label'].min()) == 0, f"Expected Label min to be 0, but got {df_merged['Label'].min()}"
    assert round(df_merged['Label'].max()) <= 7, f"Expected Label max to be 7, but got {df_merged['Label'].max()}"

    print()
    print(f"Percentage of points which are PSRs:            {np.sum(df_merged['psr'] == 1) / df_merged.shape[0]:.2%}")
    print(f"Percentage of psrs at NP:                       {np.sum((df_merged['Latitude'] >= 75) & (df_merged['psr'] == 1)) / np.sum(df_merged['psr'] == 1):.2%}")
    print(f"Percentage of psrs at SP:                       {np.sum((df_merged['Latitude'] <= -75) & (df_merged['psr'] == 1)) / np.sum(df_merged['psr'] == 1):.2%}")
    print()
    print(f"Percentage of points labelled >=2:              {np.sum(df_merged['Label'] >= 2) / df_merged.shape[0]:.2%}")
    print(f"Percentage of points labelled >=3:              {np.sum(df_merged['Label'] >= 3) / df_merged.shape[0]:.2%}")
    print(f"Percentage of points labelled >=4:              {np.sum(df_merged['Label'] >= 4) / df_merged.shape[0]:.2%}")
    print()
    print(f"Percentage of labels >=2 labeled as PSR:        {np.sum((df_merged['psr'] == 1) & (df_merged['Label'] >= 2)) / np.sum(df_merged['Label'] >= 2):.2%}")
    print(f"Percentage of labels >=3 labeled as PSR:        {np.sum((df_merged['psr'] == 1) & (df_merged['Label'] >= 3)) / np.sum(df_merged['Label'] >= 3):.2%}")
    print(f"Percentage of labels >=4 labeled as PSR:        {np.sum((df_merged['psr'] == 1) & (df_merged['Label'] >= 4)) / np.sum(df_merged['Label'] >= 4):.2%}")
    print()
    print("Percentage of each label which is a PSR:")
    for label in range(8):
        num_psrs = np.sum((df_merged['psr'] == 1) & (df_merged['Label'] == label))
        total_label = np.sum(df_merged['Label'] == label)
        percentage = (num_psrs / total_label * 100) if total_label > 0 else 0
        print(f"Label = {label}, {num_psrs:7d}  points out of {total_label:8d}  are PSR  ({percentage:.2f}%)")
    print()

    lbl = 3

    lbl_bin_df = df_merged.copy()
    lbl_bin_df['Label'] = (lbl_bin_df['Label'] >= lbl).astype(int)

    plot_polar_data(lbl_bin_df, 'Label', frac=0.01, save_path=args.plot_dir, dpi=400, graph_cat='binary')

    # Define conditions
    condition1 = (df_merged['Label'] < lbl) & (df_merged['psr'] == 1)   # psr but NOT high label            - True -> 1
    condition2 = (df_merged['Label'] >= lbl) & (df_merged['psr'] == 0)  # high label but NOT psr            - True -> 2
    condition3 = (df_merged['Label'] >= lbl) & (df_merged['psr'] == 1)  # BOTH high label and psr           - True -> 3

    df_merged['temp'] = np.select(
        [condition1, condition2, condition3],
        [1, 2, 3],
        default=0  # Default value if none of the conditions are met
    )
    
    print(f"Percentage of points with cat 0: {np.sum(df_merged['temp'] == 0) / df_merged.shape[0]:.2%}")
    print(f"Percentage of points with cat 1: {np.sum(df_merged['temp'] == 1) / df_merged.shape[0]:.2%}")
    print(f"Percentage of points with cat 2: {np.sum(df_merged['temp'] == 2) / df_merged.shape[0]:.2%}")
    print(f"Percentage of points with cat 3: {np.sum(df_merged['temp'] == 3) / df_merged.shape[0]:.2%}")
    plot_psr_data(df_merged, 'temp', graph_cat='labelled', frac=0.01, save_path=args.plot_dir, dpi=400)
    psr_eda(df_merged, args.plot_dir, lbl_thresh=lbl)

    return
    # Cast coordinates down to float32 to save memory
    # For lunar coordinates, this moves from sub-micrometer accuracy to sub-centimeter 
    # Float16 would have accuracy in the order of tens of meters    
    coordinates = df_merged[['Latitude', 'Longitude']].values.astype(np.float32)
    print(coordinates[:5])

    # Split the data into northern and southern hemispheres
    northern_mask = coordinates[:, 0] >= 75
    southern_mask = coordinates[:, 0] <= -75
    print(f"Northern hemisphere: {np.sum(northern_mask)} points")
    print(f"Southern hemisphere: {np.sum(southern_mask)} points")

    northern_coords = np.radians(coordinates[northern_mask])
    southern_coords = np.radians(coordinates[southern_mask])

    eps = 0.005
    min_samples = 10

    # Process northern hemisphere
    print("\nProcessing Northern Hemisphere..."); sys.stdout.flush()
    dbscan_north = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
    north_clusters = dbscan_north.fit_predict(northern_coords)
    print("Done processing Northern Hemisphere."); sys.stdout.flush()

    del northern_coords, dbscan_north; gc.collect()

    # Process southern hemisphere
    print("\nProcessing Southern Hemisphere..."); sys.stdout.flush()
    dbscan_south = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
    south_clusters = dbscan_south.fit_predict(southern_coords)
    print("Done processing Southern Hemisphere."); sys.stdout.flush()

    del southern_coords, dbscan_south; gc.collect()

    # AFTER MAYBE TRY HDBSCAN

    # Combine results back into the original DataFrame
    clusters = np.full(len(coordinates), -1, dtype=np.int32)  # Default cluster to -1 (noise)
    clusters[northern_mask] = north_clusters
    clusters[southern_mask] = south_clusters

    df_merged['Cluster'] = clusters

    print("\nClusters:")
    print(df_merged['Cluster'].value_counts())
    print(f"Number of clusters: {len(df_merged['Cluster'].unique())}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--plot_dir", type=str, default="../../data/plots")
    parser.add_argument("--psr_save_dir", type=str, default="../../data/CSVs/PSRs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
