import requests
import glymur
import tempfile
import numpy as np
import pandas as pd
import argparse
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import sys

from utils.utils import plot_polar_data, load_csvs_parallel


def compute_psrs(jp2_url, lbl_url, pole):
    pole = pole.lower()
    assert pole in ['north', 'south'], "Invalid pole"

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
        lon = center_lon + np.degrees(np.arctan2(y, x))
        lon = (lon + 180) % 360
    else:
        lat = 90 - np.degrees(c)
        lon = center_lon + np.degrees(np.arctan2(x, -y))
        lon = lon % 360

    binary_psr = np.array(img_data == 20000, dtype=int)  # 1 for PSR, 0 otherwise

    data = pd.DataFrame({
        'Latitude': lat.ravel(),
        'Longitude': lon.ravel(),
        'psr': binary_psr.ravel()
    })

    return data, binary_psr

def main(args):
    jp2_url_s = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/extras/illumination/jp2/lpsr_65s_240m_201608.jp2"
    lbl_url_s = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/extras/illumination/jp2/lpsr_65s_240m_201608_jp2.lbl"

    data_s, binary_psr_s = compute_psrs(jp2_url_s, lbl_url_s, 'south')

    jp2_url_n = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/extras/illumination/jp2/lpsr_65n_240m_201608.jp2"
    lbl_url_n = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/extras/illumination/jp2/lpsr_65n_240m_201608_jp2.lbl"

    data_n, binary_psr_n = compute_psrs(jp2_url_n, lbl_url_n, 'north')

    psr_df = pd.concat([data_s, data_n], ignore_index=True)
    binary_psr = np.concatenate([binary_psr_s, binary_psr_n])

    def compute_stats(name, df, binary_arr):
        return {
            'Region': name,
            'Number of points': len(df),
            'Non psr (%)': f"{np.sum(binary_arr == 0) / binary_arr.size:.2%}",
            'psr (%)': f"{np.sum(binary_arr == 1) / binary_arr.size:.2%}",
            'Lon min': f"{df['Longitude'].min():.3f}",
            'Lon max': f"{df['Longitude'].max():.3f}",
            'Lat min': f"{df['Latitude'].min():.3f}",
            'Lat max': f"{df['Latitude'].max():.3f}"
        }

    stats_s = compute_stats('South', data_s, binary_psr_s)
    stats_n = compute_stats('North', data_n, binary_psr_n)
    stats_tot = compute_stats('Total', psr_df, binary_psr)

    stats_df = pd.DataFrame([stats_s, stats_n, stats_tot])
    print(stats_df.to_string(index=False))
    print()

    plot_polar_data(psr_df, 'psr', frac=0.01, save_path=args.plot_dir, dpi=400)

    combined_df = load_csvs_parallel('../../data/CSVs/combined', n_workers=4)

    coords_1 = np.column_stack((combined_df['Latitude'].to_numpy(float), combined_df['Longitude'].to_numpy(float)))
    coords_2 = np.column_stack((psr_df['Latitude'].to_numpy(float), psr_df['Longitude'].to_numpy(float)))

    tree = cKDTree(coords_1)
    _, idxs = tree.query(coords_2, k=1)

    df_merged = psr_df.copy()
    desired_cols = [c for c in combined_df.columns if c not in ['Latitude', 'Longitude']]
    for col in desired_cols:
        df_merged[col] = combined_df.iloc[idxs][col].values

    print("\nMerged DF:")
    print(df_merged.head()) 
    print()
    print(f"Merged df cols: {df_merged.columns}")
    print(f"percentage of psr value of 1 in merged df: {np.sum(df_merged['psr'] == 1) / df_merged.shape[0]:.2%}")
    print("Proportion of each label with psr value of 1:")
    for label in range(8):
        num_psrs = np.sum((df_merged['psr'] == 1) & (df_merged['Label'] == label))
        total_label = np.sum(df_merged['Label'] == label)
        percentage = (num_psrs / total_label * 100) if total_label > 0 else 0
        print(f"num psrs w label = {label}: {num_psrs} ({percentage:.2f}%)")

    # Prepare data
    coordinates = df_merged[['Latitude', 'Longitude']].values

    # Fit DBSCAN
    dbscan = DBSCAN(eps=0.01, min_samples=5, metric='haversine')  # Use haversine for geographical distances
    clusters = dbscan.fit_predict(np.radians(coordinates))  # Convert to radians for haversine distance

    df_merged['Cluster'] = clusters

    print("\nClusters:")
    print(df_merged['Cluster'].value_counts())
    print(f"Number of clusters: {len(df_merged['Cluster'].unique())}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--plot_dir", type=str, default="../../data/plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
