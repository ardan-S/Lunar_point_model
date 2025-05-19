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
from pyproj import Transformer, CRS, Proj
from pathlib import Path

from utils.utils import load_csvs_parallel, save_by_lon_range, plot_polar, psr_eda
# from utils.utils import psr_eda


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
        lon = center_lon + np.degrees(np.arctan2(x, y)) + 180

    else:
        lat = 90 - np.degrees(c)
        lon = center_lon + np.degrees(np.arctan2(x, -y))
    
    lon = lon % 360

    binary_psr = np.array(img_data == 20000, dtype=int)  # 1 for PSR, 0 otherwise

    df = pd.DataFrame({
        'Latitude': lat.ravel(),
        'Longitude': lon.ravel(),
        'psr': binary_psr.ravel()
    })

    df = df[(df['Latitude'] <= -75) | (df['Latitude'] >= 75)]

    return df


def gen_psr_df():
    jp2_url_s = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/extras/illumination/jp2/lpsr_65s_240m_201608.jp2"
    lbl_url_s = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/extras/illumination/jp2/lpsr_65s_240m_201608_jp2.lbl"

    psr_df_s = compute_psrs(jp2_url_s, lbl_url_s, 'south')

    jp2_url_n = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/extras/illumination/jp2/lpsr_65n_240m_201608.jp2"
    lbl_url_n = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/extras/illumination/jp2/lpsr_65n_240m_201608_jp2.lbl"

    psr_df_n = compute_psrs(jp2_url_n, lbl_url_n, 'north')

    psr_df = pd.concat([psr_df_s, psr_df_n], ignore_index=True)

    assert np.all(psr_df['Latitude'] <= 90) and np.all(psr_df['Latitude'] >= -90), "Latitude values out of the allowed range"

    return psr_df


# def merge_psr_df(psr_df, args):

#     combined_df = load_csvs_parallel('../../data/CSVs/combined', n_workers=args.n_workers)

#     print()
#     print("Label proportions after combining 1:")
#     print(combined_df.value_counts('Label', normalize=True) * 100) # type: ignore
#     print()
#     print(f"Total number of points: {combined_df.shape[0]}")

#     combined_df_n = combined_df[(combined_df['Latitude'] >= 75)].reset_index(drop=True)
#     combined_df_s = combined_df[(combined_df['Latitude'] <= -75)].reset_index(drop=True)
#     psr_df_n = psr_df[(psr_df['Latitude'] >= 75)].reset_index(drop=True)
#     psr_df_s = psr_df[(psr_df['Latitude'] <= -75)].reset_index(drop=True)

#     # Planetocentric lat/long, sphere of radius 1 737 400 m (IAU-2000 Moon)
#     MOON_GEOG = "+proj=longlat +R=1737400 +no_defs"

#     MOON_PS_N = "+proj=stere +lat_0=90  +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"
#     MOON_PS_S = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"

#     # One-off transformers (lon,lat ➜ x,y in metres)
#     T_N = Transformer.from_crs(CRS.from_proj4(MOON_GEOG),
#                             CRS.from_proj4(MOON_PS_N),
#                             always_xy=True)
#     T_S = Transformer.from_crs(CRS.from_proj4(MOON_GEOG),
#                             CRS.from_proj4(MOON_PS_S),
#                             always_xy=True)

#     def _to_xy(df, transformer):
#         """Vector-project lon/lat columns → (x,y) ndarray."""
#         lon = df["Longitude"].to_numpy(float)
#         lat = df["Latitude"].to_numpy(float)
#         x, y = transformer.transform(lon, lat)  # returns 1-D arrays
#         return np.column_stack((x, y))

#     coords_n = _to_xy(combined_df_n, T_N)
#     coords_s = _to_xy(combined_df_s, T_S)
#     coords_psr_n = _to_xy(psr_df_n, T_N)
#     coords_psr_s = _to_xy(psr_df_s, T_S)
#     tree_n = cKDTree(coords_n)
#     tree_s = cKDTree(coords_s)

#     def _query_tree(tree, coords, k=1):
#         return (np.array([]), np.array([])) if tree is None else tree.query(coords, k=k)

#     # Find nearest neighbors in the combined_df for each point in psr_df
#     with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
#         dist_n, idxs_n = executor.submit(_query_tree, tree_n, coords_psr_n).result()
#         dist_s, idxs_s = executor.submit(_query_tree, tree_s, coords_psr_s).result()

#     # For each point in psr_df, find the closest point in combined_df and copy into df
#     desired_cols = [c for c in combined_df_n.columns if c not in ['Latitude', 'Longitude']]

#     df_merged_n = psr_df_n.copy()
#     if len(idxs_n):
#         df_merged_n[desired_cols] = combined_df_n.iloc[idxs_n][desired_cols].values

#     df_merged_s = psr_df_s.copy()
#     if len(idxs_s):
#         df_merged_s[desired_cols] = combined_df_s.iloc[idxs_s][desired_cols].values

#     df_merged = pd.concat([df_merged_n, df_merged_s], ignore_index=True)

#     # Assert values are only between latitudes [-75, -90] and [75, 90]
#     valid_mask = ((df_merged['Latitude'] >= -90) & (df_merged['Latitude'] <= -75)) | \
#                 ((df_merged['Latitude'] >= 75) & (df_merged['Latitude'] <= 90))
#     assert np.all(valid_mask), "Latitude values out of the allowed range"

#     print(f"\nMax/mean/min distance N: {dist_n.max():.1f} m / {dist_n.mean():.1f} m / {dist_n.min():.1f} m")
#     print(f"Max/mean/min distance S: {dist_s.max():.1f} m / {dist_s.mean():.1f} m / {dist_s.min():.1f} m")
#     print("\nLabel proportions after combining 2:")
#     print(df_merged.value_counts('Label', normalize=True) * 100) # type: ignore
#     print(f"\nTotal number of points: {df_merged.shape[0]}")
#     print(f"\nDensity of combined_df_n: {len(combined_df_n) / (psr_df_n['Latitude'].max() - psr_df_n['Latitude'].min())}")
#     print(f"Density of psr_df_n: {len(psr_df_n) / (psr_df_n['Latitude'].max() - psr_df_n['Latitude'].min())}")
#     print(f"Density of combined_df_s: {len(combined_df_s) / (psr_df_s['Latitude'].max() - psr_df_s['Latitude'].min())}")
#     print(f"Density of psr_df_s: {len(psr_df_s) / (psr_df_s['Latitude'].max() - psr_df_s['Latitude'].min())}")

#     save_by_lon_range(df_merged, args.psr_save_dir)
#     return df_merged


def merge_psr_df_2(psr_df, args, R_MOON_M=1_737_400):
    # 1. Load and split poles
    combined_df = load_csvs_parallel(Path("../../data/CSVs/combined"), n_workers=args.n_workers)

    print("Original label distribution:")
    print(combined_df["Label"].value_counts(normalize=True) * 100)

    print("Original psr distribution:")
    print(psr_df["psr"].value_counts(normalize=True) * 100)

    orig_val_cnt = combined_df["Label"].value_counts(normalize=True).mul(100).round(0).sort_index()

    combined_df_n = combined_df.loc[combined_df["Latitude"] >= 75].copy()
    combined_df_s = combined_df.loc[combined_df["Latitude"] <= -75].copy()

    psr_df_n = psr_df.loc[psr_df["Latitude"] >= 75].copy()
    psr_df_s = psr_df.loc[psr_df["Latitude"] <= -75].copy()

    # 2. Project
    proj_n = Proj(proj='aeqd', lat_0=90, lon_0=0, R=R_MOON_M, always_xy=True)
    proj_s = Proj(proj='aeqd', lat_0=-90, lon_0=0, R=R_MOON_M, always_xy=True)


    def _to_xy_aeqd(df, proj_n, proj_s):
        """Vector‑project lon/lat columns to (x, y) using AEQD per pole."""
        lon = df["Longitude"].to_numpy(np.float32)
        lat = df["Latitude"].to_numpy(np.float32)
        # boolean mask for hemisphere
        north_m = lat >= 0
        x = np.empty_like(lon, dtype=np.float32)
        y = np.empty_like(lat, dtype=np.float32)

        # project north
        if north_m.any():
            x[north_m], y[north_m] = proj_n(lon[north_m], lat[north_m])
        # project south
        south_m = ~north_m
        if south_m.any():
            x[south_m], y[south_m] = proj_s(lon[south_m], lat[south_m])

        # return np.column_stack((x, y))
        df["x"] = x
        df["y"] = y

    def _attach_grid_indices(df, x_col="x", y_col="y"):
        """Create integer row/col columns so that each unique (x, y) tuple maps to one grid cell.

        Returns
        -------
        Tuple (x0, y0, dx, dy): origin and cell size in x and y.
        """
        if x_col not in df.columns or y_col not in df.columns:
            raise KeyError(f"DataFrame missing {x_col!r} or {y_col!r} columns.")

        # The grid is assumed regular and equidistant (AEQD)
        # Compute unique sorted coordinates & their deltas to infer cell size
        x_sorted = np.sort(df[x_col].unique())
        y_sorted = np.sort(df[y_col].unique())

        if len(x_sorted) < 2 or len(y_sorted) < 2:
            raise ValueError("Combined_df does not contain a regular grid – need ≥2 unique X and Y values.")

        dx = dy = 300.0
        print(f"USING dx,dy = {dx} m")

        if np.isnan(dx) or np.isnan(dy):
            raise RuntimeError("Failed to infer dx/dy – after filtering, no gaps > 1 m.")

        x0 = x_sorted.min()
        y0 = y_sorted.min()

        df["col"] = np.floor((df[x_col] - x0) / dx).astype(np.int32)
        df["row"] = np.floor((df[y_col] - y0) / dy).astype(np.int32)

        return x0, y0, dx, dy
    
    for df, name in [(combined_df_n, "combined_df_n"), (combined_df_s, "combined_df_s"), (psr_df_n, "psr_df_n"), (psr_df_s, "psr_df_s")]:
        _to_xy_aeqd(df, proj_n, proj_s)
        if len(df) == 0:
            print(f"Warning: {name} is empty after projection.")

    # 3. Build integer grid idxs for combined catalog (master grid)
    x0_n = y0_n = dx_n = dy_n = None
    x0_s = y0_s = dx_s = dy_s = None

    if len(combined_df_n):
        x0_n, y0_n, dx_n, dy_n = _attach_grid_indices(combined_df_n, x_col="x", y_col="y")
    if len(combined_df_s):
        x0_s, y0_s, dx_s, dy_s = _attach_grid_indices(combined_df_s, x_col="x", y_col="y")

    # 4. Map PSR points onto grid and take mode
    def _grid_idxs_psr(df_psr, x0, y0, dx, dy):
        df = df_psr.copy()
        df["col"] = np.round((df["x"] - x0) / dx).astype(np.int32)
        df["row"] = np.round((df["y"] - y0) / dy).astype(np.int32)
        return df
    
    if not combined_df_n.empty and not psr_df_n.empty:
        psr_df_n = _grid_idxs_psr(psr_df_n, x0_n, y0_n, dx_n, dy_n)
        mode_n = (psr_df_n.groupby(["row", "col"])["psr"]
                  .agg(lambda x: x.mode().iat[0])
                  .reset_index())
        combined_df_n = combined_df_n.merge(mode_n, on=["row", "col"], how="left")

    if not combined_df_s.empty and not psr_df_s.empty:
        psr_df_s = _grid_idxs_psr(psr_df_s, x0_s, y0_s, dx_s, dy_s)
        mode_s = (psr_df_s.groupby(["row", "col"])["psr"]
                  .agg(lambda x: x.mode().iat[0])
                  .reset_index())
        combined_df_s = combined_df_s.merge(mode_s, on=["row", "col"], how="left")

    # 5. Merge PSR data into combined_df
    df_merged = pd.concat([combined_df_n, combined_df_s], ignore_index=True)
    df_merged.drop(columns=["row", "col"], inplace=True, errors="ignore")

    # print(f"\nLabel distribution after merging PSR data:")
    # print(df_merged["Label"].value_counts(normalize=True) * 100)
    final_val_cnt = df_merged["Label"].value_counts(normalize=True).mul(100).round(0).sort_index()
    assert np.all(final_val_cnt == orig_val_cnt), f"Label proportions do not match original data: \n{final_val_cnt} \n!= \n{orig_val_cnt}"
    print("Final label distribution:")
    print(final_val_cnt)

    if "psr" in df_merged.columns:
        print(f"\nPSR mode distribution after merging PSR data:")
        print(df_merged["psr"].value_counts(normalize=True, dropna=False) * 100)

    else:
        print("No PSR data merged.")

    if getattr(args, "save_dir", None):
        save_by_lon_range(df_merged, args.save_dir)

    return df_merged

def main(args):

    # if len([file for file in os.listdir(save_dir) if file.endswith('.csv')]) == 12:
    #     print("Files found, loading psr data from CSVs")
    #     df_merged = load_csvs_parallel(save_dir, n_workers=args.n_workers)
    #     valid_mask = ((df_merged['Latitude'] <= -80) & (df_merged['Latitude'] >= -90)) | ((df_merged['Latitude'] <= 90) & (df_merged['Latitude'] >= 80))
    #     valid_mask &= np.isfinite(df_merged['psr'])
    # else:
    #     print("No files found, generating psr data")
    #     df_merged = gen_psr_df(args)
    psr_df = gen_psr_df()

    # Compute stats
    psr_df_n = psr_df[(psr_df['Latitude'] >= 75)]
    psr_df_s = psr_df[(psr_df['Latitude'] <= -75)]

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

    # Plot PSR data
    plot_polar(psr_df, 'psr', args.plot_dir, mode='binary', frac=0.01, dpi=400, label_col='psr', name_add='_raw')

    df_merged = merge_psr_df_2(psr_df, args)

    # Plot new psr data after rasterisation
    plot_polar(df_merged, 'psr', args.plot_dir, mode='binary', frac=0.01, dpi=400, label_col='psr', name_add='_rasterised')
    
    # Check columns
    expected = {'Latitude', 'Longitude', 'psr',
                'Diviner', 'MiniRF', 'LOLA', 'M3',
                'Elevation', 'Label'}

    missing = expected - set(df_merged.columns)
    if missing:
        raise AssertionError(f"Missing columns: {sorted(missing)}")

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

    # plot_polar_data(lbl_bin_df, 'Label', frac=0.01, save_path=args.plot_dir, dpi=400, graph_cat='binary')
    plot_polar(lbl_bin_df, 'Label', args.plot_dir, mode='binary', frac=0.01, dpi=400, label_col='Label', poster=True)

    # Define conditions
    condition1 = (df_merged['Label'] < lbl) & (df_merged['psr'] == 1)   # psr but NOT high label (FN)           - True -> 1
    condition2 = (df_merged['Label'] >= lbl) & (df_merged['psr'] == 0)  # high label but NOT psr (FP)           - True -> 2
    condition3 = (df_merged['Label'] >= lbl) & (df_merged['psr'] == 1)  # BOTH high label and psr (TP)          - True -> 3

    df_merged['class'] = np.select(
        [condition1, condition2, condition3],
        [1, 2, 3],
        default=0  # Default value if none of the conditions are met
    )
    
    print(f"Percentage of points with cat 0 (TN): {np.sum(df_merged['class'] == 0) / df_merged.shape[0]:.2%}")
    print(f"Percentage of points with cat 1 (FN): {np.sum(df_merged['class'] == 1) / df_merged.shape[0]:.2%}")
    print(f"Percentage of points with cat 2 (FP): {np.sum(df_merged['class'] == 2) / df_merged.shape[0]:.2%}")
    print(f"Percentage of points with cat 3 (TP): {np.sum(df_merged['class'] == 3) / df_merged.shape[0]:.2%}")

    plot_polar(df_merged, 'class', args.plot_dir, mode='category', categories=[0, 1, 2, 3], cat_colours={0: 'black', 1: 'blue', 2: 'green', 3: 'red'}, frac=0.01, dpi=400, label_col='class', poster=True)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['True Negatives', 'False Negatives', 'False Positives', 'True Positives']
    counts = [np.sum(df_merged['class'] == 0), np.sum(df_merged['class'] == 1), np.sum(df_merged['class'] == 2), np.sum(df_merged['class'] == 3)]
    ax.bar(labels, counts, color=['black', 'blue', 'green', 'red'])
    ax.set_ylabel('Count')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(labels, rotation=45)
    plt.tight_layout()
    plt.savefig(f"{args.plot_dir}/confusion_matrix.png", dpi=400)
    plt.close(fig)

    # Drop nan for col 'psr'
    df_merged = df_merged.dropna(subset=['psr'])
    psr_eda(df_merged, args.plot_dir, lbl_thresh=lbl)


def parse_args():
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--plot_dir", type=str, default="../../data/plots")
    parser.add_argument("--psr_save_dir", type=str, default="../../data/CSVs/PSRs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
