import pandas as pd # type: ignore
import os
from pathlib import Path
import numpy as np
import requests
import matplotlib.pyplot as plt # type: ignore
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

from download_data import clear_dir
from utils.load_dfs import process_LRO_image, process_M3_image, generate_LRO_coords, generate_M3_coords
from utils.utils import (
    get_metadata_value, 
    clean_metadata_value, 
    parse_metadata_content, 
    plot_polar, 
    decode_image_file, 
    get_closest_channels, 
    save_by_lon_range, 
    create_hist, 
    from_csv_and_desc
    )

def ensure_dirs(*paths, clear=False):
    for p in map(Path, paths):
        if p is None:
            continue
        p.mkdir(parents=True, exist_ok=True)
        if clear:
            clear_dir(p)

def already_split(csv_root):
    if isinstance(csv_root, str):
        csv_root = Path(csv_root)
    return len(list(csv_root.glob("*lon*.csv"))) == 12


def load_lro(file_path, lbl_file, data_dict, address, data_type, max_val, min_val):
    lbl_path = f"{file_path}/{lbl_file}"
    metadata = parse_metadata_content(lbl_path)

    img_file = lbl_path.replace(data_dict['lbl_ext'], data_dict['img_ext'])

    image_data, output_vals = process_LRO_image(img_file, address, metadata, data_type, max_val, min_val)
    output_vals = output_vals.flatten()

    lons, lats = generate_LRO_coords(image_data.shape, metadata)
    lons = lons.flatten()
    lats = lats.flatten()

    output_vals[(output_vals < min_val) | (output_vals > max_val)] = np.nan

    return lons, lats, output_vals


def load_lola(data_dict, file, data_type):
    df_temp = pd.read_csv(os.path.join(data_dict['file_path'], file), sep=r'\s+', header=None)

    if df_temp.shape[1] < 3:
        raise ValueError(f"File {file} has only {df_temp.shape[1]} columns, expected at least 3.")

    df_temp = df_temp.iloc[:, :3]  # Select only the first three columns
    # df_temp.columns = ['Longitude', 'Latitude', data_type]
    df_temp.columns = ['Latitude', 'Longitude', data_type]

    df_temp['Longitude'] = df_temp['Longitude'].astype(np.float32)
    df_temp['Latitude'] = df_temp['Latitude'].astype(np.float32)
    df_temp[data_type] = df_temp[data_type].astype(np.float32)

    # Check for values within latitude bounds
    points_in_bounds = ((df_temp['Latitude'] <= -75) & (df_temp['Latitude'] >= -90)) | ((df_temp['Latitude'] <= 90) & (df_temp['Latitude'] >= 75))
    if not points_in_bounds.any():
        return [], [], []  

    max_val = data_dict['max']
    min_val = data_dict['min']

    # df_temp[data_type][(df_temp[data_type] < min_val) | (df_temp[data_type] > max_val)] = np.nan
    df_temp.loc[(df_temp[data_type] < min_val) | (df_temp[data_type] > max_val), data_type] = np.nan

    df_temp.drop_duplicates(subset=['Longitude', 'Latitude'], inplace=True)
    df_temp.reset_index(drop=True, inplace=True)

    lons = df_temp['Longitude'].values
    lats = df_temp['Latitude'].values
    output_vals = df_temp[data_type].values

    return lons, lats, output_vals


def load_m3(file_path, lbl_file, data_dict, address):
    lbl_path = f"{file_path}/{lbl_file}"
    metadata = parse_metadata_content(lbl_path)
    img_file = lbl_path.replace(data_dict['lbl_ext'], data_dict['img_ext'])

    image_data, output_vals = process_M3_image(img_file, address, metadata)
    lons, lats, elev = generate_M3_coords(image_data.shape, metadata, data_dict)

    lons = lons.flatten()
    lats = lats.flatten()
    elev = elev.flatten()
    output_vals = output_vals.flatten()

    return lons, lats, output_vals, elev


def load_file(file, data_dict, data_type):
    file_path = data_dict['file_path']
    address = data_dict['address']
    lbl_ext = data_dict['lbl_ext']
    csv_save_path = data_dict['save_path'] if 'save_path' in data_dict else None
    max_val = data_dict['max']
    min_val = data_dict['min']

    if data_type == 'LOLA':
        lons, lats, data = load_lola(data_dict, file, data_type)
        df_temp = pd.DataFrame({
            'Longitude': lons,
            'Latitude': lats,
            data_type: data,
        })
    elif data_type == 'M3':
        lons, lats, data, elev = load_m3(file_path, file, data_dict, address)
        df_temp = pd.DataFrame({
            'Longitude': lons,
            'Latitude': lats,
            data_type: data,
            'Elevation': elev,
        })
    else:
        lons, lats, data = load_lro(file_path, file, data_dict, address, data_type, max_val, min_val)
        df_temp = pd.DataFrame({
            'Longitude': lons,
            'Latitude': lats,
            data_type: data,
        })

    valid_mask = ((df_temp['Latitude'] <= -75) & (df_temp['Latitude'] >= -90)) | ((df_temp['Latitude'] <= 90) & (df_temp['Latitude'] >= 75))
    valid_mask &= np.isfinite(df_temp[data_type])  # Remove non-finite vals from output_vals and clip coords to poles

    assert np.all((df_temp['Longitude'] >= 0) & (df_temp['Longitude'] <= 360)), f"Some longitude values are out of bounds for {data_type}: \n{df_temp['Longitude']<0} \n{df_temp['Longitude']>360}"
    assert np.all((df_temp['Latitude'] >= -90) & (df_temp['Latitude'] <= 90)), f"Some latitude values are out of bounds for {data_type}: \n{df_temp['Latitude']<-90} \n{df_temp['Latitude']>90}"

    return df_temp[valid_mask]


def load_df(data_dict, data_type, n_workers=1, plot_frac=0.25, hist=False):
    t0 = time.time()
    if data_dict['save_path'] is not None:
        os.mkdir(data_dict['save_path']) if not os.path.exists(data_dict['save_path']) else None
    if data_dict['file_path'] is not None:
        os.mkdir(data_dict['file_path']) if not os.path.exists(data_dict['file_path']) else None
    if  data_dict['plot_path'] is not None:
        os.mkdir(data_dict['plot_path']) if not os.path.exists(data_dict['plot_path']) else None

    clear_dir(data_dict['save_path'])   # Clears dirs only

    plot_save_path = data_dict['plot_path'] if 'plot_path' in data_dict else None

    if already_split(data_dict['save_path']):
        print(f"Raw CSVs appear to exist for {data_type} data. Skipping loading step.")
        return 

    print(f"Processing {data_type} data")
    file_path = data_dict['file_path']
    csv_save_path = data_dict['save_path'] if 'save_path' in data_dict else None
    lbl_ext = data_dict['lbl_ext']

    if len(os.listdir(file_path)) == 0:
        raise ValueError(f"No files found in directory: {file_path}\nHave you downloaded the data?")

    assert csv_save_path or plot_save_path, "At least one of 'save_path' or 'plot_path' must be provided."

    file_list = [f for f in os.listdir(file_path) if f.endswith(lbl_ext)][:64]
    print(f"WARNING: file list limited to 64")
    dfs = []

    t1 = time.time()
    print(f"STAGE1: {t0 - t1:.2f} seconds")

    print(f"Multiprocessing start method: {mp.get_start_method()}\n")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(load_file, file, data_dict, data_type): file for file in file_list}
        print(f"Processing {len(futures)} files for {data_type} data with {n_workers} workers...")
        for future in as_completed(futures):
            file = futures[future]
            try:
                dfs.append(future.result())
            except Exception as e:
                print(f"Error processing file {file}")
                raise e

    if not dfs:
        raise RuntimeError(f"No valid data found in {data_type} files. Check the file format and content.")
    
    t2 = time.time()
    print(f"STAGE2: {t2 - t1:.2f} seconds")

    df = pd.concat(dfs, ignore_index=True)
    t3 = time.time()
    print(f"STAGE3: {t3 - t2:.2f} seconds")
    if csv_save_path:
        save_by_lon_range(df, csv_save_path, n_workers=n_workers)
    t4 = time.time()
    print(f"STAGE4: {t4 - t3:.2f} seconds")

    if plot_save_path or hist:
        df = from_csv_and_desc(data_dict, data_type)
        print(f"Loaded {len(df):,} points for {data_type} data.")
        t5 = time.time()
        print(f"STAGE5: {t5 - t4:.2f} seconds")
    
        if plot_save_path:
            plot_polar(df, data_type, frac=plot_frac, save_path=plot_save_path, name_add='raw')
        t6 = time.time()
        print(f"STAGE6: {t6 - t5:.2f} seconds")
        if hist:
            create_hist(df, data_type)
