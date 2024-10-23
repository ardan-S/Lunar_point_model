# type: ignore[reportPrivateImportUsage]
import re
import requests
from collections import defaultdict
import glymur
import numpy as np
import pandas as pd
import os
from urllib.parse import urljoin
import io
from requests.exceptions import ChunkedEncodingError, ConnectionError
from http.client import IncompleteRead
from matplotlib import pyplot as plt
import dask.dataframe as dd
import json



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


def load_dataset_config(json_file, args):
    with open(json_file, 'r') as f:
        dataset_dict = json.load(f)
    
    for dataset, config in dataset_dict.items():
        for key, value in config.items():
            if isinstance(value, str):
                config[key] = value.format(
                    download_dir=args.download_dir,
                    save_dir=args.save_dir,
                    interp_dir=args.interp_dir,
                    plot_dir=args.plot_dir,
                    combined_dir=args.combined_dir
                )
    return dataset_dict


def parse_metadata_content(file_path):
    metadata = defaultdict(dict)
    object_stack = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content_str = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise e

    try:
        current_object_path = ""
        for line in content_str.splitlines():
            line = line.strip()
            if line.startswith("OBJECT"):
                object_name = re.findall(r'OBJECT\s*=\s*(\w+)', line)
                if object_name:
                    object_stack.append(object_name[0])
                    current_object_path = '.'.join(object_stack)
                    metadata[current_object_path] = {}
                else:
                    raise ValueError(f"Malformed OBJECT line: {line}")
            elif line.startswith("END_OBJECT"):
                if object_stack:
                    object_stack.pop()
                    current_object_path = '.'.join(object_stack)
                else:
                    raise ValueError(f"END_OBJECT without corresponding OBJECT line: {line}")
            elif "=" in line:
                key, value = map(str.strip, line.split('=', 1))
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif re.match(r'^-?\d+$', value):
                    value = int(value)
                elif re.match(r'^-?\d+\.\d*$', value):
                    value = float(value)
                metadata[current_object_path][key] = value
    except UnicodeDecodeError:
        print("Error decoding the file content with utf-8 encoding")

    # Convert defaultdict to regular dict for compatibility
    return {k: dict(v) for k, v in metadata.items()}


def clean_metadata_value(value, string=False):
    if string:
        return str(value)
    try:
        cleaned_value = ''.join(filter(lambda x: x.isdigit() or x in ['.', '-'], str(value)))
        return float(cleaned_value)
    except ValueError:
        if value != 'PC_REAL':
            print(f"Error converting value to float: {value}")
        return str(value)


def get_metadata_value(metadata, object_path, key, string=False):
    return clean_metadata_value(metadata.get(object_path, {}).get(key), string=string)


def decode_image_file(file_path, file_extension, metadata, address):
    if file_extension == 'jp2':
        image_data = glymur.Jp2k(file_path)[:]

    elif file_extension == 'img':
        lines = int(get_metadata_value(metadata, address, 'LINES'))
        line_samples = int(get_metadata_value(metadata, address, 'LINE_SAMPLES'))
        bands = int(get_metadata_value(metadata, address, 'BANDS'))
        sample_bits = int(get_metadata_value(metadata, address, 'SAMPLE_BITS'))
        sample_type = str(get_metadata_value(metadata, address, 'SAMPLE_TYPE'))

        if sample_type == 'PC_REAL' and sample_bits == 32:
            dtype = np.float32
        else:
            raise ValueError(f"Unsupported combination of SAMPLE_TYPE: {sample_type} and SAMPLE_BITS: {sample_bits}")

        with open(file_path, 'rb') as f:
            image_data = np.fromfile(f, dtype=dtype)

            new_lines = lines
            new_size = new_lines * line_samples * bands

            if new_size != image_data.size:
                raise ValueError(f"Mismatch in data size: expected {new_size}, got {image_data.size}")

            image_data = image_data.reshape((new_lines, line_samples, bands))

            if image_data.shape[-1] == 1:
                image_data = np.squeeze(image_data, axis=-1)
            else:
                raise ValueError(f"Unsupported number of bands: {bands}")

    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    return image_data


def get_closest_channels(metadata, address, target_wavelengths):
    def fetch_url(url, retries=3):
        if os.path.isfile(url):  # Check if the source is a local file
            with open(url, 'r') as file:
                return file.read()
        else:  # Assume the source is a URL
            for attempt in range(retries):
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    return response.text
                except (ChunkedEncodingError, ConnectionError, IncompleteRead) as e:
                    if attempt < retries - 1:
                        print(f'Attempt {attempt + 1} failed for url: {url}\n Error: {e}.\nRetrying...')
                        continue
                    else:
                        raise e

    base_calib_file = 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0003/CALIB/'
    calib_file = str(get_metadata_value(metadata, '', 'CH1:SPECTRAL_CALIBRATION_FILE_NAME', string=True))
    calib_file_url = urljoin(base_calib_file, calib_file)

    response = fetch_url(calib_file_url)

    calib_data = pd.read_csv(io.StringIO(response), sep=r'\s+', header=None, names=["Channel", "Wavelength"])
    calib_data["Channel"] = calib_data["Channel"].astype(int)
    calib_data["Wavelength"] = calib_data["Wavelength"].astype(float)

    # Find the closest channel for each target wavelength
    closest_channels = []
    for wavelength in target_wavelengths:
        idx = (calib_data['Wavelength'] - wavelength).abs().argmin()
        channel = calib_data.iloc[idx]['Channel']
        closest_channels.append(channel)

    test_channels = np.array(closest_channels).astype(int)

    return test_channels


def save_by_lon_range(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    lon_ranges = [(0, 30), (30, 60), (60, 90),
                  (90, 120), (120, 150), (150, 180),
                  (180, 210), (210, 240), (240, 270),
                  (270, 300), (300, 330), (330, 360)]
    file_names = [
        os.path.join(output_dir, 'lon_000_030.csv'),
        os.path.join(output_dir, 'lon_030_060.csv'),
        os.path.join(output_dir, 'lon_060_090.csv'),
        os.path.join(output_dir, 'lon_090_120.csv'),
        os.path.join(output_dir, 'lon_120_150.csv'),
        os.path.join(output_dir, 'lon_150_180.csv'),
        os.path.join(output_dir, 'lon_180_210.csv'),
        os.path.join(output_dir, 'lon_210_240.csv'),
        os.path.join(output_dir, 'lon_240_270.csv'),
        os.path.join(output_dir, 'lon_270_300.csv'),
        os.path.join(output_dir, 'lon_300_330.csv'),
        os.path.join(output_dir, 'lon_330_360.csv')
    ]

    for lon_range, file_name in zip(lon_ranges, file_names):
        lon_min, lon_max = lon_range
        df_slice = df[(df['Longitude'] >= lon_min) & (df['Longitude'] < lon_max)]
        df_slice.to_csv(file_name, index=False)


def plot_polar_data(df, variable, graph_cat='raw', frac=None, random_state=42, save_path=None):
    # Check for required columns
    required_columns = {'Latitude', 'Longitude', variable}
    missing_cols = required_columns - set(df.columns)
    assert not missing_cols, f"Missing columns in DataFrame: {', '.join(missing_cols)}"

    # Convert to Dask DataFrame and drop missing values
    ddf = dd.from_pandas(df.dropna(subset=required_columns), npartitions=4) if isinstance(df, pd.DataFrame) else df.dropna(subset=required_columns)
    assert isinstance(ddf, dd.DataFrame), "Input 'df' must be a Dask DataFrame"

    if frac:
        ddf = ddf.sample(frac=frac, random_state=random_state)

    ave_ddf = ddf.groupby(['Longitude', 'Latitude']).mean().reset_index().compute()
    north_pole_ddf = (ave_ddf[ave_ddf['Latitude'] >= 0]).copy()
    south_pole_ddf = (ave_ddf[ave_ddf['Latitude'] < 0]).copy()

    def prepare_polar_data(ddf, pole):
        if len(ddf.index) == 0:
            return ddf
        ddf = ddf.copy()
        ddf['r'] = 90 - ddf['Latitude'] if (pole == 'north') else 90 + ddf['Latitude']
        ddf['theta'] = np.deg2rad(ddf['Longitude'])
        return ddf

    north_pole_ddf = prepare_polar_data(north_pole_ddf, 'north')
    south_pole_ddf = prepare_polar_data(south_pole_ddf, 'south')

    north_pole_df = north_pole_ddf.compute() if isinstance(north_pole_ddf, dd.DataFrame) else north_pole_ddf
    south_pole_df = south_pole_ddf.compute() if isinstance(south_pole_ddf, dd.DataFrame) else south_pole_ddf

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(20, 10))

    def set_latitude_labels(ax, pole):
        ax.set_ylim(0, 15)
        ax.set_yticks(range(0, 16, 5))
        labels = [str(90 - x) if pole == 'north' else str(-90 + x) for x in range(0, 16, 5)]
        ax.set_yticklabels(labels)

    def plot_pole_data(ax, df, pole):
        if len(df.index) == 0:
            return
        sc = ax.scatter(df['theta'], df['r'], c=df[variable], cmap='Greys_r', s=50)
        plt.colorbar(sc, ax=ax, label=variable)
        set_latitude_labels(ax, pole)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f'{variable} values - {pole.capitalize()} Pole')

    # Plot for North Pole
    if len(north_pole_df.index) != 0:
        plot_pole_data(ax1, north_pole_df, 'north')
    else:
        print('No data for North Pole')
        fig.delaxes(ax1)

    # Plot for South Pole
    if len(south_pole_df.index) != 0:
        plot_pole_data(ax2, south_pole_df, 'south')
    else:
        print('No data for South Pole')
        fig.delaxes(ax2)

    if save_path:
        plt.savefig(f"{save_path}/{variable}_{graph_cat}_plot.png")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_labeled_polar_data(df, variable, label_column, save_path=None):
    # Filter the DataFrame to include only rows where label_column > 0
    df_filtered = df[df[label_column] > 0].copy()

    # Check if there is data to plot
    if df_filtered.empty:
        print(f"No data to plot for {variable}.")
        return

    # Split into north and south poles
    north_pole_df = df_filtered[df_filtered['Latitude'] >= 0]
    south_pole_df = df_filtered[df_filtered['Latitude'] < 0]

    def prepare_polar_data(df, pole):
        if df.empty:
            return df
        df = df.copy()
        if pole == 'north':
            df['r'] = 90 - df['Latitude']
        else:
            df['r'] = 90 + df['Latitude']
        df['theta'] = np.deg2rad(df['Longitude'])
        return df

    north_pole_df = prepare_polar_data(north_pole_df, 'north')
    south_pole_df = prepare_polar_data(south_pole_df, 'south')

    # Create subplots for north and south poles
    num_plots = (not north_pole_df.empty) + (not south_pole_df.empty)
    fig, axes = plt.subplots(1, num_plots, subplot_kw={'projection': 'polar'}, figsize=(10 * num_plots, 10))

    # Ensure axes is a list even if there's only one plot
    if num_plots == 1:
        axes = [axes]

    def plot_pole_data(ax, df, pole):
        if df.empty:
            return

        # Map labels to colors directly
        label_values = df[label_column].unique()
        colour_map = {}
        if np.array_equal(label_values, [1]):
            colour_map = {1: 'blue'}
        elif np.array_equal(label_values, [2]):
            colour_map = {2: 'red'}
        elif np.array_equal(np.sort(label_values), [1, 2]):
            colour_map = {1: 'blue', 2: 'red'}
        else:
            raise ValueError(f"Unsupported label values: {label_values.unique()}. Expected [1], [2] or [1, 2]")

        # # Map label values to colors
        # colours = df[label_column].map(color_map)

        # ax.scatter(df['theta'], df['r'], c=colours, s=10)

        # Plot data points with labels
        for label, colour in colour_map.items():
            df_subset = df[df[label_column] == label]
            ax.scatter(df_subset['theta'], df_subset['r'], c=colour, s=10, label=f'Label {label}')


        ax.set_ylim(0, 15)
        ax.set_yticks(range(0, 16, 5))
        labels = [str(90 - x) if pole == 'north' else str(-90 + x) for x in range(0, 16, 5)]
        ax.set_yticklabels(labels)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f'{variable} Labels - {pole.capitalize()} Pole')

    plot_idx = 0
    # Plot for North Pole
    if not north_pole_df.empty:
        plot_pole_data(axes[plot_idx], north_pole_df, 'north')
        plot_idx += 1

    # Plot for South Pole
    if not south_pole_df.empty:
        plot_pole_data(axes[plot_idx], south_pole_df, 'south')

    if save_path:
        plt.savefig(save_path)
        print(f"Label plot saved to {save_path} for {variable}")
        plt.close(fig)  # Close the figure to free memory
    else:
        plt.show()


def generate_mesh(RESOLUTION=0.24):
    MOON_RADIUS = 1737.4  # Radius of the Moon in kilometers

    # Convert resolution to degrees (approximate, depends on latitude)- 1 degree latitude is roughly MOON_RADIUS * pi / 180 km
    resolution_deg = (RESOLUTION / (MOON_RADIUS * np.pi / 180))

    lat_ranges = [(75, 90), (-90, -75)]
    lon_slices = [(0, 30), (30, 60), (60, 90),
                  (90, 120), (120, 150), (150, 180),
                  (180, 210), (210, 240), (240, 270),
                  (270, 300), (300, 330), (330, 360)]

    def generate_grid(lat_range, lon_range, resolution_deg):
        lats = np.arange(lat_range[0], lat_range[1] + resolution_deg, resolution_deg)
        lons = np.arange(lon_range[0], lon_range[1] + resolution_deg, resolution_deg)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        return lon_grid, lat_grid

    meshes = []

    for lon_range in lon_slices:
        lon_grid_north, lat_grid_north = generate_grid(lat_ranges[0], lon_range, resolution_deg)
        lon_grid_south, lat_grid_south = generate_grid(lat_ranges[1], lon_range, resolution_deg)
        lon_lat_grid_north = np.column_stack((lon_grid_north.ravel(), lat_grid_north.ravel()))
        lon_lat_grid_south = np.column_stack((lon_grid_south.ravel(), lat_grid_south.ravel()))
        meshes.append((lon_lat_grid_north, lon_lat_grid_south))

    print(f'Meshes created. Total points: {sum(len(lon_lat_grid_north) + len(lon_lat_grid_south) for lon_lat_grid_north, lon_lat_grid_south in meshes):,}')

    return meshes
