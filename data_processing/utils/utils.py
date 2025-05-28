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
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns
from pyproj import Proj
from typing import List, Tuple

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, jaccard_score 


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

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        # Drop duplicates to handle overlap
        combined_df.drop_duplicates(inplace=True)
        return combined_df
    else:
        return pd.DataFrame()


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
        raise ValueError(f"Unsupported file extension: {file_extension} found at {file_path}")

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


def load_every_nth_line(file_path, n):
    def should_skip(row_idx):
        # Skip the row if it's not a multiple of n (excluding header)
        if row_idx == 0:
            return False
        else:
            return (row_idx-1) % n != 0

    df = pd.read_csv(
        file_path,
        skiprows=lambda x: should_skip(x)
    )
    return df


def save_by_lon_range(df, output_dir, n_workers=1):
    os.makedirs(output_dir, exist_ok=True)

    """"
    IMPORTANT NOTE: THIS FUNCTION SAVES AN OVERLAP OF 2 DEGREES BETWEEN FILES
    IMPORTANT NOTE: THIS METHOD OF SAVING CREATES DUPLICATES IN THE FILES
    """
    # Define longitude ranges with 2-degree overlaps
    lon_ranges = [
        (0, 32), (28, 62), (58, 92),
        (88, 122), (118, 152), (148, 182),
        (178, 212), (208, 242), (238, 272),
        (268, 302), (298, 332), (328, 360)
    ]

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

    tasks = []

    for lon_range, file_name in zip(lon_ranges, file_names):
        lon_min, lon_max = lon_range

        # Adjust slicing to include the overlap and handle wrap-around
        if lon_min == 0:  # Handle 0 boundary wrap-around
            df_slice = df[(df['Longitude'] >= lon_min) & (df['Longitude'] < lon_max) |
                          (df['Longitude'] >= 360 - (32 - lon_max))]
        elif lon_max == 360:  # Handle 360 boundary wrap-around
            df_slice = df[(df['Longitude'] >= lon_min) & (df['Longitude'] < lon_max) |
                          (df['Longitude'] < (lon_min - 360) + 2)]
        else:
            df_slice = df[(df['Longitude'] >= lon_min) & (df['Longitude'] < lon_max)]

        if not df_slice.empty:
            if os.path.exists(file_name):
                df_slice.to_csv(file_name, mode='a', header=False, index=False)
            else:
                df_slice.to_csv(file_name, index=False)


def plot_polar(
    df_in,
    variable,
    save_path,
    mode: str = "continuous",
    label_col: str | None = None,
    categories: list[int] | None = None,
    cat_colours: dict[int, str] | None = None,
    frac: float | None = None,
    random_state: int = 42,
    dpi: int | None = None,
    name_add: str | None = None,
    poster=False
):
    """
    Plot a north and south polar scatter map of a lunar **300 m AEQD grid**
    and save it as a PNG.

    The same plotting engine supports four modes:

    ╔════════════╤═════════════════════════════════════════════════════════════╗
    ║ Mode name  │ What it shows / when to use it                              ║
    ╠════════════╪═════════════════════════════════════════════════════════════╣
    ║ "continuous" (default)                                                   ║
    ║            │ • Greyscale map of the numeric *variable* column.           ║
    ║            │ • No label column needed.                                   ║
    ║            │ • A colour bar is added.                                    ║
    ║────────────┼─────────────────────────────────────────────────────────────╣
    ║ "labeled"                                                                ║
    ║            │ • Categorical overlay of a *label_col* whose values are     ║
    ║            │   **positive integers (1, 2 , 3 …)**.  Label 0 is treated  ║
    ║            │   as “background” and dropped.                              ║
    ║            │ • Useful for quality flags or cluster classes where only    ║
    ║            │   the flagged pixels should be shown.                       ║
    ║            │ • If *cat_colours* is omitted, colours default to           ║
    ║            │   `{1: "blue", 2: "red"}`; extend the dict for extra codes. ║
    ║────────────┼─────────────────────────────────────────────────────────────╣
    ║ "binary"                                                                 ║
    ║            │ • A special case of *labeled* where the label column        ║
    ║            │   contains **only 0 or 1**.                                 ║
    ║            │ • Defaults to `{0:"black", 1:"gray"}` unless overridden.    ║
    ║────────────┼─────────────────────────────────────────────────────────────╣
    ║ "category"                                                              ║
    ║            │ • Arbitrary integer categories.  You *must* supply          ║
    ║            │   `categories=[a,b,c…]` **and** a matching `cat_colours`    ║
    ║            │   dict that maps every listed category to a colour.         ║
    ║            │ • Use when your labels are non-binary and include 0.        ║
    ╚════════════╧═════════════════════════════════════════════════════════════╝

    Parameters
    ----------
    df_in : pandas.DataFrame
        Table that contains at least the columns
        ``"Longitude"``, ``"Latitude"`` (decimal degrees) and *variable*.
        For non-continuous modes it must also contain *label_col*.
    variable : str
        Name of the numeric column to visualise (always plotted; greyscale in
        ``continuous`` mode, ignored in the other modes except for point size
        aggregation).
    save_path : str or pathlib.Path
        Folder where the PNG will be written.
    mode : {"continuous", "labeled", "binary", "category"}, default "continuous"
        Selects the plotting behaviour described in the table above.
    label_col : str, optional
        Column holding discrete labels; **required** for every mode except
        ``continuous``. Often the same as the variable column.
    categories : sequence[int], optional
        Explicit list of category codes to plot (``category`` mode only).
    cat_colours : dict[int, str], optional
        Maps category codes to matplotlib-recognised colour strings.
        • Optional in ``labeled`` and ``binary`` (sensible defaults exist).  
        • **Required** in ``category`` mode and must cover *every* entry in
          *categories*.
    frac : float in (0, 1], optional
        If given, randomly sample this fraction of rows *before* plotting.
        Useful for very dense point clouds.
    random_state : int, default 42
        RNG seed for the sampling step.
    dpi : int, optional
        Resolution passed to ``matplotlib.pyplot.savefig``.
    name_add : str, optional
        Extra tag appended to the file name (``<variable>_<mode>_<name_add>.png``).

    """
    # ---------- Basic checks ----------
    mode = mode.lower()
    required = {'Longitude', 'Latitude', variable}
    if mode != 'continuous':
        assert label_col, "Label column is required for non-continuous mode"
        required.add(label_col)
    df = df_in.copy()
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    df = df.dropna(subset=required)

    # ---------- Sampling ----------
    if frac:
        df = df.sample(frac=frac, random_state=random_state)

    # ---------- Group dups ----------
    agg = {variable: 'mean'}
    if mode != 'continuous':
        agg[label_col] = 'first'
    ave = df.groupby(['Longitude', 'Latitude']).agg(agg).reset_index()

    # ---------- Mode setup ----------
    if mode == 'labeled':
        ave = ave[ave[label_col] > 0]
        cat_colours = cat_colours or {1: 'blue', 2: 'red'}
        categories = sorted(ave[label_col].unique())
    elif mode == 'binary':
        labels = set(ave[label_col])
        assert labels <= {0, 1}, f"Binary mode requires labels to be 0 or 1, found: {labels}"
        categories = [0, 1]
        cat_colours = cat_colours if cat_colours else {0: 'black', 1: 'gray'}
    elif mode == 'category':
        assert categories, "Categories are required for category mode"
        assert cat_colours, "Category colours are required for category mode"
    else:
        vmin, vmax = ave[variable].min(), ave[variable].max()

    # ---------- Convert to polar coordinates ----------
    def to_polar(df, pole):
        if df.empty:
            return df
        df = df.copy()
        df['r'] = 90 - df['Latitude'] if (pole == 'north') else 90 + df['Latitude']
        df['theta'] = np.deg2rad(df['Longitude'])
        return df
    
    north = to_polar(ave[ave['Latitude'] >= 0], 'north')
    south = to_polar(ave[ave['Latitude'] < 0], 'south')

    # ---------- Plotting ----------
    fig, (axN, axS) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(20, 10))

    def set_axes(ax, pole):
        ax.set_ylim(0, 15)
        ax.set_yticks(range(0, 16, 5))
        labels = [str(90 - x) if pole == 'north' else str(-90 + x) for x in range(0, 16, 5)]
        # ax.set_ylim(0, 5)
        # ax.set_yticks(range(0, 6, 1))
        # labels = [str(90 - x) if pole == 'north' else str(-90 + x) for x in range(0, 6, 1)]
    
        ax.set_yticklabels(labels)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        print("REMINDER: plotting only to 5° radius, not 15° - Change in set_axes()")

    def draw(ax, df, pole):
        if df.empty:
            fig.delaxes(ax)
            return
        set_axes(ax, pole)
        if mode == 'continuous':
            sc = ax.scatter(df['theta'], df['r'], c=df[variable], cmap='Greys_r', s=0.5, vmin=vmin, vmax=vmax)
            fig.colorbar(sc, ax=ax, label=variable)
        else:
            for cat in categories:
                sub = df[df[label_col] == cat]
                if not sub.empty:
                    ax.scatter(sub['theta'], sub['r'], label=f'{label_col}={cat}', c=cat_colours[cat], s=5)
                
            ax.legend(loc='upper right', title=label_col)
        ax.set_title(f'{variable} values - {pole.capitalize()} Pole')

    draw(axN, north, 'north')
    draw(axS, south, 'south')

    # ---------- Save or show ----------
    fname = f"{variable}_{mode}_{name_add}.png" if name_add else f"{variable}_{mode}.png"
    fpath = os.path.join(save_path, fname)

    kwargs = {'dpi': dpi or 300}
    if poster:
        kwargs['transparent'] = True

    plt.savefig(fpath, **kwargs)

    print(f"Plot saved to {fname}")
    plt.close(fig)

def plot_polar_overlay(
    base_df,
    overlay_df,
    variable,
    label_col,
    save_path,
    frac_base=0.1,
    frac_overlay=0.01,
    dpi=300,
    poster=True,
):
    # 1) Prepare the two DataFrames
    #   * sample them
    base = base_df.sample(frac=frac_base, random_state=42)
    over = overlay_df.sample(frac=frac_overlay, random_state=42)
    
    # 2) Compute polar coords
    def to_polar(df, pole):
        df = df.copy()
        df['r'] = (90 - df['Latitude']) if pole=='north' else (90 + df['Latitude'])
        df['theta'] = np.deg2rad(df['Longitude'])
        return df

    north_base = to_polar(base[base['Latitude'] >= 0], 'north')
    south_base = to_polar(base[base['Latitude'] <  0], 'south')
    north_over = to_polar(over[over['Latitude'] >= 0], 'north')
    south_over = to_polar(over[over['Latitude'] <  0], 'south')

    # 3) Make one figure
    fig, (axN, axS) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(20, 10), facecolor='none')

    for ax in (axN, axS):
        pole = 'north' if ax == axN else 'south'
        ax.set_facecolor('none')
        ax.set_ylim(0,15)
        ax.set_yticks(range(0,16,5))
        labels = [str(90 - x) if pole == 'north' else str(-90 + x) for x in range(0, 16, 5)]
        ax.set_yticklabels(labels)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    # 4) First, draw the background layer (interpolated → continuous greyscale)
    def draw_continuous(ax, df):
        sc = ax.scatter(
            df['theta'], df['r'],
            c=df[variable],
            cmap='Greys_r',
            s=5,
            alpha=0.6,
            zorder=1
        )
    
    draw_continuous(axN, north_base)
    draw_continuous(axS, south_base)

    # 5) Then, on the very same axes, draw your labeled overlay
    def draw_labels(ax, df):
        df = df[df[label_col] > 0]
        for cat, col in {1:'blue', 2:'red'}.items():    # or your cat_colours dict
            sub = df[df[label_col]==cat]
            ax.scatter(
                sub['theta'], sub['r'],
                c=col,
                s=10,
                label=f"{label_col}={cat}",
                zorder=2  # higher than the greyscale dots
            )
        ax.legend(loc='upper right', title=label_col)

    draw_labels(axN, north_over)
    draw_labels(axS, south_over)

    # 6) Save with transparent background
    fname = f"{variable}_overlay.png"
    kwargs = {'dpi':dpi}
    if poster:
        kwargs['transparent'] = True

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), **kwargs)
    plt.close(fig)
    print("Saved overlayed polar:", fname)

def generate_xy_mesh(
    RESOLUTION_M: float = 300.0,
    MOON_RADIUS_M: float = 1_737_400.0,   # metres
    LAT_MIN_DEG: float = 75.0,            # poleward limit of mesh
    MARGIN_DEG: float = 2.0               # extra margin on every 30-deg slice
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build azimuthal-equidistant (AEQD) x/y meshes for the lunar north- and
    south-pole caps in 12 longitude slices.

    Returns
    -------
    list[(mesh_north, mesh_south)]
        *mesh_north* / *mesh_south*:  (N, 2) arrays of projected (x, y) [metres]
    """

    # AEQD projectors (always_xy → order = lon, lat)
    aeqd_n = Proj(proj="aeqd", lat_0=+90, lon_0=0,
                  R=MOON_RADIUS_M, always_xy=True)
    aeqd_s = Proj(proj="aeqd", lat_0=-90, lon_0=0,
                  R=MOON_RADIUS_M, always_xy=True)

    # radial distance from the pole that corresponds to LAT_MIN_DEG
    lat_min_rad = np.deg2rad(LAT_MIN_DEG)
    r_max = MOON_RADIUS_M * (np.pi / 2 - lat_min_rad)

    # radial steps (0 … r_max) in metres
    rs = np.arange(0, r_max + RESOLUTION_M, RESOLUTION_M, dtype=np.float32)
    dtheta = RESOLUTION_M / r_max                         # angular step (rad)

    # 12 longitude slices centred every 30°
    lon_slices = [(i * 30, (i + 1) * 30) for i in range(12)]

    meshes = []
    for lon0, lon1 in lon_slices:
        lon0 = (lon0 - MARGIN_DEG) % 360      # widen slice by ±MARGIN_DEG
        lon1 = (lon1 + MARGIN_DEG) % 360
        if lon1 <= lon0:                      # handle wraparound past 360°
            lon1 += 360

        # angular positions (radians) across the slice
        thetas = np.arange(np.deg2rad(lon0),
                           np.deg2rad(lon1),
                           dtheta,
                           dtype=np.float32)

        # polar grid: R (m), θ (rad)  →  broadcast to 2-D arrays
        Rr, theta = np.meshgrid(rs, thetas, indexing="xy")

        # convert polar grid to geographic lon/lat
        lon_grid = np.rad2deg(theta) % 360                       # 0-360°
        lat_offset_deg = np.rad2deg(Rr / MOON_RADIUS_M)          # ρ → Δφ

        lat_n_grid = 90.0 - lat_offset_deg                       # north cap
        lat_s_grid = -90.0 + lat_offset_deg                      # south cap

        # forward projection to AEQD x/y (metres)
        Xn, Yn = aeqd_n(lon_grid, lat_n_grid)
        Xs, Ys = aeqd_s(lon_grid, lat_s_grid)

        mesh_north = np.column_stack((Xn.ravel(), Yn.ravel())).astype(np.float32)
        mesh_south = np.column_stack((Xs.ravel(), Ys.ravel())).astype(np.float32)

        meshes.append((mesh_north, mesh_south))

    total_pts = sum(mn.shape[0] + ms.shape[0] for mn, ms in meshes)
    print(f"XY-meshes created: {len(meshes)} slices, {total_pts:,} total points")

    return meshes



def psr_eda(data, save_dir, lbl_thresh=3):
    assert 'Label' in data.columns, "Missing 'label' column in DataFrame"
    assert 'psr' in data.columns, "Missing 'psr' column in DataFrame"

    y_true = data['psr']  # 'True' is psr values
    y_pred = (data['Label'] >= lbl_thresh).astype(int)  # 'Predicted' is label values

    # Compute performance metrics
    # HELPFUL
    precision = precision_score(y_true, y_pred, average='macro')    # How trustworthy is 'ice detected' value
    recall = recall_score(y_true, y_pred, average='macro')          # Fraction of PSRs which have ice detected
    f1 = f1_score(y_true, y_pred, average='macro')                  # optional single‑number trade‑off of precision vs recall, still informative because it is not class‑weighted
    mcc = matthews_corrcoef(y_true, y_pred)                         # Correlation coefficient between true and predicted labels
    iou = jaccard_score(y_true, y_pred)                             # Intersection over Union (IoU) score - what fraction of the union of “detected‑ice” and “PSR” pixels is common to both maps. That is exactly the overlap mission planners care about.

    # NOT HELPFUL
    # accuracy = accuracy_score(y_true, y_pred)                               # Inflated by ocean of TNs
    # acc_balanced = balanced_accuracy_score(y_true, y_pred)                  # Collapses to same as recall bc FP rate near zero
    # prec_weighted = precision_score(y_true, y_pred, average='weighted')     # Dominated by negatives so mask PSR performance
    # rec_weighted = recall_score(y_true, y_pred, average='weighted')         # Dominated by negatives so mask PSR performance
    # f1_weighted = f1_score(y_true, y_pred, average='weighted')              # Dominated by negatives so mask PSR performance

    print()
    print(f"PRECISION (How trustworthy is a positive value):        {precision:.4f}")
    print(f"RECALL (Portion of psrs with positive values):          {recall:.4f}")
    print(f"F1 SCORE (single-number trade-off between prec/rec):    {f1:.4f}")
    print(f"MCC (Correlation coefficient):                          {mcc:.4f}")
    print(f"IOU (frac of union of detected ice and PSR common to both maps): {iou:.4f}")

    print("Performance metrics not used: accuracy, acc_balanced, prec_weighted, rec_weighted, f1_weighted")



def from_csv_and_desc(data_dict, data_type, n=10):
    df_list = []

    for file in os.listdir(data_dict['save_path']):
        if file.endswith('.csv') and 'lon' in file:
            df_temp = load_every_nth_line(os.path.join(data_dict['save_path'], file), n)
            df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)

    # Cast lon and lat to float32 to save memory
    # For lunar coordinates, this moves from sub-micrometer accuracy to sub-centimeter accuracy
    df['Longitude'] = df['Longitude'].astype(np.float32)
    df['Latitude'] = df['Latitude'].astype(np.float32)

    print(f"{data_type} df:")
    print(df.describe())
    print()

    return df


def create_hist(df, name):
    plt.figure(figsize=(8, 6))
    plt.hist(df[name], bins=50, edgecolor='black')
    plt.title(f'Histogram of {name} data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(f'../../data/plots/{name}_hist.png')
    sys.stdout.flush()


if __name__ == "__main__":
    meshes = generate_xy_mesh()
    total = 0
    for i, (mesh_north, mesh_south) in enumerate(meshes):
        print(f"Slice {i}: North mesh shape: {mesh_north.shape}, South mesh shape: {mesh_south.shape}")
        total += mesh_north.shape[0] + mesh_south.shape[0]
    print(f"Total points in all slices: {total:,}")
