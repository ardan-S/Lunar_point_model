import pandas as pd
import os
from pathlib import Path
import numpy as np
import requests
import matplotlib.pyplot as plt
import sys

from utils.utils import get_metadata_value, clean_metadata_value, parse_metadata_content
from utils.utils import decode_image_file, get_closest_channels, plot_polar_data, save_by_lon_range


def process_LRO_image(image_file, address, metadata, data_type, max_val=1.0, min_val=0.0):
    # Extract image data
    file_path = Path(image_file)
    assert file_path.is_file(), f"File not found: {file_path}"

    file_extension = file_path.suffix[1:].lower()
    image_data = np.asarray(decode_image_file(file_path, file_extension, metadata, address))

    # Convert to scientific values
    scaling_factor = float(get_metadata_value(metadata, address, 'SCALING_FACTOR'))
    offset = float(get_metadata_value(metadata, address, 'OFFSET'))
    assert scaling_factor is not None or offset is not None, "Scaling factor and offset not found in metadata"

    missing_constants = {
        'LOLA': clean_metadata_value(metadata.get('MISSING_CONSTANT', -32768)),
        'MiniRF': clean_metadata_value(metadata.get('MISSING_CONSTANT', -1.7976931E+308)),
        'Diviner': clean_metadata_value(metadata.get('MISSING_CONSTANT', -32768))
    }

    missing_constant = missing_constants.get(data_type)
    assert missing_constant is not None, f"Missing constant not found for data type '{data_type}'"

    mask = (image_data != missing_constant)
    output_vals = np.where(mask, (image_data * scaling_factor) + offset, np.nan)    # Remove missing values BEFORE applying transform
    output_vals = np.where(output_vals > max_val, np.nan, output_vals)  # Remove extreme values
    output_vals = np.where(output_vals < min_val, np.nan, output_vals)

    return image_data, output_vals


def generate_LRO_coords(image_shape, metadata):
    lines, samples = image_shape
    projection_keys = ['LINE_PROJECTION_OFFSET', 'SAMPLE_PROJECTION_OFFSET', 'CENTER_LATITUDE',
                       'CENTER_LONGITUDE', 'MAP_RESOLUTION', 'MINIMUM_LATITUDE', 'MAXIMUM_LATITUDE']
    line_proj_offset, sample_proj_offset, center_lat, center_lon, map_res, min_lat, max_lat = \
        (float(get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', key)) for key in projection_keys)
    proj_type = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'MAP_PROJECTION_TYPE', string=True)
    a = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'A_AXIS_RADIUS') if center_lat == 0.0 else 1737.4   # Moon's radius in km

    # Generate pixel coordinates
    x = (np.arange(samples) - sample_proj_offset) / map_res
    y = (np.arange(lines) - line_proj_offset) / map_res
    x, y = np.meshgrid(x, y)

    if center_lat == 0.0:  # Equatorial (Mini-RF) - Simple Cylindrical
        assert proj_type == 'SIMPLE CYLINDRICAL', f"Unsupported projection type: {proj_type}"
        assert center_lat == 0.0 and center_lon == 0.0, "Center latitude and longitude must be 0.0 for Simple Cylindrical"
        lons = np.degrees(x / a)
        lon_scale = 360 / (2 * np.max(np.abs(lons)))  # Scale factor for longitudes
        lons = (lons * lon_scale + 360) % 360  # Apply scaling and wrap to [0, 360)

        # Map y directly to latitude range [-90, 90]
        lats = y * (max_lat - min_lat) / np.max(np.abs(y))  # Scale y to latitude range

    else:   # Polar Stereographic (LOLA, Diviner)
        assert proj_type == 'POLAR STEREOGRAPHIC', f"Unsupported projection type: {proj_type}"
        t = np.sqrt(x**2 + y**2)
        c = 2 * np.arctan(t / (2 * a))

        lons = center_lon + np.degrees(np.arctan2(y, x))
        lons = (center_lon + lons) % 360

        if center_lat == 90.0:  # North Pole
            lats = center_lat - np.degrees(c)
        elif center_lat == -90.0:  # South Pole
            lats = center_lat + np.degrees(c)
        else:
            raise ValueError(f"Center latitude is not supported: {center_lat}")

    # Adjust latitude range to min_lat to max_lat
    lat_scale = (max_lat - min_lat) / (np.max(lats) - np.min(lats))
    lats = min_lat + (lats - np.min(lats)) * lat_scale
    lats = np.clip(lats, min_lat, max_lat)

    return lons, lats


def load_lro_df(data_dict, data_type, debug=False):

    csvs_in_dir = [f for f in os.listdir(data_dict['file_path']) if f.endswith('.csv') and 'lon' in f]
    if len(csvs_in_dir) == 12:
        print(f"CSVs appear to exist for {data_type} data. Skipping load_lro_df.")
        return

    file_path = data_dict['file_path']
    address = data_dict['address']
    lbl_ext = data_dict['lbl_ext']
    csv_save_path = data_dict['save_path'] if 'save_path' in data_dict else None
    plot_save_path = data_dict['plot_path'] if 'plot_path' in data_dict else None
    max_val = data_dict['max']
    min_val = data_dict['min']

    assert csv_save_path or plot_save_path, "At least one of 'save_path' or 'plot_path' must be provided."
    assert isinstance(data_type, str), "data_type must be a string."

    lbl_files = [f for f in os.listdir(file_path) if f.endswith(lbl_ext)]

    all_lons = []
    all_lats = []
    all_output_vals = []

    for lbl_file in lbl_files:
        lbl_path = f"{file_path}/{lbl_file}"
        metadata = parse_metadata_content(lbl_path)

        img_file = lbl_path.replace(data_dict['lbl_ext'], data_dict['img_ext'])

        image_data, output_vals = process_LRO_image(img_file, address, metadata, data_type, max_val, min_val)
        lons, lats = generate_LRO_coords(image_data.shape, metadata)

        all_lons.extend(lons.flatten())
        all_lats.extend(lats.flatten())
        all_output_vals.extend(output_vals.flatten())

    df = pd.DataFrame({
        'Longitude': all_lons,
        'Latitude': all_lats,
        data_type: all_output_vals
    })

    assert not np.any((df['Longitude'] < 0) | (df['Longitude'] > 360)), "Some longitude values are out of bounds."
    assert not np.any((df['Latitude'] < -90) | (df['Latitude'] > 90)), "Some latitude values are out of bounds."
    df = df[((df['Latitude'] <= -75) & (df['Latitude'] >= -90)) | ((df['Latitude'] <= 90) & (df['Latitude'] >= 75))]    # Filter out non-polar data

    df.loc[(df[data_type] < min_val) | (df[data_type] > max_val), data_type] = np.nan
    print(f"Number of NaN values after clipping: {df[data_type].isna().sum()}")
    df = df.dropna(subset=[data_type])

    if csv_save_path:
        save_by_lon_range(df, csv_save_path)

    if plot_save_path:
        plot_polar_data(df, data_type, frac=0.25, save_path=plot_save_path)

    if debug:
        print(f"{data_type} df:")
        print(df.describe())
        create_hist(df, data_type)


def process_M3_image(image_file, address, metadata):

    lines = int(get_metadata_value(metadata, address, 'LINES'))
    line_samples = int(get_metadata_value(metadata, address, 'LINE_SAMPLES'))
    bands = int(get_metadata_value(metadata, address, 'BANDS'))
    sample_bits = int(get_metadata_value(metadata, address, 'SAMPLE_BITS'))
    sample_type = str(get_metadata_value(metadata, address, 'SAMPLE_TYPE'))
    invalid_constant = clean_metadata_value(metadata.get('INVALID_CONSTANT', -999.0))

    target_wavelengths = [1300, 1500, 2000]     # Target wavelengths taken from Brown et al. (2022)
    test_channels = get_closest_channels(metadata, address, target_wavelengths)

    assert len(set(test_channels)) == len(test_channels), "Adjacent channels found in the closest channels list. Not supported."
    assert np.all((test_channels >= 1) & (test_channels <= bands)), "Channel index out of bounds"
    assert image_file.split('.')[-1].lower() == 'img', f"Unsupported file extension: {image_file.split('.')[-1].lower()}"

    if sample_type == 'PC_REAL' and sample_bits == 32:
        dtype = '<f4'   # Little-endian 32-bit float (as in M3 documentation)
    else:
        raise ValueError(f"Unsupported combination of SAMPLE_TYPE: {sample_type} and SAMPLE_BITS: {sample_bits}")

    with open(image_file, 'rb') as f:
        image_data = np.fromfile(f, dtype=dtype)

    assert (lines * line_samples * bands) == image_data.size, f"Mismatch in data size: expected {lines * line_samples * bands}, got {image_data.size}"

    extracted_bands = np.empty((lines, line_samples, len(test_channels)))
    reference_band_up = np.full((lines, line_samples, len(test_channels)), np.nan)
    reference_band_down = np.full((lines, line_samples, len(test_channels)), np.nan)

    for idx, channel in enumerate(test_channels):
        channel_start_idx = channel - 1    # Convert to 0-based index

        for i in range(lines):
            start_idx = (i * bands + channel_start_idx) * line_samples
            end_idx = start_idx + line_samples
            extracted_bands[i, :, idx] = image_data[start_idx:end_idx]

            up_start_idx = (i * bands + channel_start_idx + 1) * line_samples
            up_end_idx = up_start_idx + line_samples
            reference_band_up[i, :, idx] = image_data[up_start_idx:up_end_idx]

            down_start_idx = (i * bands + channel_start_idx - 1) * line_samples
            down_end_idx = down_start_idx + line_samples
            reference_band_down[i, :, idx] = image_data[down_start_idx:down_end_idx]

    del image_data

    # Element-wise operation to handle NaNs and sum the bands
    reference_bands = np.where(np.isnan(reference_band_up), 2 * reference_band_down,
                               np.where(np.isnan(reference_band_down), 2 * reference_band_up,
                                        reference_band_up + reference_band_down))

    extracted_bands = np.where((extracted_bands == invalid_constant) | (extracted_bands == 2 * invalid_constant), np.nan, extracted_bands)
    reference_bands = np.where((reference_bands == invalid_constant) | (reference_bands == 2 * invalid_constant), np.nan, reference_bands)
    troughs = np.where((extracted_bands < 1e-6) | (extracted_bands > 1.5), np.nan, extracted_bands)
    shoulders = np.where((reference_bands < 1e-6) | (reference_bands > 1.5), np.nan, reference_bands)

    BDRs = shoulders / (2 * troughs)    # Band depth ratio
    output_vals = np.min(BDRs, axis=2)    # Take the band with the minimum BDR for each point
    max = 1.75
    output_vals = np.clip(output_vals, None, max)  # Clip values to [0, max]

    return extracted_bands, output_vals


def generate_M3_coords(image_shape, metadata, data_dict):
    text_file = 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0003_md5.txt'
    response = requests.get(text_file)
    lines = response.text.splitlines()

    loc_file_dir = data_dict['file_path']
    loc_lbl_ext = data_dict['loc_lbl_ext']
    loc_img_ext = data_dict['loc_img_ext']

    loc_img_name = str(get_metadata_value(metadata, '', 'CH1:PIXEL_LOCATION_FILE_NAME', string=True))
    loc_lbl_name = loc_img_name.replace(loc_img_ext, loc_lbl_ext)

    loc_img_path = os.path.join(loc_file_dir, loc_img_name)
    loc_lbl_path = os.path.join(loc_file_dir, loc_lbl_name)

    assert os.path.isfile(loc_img_path), f"Location image file not found: {loc_img_path}"
    assert os.path.isfile(loc_lbl_path), f"Location label file not found: {loc_lbl_path}"

    # loc_lbl_urls = [os.path.join(loc_file_dir, f) for f in os.listdir(loc_file_dir) if f.endswith(data_dict['loc_lbl_ext'])]
    # loc_img_urls = [os.path.join(loc_file_dir, f) for f in os.listdir(loc_file_dir) if f.endswith(data_dict['loc_img_ext'])]


    # loc_lbl_file = None
    # loc_img_file = None

    # # Loop through the lines and find the one that contains the loc_file_name
    # for loc_lbl_url in loc_lbl_urls:
    #     if loc_lbl_name in loc_lbl_url:
    #         loc_lbl_file = loc_lbl_url
    #         break

    # # Loop through lines in the text file and find the one that contains the loc_img_name
    # for loc_img_url in loc_img_urls:
    #     if loc_img_name in loc_img_url:
    #         loc_img_file = loc_img_url
    #         break

    # if not loc_lbl_file or not loc_img_file:
    #     raise ValueError(f"Location file not found for {loc_lbl_name} or {loc_img_name}")

    # with open(loc_lbl_path, 'rb') as f:
    #     loc_metadata = parse_metadata_content(f.read())
    loc_metadata = parse_metadata_content(loc_lbl_path)
    
    loc_address = data_dict['loc_address']

    lines = int(get_metadata_value(loc_metadata, loc_address, 'LINES'))
    line_samples = int(get_metadata_value(loc_metadata, loc_address, 'LINE_SAMPLES'))
    bands = int(get_metadata_value(loc_metadata, loc_address, 'BANDS'))
    sample_bits = int(get_metadata_value(loc_metadata, loc_address, 'SAMPLE_BITS'))
    sample_type = str(get_metadata_value(loc_metadata, loc_address, 'SAMPLE_TYPE'))

    if loc_lbl_path.split('.')[-1].lower() != 'lbl':
        raise ValueError("Unsupported file extension: {}".format(loc_lbl_path.split('.')[-1].lower()))

    dtype = {
        (32, 'PC_REAL'): np.float32,
        (64, 'PC_REAL'): np.float64
    }.get((sample_bits, sample_type))

    if dtype is None:
        raise ValueError(f"Unsupported combination of SAMPLE_TYPE: {sample_type} and SAMPLE_BITS: {sample_bits}")

    if lines is None or line_samples is None or bands is None:
        raise ValueError("Missing metadata values for lines, line_samples or bands")

    with open(loc_img_path, 'rb') as f:
        loc_data = np.fromfile(f, dtype='<f8')

    if loc_data.size != lines * line_samples * bands:
        raise ValueError(f"Mismatch in data size: expected {lines * line_samples * bands}, got {loc_data.size}")
    
    # METHOD 1
    # loc_data = loc_data.reshape((bands, lines, line_samples))
    # lons = loc_data[0]
    # lats = loc_data[1]
    # radii = loc_data[2]

    # METHOD 2
    lons = np.empty((lines, line_samples))
    lats = np.empty((lines, line_samples))
    radii = np.empty((lines, line_samples))

    index = 0

    for i in range(lines):
        for arr in (lons, lats, radii):
            arr[i, :] = loc_data[index:index + line_samples]
            index += line_samples

    # Raise if any lon or lat values are out of bounds
    if not (np.all((0 <= lons) & (lons <= 360)) and np.all((-90 <= lats) & (lats <= 90))):
        raise ValueError("Some coordinate values are out of bounds.")

    reference_elevation = 1737400   # https://pds-imaging.jpl.nasa.gov/documentation/Isaacson_M3_Workshop_Final.pdf (pg.26, accessed 30/07/2024)
    elev = radii - reference_elevation

    return lons, lats, elev


def load_m3_df(data_dict, debug=False):

    csvs_in_dir = [f for f in os.listdir(data_dict['file_path']) if f.endswith('.csv') and 'lon' in f]
    if len(csvs_in_dir) == 12:
        print(f"CSVs appear to exist for M3 data. Skipping load_lro_df.")
        return

    file_path = data_dict['file_path']
    address = data_dict['address']
    lbl_ext = data_dict['lbl_ext']
    plot_save_path = data_dict['plot_path'] if 'plot_path' in data_dict else None
    csv_save_path = data_dict['save_path'] if 'save_path' in data_dict else None

    assert plot_save_path or csv_save_path, "At least one of 'plot_path' or 'save_path' must be provided."

    lbl_files = [f for f in os.listdir(file_path) if f.endswith(lbl_ext)]

    all_lons = []
    all_lats = []
    all_elev = []
    all_output_vals = []

    for lbl_file in lbl_files:
        lbl_path = f"{file_path}/{lbl_file}"
        metadata = parse_metadata_content(lbl_path)
        img_file = lbl_path.replace(data_dict['lbl_ext'], data_dict['img_ext'])

        image_data, output_vals = process_M3_image(img_file, address, metadata)
        lons, lats, elev = generate_M3_coords(image_data.shape, metadata, data_dict)

        all_lons.extend(lons.flatten())
        all_lats.extend(lats.flatten())
        all_elev.extend(elev.flatten())
        all_output_vals.extend(output_vals.flatten())

    df = pd.DataFrame({
        'Longitude': all_lons,
        'Latitude': all_lats,
        'Elevation': all_elev,
        'M3': all_output_vals
    })

    if csv_save_path:
        save_by_lon_range(df, csv_save_path)

    if plot_save_path:
        plot_polar_data(df, 'M3', frac=0.25, save_path=plot_save_path)

    if debug:
        print(f"M3 df:")
        print(df.describe())
        create_hist(df, 'M3')


def create_hist(df, name):
    plt.figure(figsize=(8, 6))
    plt.hist(df[name], bins=50, edgecolor='black')
    plt.title(f'Histogram of {name} data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(f'../../data/plots/{name}_hist.png')
    sys.stdout.flush()
